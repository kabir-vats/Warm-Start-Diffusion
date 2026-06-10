from __future__ import annotations

import os
import shutil
import time
import traceback
from typing import Dict, Sequence

import numpy as np
import pytorch_lightning as pl
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from src.utilities.utils import get_logger


log = get_logger(__name__)


class WatchModel(Callback):
    """
    Make wandb watch model at the beginning of the run.
    This will log the gradients of the model (as a histogram for each or some weights updates).
    """

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log_type = log
        self.log_freq = log_freq
        self.has_logged = False

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger: WandbLogger = get_wandb_logger(trainer=trainer)
        if not self.has_logged:
            try:
                logger.watch(model=pl_module, log=self.log_type, log_freq=self.log_freq, log_graph=True)
            except TypeError:
                log.info(
                    f" Pytorch-lightning/Wandb version seems to be too old to support 'log_graph' arg in wandb.watch(.)"
                    f" Wandb version={wandb.__version__}"
                )
                wandb.watch(models=pl_module, log=self.log_type, log_freq=self.log_freq)  # , log_graph=True)
            self.has_logged = True

    @rank_zero_only
    def on_any_non_train_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if not self.has_logged:
            log.info("WatchModel callback has not been called yet. Calling it now & setting log_freq=0")
            self.log_freq = 0
            # self.log_type = "parameters"
            self.on_train_start(trainer, pl_module)
            # log.info(f"wandb.run.hook_handles: {wandb.run._torch_history._hook_handles.keys()}")
            param_hook_handle = wandb.run._torch_history._hook_handles.get("parameters/")  # a RemovableHandle
            # log.info(param_hook_handle.hooks_dict_ref())
            param_hook = param_hook_handle.hooks_dict_ref()[param_hook_handle.id]
            # param_hook(pl_module, None, None)
            param_hook(pl_module, None, None)
            self.has_logged = True

    @rank_zero_only
    def on_validation_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        return self.on_any_non_train_start(trainer, pl_module)

    @rank_zero_only
    def on_test_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        return self.on_any_non_train_start(trainer, pl_module)


class SummarizeBestValMetric(Callback):
    """Make wandb log in run.summary the best achieved monitored val_metric as opposed to the last"""

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger: WandbLogger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment
        # When using DDP multi-gpu training, one usually needs to get the actual model by .module, and
        # trainer.model.module.module will be the same as pl_module

        model = pl_module  # .module if isinstance(trainer.model, DistributedDataParallel) else pl_module
        experiment.define_metric(model.monitor, summary=model.hparams.mode)
        experiment.define_metric(f"{model.monitor}_epoch", summary=model.hparams.mode)
        # Store the maximum epoch at all times
        # The following leads to a weird error in wandb file:
        #   /opt/conda/lib/python3.8/site-packages/wandb/sdk/internal/handler.py
        # where, this is an example print out before the problematic line 62:
        # log.info(target, v, key_list) --> 0 0 ('epoch', 'max')
        # experiment.define_metric('epoch', summary='max')


def save_arrays_as_line_plot(
    lightning_module: pl.LightningModule,
    x_array: np.ndarray,
    key_to_array: Dict[str, np.ndarray],
    wandb_key_stem: str,
    x_label: str = "x",
    log_as_step: bool = True,
    log_as_table: bool = False,
    update_summary: bool = True,
):
    # we want to zip the arrays together into the columns of a table
    step = lightning_module.trainer.global_step
    key_to_array_actual = {f"{wandb_key_stem}/{key}".replace("//", "/"): y for key, y in key_to_array.items()}
    if log_as_table:
        if not hasattr(lightning_module, "logger") and hasattr(lightning_module.logger, "experiment"):
            return
        log_dict = dict()
        for wandb_key, y_array in key_to_array_actual.items():
            y_label = wandb_key.split("/")[-1]
            data = [[x, y] for x, y in zip(x_array, y_array)]
            table = wandb.Table(data=data, columns=[x_label + "_x", y_label])

            log_dict["plot_" + wandb_key] = wandb.plot.line(table, x_label + "_x", y_label, title=wandb_key)
        lightning_module.logger.experiment.log(log_dict, step=step)

    if log_as_step:
        # define our custom x axis metric
        wandb.define_metric(x_label)

        # define which metrics will be plotted against it
        for wandb_key, y_array in key_to_array_actual.items():
            wandb.define_metric(wandb_key, step_metric=x_label)

        # now zip the arrays together, and log each step together
        for i, (x, *y) in enumerate(zip(x_array, *key_to_array.values()), start=1):
            lightning_module.logger.experiment.log(
                {x_label: x, **{key: y for key, y in zip(key_to_array_actual.keys(), y)}}
            )
            if update_summary:
                lightning_module.logger.experiment.summary.update(
                    {
                        f"{wandb_key_stem}/t{i}/{key_name}".replace("//", "/"): y_value
                        for key_name, y_value in zip(key_to_array.keys(), y)
                    }
                )

        # y_label2 = wandb_metric_stem if y_label in wandb_metric_stem else f'{wandb_metric_stem}/{y_label}'
        # for x, y in zip(x_array, y_array):
        #     lightning_module.logger.experiment.log({y_label2: y, x_label: x})


class MyWandbLogger(pl.loggers.WandbLogger):
    """Same as pl.WandbLogger, but also saves the last checkpoint as 'last.ckpt' and uploads it to wandb."""

    def __init__(
        self,
        save_last_ckpt: bool = True,
        save_best_ckpt: bool = True,
        save_to_wandb: bool = True,
        save_to_s3_bucket: bool = False,
        s3_endpoint_url: str = None,
        s3_bucket_name: str = None,
        log_code: bool = True,
        run_path: str = None,
        train_run_path: str = None,
        training_id: str = None,
        resume_run_id: str = None,
        **kwargs,
    ):
        """
        If using S3, set save_to_s3_bucket=True and provide your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY as
        environment variables.

        If save_best_ckpt=True or save_last_ckpt=True, one of save_to_wandb or save_to_s3_bucket must be True ...
        The corresponding checkpoint will be saved to wandb or S3 bucket.
        """
        try:
            super().__init__(**kwargs)
        except Exception as e:
            log.info("-" * 100)
            log.warning(f"Failed to initialize WandbLogger. Error: {e}")
            service_wait = os.getenv("WANDB__SERVICE_WAIT", 300)
            new_service_wait = int(service_wait) + 600
            os.environ["WANDB__SERVICE_WAIT"] = str(new_service_wait)
            wandb_version = wandb.__version__
            log.info(f"Increasing WANDB__SERVICE_WAIT to {new_service_wait} seconds (wandb version={wandb_version}).")
            log.info("-" * 100)
            # Sleep for 30sec and try again
            time.sleep(30)
            super().__init__(**kwargs)
        _ = self.experiment  # Force initialize wandb run (same as wandb.init)
        if hasattr(self.experiment.project, "lower") and self.experiment.project.lower() == "debug":
            save_best_ckpt = False
            save_last_ckpt = False
            save_to_s3_bucket = False
            save_to_wandb = False
            log.info("Wandb: Running in debug mode. Disabling saving checkpoints.")
        if log_code:

            def exclude_codes(path, root):
                if path.endswith("mcvd") or path.endswith("sfno") or path.endswith("schedulers"):
                    return True  # exclude these directories
                exclude_subdirs = [
                    "plotting",
                    "pdearena_conditioned",
                    "dnnlib",
                    "torch_utils",
                    "models/mcvd",
                    "diffusion/schedulers",
                ]
                if any([subdir in path for subdir in exclude_subdirs]):
                    return True
                return False

            code_dir = os.path.join(os.getcwd(), "src")  # "../../src"
            self.experiment.log_code(code_dir, exclude_fn=exclude_codes)  #  # saves python files in src/ to wandb

        self.save_last_ckpt = save_last_ckpt
        self.save_best_ckpt = save_best_ckpt
        self._hash_of_best_ckpts = dict()
        if save_best_ckpt or save_last_ckpt:
            assert save_to_wandb or save_to_s3_bucket, "You must save to either wandb or S3 bucket."
        self.save_to_wandb = save_to_wandb
        self.save_to_s3_bucket = save_to_s3_bucket
        if save_to_s3_bucket:
            if s3_endpoint_url is not None:
                if os.getenv("S3_ENDPOINT_URL") is not None:
                    assert (
                        os.getenv("S3_ENDPOINT_URL") == s3_endpoint_url
                    ), "S3_ENDPOINT_URL environment variable mismatch."
                os.environ["S3_ENDPOINT_URL"] = s3_endpoint_url
            if s3_bucket_name is not None:
                if os.getenv("S3_BUCKET_NAME") is not None:
                    assert (
                        os.getenv("S3_BUCKET_NAME") == s3_bucket_name
                    ), "S3_BUCKET_NAME environment variable mismatch."

                os.environ["S3_BUCKET_NAME"] = s3_bucket_name

            from src.utilities.s3utils import download_s3_object, upload_s3_object

            # Save S3 checkpoints to <Wandb-project-name>/<Wandb-run-id>/checkpoints/
            self.s3_checkpoint_dir = f"{self._project}/checkpoints/{self._id}/"
            self.download_s3_object = download_s3_object
            self.upload_s3_object = upload_s3_object
            self.summary_update(
                {
                    "checkpoint/s3_dir": self.s3_checkpoint_dir,
                    "checkpoint/s3_endpoint_url": s3_endpoint_url,
                    "checkpoint/s3_bucket_name": s3_bucket_name,
                }
            )
            self.upload_wandb_files_to_s3()

    @rank_zero_only
    def summary_update(self, summary_dict: dict):
        self.experiment.summary.update(summary_dict)

    @rank_zero_only
    def upload_wandb_files_to_s3(self):
        if not self.save_to_s3_bucket:
            return
        import boto3

        # Upload all files in wandb.run.dir to S3 bucket (e.g. hydra config files)
        dir_to_upload = wandb.run.dir
        if os.path.exists(dir_to_upload):
            for file in os.listdir(dir_to_upload):
                s3_file_path = os.path.join(f"{self._project}/configs/{self._id}", file)
                try:
                    self.upload_s3_object(os.path.join(dir_to_upload, file), s3_file_path, retry=3)
                except boto3.exceptions.S3UploadFailedError as e:
                    if "hydra_config" not in file:
                        log.error(f"Failed to upload {file} to S3 bucket. Skipping.")
                    else:
                        raise e
                # log.info(f"Uploaded {file} to S3 bucket as {s3_file_path}.")
        else:
            log.warning(f"Directory {dir_to_upload} does not exist. Skipping uploading to S3 bucket.")

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        super().after_save_checkpoint(checkpoint_callback)
        self.save_last(checkpoint_callback)
        self.save_best(checkpoint_callback)

    @rank_zero_only
    def save_last(self, ckpt_cbk):
        if not self.save_last_ckpt:
            return
        if isinstance(ckpt_cbk, Sequence):
            ckpt_cbk = [c for c in ckpt_cbk if c.last_model_path]
            if len(ckpt_cbk) == 0:
                raise Exception("No checkpoint callback has a last_model_path attribute. Ckpt callback is: {ckpt_cbk}")
            ckpt_cbk = ckpt_cbk[0]

        last_ckpt = ckpt_cbk.last_model_path
        if self.save_last and last_ckpt:
            hash_last_ckpt = hash(open(last_ckpt, "rb").read())
            if hash_last_ckpt == self._hash_of_best_ckpts.get("LAST_CKPT", None):
                return
            self._hash_of_best_ckpts["LAST_CKPT"] = hash_last_ckpt
            if self.save_to_wandb:
                self.experiment.save(last_ckpt)
            if self.save_to_s3_bucket:
                # Upload to S3 bucket
                self.upload_s3_object(local_file_path=last_ckpt, s3_file_path=self.s3_checkpoint_dir)
                try:
                    self.experiment.summary.update({"checkpoint/in_s3": True})
                except Exception as e:
                    log.error(f"Failed to update wandb summary. Error: {e}")

            self.experiment.summary.update({"checkpoint/last_filepath": last_ckpt})

    @rank_zero_only
    def save_best(self, ckpt_cbk):
        if not self.save_best_ckpt:
            return
        # Save best model
        if not isinstance(ckpt_cbk, Sequence):
            ckpt_cbk = [ckpt_cbk]

        for ckpt_cbk in ckpt_cbk:
            best_ckpt = ckpt_cbk.best_model_path
            if not best_ckpt or not os.path.isfile(best_ckpt):
                continue
            # Check if the best ckpt content has changed since last time it was uploaded
            hash_best_ckpt = hash(open(best_ckpt, "rb").read())
            unique_key_for_callback = f"{ckpt_cbk.monitor}"
            if hash_best_ckpt == self._hash_of_best_ckpts.get(unique_key_for_callback, None):
                continue
            self._hash_of_best_ckpts[unique_key_for_callback] = hash_best_ckpt
            # copy best ckpt to a file called 'best.ckpt' and upload it to wandb
            monitor = ckpt_cbk.monitor.replace("/", "_") if ckpt_cbk.monitor is not None else "MONITOR_NOT_SET"
            fname_cloud = f"best-{monitor}.ckpt"
            shutil.copyfile(best_ckpt, fname_cloud)
            if self.save_to_wandb:
                self.experiment.save(fname_cloud)
                # log.info(f"Wandb: Saved best ckpt '{best_ckpt}' as '{fname_wandb}'.")
                # log.info(f"Saved best ckpt to the wandb cloud as '{fname_wandb}'.")
            if self.save_to_s3_bucket:
                # Upload to S3 bucket
                self.upload_s3_object(local_file_path=fname_cloud, s3_file_path=self.s3_checkpoint_dir)
                try:
                    self.experiment.summary.update({"checkpoint/in_s3": True})
                except Exception as e:
                    log.error(f"Failed to update wandb summary. Error: {e}")

            self.experiment.summary.update({f"checkpoint/best_filepath_{monitor}": best_ckpt})

    def restore_checkpoint(
        self,
        ckpt_filename: str,
        local_file_path: str,
        run_path: str = None,
        root: str = None,
        restore_from: str = None,
    ):
        """Restore a checkpoint from cloud to local file path.

        Args:
            ckpt_filename: The name of the checkpoint file.
            local_file_path: The local file path to save the checkpoint to.
            run_path: The path to the wandb run where the checkpoint is stored (or in corresponding S3 bucket).
            root: The root directory to save the checkpoint, if using wandb restore.
            restore_from: The source to restore the checkpoint from. Can be 's3', 'wandb' or 'any'.

        Note:
            If save_to_s3_bucket is True, the checkpoint will be downloaded from the S3 bucket.
            Otherwise, if save_to_wandb is True, the checkpoint will be downloaded from wandb.

        """
        if run_path is None:
            run_path = f"{self.experiment.entity}/{self._project}/{self._id}"
        entity, project, run_id = run_path.split("/")

        if restore_from is None:
            if self.save_to_s3_bucket:
                restore_from = "s3"
            elif self.save_to_wandb:
                restore_from = "wandb"
            else:
                raise RuntimeError(
                    "Cannot restore checkpoint since neither save_to_wandb nor save_to_s3_bucket is True."
                    "Alternatively, set ``restore_from`` to 's3' or 'wandb' or 'any' to restore from either source."
                )
        assert restore_from in ["s3", "wandb", "any"], f"Invalid value for restore_from: {restore_from}"

        ckpt_filename = os.path.basename(ckpt_filename)  # remove any path prefix (e.g. local dir)
        if restore_from in ["s3", "any"]:
            s3_file_path = f"{project}/checkpoints/{run_id}/{ckpt_filename}"
            retries = 3
            for i in range(retries):
                try:
                    self.download_s3_object(s3_file_path, local_file_path)
                    return local_file_path
                except Exception as e1:
                    if i == retries - 1:
                        log.error(
                            f"Attempt {i}: Failed to download checkpoint from S3 bucket path ``{s3_file_path}``."
                            f"Error: {e1}.\n{traceback.format_exc()}"
                        )
                        e1_str = traceback.format_exc()
                    else:
                        log.warning(
                            f"Attempt {i}: Failed to download checkpoint from S3 bucket path ``{s3_file_path}``. Retrying..."
                        )

        if restore_from in ["wandb", "any"]:
            root = root or os.getcwd()
            # Download model checkpoint from wandb
            try:
                ckpt_path_tmp = wandb.restore(ckpt_filename, run_path=run_path, replace=True, root=root).name
                os.rename(ckpt_path_tmp, local_file_path)
                return local_file_path
            except Exception as e2:
                log.error(f"Failed to download checkpoint from wandb run {run_path}. Error: {e2}")
                e2_str = traceback.format_exc()

        s3_path_str = f"S3 bucket path: {s3_file_path}"
        wb_path_str = f"wandb run path: {run_path}"
        if restore_from == "any":
            suffix = f"S3 bucket or wandb. {s3_path_str} and {wb_path_str}.\nError S3: {e1_str}\n:Error WB: {e2_str}"
        elif restore_from == "s3":
            suffix = f"S3 bucket. {s3_path_str}. Error: {e1_str}"
        elif restore_from == "wandb":
            suffix = f"Wandb run. {wb_path_str}. Error: {e2_str}"
        raise RuntimeError(f"Failed to restore checkpoint from {suffix}.")


def get_wandb_logger(trainer: Trainer) -> WandbLogger | MyWandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, (WandbLogger, MyWandbLogger)):
        return trainer.logger

    if isinstance(trainer.loggers, list):
        for logger in trainer.loggers:
            if isinstance(logger, (WandbLogger, MyWandbLogger)):
                return logger

    raise Exception("You are using wandb related callback, but WandbLogger was not found for some reason...")
