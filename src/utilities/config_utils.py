import os
import sys
import warnings
from datetime import datetime
from typing import List, Sequence, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import requests
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only
from copy import deepcopy
from omegaconf.errors import ConfigAttributeError, ConfigKeyError, ValidationError
import re


from src.utilities import wandb_api
from src.utilities.checkpointing import reload_checkpoint
from src.utilities.naming import clean_name, get_detailed_name, get_group_name
from src.utilities.utils import get_logger
from src.utilities.wandb_api import get_existing_wandb_group_runs, get_run_api

_OVERRIDE_RE = re.compile(r"^(?P<key>[^=]+?)(?:=(?P<val>.*))?$")

def _split_override(override: str):
    m = _OVERRIDE_RE.match(override.strip())
    if not m:
        return None, None
    return m.group("key").strip(), m.group("val")

def filter_overrides_like_hydra(cfg, overrides, verbose=True):
    """
    Returns (kept_overrides, dropped_overrides).

    Strategy:
      - Keep Hydra config-group overrides (e.g. experiment=foo, trainer=default) via heuristic.
      - Keep package overrides (x@y=z).
      - Keep + / ++ add-overrides (explicitly allowed to create new keys).
      - For remaining key=value overrides, attempt OmegaConf.update on a copy; drop on struct errors.
    """
    kept, dropped = [], []
    test_cfg = deepcopy(cfg)

    for o in overrides:
        key, val = _split_override(o)

        # Non key=value forms: keep (Hydra may parse them specially)
        if key is None or val is None:
            kept.append(o)
            continue

        key = key.strip()

        # Hydra package override: scheduler@module.scheduler=cosine
        if "@" in key:
            kept.append(o)
            continue

        # Explicit "add" overrides are intended to create new keys; keep them
        if key.startswith("+"):
            kept.append(o)
            continue

        raw_key = key

        # Heuristic: config-group override like experiment=foo, trainer=default, model=adm_old, datamodule=_old...
        # These often do NOT correspond to an OmegaConf node yet; Hydra resolves them via config groups/defaults.
        if "." not in raw_key:
            # if it's already a real node in cfg, we can validate normally; otherwise assume config-group and keep
            if OmegaConf.select(test_cfg, raw_key, default="__MISSING__") == "__MISSING__":
                kept.append(o)
                continue

        # Validate by applying to a running copy (closest to Hydra behavior in struct mode)
        try:
            OmegaConf.update(test_cfg, raw_key, val, merge=True)
            kept.append(o)
        except (ConfigAttributeError, ConfigKeyError, ValidationError) as e:
            dropped.append(o)
            if verbose:
                print(f"Dropping invalid override: {o}  [{type(e).__name__}]")

    return kept, dropped

log = get_logger(__name__)


@rank_zero_only
def print_config(
    config,
    fields: Union[str, Sequence[str]] = (
        "datamodule",
        "model",
        "trainer",
        # "callbacks",
        # "logger",
        "seed",
    ),
    resolve: bool = True,
    rich_style: str = "magenta",
    max_width: int = 128,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure (if installed: ``pip install rich``).

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Args:
        config (ConfigDict): Configuration
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        rich_style (str, optional): Style of Rich library to use for printing. E.g "magenta", "bold", "italic", etc.
    """
    import importlib

    if not importlib.util.find_spec("rich") or not importlib.util.find_spec("omegaconf"):
        # no pretty printing
        log.info(OmegaConf.to_yaml(config, resolve=resolve))
        return
    import rich.syntax  # IMPORTANT to have, otherwise errors are thrown
    import rich.tree

    tree = rich.tree.Tree(":gear: CONFIG", style=rich_style, guide_style=rich_style)
    if isinstance(fields, str):
        if fields.lower() == "all":
            fields = config.keys()
        else:
            fields = [fields]

    for field in fields:
        branch = tree.add(field, style=rich_style, guide_style=rich_style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    console = rich.console.Console(width=max_width)
    console.print(tree)


def extras(
    config: DictConfig,
    if_wandb_run_already_exists: str = "resume",
    allow_permission_error: bool = False,
) -> DictConfig:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration
    - checking if config values are valid
    - init wandb if wandb logging is being used
    - Merge config with wandb config if resuming a run

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    While this method modifies DictConfig mostly in place,
    please make sure to use the returned config as the new config, especially when resuming a run.

    Args:
        if_wandb_run_already_exists (str): What to do if wandb run already exists. Wandb logger must be enabled!
            Options are:
            - 'resume': resume the run
            - 'new': create a new run
            - 'abort': raise an error if run already exists and abort
        allow_permission_error (bool): Whether to allow PermissionError when creating working dir.
    """
    USE_WANDB = "logger" in config.keys() and config.logger.get("wandb") and hasattr(config.logger.wandb, "_target_")
    if USE_WANDB:
        run_api = None
        os.environ["WANDB_HTTP_TIMEOUT"] = "200"  # Increase timeout for slow connections
        wandb_cfg = config.logger.wandb
        wandb_api.PROJECT = config.logger.wandb.project = wandb_cfg.get("project", wandb_api.PROJECT)
        wandb_api._ENTITY = config.logger.wandb.entity = wandb_api.get_entity(wandb_cfg.get("entity"))

        if wandb_cfg.get("id") or wandb_cfg.get("resume_run_id"):
            wandb_status = "resume"
            if wandb_cfg.get("id"):
                assert not wandb_cfg.get(
                    "resume_run_id"
                ), "Both wandb.id and wandb.resume_run_id are set. Only one should be set."
                resume_run_id = str(wandb_cfg.id)
                config.logger.wandb.id = resume_run_id
                log.info(f"Resuming experiment with wandb run ID = `{resume_run_id}`")
            else:
                assert not wandb_cfg.get(
                    "id"
                ), "Both wandb.id and wandb.resume_run_id are set. Only one should be set."
                resume_run_id = str(wandb_cfg.resume_run_id)
                config.logger.wandb.id = wandb_api.get_wandb_id_for_run(config.logger.wandb)
                log.info(
                    f"Resuming experiment with wandb run ID = {resume_run_id} on NEW run: `{config.logger.wandb.id}`"
                )

            run_api = get_run_api(
                run_id=resume_run_id, entity=config.logger.wandb.entity, project=config.logger.wandb.project
            )
            # Set config wandb keys in case they were none, to the wandb defaults
            keys_to_set = [k for k in wandb_cfg.keys() if k != "id"]
            for k in keys_to_set:
                config.logger.wandb[k] = getattr(run_api, k) if hasattr(run_api, k) else wandb_cfg[k]
            if resume_run_id != wandb_cfg.id:
                # Give a new name to the run based on config values (will be updated later)
                config.logger.wandb.name = None

        else:
            if not wandb_cfg.get("group"):  # no wandb group has been assigned yet
                group_name = get_group_name(config)
                # potentially truncate the group name to 128 characters (W&B limit)
                if len(group_name) >= 128:
                    group_name = group_name.replace("-fcond", "").replace("DynTime", "DynT")
                    group_name = group_name.replace("UNetResNet", "UNetRN")
                    group_name = group_name.replace("NavierStokes", "NS")
                    group_name = group_name.replace("DYffusion", "DY2s")
                    group_name = group_name.replace("SimpleUnet", "sUNet")
                    group_name = group_name.replace("lRecs_", "lRs_")
                if len(group_name) >= 128:
                    group_name = group_name.replace("_cos_LC10:400", "cosSTD")
                    group_name = group_name.replace("1-2-2-3-4", "12234")
                if len(group_name) >= 128:
                    group_name = group_name.replace("_cos_LC", "_cL")
                if len(group_name) >= 128:
                    group_name = group_name.replace("Kolmogorov-", "Kolg-")

                if len(group_name) > 128:
                    raise ValueError(f"Group name is too long, ({len(group_name)} > 128): {group_name}")
                config.logger.wandb.group = group_name
            group = config.logger.wandb.group

            if if_wandb_run_already_exists in ["abort", "resume"]:
                wandb_status = "new"
                runs_in_group = get_existing_wandb_group_runs(config, ckpt_must_exist=True, only_best_and_last=False)
                if len(runs_in_group) > 0:
                    log.info(f"Found {len(runs_in_group)} runs for group {group}")
                for other_run in runs_in_group:
                    import json
                    other_seed = json.loads(other_run.config).get("seed")
                    if other_seed is None:
                        # Name follows Kolmogorov-MH16_ar2_UNetR_EMA_64x1-2-3-4d_54lr_30at30b10b1Dr_14wd_cos_LC10:400_11seed_23h21mJul01_1634972
                        try:
                            split_seed = other_run.name.split("seed_")[0].split("_")[-1]
                            other_seed = int(split_seed) if split_seed.isdigit() else None
                        except Exception:
                            continue
                        if other_seed is None:
                            continue

                    if isinstance(other_seed, dict):
                        other_seed = other_seed.get("value")
                    if int(other_seed) != int(config.seed):
                        continue
                    # seeds are the same, so we treat this as a duplicate run
                    state = other_run.state
                    if if_wandb_run_already_exists == "abort":
                        raise RuntimeError(
                            f"Run with seed {config.seed} already exists in group {group}. State: {state}"
                        )
                    elif if_wandb_run_already_exists == "resume":
                        wandb_status = "resume"
                        run_id = other_run.id
                        config.logger.wandb.resume = "allow"  # was "allow" but "must" is more clear (?)
                        config.logger.wandb.id = run_id
                        config.logger.wandb.name = other_run.name
                        ckpt_filename = config.ckpt_path or "last.ckpt"
                        config.ckpt_path = reload_checkpoint(
                            run_id,
                            config=config,
                            ckpt_filename=ckpt_filename,
                            local_checkpoint_path=True,
                            ckpt_path_only=True,
                        )
                        log.info(f"Resuming run {run_id} from {group=}. Seed={other_seed}; State was: ``{state}``")
                    else:
                        raise ValueError(f"if_wandb_run_already_exists={if_wandb_run_already_exists} is not supported")
                    break
            elif if_wandb_run_already_exists in [None, "ignore"]:
                wandb_status = "resume"
            else:
                wandb_status = "???"

            if config.logger.wandb.get("id") is None:
                # no wandb id has been assigned yet
                config.logger.wandb.id = wandb_api.get_wandb_id_for_run(config.logger.wandb)

    elif if_wandb_run_already_exists in ["abort", "resume"]:
        wandb_status = "not_used"
        log.warning("Not checking if run already exists, since wandb logging is not being used")

    else:
        wandb_status = None

    if wandb_status == "resume":
        # Reload config from wandb
        run_path = f"{config.logger.wandb.entity}/{config.logger.wandb.project}/{config.logger.wandb.id}"

        # NEW CODE:
        if run_api is None:
            run_api = get_run_api(run_path=run_path)
        # original overrides + command line overrides (latter take precedence)

        overrides = ([] if (run_api.metadata is None or run_api.metadata.get("args") is None) else run_api.metadata["args"]) + (
            sys.argv[1:] if len(sys.argv) > 1 else []
        )
        # Drop legacy scheduler override keys that may be present in old run metadata but
        # are no longer valid in the current structured config schema.
        stale_resume_override_keys = {"module.scheduler.warmup_epochs"}
        overrides_kept, overrides_dropped = [], []
        for o in overrides:
            key, _ = _split_override(o)
            key_no_prefix = key.lstrip("+") if key is not None else None
            if key_no_prefix in stale_resume_override_keys:
                overrides_dropped.append(o)
                continue
            overrides_kept.append(o)
        overrides = overrides_kept
        if len(overrides_dropped) > 0:
            log.warning(f"Dropping stale resume overrides: {overrides_dropped}")

        GlobalHydra.instance().clear()
        with hydra.initialize(version_base=None, config_path="../configs"):
            # base_cfg = hydra.compose(config_name="main_config.yaml")  # no overrides yet

            # overrides, dropped = filter_overrides_like_hydra(base_cfg, overrides, verbose=True)

            new_config = hydra.compose(config_name="main_config.yaml", overrides=overrides)
        with open_dict(new_config):
            new_config.logger.wandb = config.logger.wandb
            new_config.ckpt_path = config.ckpt_path
        config = new_config

        # OLD CODE:
        # override_config = get_only_overriden_config(config)
        # with open_dict(override_config):
        # override_config.logger.wandb.resume = config.logger.wandb.resume
        # override_config.ckpt_path = config.ckpt_path
        # config = wandb_api.load_hydra_config_from_wandb(run_path, override_config=override_config)
        # END OF OLD CODE
        with open_dict(config):
            config.logger.wandb.run_path = run_path

    if USE_WANDB and not config.logger.wandb.get("name"):  # no wandb name has been assigned yet
        config.logger.wandb.name = get_detailed_name(config)
    # Edit some config values
    # Create working dir if it does not exist yet
    if config.get("work_dir"):
        if config.get("logger") and config.logger.get("wandb") and config.logger.wandb.get("id"):
            if str(config.logger.wandb.id) not in config.work_dir:
                config.work_dir = os.path.join(config.work_dir, str(config.logger.wandb.id))
                log.info(f"Changing work_dir to {config.work_dir} since wandb id is not in it.")
        try:
            os.makedirs(name=config.get("work_dir"), exist_ok=True)
        except PermissionError as e:
            if allow_permission_error:
                log.warning(f"PermissionError: {e}")
            else:
                log.info(
                    f"Please set ``work_dir`` to a valid path for which you have write permissions. Current: {config.work_dir}"
                )
                raise e

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug_mode"):
        log.info("Running in debug mode! <config.debug_mode=True>")
        if "fast_dev_run" in config.trainer:
            config.trainer.fast_dev_run = True
        os.environ["HYDRA_FULL_ERROR"] = "1"
        os.environ["OC_CAUSE"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        torch.autograd.set_detect_anomaly(True)
        with open_dict(config):
            config.datamodule.debug_mode = config.datamodule.get("debug_mode", True)
            config.model.debug_mode = config.model.get("debug_mode", True)

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("devices"):
            config.trainer.devices = 0
            config.trainer.accelerator = "cpu"
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
    elif config.datamodule.get("num_workers") == -1:
        # set num_workers to #CPU cores if <config.datamodule.num_workers=-1>
        config.datamodule.num_workers = os.cpu_count()
        log.info(f"Automatically setting num_workers to {config.datamodule.num_workers} (CPU cores).")

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    strategy = config.trainer.get("strategy", "")
    strategy_name = strategy if isinstance(strategy, str) else strategy._target_.lower().split(".")[-1]
    if strategy_name.startswith("ddp") or strategy_name.startswith("dp"):
        if config.datamodule.get("pin_memory"):
            if config.datamodule.get("pin_memory") == "force_true":
                config.datamodule.pin_memory = True
            else:
                if config.datamodule.pin_memory is True:
                    log.info(
                        f"Forcing pin_memory to False for multi-GPU training ({strategy_name=}). "
                        f"Force True with `pin_memory=force_true`."
                    )
                config.datamodule.pin_memory = False

    torch_matmul_precision = config.get("torch_matmul_precision", "highest")
    if torch_matmul_precision != "highest":
        log.info(f"Setting torch matmul precision to ``{torch_matmul_precision}``.")
        torch.set_float32_matmul_precision(torch_matmul_precision)

    try:
        _ = config.datamodule.get("data_dir")
    except omegaconf.errors.InterpolationResolutionError as e:
        # Provide more helpful error message for e.g. Windows users where $HOME does not exist by default
        raise ValueError(
            "Could not resolve ``datamodule.data_dir`` in config. See error message above for details.\n"
            "   If this is a Windows machine, you may need to set ``data_dir`` to an absolute path, e.g. ``C:/data``.\n"
            "       You can do so in ``src/configs/datamodule/_base_data_config.yaml`` or with the command line."
        ) from e

    if config.module.get("num_predictions", 1) > 1:
        monitor = config.module.get("monitor", "") or ""
        if "crps" not in monitor and "rmse" in monitor:
            # is_ipol_exp = "InterpolationExperiment" in config.module.get("_target_", "")
            # new_monitor = "val/" + ("ipol/avg/crps_normed" if is_ipol_exp else "avg/crps_normed")
            config.module.monitor = config.module.monitor.replace("rmse", "crps")
            log.info(f"Setting {config.module.monitor=} since num_predictions > 1")

    # fix monitor for model_checkpoint and early_stopping callbacks
    monitor = config.module.get("monitor", "") or ""
    if config.get("callbacks") is not None and monitor:
        clbk_ckpt = config.callbacks.get("model_checkpoint", None)
        clbk_es = config.callbacks.get("early_stopping", None)
        if clbk_ckpt is not None and clbk_ckpt.get("monitor"):
            config.callbacks.model_checkpoint.monitor = monitor
        if clbk_es is not None and clbk_es.get("monitor"):
            config.callbacks.early_stopping.monitor = monitor

    # Set a short name for the model
    if config.get("model"):  # Some "naive" baselines don't have a model
        model_name = config.model.get("name")
        if model_name is None or model_name == "":
            model_class = config.model.get("_target_")
            mixer = config.model.mixer.get("_target_") if config.model.get("mixer") else None
            dm_type = config.datamodule.get("_target_")
            with open_dict(config):
                config.model.name = clean_name(model_class, mixer=mixer, dm_type=dm_type)

    # Detect if using SLURM cluster
    if "SLURM_JOB_ID" in os.environ:
        with open_dict(config):
            config.slurm_job_id = str(os.environ["SLURM_JOB_ID"])
        log.info(f"Detected SLURM job ID: {config.slurm_job_id}")
        if "WANDB__SERVICE_WAIT" not in os.environ.keys():
            os.environ["WANDB__SERVICE_WAIT"] = "300"
        else:
            log.info(f"WANDB__SERVICE_WAIT already set to {os.environ['WANDB__SERVICE_WAIT']}")

    if "SCRIPT_NAME" in os.environ:
        script_path = os.environ["SCRIPT_NAME"]
        with open_dict(config):
            config.script_name = script_path.split("/")[-1]  # get only the script name
            config.script_path = script_path
        log.info(f"Detected script name: {config.script_name}")

    check_config_values(config)

    with open_dict(config):
        config.wandb_status = wandb_status
    if USE_WANDB:
        with open_dict(config):
            config.logger.wandb.training_id = config.logger.wandb.get("resume_run_id") or config.logger.wandb.id
            if config.logger.wandb.get("resume_run_id"):
                train_run_path = (
                    f"{config.logger.wandb.entity}/{config.logger.wandb.project}/{config.logger.wandb.training_id}"
                )
                config.logger.wandb.train_run_path = wandb_api._TRAINING_RUN_PATH = train_run_path

        eval_modes_opts = ["test", "predict", "validate", None]
        if True:  # config.get("eval_mode"):
            assert config.get("eval_mode") in eval_modes_opts, f"{config.get('eval_mode')=} not in {eval_modes_opts}!"
            tags = list(config.logger.wandb.tags or [])
            # Add command line kwargs to wandb tags. (we remove + or ++ from the kwargs)
            config.logger.wandb.tags = tags + [
                cli_arg.replace("+", "") for cli_arg in sys.argv[1:] if "=" in cli_arg and len(cli_arg) <= 64
            ]  # wandb tag limit is 64 chars
            if config.logger.wandb.get("project_test") is None and config.model is not None:
                config.logger.wandb.resume = "allow"  # "must"
                config.logger.wandb.training_id = config.logger.wandb.id
                wandb_api._PROJECT_TRAIN = config.logger.wandb.project
                train_run_path = (
                    f"{config.logger.wandb.entity}/{wandb_api._PROJECT_TRAIN}/{config.logger.wandb.training_id}"
                )
                with open_dict(config):
                    _ = config.logger.wandb.pop("project_test", None)  # remove project_test
                    config.logger.wandb.train_run_path = wandb_api._TRAINING_RUN_PATH = train_run_path

            elif config.logger.wandb.get("project_test") is not None and config.model is None:
                # E.g. for non-ML models like climatology
                with open_dict(config):
                    config.logger.wandb.project = wandb_api.PROJECT = config.logger.wandb.pop("project_test")
            elif config.logger.wandb.get("project_test") is not None:
                config.logger.wandb.resume = "allow"  # no resuming likely, since different project
                config.logger.wandb.training_id = config.logger.wandb.id
                wandb_api._PROJECT_TRAIN = config.logger.wandb.project
                train_run_path = (
                    f"{config.logger.wandb.entity}/{wandb_api._PROJECT_TRAIN}/{config.logger.wandb.training_id}"
                )
                train_run = get_run_api(run_path=train_run_path)
                with open_dict(config):
                    config.logger.wandb.train_run_path = wandb_api._TRAINING_RUN_PATH = train_run_path
                    config.logger.wandb.project = wandb_api.PROJECT = config.logger.wandb.pop("project_test")
                    config.effective_batch_size = train_run.config.get("effective_batch_size")

                # Set a new run ID (!= training run ID)
                config.logger.wandb.id = wandb_api.get_wandb_id_for_run(config.logger.wandb)
                # Check if a test run already exists
                try:
                    runs_in_group = []
                except requests.exceptions.HTTPError as e:
                    # 500 Server Error: Internal Server Error for url: https://api.wandb.ai/graphql
                    log.warning(f"Error when getting runs in group. Not checking for existing test runs. Error: {e}")
                    runs_in_group = []
                for other_run in runs_in_group:
                    import json
                    if json.loads(other_run.config).get("seed") is None or config.get("eval_mode") == "predict":
                        continue
                    elif other_run.tags != config.logger.wandb.tags:
                        continue
                    elif (
                        other_run.tags == config.logger.wandb.tags
                    ):  # or all(tag in other_run.tags for tag in config.logger.wandb.tags):
                        # Should do same to set id and name to other_run.id and other_run.name (?!)
                        raise ValueError(
                            f"Test run with same tags already exists: {other_run.id}. Tags: {other_run.tags} vs {config.logger.wandb.tags}"
                        )
                    elif other_run.state == "running":
                        continue
                    elif other_run.summary.get("TESTED"):  # already ran full test
                        continue
                    elif int(other_run.config.get("seed")) == int(config.seed):
                        # If so, resume it
                        log.info(f">>>>>> Resuming test run {other_run.id} from group {other_run.group}.")
                        config.logger.wandb.resume = True
                        config.logger.wandb.id = other_run.id
                        config.logger.wandb.name = other_run.name
                    else:
                        pass  # log.info(f"Seed {other_run.config.get('seed')} != {config.seed}")

    if config.get("wandb_status") == "resume":
        ckpt_filename = config.ckpt_path or "last.ckpt"
        run_id = config.logger.wandb.training_id
        config.ckpt_path = reload_checkpoint(
            run_id=run_id, config=config, ckpt_filename=ckpt_filename, local_checkpoint_path=True, ckpt_path_only=True
        )
    return config


def get_only_overriden_config(config: DictConfig) -> DictConfig:
    """
    Get only the config values that are different from the default values in configs/main_config.yaml

    Args:
        config: Hydra config object with all the config values.

    Returns:
        DictConfig: Hydra config object with only the config values that are different from the default values.
    """
    from hydra.core.global_hydra import GlobalHydra

    # OLD code:
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../configs"):
        config_default = hydra.compose(config_name="main_config.yaml", overrides=[])
        # config_overriden = hydra.compose(config_name="main_config.yaml", overrides=OmegaConf.from_cli())
    diff = get_difference_between_configs(config_default, config, one_sided=True)
    # Merge with explicit CLI args in case they happened to be equal to the default values.
    # This is needed because the default values may differ from the ones in a reloaded run config.
    args_list = [
        arg.replace("+", "") for arg in sys.argv[1:]
    ]  # this is plain omegaconf, so we need to remove + or ++ from the kwargs
    cli_kwargs = OmegaConf.from_cli(args_list=args_list)
    log.info("CLI KWARGS:2", cli_kwargs, "\nDIFF:", OmegaConf.to_yaml(diff))
    diff = OmegaConf.merge(diff, cli_kwargs)
    log.info("DIFF+CLI:", OmegaConf.to_yaml(diff))
    log.info(
        f"modules... Defaul=\n{OmegaConf.to_yaml(config_default.module)}\n\nDiff=\n{OmegaConf.to_yaml(diff.module)}\n\nConfig=\n{OmegaConf.to_yaml(config.module)}"
    )
    return diff


def get_difference_between_configs(config1: DictConfig, config2: DictConfig, one_sided: bool = False) -> DictConfig:
    """
    Get the difference between two OmegaConf DictConfig objects (potentially use the values of config2).

    Args:
        config1: OmegaConf DictConfig object.
        config2: OmegaConf DictConfig object. Use the values of this config if they are different from config1.
        one_sided: If False, values of config1 are included if they don't exist in config2. If True, they are not.

    Returns:
        DictConfig: OmegaConf DictConfig object with only the config values that are different between config1 and config2.
            That is, values that are either contained in config1 but not config2, or vice versa, or have different values.
    """
    # We can convert the DictConfig to a simple dict, and then use set operations to get the difference
    # However, we need to resolve the DictConfig first, otherwise we get a TypeError
    config1 = OmegaConf.to_container(config1, resolve=True)
    config2 = OmegaConf.to_container(config2, resolve=True)
    # Get the difference between the two configs
    diff = get_difference_between_dicts_nested(config1, config2, one_sided=one_sided)
    # Convert back to DictConfig
    diff = OmegaConf.create(diff)
    return diff


def get_difference_between_dicts_nested(dict1: dict, dict2: dict, one_sided: bool = False) -> dict:
    """
    Get the difference between two nested dictionaries (potentially use the values of dict2).

    Args:
        dict1: Nested dictionary.
        dict2: Nested dictionary. Use the values of this dictionary if they are different from dict1.
        one_sided: If False, values of config1 are included if they don't exist in config2. If True, they are not.

    Returns:
        dict: Nested dictionary with only the values that are different between dict1 and dict2.
            That is, values that are either contained in dict1 but not dict2, or vice versa, or have different values.
    """
    if dict1 is None:
        return dict2
    if dict2 is None:
        return dict1
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise TypeError(
            f"dict1 and dict2 must be dictionaries! \nGot {type(dict1)}:{dict1}\n and {type(dict2)}:{dict2}."
        )
    # Get the difference between the two dicts
    if one_sided:
        diff = dict()
    else:
        diff = {k: dict1[k] for k in set(dict1.keys()) - set(dict2.keys())}  # keys in dict1 but not dict2
    diff.update({k: dict2[k] for k in set(dict2.keys()) - set(dict1.keys())})  # keys in dict2 but not dict1
    # Keys in both dicts but with different values (use the values of dict2)
    for k in set(dict1.keys()) & set(dict2.keys()):
        if dict1[k] != dict2[k]:
            # If the value is a dict, recursively get the difference between the nested dicts
            diff[k] = dict() if isinstance(dict2[k], dict) else dict2[k]
    # Recursively get the difference between the nested dicts
    for k in diff:
        if isinstance(diff[k], dict):
            if isinstance(dict1.get(k), dict) and isinstance(dict2.get(k), dict):
                diff[k] = get_difference_between_dicts_nested(dict1.get(k), dict2.get(k), one_sided=one_sided)
            else:
                diff[k] = dict2[k]
    return diff


def check_config_values(config: DictConfig):
    """Check if config values are valid."""
    with open_dict(config):
        if config.get("model", default_value=False):
            if "net_normalization" in config.model.keys():
                if config.model.net_normalization is None:
                    config.model.net_normalization = "none"
                config.model.net_normalization = config.model.net_normalization.lower()

        if config.get("diffusion", default_value=False):
            # Check that diffusion model has same hparams as the model it is based on
            config.model.loss_function = None
            for k, v in config.model.items():
                if k in config.diffusion.keys() and k not in ["_target_", "name", "loss_function"]:
                    assert v == config.diffusion[k], f"Diffusion model and model have different values for {k}!"

            # ipolator_id = config.diffusion.get("interpolator_run_id")
            # if ipolator_id is not None:
            #     get_run_api(ipolator_id)

        scheduler_cfg = config.module.get("scheduler")
        if scheduler_cfg and "LambdaWarmUpCosineScheduler" in scheduler_cfg.get("_target_", ""):
            # set base LR of optim to 1.0, since we will scale it by the warmup factor
            config.module.optimizer.lr = 1.0

        if config.get("ckpt_path") is not None and "reconstructed_" in config.ckpt_path:
            if config.module.get("use_ema"):
                config.module.use_ema = False
                log.info(f"Disabling EMA since you are loading a reconstructed checkpoint ({config.ckpt_path}).")

        USE_WANDB = (
            "logger" in config.keys() and config.logger.get("wandb") and hasattr(config.logger.wandb, "_target_")
        )
        if USE_WANDB:
            config.logger.wandb.id = str(config.logger.wandb.id)  # convert to string
            if config.logger.wandb.get("project_test") is not None and not config.get("eval_mode"):
                log.warning(f"You are trying to override the wandb project, but {config.get('eval_mode')=}!")

            if "callbacks" in config and not config.get("eval_mode"):
                # Add wandb run ID to model checkpoint dir as a subfolder
                for name in config.callbacks.keys():
                    if "model_checkpoint" not in name or config.callbacks.get(name) is None:
                        continue
                    wandb_model_run_id = config.logger.wandb.get("id")
                    d = config.callbacks[name].dirpath
                    if not os.path.exists(d) and not os.path.exists(os.path.dirname(d)):
                        # Run on different system, use config.work_dir/checkpoints
                        if os.path.exists(config.work_dir):
                            log.info(f"Changing dirpath={d} to {config.work_dir}/checkpoints for callback {name}.")
                            d = os.path.join(config.work_dir, "checkpoints")

                    if wandb_model_run_id is not None and wandb_model_run_id not in d:
                        # Save model checkpoints to special folder <ckpt-dir>/<wandb-run-id>/
                        new_dir = os.path.join(d, wandb_model_run_id)
                        config.callbacks[name].dirpath = new_dir
                        try:
                            os.makedirs(new_dir, exist_ok=True)
                        except PermissionError as e:
                            raise PermissionError(
                                f"PermissionError when creating {new_dir} for callback {name}"
                            ) from e
                        log.info(f"Model checkpoints for ``{name}`` will be saved in: {os.path.abspath(new_dir)}")
        else:
            if config.get("callbacks") and "wandb" in config.callbacks:
                raise ValueError("You are trying to use wandb callbacks but you aren't using a wandb logger!")
            # log.warning("Model checkpoints will not be saved because you are not using wandb!")
            config.save_config_to_wandb = False

        # Adjust global batch size, batch size per GPU, and accumulate_grad_batches based on the number of GPUs and nodes
        n_gpus = config.trainer.get("devices", 1)
        n_nodes = int(config.trainer.get("num_nodes", 1))
        if n_gpus == "auto":
            n_gpus = int(torch.cuda.device_count())
        elif isinstance(n_gpus, str) and "," in n_gpus:
            n_gpus = len(n_gpus.split(","))
        elif isinstance(n_gpus, Sequence):
            n_gpus = len(n_gpus)
        world_size = int(n_gpus * n_nodes) if n_gpus > 0 else 1

        bs, ebs = config.datamodule.batch_size, config.datamodule.eval_batch_size
        max_val_samples = config.datamodule.get("max_val_samples")
        if max_val_samples is not None and config.eval_mode in [None, "validate"]:
            # Check that max_val_samples is a multiple of the eval batch size * world_size
            if max_val_samples % (ebs * world_size) != 0:
                # try to set eval batch size so that max_val_samples is a multiple of it and ebs is not too large
                max_ebs = max_val_samples // world_size
                # Find the largest factor of max_ebs that is <= ebs
                ebs_new = None
                for i in range(max_ebs, 0, -1):
                    if max_ebs % i == 0 and i <= ebs:
                        ebs_new = i
                        break
                if ebs_new is not None:
                    config.datamodule.eval_batch_size = ebs_new
                    log.info(
                        f"Scaled eval_batch_size from {ebs} to {ebs_new} so that {max_val_samples=} "
                        f"are evenly distributed over {world_size=} devices ({ebs_new*world_size=})!"
                    )
                else:
                    raise ValueError(
                        f"max_val_samples={max_val_samples} must be a multiple of eval_batch_size={ebs} * "
                        f"{world_size=} (={ebs*world_size})!"
                    )

        if config.module.get("num_predictions", 1) > 1 and ebs >= bs:
            effective_ebs = ebs * config.module.num_predictions
            log.info(
                f"Note that the effective evaluation batch size will be multiplied by the number of "
                f"predictions={config.module.num_predictions} for a total of {effective_ebs}!"
            )

        if config.datamodule.get("num_workers") == "auto":
            if world_size >= 2:
                log.info(f"Setting datamodule.num_workers to the number of GPUs (={world_size})!")
                config.datamodule.num_workers = world_size  # 1 worker per GPU
            elif world_size == 1:
                log.info("Setting datamodule.num_workers to 8. This might not be optimal for a single GPU!")
                config.datamodule.num_workers = 8
            else:
                log.warning("Setting datamodule.num_workers to 0. This might not be optimal for CPU training!")
                config.datamodule.num_workers = 0
        if config.datamodule.num_workers == 0 and config.datamodule.get("prefetch_factor") is not None:
            # Not using workers, so prefetch_factor will be ignored
            log.warning(f"{config.datamodule.prefetch_factor=} will be ignored since num_workers=0! Set to None.")
            config.datamodule.prefetch_factor = None
        if config.datamodule.num_workers == 0 and config.datamodule.get("persistent_workers") is True:
            # Not using workers, so persistent_workers will be ignored
            log.warning("datamodule.persistent_workers=True will be ignored since num_workers=0! Set to False.")
            config.datamodule.persistent_workers = False

        if config.get("eval_mode"):
            if config.datamodule.get("batch_size_per_gpu") is not None:
                log.warning("Ignoring batch_size_per_gpu in eval mode. Use ``datamodule.eval_batch_size`` instead.")
            config.datamodule.batch_size_per_gpu = None
        else:
            # Set the batch size per GPU, and accumulate_grad_batches based on the number of GPUs and nodes
            batch_size = int(config.datamodule.get("batch_size", 1))  # Global batch size
            bs_per_gpu_total = batch_size // world_size  # effective batch size per GPU
            bs_per_gpu = config.datamodule.get("batch_size_per_gpu")
            if bs_per_gpu is None or bs_per_gpu > bs_per_gpu_total:
                bs_per_gpu = bs_per_gpu_total
            elif bs_per_gpu_total % bs_per_gpu != 0:
                # Set bs_per_gpu to the largest factor of bs_per_gpu_total that is <= bs_per_gpu
                for i in range(bs_per_gpu, 0, -1):
                    if bs_per_gpu_total % i == 0:
                        bs_per_gpu = i
                        break
            acc = bs_per_gpu_total // bs_per_gpu
            acc2 = config.trainer.get("accumulate_grad_batches")
            assert (
                acc2 in [None, 1] or acc2 == acc
            ), f"trainer.accumulate_grad_batches={acc2} must be equal to {acc}! (bs_per_gpu_total={bs_per_gpu_total})"
            if acc != acc2:
                log.warning(
                    f"trainer.accumulate_grad_batches={acc2} will be set to {acc} to compensate for the number of GPUs and nodes. (bs_per_gpu_total={bs_per_gpu_total})"
                )
            effective_ebs = bs_per_gpu * acc * world_size
            if effective_ebs != batch_size:
                # Check if within 10% of the batch size
                calc_str = (
                    f"{effective_ebs}={bs_per_gpu} * {acc} * {world_size} (bs_per_gpu * n_acc_grads * world_size)"
                )
                bs_warn_suffix = f"global batch size {batch_size}! ({n_gpus=}, {n_nodes=}, {config.trainer.devices=})"
                if abs(effective_ebs - batch_size) > 0.1 * batch_size:
                    raise ValueError(f"Effective batch size {calc_str} must be equal to {bs_warn_suffix}")
                else:
                    log.warning(f"Effective batch size {calc_str} is not equal to {bs_warn_suffix}")
            config.n_gpus = n_gpus
            config.world_size = world_size
            config.effective_batch_size = effective_ebs  # * acc * n_gpus
            config.datamodule.batch_size = bs_per_gpu
            config.trainer.accumulate_grad_batches = acc
            config.datamodule.batch_size_per_gpu = None
            config.datamodule.pop("batch_size_per_gpu", None)  # Remove batch_size_per_gpu from config

        # Check if CUDA is available. If not, switch to CPU.
        if not torch.cuda.is_available():
            if config.trainer.get("accelerator") == "gpu":
                config.trainer.accelerator = "cpu"
                config.trainer.devices = 1  # devices = num_processes for CPU
                log.warning(
                    "CUDA is not available, switching to CPU.\n"
                    "\tIf you want to use GPU, please re-install pytorch: https://pytorch.org/get-started/locally/."
                    "\n\tIf you want to use a different accelerator, specify it with ``trainer.accelerator=...``."
                )
            # Check if MPS is available
            if torch.backends.mps.is_available():
                config.trainer.accelerator = "mps"
                config.trainer.devices = 1
                log.warning(
                    "CUDA is not available, switching to MPS.\n"
                    "\tIf you want to use GPU, please re-install pytorch: https://pytorch.org/get-started/locally/."
                    "\n\tIf you want to use a different accelerator, specify it with ``trainer.accelerator=...``."
                )


def get_all_instantiable_hydra_modules(config, module_name: str):
    modules = []
    if config.get(module_name):
        for _, module_config in config[module_name].items():
            if module_config is not None and "_target_" in module_config:
                if "early_stopping" in module_config.get("_target_"):
                    diffusion = config.get("diffusion", default_value=False)
                    monitor = module_config.get("monitor", "")
                    # If diffusion model: Add _step to the early stopping callback key
                    if diffusion and "step" not in monitor and "epoch" not in monitor:
                        module_config.monitor += "_step"
                        log.info("*** Early stopping monitor changed to: ", module_config.monitor)
                        log.info("----------------------------------------\n" * 20)

                try:
                    modules.append(hydra.utils.instantiate(module_config))
                except omegaconf.errors.InterpolationResolutionError as e:
                    log.warning(f" Hydra could not instantiate {module_config} for module_name={module_name}")
                    raise e
    return modules


@rank_zero_only
def log_hyperparameters(
    config,
    model: pl.LightningModule,
    data_module: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Additionally saves:
        - number of {total, trainable, non-trainable} model parameters
    """

    def copy_and_ignore_keys(dictionary, *keys_to_ignore):
        if dictionary is None:
            return None
        new_dict = dict()
        for k in dictionary.keys():
            if k in keys_to_ignore:
                continue
            if isinstance(dictionary[k], DictConfig):
                # Convert DictConfig to dict for better readability on wandb dashboard
                new_dict[k] = copy_and_ignore_keys(dictionary[k], *keys_to_ignore)
            else:
                new_dict[k] = dictionary[k]
        return new_dict

    log_params = dict()
    log_params["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "seed" in config:
        log_params["seed"] = config["seed"]

    # Remove redundant keys or those that are not important to know after training -- feel free to edit this!
    log_params["datamodule"] = copy_and_ignore_keys(config["datamodule"])
    log_params["model"] = copy_and_ignore_keys(config["model"])
    log_params["exp"] = copy_and_ignore_keys(config["module"], "optimizer", "scheduler")
    log_params["trainer"] = copy_and_ignore_keys(config["trainer"])
    # encoder, optims, and scheduler as separate top-level key
    if "n_gpus" in config.keys():
        log_params["trainer"]["gpus"] = config["n_gpus"]
    log_params["optim"] = copy_and_ignore_keys(config["module"]["optimizer"])
    if "base_lr" in config.keys():
        log_params["optim"]["base_lr"] = config["base_lr"]
    if "effective_batch_size" in config.keys():
        log_params["optim"]["effective_batch_size"] = config["effective_batch_size"]
    if "diffusion" in config:
        log_params["diffusion"] = copy_and_ignore_keys(config["diffusion"])
    log_params["scheduler"] = copy_and_ignore_keys(config["module"].get("scheduler", None))
    if config.get("model"):
        # Add a clean name for the model, for easier reading (e.g. src.model.MLP.MLP -> MLP)
        model_class = config.model.get("_target_")
        mixer = config.model.mixer.get("_target_") if config.model.get("mixer") else None
        log_params["model/name_id"] = clean_name(model_class, mixer=mixer)
    if config.get("logger"):
        log_params["wandb"] = copy_and_ignore_keys(config.logger.get("wandb"))

    if config.get("callbacks"):
        skip_callbacks = ["summarize_best_val_metric", "learning_rate_logging"]
        for k, v in config["callbacks"].items():
            if k in skip_callbacks:
                continue
            elif k == "model_checkpoint":
                log_params[k] = copy_and_ignore_keys(v, "save_top_k")
            else:
                log_params[k] = copy_and_ignore_keys(v)

    # save number of model parameters
    log_params["model/params_total"] = sum(p.numel() for p in model.parameters())
    log_params["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_params["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log_params["dirs/work_dir"] = config.get("work_dir")
    log_params["dirs/ckpt_dir"] = config.get("ckpt_dir")
    log_params["dirs/wandb_save_dir"] = (
        config.logger.wandb.get("save_dir") if (config.get("logger") and config.logger.get("wandb")) else None
    )
    if "BEAKER_EXPERIMENT_ID" in os.environ:
        log_params["beaker"] = {
            "experiment_id": os.environ["BEAKER_EXPERIMENT_ID"],
            "job_id": os.environ["BEAKER_JOB_ID"],
            "task_id": os.environ.get("BEAKER_TASK_ID", None),
        }
    # Add all values that are not dictionaries
    for k, v in config.items():
        if not isinstance(v, dict) and k not in log_params.keys():
            log_params[k] = v

    # send hparams to all loggers (if any logger is used)
    if trainer.logger is not None:
        log.info("Logging hyperparameters to the PyTorch Lightning loggers.")
        trainer.logger.log_hyperparams(log_params)

        # disable logging any more hyperparameters for all loggers
        # this is just a trick to prevent trainer from logging hparams of model,
        # since we already did that above
        # trainer.logger.log_hyperparams = no_op


@rank_zero_only
def save_hydra_config_to_wandb(config: DictConfig):
    # Save the config to the Wandb cloud (if wandb logging is enabled)
    import wandb

    if config.get("save_config_to_wandb"):
        filename = "hydra_config.yaml"
        # Check if ``filename`` already exists in wandb cloud. If so, append a version number to it.
        run_api = get_run_api(run_path=wandb.run.path)
        version = 2
        try:
            run_api_files = [f.name for f in run_api.files()]
        except TypeError as e:
            log.warning(
                f"Could not get list of files in wandb run. Not checking for existing config files. Error: {e}"
            )
            run_api_files = []
        while filename in run_api_files:
            filename = f"hydra_config-v{version}.yaml"
            version += 1

        log.info(f"Config will be saved to wandb as {filename} and in wandb.run.dir: {os.path.abspath(wandb.run.dir)}")
        # files in wandb.run.dir folder get directly uploaded to wandb
        filepath = os.path.join(wandb.run.dir, filename)
        with open(filepath, "w") as fp:
            OmegaConf.save(config, f=fp.name, resolve=True)
        wandb.save(filename)
    else:
        log.info("Hydra config will NOT be saved to WandB.")


def get_config_from_hydra_compose_overrides(
    overrides: List[str],
    config_path: str = "../configs",
    config_name: str = "main_config.yaml",
) -> DictConfig:
    """
    Function to get a Hydra config manually based on a default config file and a list of override strings.
    This is an alternative to using hydra.main(..) and the command-line for overriding the default config.

    Args:
        overrides: A list of strings of the form "key=value" to override the default config with.
        config_path: Relative path to the folder where the default config file is located.
        config_name: Name of the default config file (.yaml ending).

    Returns:
        The resulting config object based on the default config file and the overrides.

    Examples:

    .. code-block:: python

        config = get_config_from_hydra_compose_overrides(overrides=['model=mlp', 'model.optimizer.lr=0.001'])
        log.info(f"Lr={config.model.optimizer.lr}, MLP hidden_dims={config.model.hidden_dims}")
    """
    from hydra.core.global_hydra import GlobalHydra

    overrides = list(set(overrides))
    if "-m" in overrides:
        overrides.remove("-m")  # if multiruns flags are mistakenly in overrides
    # Not true?!: log.info(f" Initializing Hydra from {os.path.abspath(config_path)}/{config_name}")
    GlobalHydra.instance().clear()  # clear any previous hydra config
    hydra.initialize(config_path=config_path, version_base=None)
    try:
        config = hydra.compose(config_name=config_name, overrides=overrides)
    finally:
        GlobalHydra.instance().clear()  # always clean up global hydra
    return config


def get_model_from_hydra_compose_overrides(overrides: List[str]):
    """
    Function to get a torch model manually based on a default config file and a list of override strings.

    Args:
        overrides: A list of strings of the form "key=value" to override the default config with.

    Returns:
        The model instantiated from the resulting config.

    Examples:

    .. code-block:: python

        mlp_model = get_model_from_hydra_compose_overrides(overrides=['model=mlp'])
        random_mlp_input = torch.randn(1, 100)
        random_prediction = mlp_model(random_mlp_input)
    """
    from src.interface import get_lightning_module

    cfg = get_config_from_hydra_compose_overrides(overrides)
    return get_lightning_module(cfg)
