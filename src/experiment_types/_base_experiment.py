from __future__ import annotations

import inspect
import logging
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import xarray as xr
from omegaconf import DictConfig
from pytorch_lightning.utilities import grad_norm
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from src.datamodules._dataset_dimensions import get_dims_of_dataset
from src.datamodules.abstract_datamodule import BaseDataModule
from src.evaluation.aggregators.main import LossAggregator
from src.models._base_model import BaseModel
from src.models.ema.ema import LitEma
from src.models.modules import padding
from src.optimization.hybrid_optim import HybridOptim
from src.optimization.lr_scheduler import get_scheduler
from src.optimization.muon import Muon, get_muon_momentum
from src.utilities.checkpointing import reload_checkpoint
from src.utilities.timing import log_timing
from src.utilities.utils import (
    AlreadyLoggedError,
    concatenate_array_dicts,
    freeze_model,
    get_logger,
    log_peak_memory,
    print_gpu_memory_usage,
    raise_error_if_invalid_value,
    rrearrange,
    run_func_in_sub_batches_and_aggregate,
    to_DictConfig,
    to_tensordict,
    torch_to_numpy,
)


class StopTraining(Exception):
    pass


class BaseExperiment(pl.LightningModule):
    r"""This is a template base class, that should be inherited by any stand-alone ML model.
    Methods that need to be implemented by your concrete ML model (just as if you would define a :class:`torch.nn.Module`):
        - :func:`__init__`
        - :func:`forward`

    The other methods may be overridden as needed.
    It is recommended to define the attribute
        >>> self.example_input_array = torch.randn(<YourModelInputShape>)  # batch dimension can be anything, e.g. 7


    .. note::
        Please use the function :func:`predict` at inference time for a given input tensor, as it postprocesses the
        raw predictions from the function :func:`raw_predict` (or model.forward or model())!

    Args:
        optimizer: DictConfig with the optimizer configuration (e.g. for AdamW)
        scheduler: DictConfig with the scheduler configuration (e.g. for CosineAnnealingLR)
        monitor (str): The name of the metric to monitor, e.g. 'val/mse'
        mode (str): The mode of the monitor. Default: 'min' (lower is better)
        use_ema (bool): Whether to use an exponential moving average (EMA) of the model weights during inference.
        ema_decay (float): The decay of the EMA. Default: 0.9999 (only used if use_ema=True)
        enable_inference_dropout (bool): Whether to enable dropout during inference. Default: False
        conv_padding_mode_global (str): If set, this padding mode is used for all convolutional layers globally.
                Default: None (i.e. use specific padding modes for each layer or torch's default padding mode: 'zeros')
        name (str): optional string with a name for the model
        num_predictions (int): The number of predictions to make for each input sample
        prediction_inputs_noise (float): The amount of noise to add to the inputs before predicting
        log_every_step_up_to (int): Logging is performed at every step up to this number. Default: 1000.
            After that, logging interval corresponds to the lightning Trainer's log_every_n_steps parameter (default: 50)
        stop_after_n_epochs (int): Stop training after this number of epochs (re-initialized every time this module is loaded)
            This can be useful on clusters with a maximum time limit for a job.
        verbose (bool): Whether to print/log or not

    Read the docs regarding LightningModule for more information:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    CHANNEL_DIM = -3  # assumes 2 spatial dimensions for everything
    SUPPORT_RESIDUAL_PREDICTION = False  # Whether the experiment supports residual prediction

    def __init__(
        self,
        model_config: DictConfig,
        datamodule_config: DictConfig,
        diffusion_config: Optional[DictConfig] = None,
        optimizer: Optional[DictConfig] = None,
        scheduler: Optional[DictConfig] = None,
        monitor: Optional[str] = None,
        mode: str = "min",
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        enable_inference_dropout: bool = False,
        stack_window_to_channel_dim=True,
        conv_padding_mode_global: Optional[str] = None,
        learned_channel_variance_loss: bool = False,
        learned_spatial_variance_loss: bool = False,
        reset_optimizer: bool = False,
        from_pretrained_checkpoint_run_id: Dict[str, Any] = None,
        from_pretrained_local_path: Optional[str] = None,
        from_pretrained_checkpoint_filename: Optional[str] = "last.ckpt",
        from_pretrained_load_ema: bool = False,
        from_pretrained_frozen: bool = False,
        from_pretrained_exclude_keys: Optional[List[str]] = None,
        from_pretrained_lr_multiplier: float = None,
        torch_compile: str = None,
        use_random_channel_for_stochasticity: bool = False,
        residual_pred: bool = False,
        num_predictions: int = 1,
        num_predictions_in_memory: int = None,
        allow_validation_size_indivisible_on_ddp: bool = False,  # Throw error if False, else only log warning
        logging_infix: str = "",
        prediction_inputs_noise: float = 0.0,
        save_predictions_filename: Optional[str] = None,
        save_prediction_batches: int = 0,
        log_every_step_up_to: int = 1000,
        stop_after_n_epochs: int = None,
        seed: int = None,
        name: str = "",
        work_dir: str = "",
        verbose: bool = True,
        log_mem: bool = False,
    ):
        super().__init__()
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.monitor
        self.save_hyperparameters(ignore=["model_config", "datamodule_config", "diffusion_config", "verbose"])
        # Get a logger
        self.log_text = get_logger(name=self.__class__.__name__ if name == "" else name)
        self.name = name
        self._datamodule = None
        self.verbose = verbose
        self.logging_infix = logging_infix
        if not self.verbose:  # turn off info level logging
            self.log_text.setLevel(logging.WARN)

        # temporal_models = ["unet_semi_temp_rdm"]
        # self.stack_window_to_channel_dim = any([m in model_config.get("_target_") for m in temporal_models])
        # if not self.stack_window_to_channel_dim:
        #     self.log_text.info(f"Using stack_window_to_channel_dim={self.stack_window_to_channel_dim}. Inferred a temporal architecture from model name {model_config.get('_target_')}")

        if conv_padding_mode_global is not None:
            padding.set_global_padding_mode(padding_mode=conv_padding_mode_global)

        self.stack_window_to_channel_dim = stack_window_to_channel_dim
        self.model_config = model_config
        self.datamodule_config = datamodule_config
        self.diffusion_config = diffusion_config
        self.num_predictions = num_predictions
        if num_predictions_in_memory is None or num_predictions_in_memory > num_predictions:
            self.num_predictions_in_mem = num_predictions
        else:
            self.num_predictions_in_mem = num_predictions_in_memory
        assert (
            num_predictions % self.num_predictions_in_mem == 0
        ), f"{num_predictions_in_memory=} % {num_predictions=} != 0"
        diffusion_class = None if diffusion_config is None else diffusion_config.get("_target_", None)
        self.is_diffusion_model = diffusion_config is not None and diffusion_class is not None
        self.is_standard_diffusion = self.is_diffusion_model and "dyffusion" not in diffusion_class.lower()
        # Infer input, output, spatial dimensions from datamodule
        self.dims = get_dims_of_dataset(self.datamodule_config)
        self._instantiate_auxiliary_modules()
        self.use_ema = use_ema and ema_decay is not None
        self.update_ema = self.use_ema or (
            self.is_diffusion_model and diffusion_config.get("consistency_strategy") == "ema"
        )
        # Instantiate the model
        self.model = self.instantiate_model()
        # Potentially, reload some weights and/or freeze some layers
        #   Do this before initializing the EMA model, as the frozen layers should not be part of the EMA
        self.reloaded_state_dict_keys = None
        self.reload_weights_or_freeze_some()

        # Initialize the EMA model, if needed
        if self.update_ema:
            self.model_ema = LitEma(self.model_handle_for_ema, decay=ema_decay)
            self.log_text.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        if not self.use_ema:
            self.log_text.info("Not using EMA.")

        if self.model is not None:
            self.model.ema_scope = self.ema_scope

        if enable_inference_dropout:
            self.log_text.info("Enabling dropout during inference!")

        if residual_pred:
            assert self.SUPPORT_RESIDUAL_PREDICTION, f"{self.__class__.__name__} does not support residual prediction!"

        # Timing variables to track the training/epoch/validation time
        self._start_validation_epoch_time = self._start_test_epoch_time = self._start_epoch_time = None
        self.training_step_outputs = []
        self._validation_step_outputs, self._predict_step_outputs = [], []
        self._test_step_outputs = defaultdict(list)

        # Epoch and global step defaults. When only doing inference, the current_epoch of lightning may be 0, so you can set it manually.
        self._default_epoch = self._default_global_step = 0
        self._n_epochs_since_init = 0
        assert stop_after_n_epochs is None or stop_after_n_epochs > 0, f"{stop_after_n_epochs=}"

        # Check that the args/hparams are valid
        self._check_args()

        if self.use_ensemble_predictions("val"):
            self.log_text.info(f"Using a {num_predictions}-member ensemble for validation.")

        # Example input array, if set
        if hasattr(self.model, "example_input_array"):
            self.example_input_array = self.model.example_input_array

        if save_predictions_filename is not None:
            assert (
                save_prediction_batches == "all" or save_prediction_batches > 0
            ), "save_prediction_batches must be > 0 if save_predictions_filename is set."

    @property
    def model_handle_for_ema(self) -> torch.nn.Module:
        """Return the model handle that is used for the EMA. By default, this is the model itself.
        But it can be overridden in subclasses, e.g. for GANs, where the EMA is only applied to the generator."""
        return self.model

    @property
    def current_epoch(self) -> int:
        """The current epoch in the ``Trainer``, or 0 if not attached."""
        if self._trainer and self.trainer.current_epoch != 0:
            return self.trainer.current_epoch
        return self._default_epoch

    @property
    def global_step(self) -> int:
        """Total training batches seen across all epochs.

        If no Trainer is attached, this propery is 0.

        """
        if self._trainer and self.trainer.global_step != 0:
            return self.trainer.global_step

        return self._default_global_step

    # --------------------------------- Interface with model
    def actual_spatial_shapes(self, spatial_shape_in: Tuple[int, int], spatial_shape_out: Tuple[int, int]) -> Tuple:
        return spatial_shape_in, spatial_shape_out

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        return num_input_channels

    def actual_num_output_channels(self, num_output_channels: int) -> int:
        return num_output_channels

    @property
    def num_conditional_channels(self) -> int:
        """The number of channels that are used for conditioning as auxiliary inputs."""
        nc = self.dims.get("conditional", 0)
        if self.is_diffusion_model:
            diffusion_target = str(self.diffusion_config.get("_target_", "")).lower()

            # Guided regression can be configured so the backbone only consumes
            # guidance predictions (no explicit condition tensor).
            if "guided_regression" in diffusion_target and not self.diffusion_config.get(
                "pass_condition_to_model", False
            ):
                return 0

            if self.is_standard_diffusion:
                if self.diffusion_config.get("force_unconditional", False) is True:
                    pass
                elif (
                    self.stack_window_to_channel_dim
                    or self.diffusion_config.get("when_3d_concat_condition_to") == "channel"
                ):
                    # we use the data from the past window frames as conditioning (unless we use a temporal model)
                    # Would be easier to check if not self.model.is_3d here, but model is not set yet
                    nc += self.window * self.dims["input"]
            else:
                fwd_cond = self.diffusion_config.get("forward_conditioning", "").lower()
                if fwd_cond == "":
                    pass  # no forward conditioning, i.e. don't add anything
                elif fwd_cond == "data|noise":
                    nc += 2 * self.window * self.dims["input"]
                elif fwd_cond in ["none", None]:
                    pass
                else:
                    nc += self.window * self.dims["input"]
        return nc

    @property
    def num_conditional_channels_non_spatial(self) -> int:
        """The number of channels that are used for conditioning as auxiliary inputs for cross-attention."""
        return self.dims.get("conditional_non_spatial_dim", 0)

    @property
    def num_temporal_channels(self) -> Optional[int]:
        """The number of temporal dimensions."""
        return None

    @property
    def window(self) -> int:
        return self.datamodule_config.get("window", 1)

    @property
    def horizon(self) -> int:
        return self.datamodule_config.get("horizon", 1)

    @property
    def inputs_noise(self):
        # internally_probabilistic = isinstance(self.model, (GaussianDiffusion, DDPM))
        # return 0 if internally_probabilistic else self.hparams.prediction_inputs_noise
        return self.hparams.prediction_inputs_noise

    @property
    def num_prediction_loops(self):
        return self.num_predictions // self.num_predictions_in_mem

    @property
    def datamodule(self) -> BaseDataModule:
        if self._datamodule is None:  # alt: set in ``on_fit_start``  method
            if self._trainer is None:
                return None
            self._datamodule = self.trainer.datamodule
            # Make sure that normalizer means and stds are on same device as model
            for b_key in set(self.main_data_keys + self.main_data_keys_val + [""]):
                normalizer_name = f"normalizer_{b_key}" if b_key else "normalizer"
                if hasattr(self._datamodule, normalizer_name):
                    self.log_text.info(
                        f"Moving {normalizer_name} means and stds to same device={self.device} as model"
                    )
                    getattr(self._datamodule, normalizer_name).to(self.device)

        return self._datamodule

    def _instantiate_auxiliary_modules(self):
        """Instantiate auxiliary modules that need to exist before the model is instantiated.
        This is necessary because it is not possible to instantiate modules before calling super().__init__().
        """
        pass

    def extra_model_kwargs(self) -> dict:
        """Return extra kwargs for the model instantiation."""
        return {}

    @property
    def channel_dim(self):
        channel_dim = self.CHANNEL_DIM
        if self.datamodule_config.get("spatial_crop_during_training") is True:
            channel_dim += 1
        return channel_dim

    def instantiate_model(self, *args, **kwargs) -> BaseModel:
        r"""Instantiate the model, e.g. by calling the constructor of the class :class:`BaseModel` or a subclass thereof."""
        spatial_shape_in, spatial_shape_out = self.actual_spatial_shapes(
            self.dims["spatial_in"], self.dims["spatial_out"]
        )
        in_channels = self.actual_num_input_channels(self.dims["input"])
        out_channels = self.actual_num_output_channels(self.dims["output"])
        cond_channels = self.num_conditional_channels

        if self.is_diffusion_model and self.diffusion_config.get("residual_diffusion", False):
            residual_mode = str(self.diffusion_config.get("residual_condition_mode", "none")).lower()
            if residual_mode in ["concat", "replace"]:
                if not isinstance(cond_channels, int) or not isinstance(out_channels, int):
                    raise TypeError(
                        "Residual condition mode requires integer channel counts, got "
                        f"num_conditional_channels={type(cond_channels)} and num_output_channels={type(out_channels)}"
                    )
                if residual_mode == "concat":
                    cond_channels = cond_channels + out_channels
                else:  # replace
                    cond_channels = out_channels
                self.log_text.info(
                    "Residual conditioning: setting model num_conditional_channels=%s "
                    "(mode=%s, base=%s, guidance=%s)",
                    cond_channels,
                    residual_mode,
                    self.num_conditional_channels,
                    out_channels,
                )
            elif residual_mode != "none":
                raise ValueError(
                    f"Unknown residual_condition_mode={residual_mode!r}. "
                    "Expected one of: 'none', 'replace', 'concat'."
                )

        assert isinstance(in_channels, (int, dict)), f"Expected int, got {type(in_channels)} for in_channels."
        assert isinstance(out_channels, (int, dict)), f"Expected int, got {type(out_channels)} for out_channels."
        kwargs["datamodule_config"] = self.datamodule_config
        kwargs["learned_channel_variance_loss"] = self.hparams.learned_channel_variance_loss
        kwargs["learned_spatial_variance_loss"] = self.hparams.learned_spatial_variance_loss
        kwargs["channel_dim"] = self.channel_dim
        model = hydra.utils.instantiate(
            self.model_config,
            num_input_channels=in_channels,
            num_output_channels=out_channels,
            num_output_channels_raw=self.dims["output"],
            num_conditional_channels=cond_channels,
            num_conditional_channels_non_spatial=self.num_conditional_channels_non_spatial,
            num_temporal_channels=self.num_temporal_channels,
            spatial_shape_in=spatial_shape_in,
            spatial_shape_out=spatial_shape_out,
            _recursive_=False,
            **kwargs,
            **self.extra_model_kwargs(),
        )
        if self.is_diffusion_model:
            model = hydra.utils.instantiate(self.diffusion_config, model=model, _recursive_=False, **kwargs)
            self.log_text.info(
                f"Instantiated diffusion model: {model.__class__.__name__}, with"
                f" #diffusion steps={model.num_timesteps}"
            )

        # Compile torch model if needed
        torch_compile = self.hparams.torch_compile
        raise_error_if_invalid_value(torch_compile, [False, None, "model", "module"], name="torch_compile")
        if torch_compile == "model":
            self.log_text.info("Compiling the model (but not the LightningModule)...")
            model = torch.compile(model)

        return model

    def reload_weights_or_freeze_some(self) -> Dict[str, Any]:
        """Reload weights from a pretrained model, potentially freezing some layers."""
        reloaded_pretrained_ckpt = None
        if self.hparams.from_pretrained_checkpoint_run_id is not None:
            local_checkpoint_path = self.hparams.from_pretrained_local_path or True
            reloaded_pretrained_ckpt = reload_checkpoint(
                run_id=self.hparams.from_pretrained_checkpoint_run_id,
                local_checkpoint_path=local_checkpoint_path,  # If True, search in file system
                ckpt_filename=self.hparams.from_pretrained_checkpoint_filename,
                model=self,
                exclude_state_dict_keys=self.hparams.from_pretrained_exclude_keys,
                use_ema_weights_only=self.hparams.from_pretrained_load_ema,
                print_name="Pretrained model",
            )
            self.reloaded_state_dict_keys = reloaded_pretrained_ckpt["state_dict"]
            if self.hparams.from_pretrained_frozen:
                assert self.hparams.from_pretrained_lr_multiplier is None, "Cannot freeze and change LR"
                # Freeze the reloaded parameters (those that were pretrained and not in the exclude list)
                params_to_freeze = [k for k in self.reloaded_state_dict_keys if "model_ema." not in k]
                # params_to_freeze = [k.replace("_orig_mod.", "") for k in params_to_freeze]
                freeze_model(self, params_subset=params_to_freeze)
        return reloaded_pretrained_ckpt

    def forward(self, *args, **kwargs) -> Any:
        y = self.model(*args, **kwargs)
        return y

    # --------------------------------- Names
    @property
    def short_description(self) -> str:
        return self.name if self.name else self.__class__.__name__

    @property
    def WANDB_LAST_SEP(self) -> str:
        """Used to separate metrics. Base classes may use an additional prefix, e.g. '/ipol/'"""
        return "/"

    @property
    def validation_set_names(self) -> List[str]:
        if hasattr(self.datamodule, "validation_set_names") and self.datamodule.validation_set_names is not None:
            return self.datamodule.validation_set_names
        elif hasattr(self, "aggregators_val") and self.aggregators_val is not None:
            n_aggs = len(self.aggregators_val)
            if n_aggs > 1:
                self.log_text.warning(
                    "Datamodule has no attribute ``validation_set_names``. Using default names ``val_{i}``!"
                )
                return [f"val_{i}" for i in range(n_aggs)]
        return ["val"]

    @property
    def test_set_names(self) -> List[str]:
        if self._trainer is None:
            return ["???"]
        if hasattr(self.datamodule, "test_set_names"):
            return self.datamodule.test_set_names
        return ["test"]

    @property
    def prediction_set_name(self) -> str:
        return self.datamodule.prediction_set_name if hasattr(self.datamodule, "prediction_set_name") else "predict"

    # --------------------------------- Metrics
    def get_epoch_aggregators(self, split: str, dataloader_idx: int = None) -> dict:
        """Return a dictionary of epoch aggregators, i.e. functions that aggregate the metrics over the epoch.
        The keys are the names of the metrics, the values are the aggregator functions.
        """
        assert split in ["val", "test", "predict"], f"Invalid split {split}"
        save_to_path = self.prediction_outputs_filepath
        aggregators = self.datamodule.get_epoch_aggregators(
            split=split,
            dataloader_idx=dataloader_idx,
            is_ensemble=self.use_ensemble_predictions(split),
            experiment_type=self.__class__.__name__,
            device=self.device,
            verbose=self.current_epoch == 0,
            save_to_path=save_to_path,
            lit_module_handle=self,
        )
        if dataloader_idx in [0, None]:
            # Add aggregator for loss aggregated over batches of validation dataloader
            self._set_loss_weights()  # Set the loss weights (might be needed if only doing validation)
            aggregators["loss"] = LossAggregator()
        for v in aggregators.values():
            v.to(self.device)
        return aggregators

    def get_dataset_attribute(self, attribute: str, split: str = "train") -> Any:
        """Return the attribute of the dataset."""
        split = "train" if split in ["fit", None] else split
        if hasattr(self, f"_dataset_{split}_{attribute}"):
            # Return the cached attribute
            return getattr(self, f"_dataset_{split}_{attribute}")

        if self.datamodule is None:
            raise ValueError("Cannot get dataset attribute if datamodule is None. Please set datamodule first.")

        dl = {
            "train": self.datamodule.train_dataloader(),
            "val": self.datamodule.val_dataloader(),
            "test": self.datamodule.test_dataloader(),
            "predict": self.datamodule.predict_dataloader(),
        }[split]
        if dl is None:
            return None

        # Try to get the attribute from the dataset
        ds = dl.dataset if isinstance(dl, torch.utils.data.DataLoader) else dl[0].dataset
        if isinstance(ds, torch.utils.data.Subset):
            ds = ds.dataset
        elif isinstance(ds, torch.utils.data.ConcatDataset):
            ds = ds.datasets[0]
        # log.info(ds.__dict__, getattr(ds, attribute), attribute)
        attr_value = getattr(ds, attribute, getattr(ds, f"_{attribute}", None))
        if attr_value is not None:
            # Cache the attribute
            setattr(self, f"_dataset_{split}_{attribute}", attr_value)
        return attr_value

    # --------------------------------- Check arguments for validity
    def _check_args(self):
        """Check if the arguments are valid."""
        pass

    @contextmanager
    def ema_scope(self, context=None, force_non_ema: bool = False, condition: bool = None):
        """Context manager to switch to EMA weights."""
        condition = self.use_ema if condition is None else condition
        if condition and not force_non_ema:
            self.model_ema.store(self.model_handle_for_ema.parameters())
            self.model_ema.copy_to(self.model_handle_for_ema)
            if context is not None:
                self.log_text.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if condition and not force_non_ema:
                self.model_ema.restore(self.model_handle_for_ema.parameters())
                if context is not None:
                    self.log_text.info(f"{context}: Restored training weights")

    @contextmanager
    def inference_dropout_scope(self, condition: bool = None, context=None):
        """Context manager to switch to inference dropout mode.
        Args:
            condition (bool, optional): If True, switch to inference dropout mode. If False, switch to training mode.
                If None, use the value of self.hparams.enable_inference_dropout.
                Important: If not None, self.hparams.enable_inference_dropout is ignored!
            context (str, optional): If not None, print this string when switching to inference dropout mode.
        """
        condition = self.hparams.enable_inference_dropout if condition is None else condition
        if condition:
            self.model.enable_inference_dropout()
            if context is not None:
                self.log_text.info(f"{context}: Switched to enabled inference dropout")
        try:
            yield None
        finally:
            if condition:
                self.model.disable_inference_dropout()
                if context is not None:
                    self.log_text.info(f"{context}: Switched to disabled inference dropout")

    @contextmanager
    def timing_scope(self, context="", no_op=True, precision=2):
        """Context manager to measure the time of the code inside the context. (By default, does nothing.)
        Args:
            context (str, optional): If not None, print time elapsed in this context.
        """
        start_time = time.time() if not no_op else None
        try:
            yield None
        finally:
            if not no_op:
                context = f"``{context}``:" if context else ""
                self.log_text.info(f"Elapsed time {context} {time.time() - start_time:.{precision}f}s")

    def get_normalizer(self, batch_key: str):
        if hasattr(self, f"_normalizer_{batch_key}"):
            return getattr(self, f"_normalizer_{batch_key}")

        possible_keys = [
            f"normalizer_{batch_key}",  # Allow for different normalizers for different batch keys
            "normalizer",
        ]
        for normalizer_name in possible_keys:
            if hasattr(self.datamodule, normalizer_name):
                setattr(self, f"_normalizer_{batch_key}", getattr(self.datamodule, normalizer_name))
                return getattr(self.datamodule, normalizer_name)
        return None

    def normalize_data(self, x: Dict[str, Tensor], batch_key: str) -> TensorDict:
        """Normalize the data."""
        normalizer = self.get_normalizer(batch_key)
        if normalizer is not None:
            x = normalizer.normalize(x)

        return to_tensordict(x)  #  to_tensordict(x) is no-op if x is a tensor

    def normalize_batch(
        self, batch: Dict[str, Dict[str, Tensor]] | Dict[str, Tensor] | Tensor, batch_key: str
    ) -> Dict[str, TensorDict] | TensorDict:
        """Normalize the batch. If the batch is a nested dictionary, normalize each nested dictionary separately."""
        if torch.is_tensor(batch) or isinstance(next(iter(batch.values())), Tensor):
            return self.normalize_data(batch, batch_key)
        elif isinstance(batch, TensorDict):
            return TensorDict(
                {k: self.normalize_data(v, batch_key) for k, v in batch.items()}, batch_size=batch.batch_size
            )
        else:
            return {k: self.normalize_data(v, batch_key) for k, v in batch.items()}

    def denormalize_data(self, x: Dict[str, Tensor], batch_key: str) -> TensorDict:
        """Denormalize the data."""
        normalizer = self.get_normalizer(batch_key)
        if normalizer is not None:
            x = normalizer.denormalize(x)
        return to_tensordict(x)

    def denormalize_batch(
        self, x: Dict[str, Dict[str, Tensor]] | Dict[str, Tensor], batch_key: str
    ) -> Dict[str, TensorDict] | TensorDict:
        if torch.is_tensor(x) or isinstance(next(iter(x.values())), Tensor):
            return self.denormalize_data(x, batch_key)
        elif isinstance(x, TensorDict):
            return TensorDict({k: self.denormalize_data(v, batch_key) for k, v in x.items()}, batch_size=x.batch_size)
        else:
            return {k: self.denormalize_data(v, batch_key) for k, v in x.items()}

    def predict_packed(self, *inputs: Tensor, **kwargs) -> Dict[str, Tensor]:
        # check if model has sample_loop method with argument num_predictions
        if (
            hasattr(self.model, "sample_loop")
            and "num_predictions" in inspect.signature(self.model.sample_loop).parameters
        ):
            kwargs["num_predictions"] = self.num_predictions_in_mem

        results = self.model.predict_forward(*inputs, **kwargs)  # by default, just call the forward method
        if torch.is_tensor(results):
            results = {"preds": results}
        return results

    def _predict(
        self,
        *inputs: Tensor,
        num_predictions: Optional[int] = None,
        predictions_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        This should be the main method to use for making predictions/doing inference.

        Args:
            inputs (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`.
                This is the same tensor one would use in :func:`forward`.
            num_predictions (int, optional): Number of predictions to make. If None, use the default value.
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Tensor]: The model predictions (in a post-processed format), i.e. a dictionary output_var -> output_var_prediction,
                where each output_var_prediction is a Tensor of shape :math:`(B, *)` in original-scale (e.g.
                in Kelvin for temperature), and non-negativity has been enforced for variables such as precipitation.

        Shapes:
            - Input: :math:`(B, *, C_{in})`
            - Output: Dict :math:`k_i` -> :math:`v_i`, and each :math:`v_i` has shape :math:`(B, *)` for :math:`i=1,..,C_{out}`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{out}` is the number of output features.
        """
        base_num_predictions = self.num_predictions
        self.num_predictions = num_predictions or base_num_predictions

        # By default, we predict the entire batch at once (i.e. num_prediction_loops=1)
        results = run_func_in_sub_batches_and_aggregate(
            self.predict_packed,
            *inputs,
            num_prediction_loops=self.num_prediction_loops,
            predictions_mask=predictions_mask,
            **kwargs,
        )
        # results = TensorDict(results, batch_size=(full_batch_size,))
        # results = to_tensordict({k: torch.cat(v, dim=0) for k, v in results.items()}, find_batch_size_max=True)
        # log.info(results["preds2d"].shape, "after cat")
        self.num_predictions = base_num_predictions
        results = self.postprocess_predictions(results, *inputs)
        return results

    def postprocess_predictions(self, results: Dict[str, Tensor], *inputs) -> Dict[str, Tensor]:
        # log.info(results["preds2d"].shape, "after reshape")
        results = self.unpack_predictions(results)
        if self.hparams.residual_pred:
            # inputs is normalized (with normal mean and std), but preds are normalized by residual std
            # --> Need to first convert preds to normal normalization space and then add back the inputs to get normalized space full state preds
            assert len(inputs) == 1, "Residual prediction requires one input tensor."
            inputs = inputs[0]
            if inputs.ndim == 4:
                ndims = len(next(iter(results.values())).keys())  # ndims for one frame
                inputs = inputs[:, -ndims:]  # b, c, h, w; take the channels of latest frame
            elif inputs.ndim == 5:
                inputs = inputs[:, -1]  # b, t, c, h, w
            else:
                raise ValueError(f"Expected input tensor to have 4 or 5 dimensions, got {inputs.ndim}.")
            inputs = self.unpack_data(inputs, input_or_output="input")

            for k in results.keys():
                if "preds" not in k:
                    continue
                results[k] = self.get_normalizer("dynamics").normalized_residual_to_normalized(results[k])
                results[k] = to_tensordict({kk: vv + inputs.get(kk, 0) for kk, vv in results[k].items()})

        results = self.reshape_predictions(results)

        for k in list(results.keys()):
            if "preds" in k:  # Rename the keys from <var> to <var>_normed
                results[f"{k}_normed"] = results.pop(k)  # unpacked and normalized

        # results['preds_packed'] = packed_preds  # packed and normalized
        if self.datamodule is not None:
            # Unpack and denormalize the predictions. Keys are renamed from <var>_normed to <var>
            for k in list(results.keys()):
                if "preds" in k:
                    batch_p_key = self.target_key
                    unnormed_key = k.replace("_normed", "")
                    results[unnormed_key] = self.denormalize_batch(results[k], batch_key=batch_p_key)
        # for k, v in results.items(): print(k, v.shape if torch.is_tensor(v) else v)
        return results

    def reshape_predictions(self, results: TensorDict) -> TensorDict:
        """Reshape and unpack the predictions from the model. This modifies the input dictionary in-place.
        Args:
           results (Dict[str, Tensor]): The model outputs. Access the predictions via results['preds'].
        """
        pred_keys = [k for k in results.keys() if "preds" in k]
        preds_shape = results[pred_keys[0]].shape
        if preds_shape[0] > 1:
            if self.num_predictions > 1 and preds_shape[0] % self.num_predictions == 0:
                for k in pred_keys:
                    results[k] = self._reshape_ensemble_preds(results[k])
                # results = self._reshape_ensemble_preds(results)
        if len(pred_keys) != len(results):
            skipped_keys = [k for k in results.keys() if k not in pred_keys]
            self.log_text.info(
                f"Skipping reshaping for keys {skipped_keys} as they do not contain 'preds' in their name."
            )
        return results

    def predict(self, inputs: Union[Tensor, TensorDictBase], **kwargs) -> Dict[str, Tensor]:
        """Wrapper around the main predict method, to allow inputs to be a TensorDictBase or a Tensor."""
        if torch.is_tensor(inputs):
            return self._predict(inputs, **kwargs)
        else:
            return self._predict(**inputs, **kwargs)

    def pack_data(self, data: Dict[str, Tensor], input_or_output: str) -> Tensor:
        """Pack the data into a single tensor."""
        if input_or_output == "input":
            packer_name = "in_packer"
        elif input_or_output == "output":
            packer_name = "out_packer"
        else:
            raise ValueError(f"Unknown input_or_output: {input_or_output}")
        if not hasattr(self.datamodule, packer_name):
            assert torch.is_tensor(data), f"Expected tensor, got {type(data)}. ({self.__class__.__name__=})"
            return data
            # return torch.tensor(data) if not torch.is_tensor(data) else data
        packer = getattr(self.datamodule, packer_name)
        return packer.pack(data)

    def unpack_data(
        self, results: Dict[str, Tensor], input_or_output: str, axis=None, func="unpack"
    ) -> Dict[str, Tensor]:
        """Unpack the predictions from the model. This modifies the input dictionary in-place.
        Args:
           results (Dict[str, Tensor]): The model outputs. Access the predictions via results['preds'].
           input_or_output (str): Whether to unpack the input or output data.
           axis (int, optional): The axis along which to unpack the data. Default: None (use the default axis).
        """
        #  As of now, only keys with ``preds`` in them are unpacked.
        if input_or_output == "input":
            packer_name = "in_packer"
        elif input_or_output == "output":
            packer_name = "out_packer"
        else:
            raise ValueError(f"Unknown input_or_output: {input_or_output}")
        if not hasattr(self.datamodule, packer_name):
            return results

        packer = getattr(self.datamodule, packer_name)
        packer_func = getattr(packer, func)  # basically packer.unpack
        if torch.is_tensor(results):
            results = packer_func(results, axis=axis)
        elif "preds" in results.keys():
            preds_shape = results["preds"].shape
            results["preds"] = packer_func(results["preds"], axis=axis)
            for k in results.keys():
                if "preds" in k and k != "preds":
                    axis_here = axis or packer.axis_unpack
                    if results[k].shape != preds_shape:
                        # Find axis where same dim as preds_shape[axis_here]
                        axis_here = results[k].shape.index(preds_shape[axis_here])
                    results[k] = packer_func(results[k], axis=axis_here)

            # results["preds"] = packer.unpack(results["preds"], axis=axis)
        elif hasattr(packer, "packer_names") and packer.packer_names == set(
            packer.k_to_base_key(k) for k in results.keys()
        ):
            results = packer_func(results, axis=axis)
        else:
            for k, v in results.items():
                if k == "condition_non_spatial":
                    results[k] = v  # no unpacking for non-spatial predictions
                elif "preds" in k:
                    packer_k = packer[k.replace("preds", "")] if isinstance(packer, dict) else packer
                    results[k] = packer_k.unpack(v, axis=axis)
                else:
                    raise ValueError(f"Unknown key {k} in results for unpacking.")
        return results

    def unpack_predictions(self, results: Dict[str, Tensor], axis=None, **kwargs) -> Dict[str, Tensor]:
        return self.unpack_data(results, input_or_output="output", axis=axis, **kwargs)

    def get_target_variants(self, targets: Tensor, is_normalized: bool = False) -> Dict[str, Tensor]:
        if is_normalized:
            targets_normed = targets.contiguous()
            targets_raw = self.denormalize_batch(targets_normed, batch_key=self.target_key)
        else:
            targets_raw = targets.contiguous()
            try:
                targets_normed = self.normalize_batch(targets_raw, batch_key=self.target_key).contiguous()
            except RuntimeError:
                targets_normed = None
        return {
            "targets": targets_raw,
            "targets_normed": targets_normed,
        }

    # --------------------- training with PyTorch Lightning
    def on_any_start(self, stage: str = None) -> None:
        # Check if model has property ``sigma_data`` and set it to the data's std
        if hasattr(self.model, "sigma_data") and getattr(self.model, "_USE_SIGMA_DATA", False):
            self.model.sigma_data = self.datamodule.sigma_data

    def on_fit_start(self) -> None:
        self.on_any_start(stage="fit")

    def on_validation_start(self) -> None:
        self.on_any_start(stage="val")

    def on_test_start(self) -> None:
        self.on_any_start(stage="test")

    def on_train_start(self) -> None:
        """Log some info about the model/data at the start of training"""
        assert "/" in self.WANDB_LAST_SEP, f'Please use a separator that contains a "/" in {self.WANDB_LAST_SEP}'
        # Find size of the validation set(s)
        dl_val = self.datamodule.val_dataloader()
        val_sizes = [len(dl.dataset) for dl in (dl_val if isinstance(dl_val, list) else [dl_val])]
        # Compute the effective batch size
        # bs * acc * n_gpus
        train_dl = self.datamodule.train_dataloader()
        bs = train_dl.batch_size
        acc = self.trainer.accumulate_grad_batches
        n_gpus = max(1, self.trainer.num_devices)
        n_nodes = max(1, self.trainer.num_nodes)
        eff_bs = bs * acc * n_gpus * n_nodes
        # compute number of steps per epoch
        n_steps_per_epoch = len(train_dl)
        n_steps_per_epoch_per_gpu = n_steps_per_epoch / n_gpus
        to_log = {
            "Parameter count": float(self.model.num_params),
            "Training set size": float(len(train_dl.dataset)),
            "Validation set size": float(sum(val_sizes)),
            "Effective batch size": float(eff_bs),
            "Dataloader batch size": float(bs),
            "Steps per epoch": float(n_steps_per_epoch),
            "Steps per epoch per GPU": float(n_steps_per_epoch_per_gpu),
            "n_gpus": n_gpus,
            "world_size": self.trainer.world_size,
            "accumulate_grad_batches": acc,
            "TESTED": False,
        }
        # Log some dataloader args (useful for debugging/optimizing dataloader speed)
        dataloader_args_to_log = [
            "batch_size",
            "num_workers",
            "pin_memory",
            "drop_last",
            "persistent_workers",
            "prefetch_factor",
        ]
        for arg in dataloader_args_to_log:
            if hasattr(train_dl, arg) and getattr(train_dl, arg) is not None:
                val = getattr(train_dl, arg)
                if val == "force_true":
                    val = True
                to_log[f"train_dataloader/{arg}"] = val

        self.log_dict(to_log, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # provide access to trainer to the model
        self.model.trainer = self.trainer
        self._n_steps_per_epoch = n_steps_per_epoch
        self._n_steps_per_epoch_per_gpu = n_steps_per_epoch_per_gpu
        if self.global_step <= self.hparams.log_every_step_up_to:
            self._original_log_every_n_steps = self.trainer.log_every_n_steps
            self.trainer.log_every_n_steps = 1

        # Set the loss weights, if needed
        self._set_loss_weights()

        # Print the world size, rank, and local rank
        world_size = self.trainer.world_size
        if world_size > 1:
            self.log_text.info(
                f"World size: {world_size}, Rank: {self.trainer.global_rank}, Local rank: {self.trainer.local_rank}"
            )
            # Check that validation dataset sizes are divisible by the world size
            self.check_eval_dataset_divisibility(dl_val, "validation")

        # Set local rank and world size in os.env
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(self.trainer.local_rank)
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(world_size)

    def check_eval_dataset_divisibility(self, eval_loaders, split: str) -> None:
        eval_loaders = eval_loaders if isinstance(eval_loaders, list) else [eval_loaders]
        world_size = self.trainer.world_size
        if world_size <= 1:
            return  # No need to check divisibility if using a single GPU or CPU
        for i, eval_loader in enumerate(eval_loaders):
            eval_size = len(eval_loader.dataset)
            if eval_size % (eval_loader.batch_size * world_size) != 0:
                message = (
                    f"{split.capitalize()}_{i} set size ({eval_size}) is not divisible by "
                    f"{eval_loader.batch_size * world_size=} ({eval_loader.batch_size=}, {world_size=}). "
                    f"This will cause data point duplications across GPUs, leading to (slightly) incorrect metrics. "
                )
                if "val" in split and self.hparams.allow_validation_size_indivisible_on_ddp:
                    message += "Ignoring this warning because `module.allow_validation_size_indivisible_on_ddp=True`."
                    self.log_text.warning(message)
                else:
                    message += (
                        "Please set `datamodule.eval_batch_size` to a value that divides the validation set size. "
                        "If you prefer ignoring this warning for the validation dataloaders, "
                        "set `module.allow_validation_size_indivisible_on_ddp=True`."
                        "Alternatively, you may use a single GPU to get correct results. "
                    )
                    raise ValueError(message)

    @property
    def channels_logvar(self):
        if self.hparams.learned_channel_variance_loss:
            return self.get_logvar("channels")

    @property
    def spatial_logvar(self):
        if self.hparams.learned_spatial_variance_loss:
            return self.get_logvar("spatial")

    def get_logvar(self, dim_name: str):
        if isinstance(self.model.criterion, (dict, torch.nn.ModuleDict)):
            criterions = self.model.criterion.values()
        else:
            criterions = [self.model.criterion]
        for criterion in criterions:
            if dim_name == "spatial" and hasattr(criterion, "spatial_logvar"):
                return criterion.spatial_logvar
            if hasattr(criterion, "logvar_vector"):
                return criterion.logvar_vector(dim_name)

    def _reshape_loss_weights(self, loss_weights: Tensor) -> Tensor:
        return loss_weights

    def _set_loss_weights(self, split: str = "fit") -> None:
        """
        Configure weights for the weighted loss function based on dataset attribute "loss_weights_tensor".

        Args:
            split (str): The dataset split to use for getting loss weights.
        """

        def _set_criterion_weights(criterion):
            if not hasattr(criterion, "weights") or criterion.weights is not None:
                return False

            loss_weights = self.get_dataset_attribute("loss_weights_tensor", split)
            if loss_weights is None:
                self.log_text.warning(
                    f"Criterion supports loss weights but dataset has no loss_weights_tensor for {split=}."
                )
                return False

            weights = self._reshape_loss_weights(loss_weights.to(self.device))
            self.log_text.info(f"Setting loss weights of shape {weights.shape} for weighted loss function.")
            assert (weights > 1e-4).all(), f"{weights=}"
            criterion.weights = weights
            return True

        if not isinstance(self.model.criterion, (dict, torch.nn.ModuleDict)):
            boo = _set_criterion_weights(self.model.criterion)  # Handle single criterion case
            assert not boo or self.model.criterion.weights is not None, "Loss weights must be set."
        else:
            for key, criterion in self.model.criterion.items():
                boo = _set_criterion_weights(criterion)  # Handle ModuleDict case
                assert not boo or self.model.criterion[key].weights is not None, "Loss weights must be set."

    def on_train_epoch_start(self) -> None:
        self._start_epoch_time = time.time()
        if (
            self.hparams.stop_after_n_epochs is not None
            and self._n_epochs_since_init >= self.hparams.stop_after_n_epochs
        ):
            self.log_text.info(f"Reached {self.hparams.stop_after_n_epochs} epochs, stopping training...")
            time.sleep(10)
            raise StopTraining(
                f"Stopping training after {self.hparams.stop_after_n_epochs} epochs. "
                f"To disable this, set `module.stop_after_n_epochs=None`."
            )

    def train_step_initial_log_dict(self) -> dict:
        return dict()

    @property
    def target_key(self) -> str:
        return "dynamics"

    @property
    def main_data_keys(self) -> List[str]:
        return [self.target_key]

    @property
    def main_data_keys_val(self) -> List[str]:
        return self.main_data_keys

    @property
    def normalize_data_keys_val(self) -> List[str]:
        return self.main_data_keys_val  # by default, normalize all the main data keys

    @property
    def inputs_data_key(self) -> str:
        return self.main_data_keys_val[0]

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch"""
        raise NotImplementedError(f"Please implement the get_loss method for {self.__class__.__name__}")

    @log_timing(name="training_step", log_every_n=50)
    def training_step(self, batch: Any, batch_idx: int):
        r"""One step of training (backpropagation is done on the loss returned at the end of this function)"""
        if self.global_step == self.hparams.log_every_step_up_to:
            # Log on rank 0 only
            if self.trainer.global_rank == 0:
                self.log_text.info(f"Logging every {self._original_log_every_n_steps} steps from now on")
            self.trainer.log_every_n_steps = self._original_log_every_n_steps

        time_start = time.time()
        for main_data_key in self.main_data_keys:
            if isinstance(batch[main_data_key], dict):
                batch[main_data_key] = {k: to_tensordict(v) for k, v in batch[main_data_key].items()}
                batch[main_data_key] = to_tensordict(batch[main_data_key], find_batch_size_max=True)
            else:
                batch[main_data_key] = to_tensordict(batch[main_data_key])
            batch[main_data_key] = self.normalize_batch(batch[main_data_key], batch_key=main_data_key)

        # Normalize data and convert to tensor dict (if it's a dict)
        # Print mean and std of the data before normalization
        # if self.global_step == 0 and self.trainer.global_rank == 0:
        # to_float = lambda x: float(x) if torch.is_tensor(x) else {k: float(v) for k, v in x.items()}
        # self.log_text.info(f"Mean/std of the data before normalization: {to_float(batch[self.main_data_key].mean())} / {to_float(batch[self.main_data_key].std())}")
        # if self.global_step == 0 and self.trainer.global_rank == 0:
        # self.log_text.info(f"Mean/std of the data after normalization: {to_float(batch[self.main_data_key].mean())} / {to_float(batch[self.main_data_key].std())}")
        # Should be close to 0 and 1, respectively
        # Compute main loss - track forward pass memory
        with log_peak_memory(
            log_mem=self.hparams.log_mem and self.do_log_training_diagnostics(batch_idx),
            logger_instance=self,
            name="forward_peak_gb",
            prefix="memory/train/",
        ):
            loss_output = self.get_loss(batch)  # either a scalar or a dict with key 'loss'
        if isinstance(loss_output, dict):
            self.log_dict(
                {k: float(v) for k, v in loss_output.items()}, prog_bar=True, logger=True, on_step=True, on_epoch=True
            )
            loss = loss_output.pop("loss")
            # train_log_dict.update(loss_output)
        else:
            loss = loss_output
            # Train logs (where on_step=True) will be logged at all steps defined by trainer.log_every_n_steps
            self.log("train/loss", float(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Count number of zero gradients as diagnostic tool
        train_log_dict = {"time/train/step": time.time() - time_start}
        # train_log_dict["time/train/step_ratio"] = time_per_step / self.trainer.accumulate_grad_batches
        self.log_dict(train_log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss  # {"loss": loss}

    def do_log_training_diagnostics(self, batch_idx: int = None) -> bool:
        return self.global_step % (self.trainer.log_every_n_steps * 2) == 0

    def on_train_batch_end(self, outputs=None, batch=None, batch_idx: int = None):
        if self.update_ema:
            self.model_ema(self.model_handle_for_ema)  # update the model EMA
        if not self.do_log_training_diagnostics(batch_idx):
            return  # Do not log the following things at every step

        log_dict = {}
        # Log logvar of the channels
        channel_logvars = self.channels_logvar
        if channel_logvars is not None:
            channel_vars = channel_logvars.exp().detach()
            # Unpack to map channel index to semantic channel name
            channel_vars = self.unpack_data(
                results=channel_vars, input_or_output="output", axis=0, func="unpack_simple"
            )
            if torch.is_tensor(channel_vars):
                # When no packing is used
                channel_vars = {f"channel_{i}": v for i, v in enumerate(channel_vars)}
            # Pre-pend with "train/learned_var/" and make float
            channel_vars = {f"train/learned_var/{k}": float(v) for k, v in channel_vars.items()}
            log_dict.update(channel_vars)

        if self.spatial_logvar is not None:
            spatial_vars = self.spatial_logvar.exp().detach()
            assert len(spatial_vars.shape) == 2, f"Expected 2D tensor, got {spatial_vars.shape=}"
            # Make a heatmap plot of the 2D spatial variance
            spatial_lv_log = {
                "train/learned_var/spatial": wandb.Image(spatial_vars),
                "train/learned_var/spatial_mean": float(spatial_vars.mean()),
                "train/learned_var/spatial_std": float(spatial_vars.std()),
                "train/learned_var/spatial_min": float(spatial_vars.min()),
                "train/learned_var/spatial_max": float(spatial_vars.max()),
            }
            self.logger.experiment.log(spatial_lv_log)

        if hasattr(self, "_ar_logvars"):
            ar_logvars = self._ar_logvars.exp().detach()
            ar_logvars = {f"train/learned_var/ar_{i}": float(v) for i, v in enumerate(ar_logvars)}
            log_dict.update(ar_logvars)

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)

    def on_before_optimizer_step(self, optimizer, optimizer_idx: int = None):
        if self.do_log_training_diagnostics():
            # Compute number of zero gradients as diagnostic tool
            diags = {
                "n_zero_gradients": (
                    sum(
                        [
                            int(torch.count_nonzero(p.grad == 0))
                            for p in self.model.get_parameters()
                            if p.grad is not None
                        ]
                    )
                    / self.model.num_params
                )
            }
            # log some things less often:
            if self.global_step % (self.trainer.log_every_n_steps * 20) == 0:
                # Compute the 2-norm for each layer (and total) and log it
                diags.update(**grad_norm(self.model, norm_type=2))
                # In the logs, look for f"grad_{2.0}_norm_total" = torch.tensor(list(norms.values())).norm(norm_type)

            self.log_dict(diags)

    def on_train_epoch_end(self) -> None:
        log_dict = {"epoch": float(self.current_epoch)}
        if self._start_epoch_time is not None:  # sometimes there's a weird issue in DDP mode where this is not set.
            log_dict["time/train"] = time.time() - self._start_epoch_time
        self.log_dict(log_dict, sync_dist=True)
        self._n_epochs_since_init += 1

    # --------------------- evaluation with PyTorch Lightning
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        aggregators: Dict[str, Callable] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        One step of evaluation (forward pass, potentially metrics computation, logging, and return of results)
        Returns:
            results_dict: Dict[str, Tensor], where for each semantically different result, a separate prefix key is used
                Then, for each prefix key <p>, results_dict must contain <p>_preds and <p>_targets.
        """
        raise NotImplementedError(f"Please implement the _evaluation_step method for {self.__class__.__name__}")

    def evaluation_step(self, batch: Any, batch_idx: int, split: str, **kwargs) -> Dict[str, Tensor]:
        # Handle boundary conditions
        if "boundary_conditions" in inspect.signature(self._evaluation_step).parameters.keys():
            kwargs["boundary_conditions"] = self.datamodule.boundary_conditions
            kwargs.update(self.datamodule.get_boundary_condition_kwargs(batch, batch_idx, split))

        for k in self.main_data_keys_val:
            if k not in batch.keys():
                raise ValueError(
                    f"Could not find key {k} in batch. You need to either return it in your pytorch dataset or need to edit main_data_keys{{_val}} of this module."
                )
            if isinstance(batch[k], dict):
                batch[k] = {k: to_tensordict(v) for k, v in batch[k].items()}
                batch[k] = to_tensordict(batch[k], find_batch_size_max=True)
            else:
                batch[k] = to_tensordict(batch[k])

        for k in self.normalize_data_keys_val:
            if k == self.target_key:
                # Store the raw data, if needed for post-processing/using ground truth data
                batch[f"raw_{k}"] = batch[k].clone()

            # Normalize data
            batch[k] = self.normalize_batch(batch[k], batch_key=k)

        with self.ema_scope():  # use the EMA parameters for the validation step (if using EMA)
            with self.inference_dropout_scope():  # Enable dropout during inference
                results = self._evaluation_step(batch, batch_idx, split, **kwargs)

        return results

    def get_batch_shape(self, batch: Any) -> Tuple[int, ...]:
        """Get the shape of the batch"""
        for k in self.main_data_keys + self.main_data_keys_val:
            if k in batch.keys():
                if torch.is_tensor(batch[k]):
                    return batch[k].shape
                else:
                    # add singleton dim for channel
                    return batch[k].unsqueeze(self.channel_dim).shape
        raise ValueError(f"Could not find any of the keys {self.main_data_keys=}, {self.main_data_keys_val=}")

    def evaluation_results_to_xarray(self, results: Dict[str, np.ndarray], **kwargs) -> xr.Dataset:
        # if hasattr(self.model, "evaluation_results_to_xarray"):
        # self.log_text.info("Using model's evaluation_results_to_xarray method")
        # return self.model.evaluation_results_to_xarray(results, **kwargs)
        has_ens_dim = self.use_ensemble_predictions("predict")
        # remove the prefix. We will concatenate those with same suffix
        unique_keys = set(["_".join(k.split("_")[1:]) for k in results.keys()])
        self.log_text.info(f"unique_keys that will be concatenated: {unique_keys}.")  # All keys: {results.keys()}")
        any_target_key = [k for k in results.keys() if "targets" in k][0]
        any_target = results[any_target_key]
        if isinstance(any_target, dict):  # dict of tensors (per channel)
            is_dict = True
            n_spatial_dims = len(list(any_target.values())[0].shape) - 1  # remove b
        else:
            is_dict = False
            n_spatial_dims = len(any_target.shape) - 2  # remove b, c
        spatial_dims = ["Height", "H", "W"][-n_spatial_dims:]
        results_xr = dict()
        for base_key in unique_keys:
            coords = dict()
            keys = [k for k in results.keys() if k.endswith(base_key)]
            prefixes = [k.split("_")[0] for k in keys]
            if has_ens_dim and "preds" in keys[0]:
                cat_dim = 2
                dims = ["ens", "B", "T"]
                coords["ens"] = np.arange(1, self.num_predictions + 1)
            else:
                cat_dim = 1
                dims = ["B", "T"]
            if prefixes[0].startswith("t") and len(prefixes[0]) <= 3:
                coords["T"] = [int(p[1:]) for p in prefixes]
            else:
                coords["T"] = prefixes

            if is_dict:
                dims += spatial_dims
                # Stack per-channel into an xr dataset
                for vari in any_target.keys():
                    name_vari = f"{base_key}_{vari}"
                    cat_values = np.stack([results[k][vari] for k in keys], axis=cat_dim).astype(np.float32)
                    results_xr[name_vari] = xr.DataArray(cat_values, dims=dims, name=name_vari, coords=coords)

            else:
                dims += ["C"] + spatial_dims
                # we want (ens, B, T, C, H, W) or (B, T, C, H, W)
                cat_values = np.stack([results[k] for k in keys], axis=cat_dim).astype(np.float32)
                # print(f"cat_values.shape: {cat_values.shape}, original shape: {results[keys[0]].shape}, dims={dims}")
                results_xr[base_key] = xr.DataArray(cat_values, dims=dims, name=base_key, coords=coords)

        # to xr_dataset
        xr_dataset = xr.Dataset(results_xr)
        return xr_dataset

    def use_ensemble_predictions(self, split: str) -> bool:
        return self.num_predictions > 1 and split in ["val", "test", "predict"] + self.test_set_names

    def use_stacked_ensemble_inputs(self, split: str) -> bool:
        return True

    def get_ensemble_inputs(
        self, inputs_raw: Optional[Tensor], split: str, add_noise: bool = True, flatten_into_batch_dim: bool = True
    ) -> Optional[Tensor]:
        """Get the inputs for the ensemble predictions"""
        if inputs_raw is None:
            return None
        elif not self.use_stacked_ensemble_inputs(split):
            return inputs_raw  # we can sample from the Gaussian distribution directly after the forward pass
        elif self.use_ensemble_predictions(split):
            # create a batch of inputs for the ensemble predictions
            num_predictions = self.num_predictions
            if isinstance(inputs_raw, (dict, TensorDictBase)):
                inputs = {
                    k: self.get_ensemble_inputs(v, split, add_noise, flatten_into_batch_dim)
                    for k, v in inputs_raw.items()
                }
                if isinstance(inputs_raw, TensorDictBase):
                    # Transform back to TensorDict
                    original_bs = inputs_raw.batch_size
                    inputs = TensorDict(inputs, batch_size=[num_predictions * original_bs[0]] + list(original_bs[1:]))
            else:
                if isinstance(inputs_raw, Sequence):
                    inputs = np.array([inputs_raw] * num_predictions)
                elif add_noise:
                    inputs = torch.stack(
                        [
                            inputs_raw + self.inputs_noise * torch.randn_like(inputs_raw)
                            for _ in range(num_predictions)
                        ],
                        dim=0,
                    )
                else:
                    inputs = torch.stack([inputs_raw for _ in range(num_predictions)], dim=0)

                if flatten_into_batch_dim:
                    # flatten num_predictions and batch dimensions
                    inputs = rrearrange(inputs, "N B ... -> (N B) ...")
        else:
            inputs = inputs_raw
        return inputs

    def _reshape_ensemble_preds(self, results: TensorDict) -> TensorDict:
        r"""
        Reshape the predictions of an ensemble so that the first dimension is the ensemble dimension, N.

         Args:
                results: Model outputs with shape (N * B, ...), where N is the number of ensemble members and B is the batch size.

        Returns:
            The reshaped predictions (i.e. each output_var_prediction has shape (N, B, *)).
        """
        batch_size = results.shape[0] // self.num_predictions
        results = results.reshape(self.num_predictions, batch_size, *results.shape[1:])
        return results

    def _evaluation_get_preds(
        self, outputs: List[Any], split: str
    ) -> Dict[str, Union[torch.distributions.Normal, np.ndarray]]:
        if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], list):
            outputs = outputs[0]
        use_ensemble = self.use_ensemble_predictions(split)
        outputs_keys, results = outputs[0].keys(), dict()
        for key in outputs_keys:
            # print(key, outputs[0][key].keys())   # e.g. t3_preds_normed, ['inputs3d', 'inputs2d']
            batch_axis = 1 if (use_ensemble and "targets" not in key and "true" not in key) else 0
            results[key] = concatenate_array_dicts(outputs, batch_axis, keys=[key])[key]
        return results

    def on_validation_epoch_start(self) -> None:
        self._start_validation_epoch_time = time.time()
        val_loaders = self.datamodule.val_dataloader()
        n_val_loaders = len(val_loaders) if isinstance(val_loaders, list) else 1
        self.aggregators_val = []
        for i in range(n_val_loaders):
            self.aggregators_val.append(self.get_epoch_aggregators(split="val", dataloader_idx=i))

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        kwargs["aggregators"] = self.aggregators_val[dataloader_idx or 0]
        results = self.evaluation_step(batch, batch_idx, split="val", dataloader_idx=dataloader_idx, **kwargs)
        results = torch_to_numpy(results)
        # self._validation_step_outputs.append(results)  # uncomment to save all val predictions
        return results

    def ensemble_logging_infix(self, split: str) -> str:
        """No '/' in front of the infix! But '/' at the end!"""
        s = "" if self.logging_infix == "" else f"{self.logging_infix}/".replace("//", "/")
        # if self.inputs_noise > 0.0 and split != "val":
        # s += f"{self.inputs_noise}eps/"
        # s += f"{self.num_predictions}ens_mems{self.WANDB_LAST_SEP}"
        s += f"{self.WANDB_LAST_SEP}"
        return s

    def on_validation_epoch_end(self) -> None:
        # val_outputs = self._evaluation_get_preds(self._validation_step_outputs)
        self._validation_step_outputs = []
        val_stats, total_mean_metrics_all = self._on_eval_epoch_end(
            "val",
            time_start=self._start_validation_epoch_time,
            data_split_names=self.validation_set_names,
            aggregators=self.aggregators_val,
        )

        # If monitoring is enabled, check that it is one of the monitored metrics
        if self.trainer.sanity_checking:
            monitors = [self.monitor]
            for ckpt_callback in self.trainer.checkpoint_callbacks:
                if hasattr(ckpt_callback, "monitor") and ckpt_callback.monitor is not None:
                    monitors.append(ckpt_callback.monitor)
            val_stats_keys = list(val_stats.keys()) + ["val/loss"]
            for monitor in monitors:
                assert monitor in val_stats_keys, (
                    f"Monitor metric {monitor} not found in {val_stats.keys()}. "
                    f"\nTotal mean metrics: {total_mean_metrics_all}"
                )
        return val_stats

    def _on_eval_epoch_end(
        self,
        split: str,
        time_start: float,
        data_split_names: List[str] = None,
        aggregators: List[Dict[str, Callable]] = None,
    ) -> Tuple[Dict[str, float], List[str]]:
        logging_infix = self.ensemble_logging_infix(split=split).rstrip("/")
        is_global_zero = getattr(self.trainer, "is_global_zero", True)
        # Need the following to log some metrics for consistency with old formats
        is_ps_data = "PhysicalSystemsBenchmarkDataModule" in self.datamodule_config.get("_target_", "None")

        val_time = time.time() - time_start
        split_name = "val" if split == "val" else split
        val_stats = {
            f"time/{split_name}": val_time,
            "num_predictions": self.num_predictions,
            "noise_level": self.inputs_noise,
            "epoch": float(self.current_epoch),
            "global_step": self.global_step,
            "eval_batch_size": self.datamodule_config.get("eval_batch_size"),
            "world_size": self.trainer.world_size,
        }
        val_media = {"epoch": self.current_epoch, "global_step": self.global_step}
        data_split_names = data_split_names or [split]

        skip_temporal_metrics_after = 64
        total_mean_metrics_all = []
        # Loop over dataloader's
        for prefix, aggregators in zip(data_split_names, aggregators):
            label = f"{prefix}/{logging_infix}".rstrip("/")  # e.g. "val/5ens_mems" or "val"
            per_variable_mean_metrics = defaultdict(list)
            temporal_metrics_logged = 0
            # Loop over aggregators
            for agg_name, agg in aggregators.items():
                agg_start = time.time()
                if agg_name == "save_to_disk":
                    metadata = self.get_logger_metadata()
                    agg.compute(prefix=label, epoch=self.current_epoch, metadata=metadata)
                    agg_dt = time.time() - agg_start
                    if is_global_zero and agg_dt > 10:
                        self.log_text.info(f"Eval aggregator '{label}/{agg_name}' took {agg_dt:.1f}s")
                    continue
                # agg.name takes precedence over agg_name as it may better specify the lead time
                agg_name_substrings = agg.name.split("/") if agg.name is not None else []
                agg_name_substrings += agg_name.split("/") if agg_name is not None else []
                agg_name_part_with_t, lead_time = None, None
                for agg_name_substring in agg_name_substrings:
                    if agg_name_substring.startswith("t") and agg_name_substring[1:].isdigit():
                        agg_name_part_with_t = agg_name_substring
                        lead_time = int(agg_name_substring[1:])
                        lead_time_name = "lead_time" if not is_ps_data or split == "val" else "time"
                        break
                    elif all(c.isdigit() for c in agg_name_substring.split("-")):
                        agg_name_part_with_t = agg_name_substring
                        lead_time = agg_name_substring
                        if len(agg_name_substrings.split("-")) == 1:
                            lead_time_name = "Year"
                        elif len(agg_name_substrings.split("-")) == 2:
                            lead_time_name = "Year-Month"
                        else:
                            raise ValueError(f"Unknown lead time format: {agg_name_substring}")
                        break
                    elif agg_name_substring != "" and all(k not in agg_name_substring for k in ["loss", "time_mean"]):
                        print(f"agg_name_substring={agg_name_substring} does not start with 't' and is not a number.")

                # if agg.name is None:  # does not work when using a listaggregator
                #     label = f"{label}/{agg_name}"   # e.g. "val/5ens_mems/t3" or "val/t1"
                logs_metrics, logs_media, logs_own_xaxis = agg.compute(prefix=label, epoch=self.current_epoch)
                agg_dt = time.time() - agg_start
                if is_global_zero and agg_dt > 10:
                    self.log_text.info(f"Eval aggregator '{label}/{agg_name}' took {agg_dt:.1f}s")
                # log.info(f"Aggregator {agg_name} has logs: {logs_metrics.keys()}, {agg_name_substrings=}")
                val_media.update(logs_media)
                if lead_time is not None:
                    # Don't overload the logs with too many temporal metrics (they will be logged as lines below too)
                    if temporal_metrics_logged <= skip_temporal_metrics_after:
                        val_stats.update(logs_metrics)
                else:
                    val_stats.update(logs_metrics)

                # Log the custom x-axis metrics
                for x_axis_name, values_list in logs_own_xaxis.items():
                    # values_list is e.g. wavenumber -> {wv_1: {wv_1: 1, pow: 2}, wv_2: {wv_2: 2, pow: 3}, ...}
                    x_axes = values_list.pop("x_axes")  # Dict of <x_axis_name> -> [<x_values>]
                    x_axes_keys = list(x_axes.keys()) if isinstance(x_axes, dict) else list(x_axes)
                    first_x_axis = x_axes_keys[0]
                    for x_axis in x_axes_keys:
                        # define our custom x axis metric
                        if is_global_zero:
                            try:
                                wandb.define_metric(x_axis)
                            except wandb.errors.errors.Error as e:
                                self.log_text.warning(f"Could not define metric '{x_axis}' in wandb: {e}.")
                    for x_axis_value, values in values_list.items():
                        assert not isinstance(x_axis_value, str), f"{type(x_axis_value)=}. Use a number or timestamp."
                        for value_i_k, values_i in values.items():
                            if value_i_k not in x_axes_keys:
                                for custom_x_axis in x_axes_keys:
                                    # define which metrics will be plotted against it
                                    if is_global_zero:
                                        try:
                                            wandb.define_metric(value_i_k, step_metric=custom_x_axis)
                                        except wandb.errors.errors.Error as e:
                                            self.log_text.warning(
                                                f"Could not define metric '{value_i_k}' in wandb: {e}."
                                            )
                        if first_x_axis not in values.keys():
                            # alternatively, specify exact x-axis value inside values
                            values[first_x_axis] = x_axis_value  # e.g. "sigma" -> 0.02
                        if is_global_zero and self.logger is not None and hasattr(self.logger, "experiment"):
                            self.logger.experiment.log(values)

                if lead_time is None:  # Don't use these aggregators for the mean metrics (not temporal)
                    if "loss" not in agg_name:
                        self.log_text.info(f"Skipping aggregator ``{agg_name}`` for mean metrics.")
                    continue

                # Log the temporal metrics with t<number> as the x-axis
                logs_metrics_no_t = {
                    k.replace(f"{agg_name_part_with_t}/", "").replace("//", "/"): v for k, v in logs_metrics.items()
                }
                if temporal_metrics_logged == 0:
                    if is_global_zero:
                        try:
                            wandb.define_metric(lead_time_name)
                            for k in logs_metrics_no_t.keys():
                                wandb.define_metric(k, step_metric=lead_time_name)
                        except Exception as e:
                            self.log_text.warning(f"Could not define metric '{lead_time_name}' in wandb: {e}.")
                if is_global_zero and self.logger is not None and hasattr(self.logger, "experiment"):
                    self.logger.experiment.log({lead_time_name: lead_time, **logs_metrics_no_t})
                temporal_metrics_logged += 1

                # Compute average metrics over all aggregators I
                for k, v in logs_metrics.items():
                    k_base = k.replace(f"{label}/", "")
                    k_base = re.sub(r"t\d+/", "", k_base)  # remove the /t{t} infix
                    per_variable_mean_metrics[k_base].append(v)

            # Compute average metrics over all aggregators II
            total_mean_metrics = defaultdict(list)
            for k, v in per_variable_mean_metrics.items():
                if logging_infix != "":
                    assert logging_infix not in k, f"Logging infix {logging_infix} found in {k}"
                aggs_mean = np.mean(v)  # Mean over timesteps
                # If there is a "/" separator, remove the variable name into "k_base" stem
                # Split k such that variable is dropped e.g. k= global/rmse/z500 and k_base=global/rmse
                k_base = "/".join(k.split("/")[:-1])
                val_stats[f"{label}/avg/{k}"] = aggs_mean
                if is_ps_data and "vorticity" in k or "velocity" in k:
                    continue  # For consistency, we do not use these metrics for cross-variable averaging

                total_mean_metrics[f"{label}/avg/{k_base}"].append(aggs_mean)

            # Total mean metrics: ['val/avg/l1', 'val/avg/ssr', 'val/avg/rmse', 'val/avg/bias', 'val/avg/grad_mag_percent_diff', 'val/avg/crps', 'inference/avg/l1', etc...]
            # Compute average metrics over all aggregators and variables III
            total_mean_metrics = {k: np.mean(v) for k, v in total_mean_metrics.items()}
            val_stats.update(total_mean_metrics)
            total_mean_metrics_all += list(total_mean_metrics.keys())
        # print(f"Total mean metrics: {total_mean_metrics_all}, 10 values: {dict(list(val_stats.items())[:10])}")
        # NOTE: Sync_dist is False because we assume that aggregators are correctly synchronized across devices
        #  This is true if you use the torchmetrics.Metric class or our own aggregators that base off it
        #  If you do not, sync_dist=True will synchronize the metrics across devices
        #  We set it to False because some weird issues occur when syncronizing the loss aggregator
        self.log_dict(val_stats, sync_dist=False, prog_bar=False)
        # log to experiment
        if is_global_zero and self.logger is not None and hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log(val_media)
        return val_stats, total_mean_metrics_all

    def on_test_epoch_start(self) -> None:
        self._start_test_epoch_time = time.time()
        test_loaders = self.datamodule.test_dataloader()
        self.check_eval_dataset_divisibility(test_loaders, "test")
        n_test_loaders = len(test_loaders) if isinstance(test_loaders, list) else 1
        self.aggregators_test = [
            self.get_epoch_aggregators(split="test", dataloader_idx=i) for i in range(n_test_loaders)
        ]
        test_name = self.test_set_names[0] if len(self.test_set_names) == 1 else "test"
        example_metric = f"{test_name}/{self.ensemble_logging_infix(test_name)}avg/crps"
        if wandb.run is not None and example_metric in wandb.run.summary.keys():
            raise AlreadyLoggedError(f"Testing for ``{test_name}`` data already done.")
        self.log_text.info(f"Starting testing for ``{test_name}`` data.")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        split = self.test_set_names[0 if dataloader_idx is None else dataloader_idx]
        agg = self.aggregators_test[0] if dataloader_idx is None else self.aggregators_test[dataloader_idx]
        results = self.evaluation_step(
            batch, batch_idx, dataloader_idx=dataloader_idx, split=split, aggregators=agg, **kwargs
        )
        results = torch_to_numpy(results)
        self._test_step_outputs[split].append(results)
        return results

    def on_test_epoch_end(self) -> None:
        # for test_split in self._test_step_outputs.keys():
        # self._eval_ensemble_predictions(self._test_step_outputs[test_split], split=test_split)
        self._test_step_outputs = defaultdict(list)
        self._on_eval_epoch_end(
            "test",
            time_start=self._start_test_epoch_time,
            data_split_names=self.test_set_names,
            aggregators=self.aggregators_test,
        )
        self.log_dict({"TESTED": True}, prog_bar=False, sync_dist=False)

    # ---------------------------------------------------------------------- Inference
    def on_predict_start(self) -> None:
        self.on_any_start(stage="predict")
        pdls = self.trainer.predict_dataloaders
        pdls = [pdls] if isinstance(pdls, torch.utils.data.DataLoader) else pdls
        for pdl in pdls:
            assert pdl.dataset.dataset_id == "predict", f"dataset_id is not 'predict', but {pdl.dataset.dataset_id}"

        n_preds = self.num_predictions
        if n_preds > 1:
            self.log_text.info(f"Generating {n_preds} predictions per input with noise level {self.inputs_noise}")

    def on_predict_epoch_start(self) -> None:
        if self.inputs_noise > 0:
            self.log_text.info(f"Adding noise to inputs with level {self.inputs_noise}")
        if self.prediction_outputs_filepath is not None:
            self.log_text.info(f"Predictions will be saved at {self.prediction_outputs_filepath}")
            if self.prediction_outputs_filepath.endswith(".npz"):
                # try to write it with dummy data to make sure it works
                np.savez_compressed(self.prediction_outputs_filepath, dummy=np.zeros((1, 1)))
                os.remove(self.prediction_outputs_filepath)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None, **kwargs):
        """Anything returned here, will be returned when calling trainer.predict(model, datamodule)."""
        results = dict()
        if (
            self.hparams.save_prediction_batches is None
            or self.hparams.save_prediction_batches == "all"
            or batch_idx < self.hparams.save_prediction_batches
        ):
            results = self.evaluation_step(batch, batch_idx, split="predict", **kwargs)
            results = torch_to_numpy(results)  # self._reshape_ensemble_preds(results, split='predict')
            # print(f"batch_idx={batch_idx}", results.keys(), type(results), type(results[list(results.keys())[0]])) # where
            self._predict_step_outputs.append(results)

        return results

    @property
    def prediction_outputs_filepath(self):
        fname = self.hparams.save_predictions_filename
        if not self.trainer.is_global_zero:  # Non-master processes get a dummy filename
            if fname in [True, "True", "auto", "xarray", "zarr"]:
                return f"DUMMY.{fname}"
            elif isinstance(fname, str):
                ending = fname.split(".")[-1]
                return f"DUMMY.{ending}"
            return None

        if fname is not None:
            if fname in [True, "True", "auto", "xarray", "zarr"]:
                ending = "nc" if fname == "xarray" else "zarr" if fname == "zarr" else "npz"
                fname = self.name or ""
                if self.logger is not None and hasattr(self.logger, "experiment"):
                    # fname += f"-{self.logger.experiment.name}"
                    fname += f"-{self.logger.experiment.name.split('_')[-1]}"
                    run_id = self.logger.experiment.id
                    if run_id not in fname:
                        fname += f"-{run_id}"
                if hasattr(self, "prediction_horizon"):
                    fname += f"-hor{self.prediction_horizon}"
                tags = self.logger.experiment.tags if hasattr(self.logger.experiment, "tags") else []
                skip_tags = [
                    "prediction_horizon",
                    "prediction_horizon_long",
                    "ckpt_path",
                    "lookback_window",
                    "logger.wandb",
                    "mode",
                    "regression_overrides",
                    "regression_use_ema",
                    "trainer",
                    "denoiser_clip",
                    "save_prediction",
                    "num_predictions_in_memory",
                    "batch_size",
                    "force_pure_noise_last_frame",
                    "compute_loss_per_sigma",
                    "val_slice",
                    "data_dir",
                    "log_normed",
                    "log_spectra",
                    "log_images",
                    "log_abs_values",
                    "empty_cache_at_autoregressive_step",
                    "prefetch_factor",
                    "num_workers",
                    "from_pretrained_checkpoint_run_id",
                    "from_pretrained_checkpoint_filename",
                    "monitor",
                    "dyn_cond_from_inputs",
                ]
                skip_tags_with_value = [
                    "initialize_window=regression",
                    "regression_ckpt_filename=latest_epoch",
                    "loss_latitude_weighting=True",
                ]
                tags = [
                    t
                    for t in tags
                    if "=" in t
                    and "=null" not in t
                    and not any([st in t for st in skip_tags])
                    and not any([st in t for st in skip_tags_with_value])
                ]
                tags_to_short = dict(
                    regression_run_id="rID",
                    regression_ckpt_filename="rCfname",
                    S_churn="ch",
                    shift_test_times_by="shift",
                    test_filename="fn",
                    num_predictions="ENS",
                    num_steps="N",
                    subsample_predict="subs",
                    yield_denoised="yd",
                    sigma_max_inf="Smax",
                    sigma_min="Smin",
                    possible_initial_times="IC",
                    use_same_dropout_state_for_sampling="DropState",
                    use_cold_sampling_for_intermediate_steps="cI",
                    use_cold_sampling_for_last_step="cL",
                    use_cold_sampling_for_init_of_ar_step="cIAR",
                    refine_intermediate_predictions="rip",
                    max_val_samples="Nval",
                    window="w",
                    enable_inference_dropout="MCd",
                    learned_channel_variance_loss="lvC",
                    residual_pred="res",
                    hourly_resolution="hr",
                    eval_resolution="evalRes",
                )
                tags_to_short["True"] = "T"
                tags_to_short["False"] = "F"
                tags_to_short["optimizer.lr"] = "lr"
                tags_to_short["optimizer.weight_decay"] = "wd"
                tags_to_short["optimizer.muon.lr"] = "mlr"
                tags_to_short["optimizer.muon.wd"] = "mwd"
                tags_to_short["scheduler.warmup_steps"] = "wup"
                tags_to_short["kolmogorov-N256-n_inits16-T250.nc"] = "V1"
                tags_to_short["kolmogorov-N256-n_inits16-T1000.nc"] = "V2"
                tags_clean = []
                for t in tags:
                    t = ".".join(t.split(".")[1:]).replace("-", "").replace(",", "_")
                    t = t.replace("[", "").replace("]", "").replace(" ", "")
                    # Replace apostrophes with nothing
                    t = t.replace("'", "").replace('"', "")

                    for k, v in tags_to_short.items():
                        t = t.replace(k, v)
                    tags_clean.append(t)
                fname += "-TAG--" + "-".join(tags_clean) + "--TAG"
                fname = f"{fname}-epoch{self.current_epoch}.{ending}".lstrip("-").replace("--", "-")
                fname = fname.strip("'")
            work_dir = self.hparams.work_dir if self.hparams.work_dir is not None else "."
            fname = os.path.join(work_dir, "predictions", fname)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "summary"):
                self.logger.experiment.summary["predictions_outputs_filepath"] = str(fname)
            # self.log_dict({"predictions_outputs_filepath": str(fname)}, prog_bar=False, logger=True)
        return fname

    def on_predict_epoch_end(self):
        numpy_results = self._evaluation_get_preds(self._predict_step_outputs, split="predict")
        # for k, v in numpy_results.items(): print(k, v.shape)
        self._predict_step_outputs = []
        fname = self.prediction_outputs_filepath
        if fname is not None:
            self.log_text.info(f"Saving predictions to {os.path.abspath(fname)}")
            if fname.endswith(".nc"):
                self.evaluation_results_to_xarray(numpy_results).to_netcdf(fname)
            else:
                np.savez_compressed(fname, **numpy_results)

        return numpy_results

    # ---------------------------------------------------------------------- Optimizers and scheduler(s)
    @property
    def lr_groups(self):
        """Get the learning rate groups for the optimizer. If None, all parameters have the same lr.
        If a dict, the keys are patterns to match the parameter names and the values are the lr multipliers.
          (i.e. a Dict mapping parameter patterns to learning rate multipliers)
        """
        if self.hparams.from_pretrained_lr_multiplier is not None:
            assert not self.hparams.from_pretrained_frozen, "If frozen, do not change LR"
            # Use a negative match to train the pretrained model with a different (lower) LR
            return {k: self.hparams.from_pretrained_lr_multiplier for k in self.reloaded_state_dict_keys}
        return None

    def _get_optim(
        self,
        optim_name: str,
        model_handle=None,
        lr_groups=None,
        muon: DictConfig = None,
        **kwargs,
    ):
        """
        Method that returns the torch.optim optimizer object.
        May be overridden in subclasses to provide custom optimizers.

        Args:
            optim_name: Name of the optimizer to use
            model_handle: Optional model to optimize (defaults to self)
            lr_groups: Dict mapping parameter patterns to learning rate multipliers
                      e.g. {"temporal_": 2.0, "spatial_": 0.1}
            **kwargs: Additional optimizer arguments
        """
        if optim_name.lower() == "fusedadam":
            try:
                from apex import optimizers
            except ImportError as e:
                raise ImportError(
                    "To use FusedAdam, please install apex. Alternatively, use normal AdamW with ``module.optimizer.name=adamw``"
                ) from e

            optimizer = optimizers.FusedAdam  # set adam_w_mode=False for Adam (by default: True => AdamW)
        elif optim_name.lower() == "adamw":
            optimizer = torch.optim.AdamW
        elif optim_name.lower() == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"Unknown optimizer type: {optim_name}")
        self.log_text.info(f"{optim_name} optim with kwargs: " + str(kwargs))
        model_handle = self if model_handle is None else model_handle
        assert all(k in ["lr", "momentum", "wd"] for k in muon.keys())
        muon_lr = muon.get("lr")
        self.use_muon = muon_lr is not None

        # Handle weight decay setup
        wd_orig = kwargs.get("weight_decay", 0)
        base_lr = kwargs["lr"]
        allow_disable_weight_decay = kwargs.pop("allow_disable_weight_decay", True)

        if allow_disable_weight_decay:
            no_decay_params = {"channel_embed", "pos_embed", "_logvar", "logvars"}
            no_decay_params |= {k for k, _ in model_handle.named_parameters() if "weight" not in k or "norm" in k}
        else:
            no_decay_params = set()

        if hasattr(self.model, "no_weight_decay"):
            no_decay_params = no_decay_params.union(set(self.model.no_weight_decay()))
        if hasattr(self.model, "model") and hasattr(self.model.model, "no_weight_decay"):
            no_decay_params = no_decay_params.union(set(self.model.model.no_weight_decay()))

        non_muon_weights = ["embed", "gate", "output_layer", "out_layer", "out_conv", "output_conv"]
        hidden_matrix_params = [
            n
            for n, p in model_handle.named_parameters()
            if p.ndim >= 2 and all(nw not in n for nw in non_muon_weights)
        ]

        # Initialize parameter groups
        param_groups = {}  #  start empty to ensure that only groups with parameters are created
        muon_groups = {}
        no_grad_params = 0

        # Process each parameter
        unexpect_no_wd_params = []
        for name, param in model_handle.named_parameters():
            if not param.requires_grad:
                no_grad_params += 1
                continue

            # Determine learning rate multiplier
            is_muon = self.use_muon and name in hidden_matrix_params
            curr_lr = muon_lr if is_muon else base_lr
            if lr_groups:
                for pattern, multiplier in lr_groups.items():
                    # allow for negative pattern with "!" prefix, which means "not in"
                    if pattern.startswith("!") and pattern[1:] not in name:
                        curr_lr = base_lr * multiplier
                        break
                    elif pattern in name:
                        curr_lr = base_lr * multiplier
                        break

            # Determine weight decay group
            use_wd = not (wd_orig > 0 and any(nd in name for nd in no_decay_params))
            if is_muon:
                use_wd = use_wd and muon.wd > 0
                group_key = (curr_lr, use_wd, "MUON")
            else:
                # Create group key based on lr and weight decay
                group_key = (curr_lr, use_wd)

            # Initialize group if needed
            param_groups_here = param_groups if not is_muon else muon_groups
            if group_key not in param_groups_here:
                group_kwargs = kwargs.copy()
                group_kwargs["lr"] = curr_lr
                if not use_wd:
                    group_kwargs["weight_decay"] = 0
                    if "bias" not in name and "norm" not in name:
                        unexpect_no_wd_params.append(name)
                if is_muon:
                    group_kwargs["momentum"] = muon.momentum
                    group_kwargs["weight_decay"] = muon.wd
                self.log_text.info(f"{group_key=}, {group_kwargs=}, {name=}")
                param_groups_here[group_key] = {"params": [], **group_kwargs}

            param_groups_here[group_key]["params"].append(param)

        # Log parameter statistics
        total_params_count = len(list(model_handle.parameters()))
        print_txt = f"Found {total_params_count} parameters"
        no_wd_count = sum(len(g["params"]) for g in param_groups.values() if g.get("weight_decay", 0) == 0)
        if no_wd_count > 0:
            print_txt += f", of which {no_wd_count} won't use weight decay ({unexpect_no_wd_params=})"
        if no_grad_params > 0:
            print_txt += f", and {no_grad_params} do not require gradients."
        if len(param_groups) > 1:
            pg_to_n_params = {k: len(v["params"]) for k, v in param_groups.items()}
            print_txt += f"\nUsing {len(param_groups)} parameter groups with (lr, wd) settings: {pg_to_n_params}."
        self.log_text.info(print_txt)

        # Create optimizer with all parameter groups
        optim = optimizer(list(param_groups.values()))
        if self.use_muon:
            assert len(muon_groups) == 1, f"{muon_groups}"
            optim = HybridOptim([optim, Muon(**list(muon_groups.values())[0])])
        return optim

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        if "name" not in to_DictConfig(self.hparams.optimizer).keys():
            self.log_text.info("No optimizer was specified, defaulting to AdamW.")
            self.hparams.optimizer.name = "adamw"

        optim_kwargs = {k: v for k, v in self.hparams.optimizer.items() if k not in ["name", "_target_"]}
        is_gan_model = any(cls.__name__ == "BaseGAN" for cls in type(self.model).__mro__)
        if is_gan_model:
            optimizer = [
                self._get_optim(self.hparams.optimizer.name, model_handle=self.model.generator, **optim_kwargs),
                self._get_optim(self.hparams.optimizer.name, model_handle=self.model.discriminator, **optim_kwargs),
            ]
        else:
            optimizer = self._get_optim(self.hparams.optimizer.name, lr_groups=self.lr_groups, **optim_kwargs)

        # Build the scheduler
        if self.hparams.scheduler is None:
            return optimizer  # no scheduler
        else:
            scheduler_params = to_DictConfig(self.hparams.scheduler)
            if "_target_" not in scheduler_params.keys() and "name" not in scheduler_params.keys():
                raise ValueError(f"Please provide a _target_ or ``name`` for module.scheduler={scheduler_params}!")
            interval = scheduler_params.pop("interval", "step")
            scheduler_target = scheduler_params.get("_target_")
            if (
                scheduler_target is not None
                and "torch.optim" not in scheduler_target
                and ".lr_scheduler." not in scheduler_target
            ):
                # custom LambdaLR scheduler
                scheduler = hydra.utils.instantiate(scheduler_params)
                scheduler = {
                    "scheduler": LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    "interval": interval,
                    "frequency": 1,
                }
            else:
                # To support interval=step, we need to multiply the number of epochs by the number of steps per epoch
                multiply_ep_keys = ["warmup_epochs", "max_epochs", "T_max"]
                if interval == "step" and any(scheduler_params.get(k) is not None for k in multiply_ep_keys):
                    trainer_nsteps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
                    if self.trainer.max_epochs in [None, -1]:
                        pass
                    elif self.trainer.limit_train_batches is None or self.trainer.limit_train_batches == 1:
                        # Sanity check that we get the same
                        n_steps_per_machine = len(self.datamodule.train_dataloader())

                        n_steps = n_steps_per_machine // (
                            self.trainer.num_devices * self.trainer.num_nodes * self.trainer.accumulate_grad_batches
                        )
                        assert n_steps in [
                            trainer_nsteps_per_epoch,
                            trainer_nsteps_per_epoch - 1,
                        ], f"{n_steps}, {trainer_nsteps_per_epoch=} {n_steps_per_machine=}, {self.trainer.num_devices=}, {self.trainer.num_nodes=}, {self.trainer.accumulate_grad_batches=}, {self.trainer.estimated_stepping_batches=}, {self.trainer.max_epochs}"
                    n_steps = trainer_nsteps_per_epoch
                    for key in multiply_ep_keys:
                        if key in scheduler_params:
                            scheduler_params[key] *= n_steps

                wup_epochs = scheduler_params.pop("warmup_epochs", None)
                max_epochs = scheduler_params.pop("max_epochs", None)
                if wup_epochs is not None:
                    assert scheduler_params.get("warmup_steps") is None
                    self.hparams.scheduler.warmup_steps = scheduler_params["warmup_steps"] = wup_epochs

                if max_epochs is not None:
                    assert scheduler_params.get("max_steps") in [None, -1]
                    scheduler_params["max_steps"] = max_epochs
                # Instantiate scheduler
                if scheduler_target is not None:
                    scheduler = hydra.utils.instantiate(scheduler_params, optimizer=optimizer)
                else:
                    assert scheduler_params.get("name") is not None, "Please provide a name for the scheduler."
                    scheduler = get_scheduler(optimizer, **scheduler_params)
                scheduler = {"scheduler": scheduler, "interval": interval, "frequency": 1}

        if self.hparams.monitor is None:
            self.log_text.info(f"No ``monitor`` was specified, defaulting to {self.default_monitor_metric}.")
        if not hasattr(self.hparams, "mode") or self.hparams.mode is None:
            self.hparams.mode = "min"

        if isinstance(scheduler, dict):
            lr_dict = {**scheduler, "monitor": self.monitor}  # , 'mode': self.hparams.mode}
        else:
            lr_dict = {"scheduler": scheduler, "monitor": self.monitor}  # , 'mode': self.hparams.mode}
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    @property
    def monitor(self):
        return self.hparams.monitor

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        super().lr_scheduler_step(scheduler, *args, **kwargs)
        if self.use_muon:
            wup = self.hparams.scheduler.get("warmup_steps", 1000)
            cooldown = wup // 10
            mom_max = self.hparams.optimizer.get("muon", {}).get("momentum", 0.95)
            mom_min = mom_max - 0.1
            num_iterations = self.trainer.estimated_stepping_batches
            mom = get_muon_momentum(
                self.global_step,
                num_iterations,
                muon_warmup_steps=wup,
                muon_cooldown_steps=cooldown,
                momentum_min=mom_min,
                momentum_max=mom_max,
            )
            # optimizer will be a HybridOptim
            for group in scheduler.optimizer.optimizers[-1].param_groups:
                assert "momentum" in group.keys()
                group["momentum"] = mom
            if self.global_step % 100 == 0 and self.logger.experiment is not None:
                self.logger.experiment.log({"muon/momentum": mom})

    def get_logger_metadata(self) -> Dict[str, Any]:
        metadata = dict()
        if self.logger is not None and hasattr(self.logger, "experiment") and self.logger.experiment is not None:
            metadata["id"] = self.logger.experiment.id
            metadata["name"] = self.logger.experiment.name
            metadata["group"] = self.logger.experiment.group
            metadata["project"] = self.logger.experiment.project
            metadata["entity"] = self.logger.experiment.entity
        return metadata

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if not self.use_ema:
            # Remove the model EMA parameters from the state_dict (since unwanted here)
            state_dict = {k: v for k, v in state_dict.items() if "model_ema" not in k}

        if not bool(getattr(self.model.hparams, "residual_diffusion", False)):
            residual_buffer_keys = {"model.residual_rms_ema", "model.residual_rms_initialized"}
            if any(k in state_dict for k in residual_buffer_keys):
                state_dict = {k: v for k, v in state_dict.items() if k not in residual_buffer_keys}
                self.log_text.info(
                    "Ignoring residual-diffusion RMS buffers from checkpoint because "
                    "diffusion.residual_diffusion=False: %s",
                    sorted(residual_buffer_keys),
                )

        logvar_c_key = None
        for k in state_dict.keys():
            if "criterion.preds.channels_logvar" in k:
                logvar_c_key = k
                if self.hparams.learned_spatial_variance_loss and not self.model.hparams.log_vars_learn_per_dim:
                    # Repeat into spatial dims self.model.spatial_shape_out
                    logvar_all_k = k.replace("channels_logvar", "logvars")
                    state_dict[logvar_all_k] = state_dict[k].unsqueeze(-1).unsqueeze(-1)
                    state_dict[logvar_all_k] = state_dict[logvar_all_k].repeat(1, *self.model.spatial_shape_out)
                    state_dict.pop(k)
                    self.log_text.info(f"Repeated {k} into shape {state_dict[logvar_all_k].shape=}")
                elif state_dict[k].ndim == 3 and tuple(state_dict[k].shape[1:]) == (1, 1):
                    state_dict[k] = state_dict[k].squeeze(-1)
                break

        if self.hparams.reset_optimizer:
            strict = False  # Allow loading of partial state_dicts (e.g. fine-tune new layers)
        try:
            super().load_state_dict(state_dict, strict=strict)
        except Exception:
            if self.model.is_3d or (self.is_diffusion_model and self.model.model.is_3d):
                # Unsqueeze third dimension for 3D models
                keys_to_unsqueeze = ["conv", "skip", "qkv", "proj"]
                for k in state_dict.keys():
                    if any(ku in k for ku in keys_to_unsqueeze) and ("weight" in k or "resample_filter" in k):
                        state_dict[k] = state_dict[k].unsqueeze(2)
            # try adding 2 singleton dims to criterion.preds.channels_logvar key
            if self.model.hparams.log_vars_learn_per_dim and logvar_c_key is not None:
                state_dict[logvar_c_key] = state_dict[logvar_c_key].unsqueeze(-1)  # .unsqueeze(-1)
            super().load_state_dict(state_dict, strict=strict)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save a model checkpoint with extra info"""
        script_path = os.environ.get("SCRIPT_NAME", None)
        if script_path is not None:
            checkpoint["script_path"] = script_path.split("/")[-1]  # get only the script name

        # Save the full class name with module path. This is useful to instantiating the correct class when loading
        checkpoint["_target_"] = f"{self.__class__.__module__}.{self.__class__.__name__}"

        # Save wandb run info, if available
        checkpoint["wandb"] = self.get_logger_metadata()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Log the epoch and global step of the loaded checkpoint."""
        if "epoch" in checkpoint.keys():
            self.log_text.info(f"Checkpoint epoch={checkpoint['epoch']}; global_step={checkpoint['global_step']}.")
        if self.hparams.reset_optimizer:
            self.log_text.info("================== Resetting optimizer states ===================")
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
        # else:
        #     # Remove param groups without parameters (e.g. when using EMA)
        #     #  This is due to unclean versions of old code, where optimizer groups with no parameters were saved
        #     if "optimizer_states" in checkpoint.keys():
        #         optimizer_states = checkpoint["optimizer_states"]
        #         for i, state in enumerate(optimizer_states):  # Loop over all optimizers
        #             if "param_groups" in state.keys():
        #                 state["param_groups"] = [pg for pg in state["param_groups"] if len(pg["params"]) > 0]
        #             optimizer_states[i] = state
        #         checkpoint["optimizer_states"] = optimizer_states

    # Monitor GPU Usage
    def print_gpu_memory_usage(
        self,
        prefix: str = "",
        tqdm_bar=None,
        add_description: bool = True,
        keep_old: bool = False,
        empty_cache: bool = False,
    ):
        """Use this function to print the GPU memory usage (logged or in a tqdm bar).
        Use this to narrow down memory leaks, by printing the GPU memory usage before and after a function call
        and checking if the available memory is the same or not.
        Recommended to use with 'empty_cache=True' to get the most accurate results during debugging.
        """
        print_gpu_memory_usage(prefix, tqdm_bar, add_description, keep_old, empty_cache, log_func=self.log_text.info)

    def time_it(self, func: Callable, *args, **kwargs):
        """Time a function call"""
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        # self.log_text.info(f"Function {func.__name__} took {duration:.2f} seconds.")
        return result, duration
