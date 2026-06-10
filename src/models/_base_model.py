from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import hydra
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor

from src.optimization.losses import get_loss
from src.utilities.timing import log_timing
from src.utilities.utils import (
    disable_inference_dropout,
    enable_inference_dropout,
    get_logger,
    raise_error_if_invalid_value,
)


class BaseModel(LightningModule):
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
        name (str): optional string with a name for the model
        verbose (bool): Whether to print/log or not

    Read the docs regarding LightningModule for more information:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    is_3d = False

    def __init__(
        self,
        num_input_channels: int = None,
        num_output_channels: int = None,
        num_output_channels_raw: int = None,  # actual channels. output_channels may be larger when stacking dims
        num_conditional_channels: int = 0,
        num_conditional_channels_non_spatial: int = 0,
        num_temporal_channels: int = None,
        spatial_shape_in: Union[Sequence[int], int] = None,
        spatial_shape_out: Union[Sequence[int], int] = None,
        loss_function: str = "mean_squared_error",
        loss_function_weights: Optional[Dict[str, float]] = None,
        num_training_ensemble_members: int = 1,
        train_ensemble_type: str = "loop",
        noise_dim: int = 0,
        predict_non_spatial_condition: bool = False,
        learned_channel_variance_loss: bool = False,  # Please use module.learned_channel_variance_loss to set this
        learned_spatial_variance_loss: bool = False,  # Please use module.learned_spatial_variance_loss to set this
        log_vars_reduce_op: str = "add",  # "add" or "mul"
        log_vars_learn_per_dim: bool = True,
        upsample_condition_by: int = 1,
        datamodule_config: Optional[DictConfig] = None,
        channel_dim: int = 1,
        debug_mode: bool = False,
        name: str = "",
        verbose: bool = True,
    ):
        super().__init__()
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.monitor
        self.save_hyperparameters(ignore=["verbose", "model"])
        # Get a logger
        self.log_text = get_logger(name=self.__class__.__name__ if name == "" else name)
        self.name = name
        self.verbose = verbose
        if not self.verbose:  # turn off info level logging
            self.log_text.setLevel(logging.WARN)

        self._channel_dim = channel_dim
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_output_channels_raw = num_output_channels_raw
        self.num_temporal_channels = num_temporal_channels
        self.num_conditional_channels = num_conditional_channels
        self.num_conditional_channels_non_spatial = num_conditional_channels_non_spatial
        self.spatial_shape_in = spatial_shape_in
        self.spatial_shape_out = spatial_shape_out or spatial_shape_in
        self.datamodule_config = datamodule_config
        self.predict_non_spatial_condition = predict_non_spatial_condition

        print_text = f"Model: {self.__class__.__name__} with {self.num_input_channels=}, {self.num_output_channels=}"
        if self.spatial_shape_in == self.spatial_shape_out:
            print_text += f", {self.spatial_shape_in=}"
        else:
            print_text += f", {self.spatial_shape_in=}, {self.spatial_shape_out=}"
        print_text += f", {self.num_temporal_channels=}" if self.num_temporal_channels is not None else ""
        print_text += f", {self.num_conditional_channels=}" if self.num_conditional_channels > 0 else ""
        print_text += (
            f", {self.num_conditional_channels_non_spatial=}" if self.num_conditional_channels_non_spatial > 0 else ""
        )
        self.log_text.info(print_text)

        if upsample_condition_by > 1:
            self.upsample_condition = nn.Upsample(scale_factor=upsample_condition_by)

        if loss_function is not None:
            # Get the loss function
            loss_function_name = (
                loss_function if isinstance(loss_function, str) else loss_function.get("_target_", "").split(".")[-1]
            )
            self.loss_function_name = loss_function_name.lower()
            self.loss_function_weights = loss_function_weights if loss_function_weights is not None else {}
            for k in self.loss_function_weights.keys():
                assert k in ["preds", "condition_non_spatial"], f"Invalid loss function key: {k}"

            criterion = self.get_loss_callable()
            print_text = (
                f"Criterion: {criterion} with weights: {self.loss_function_weights}"
                if loss_function_weights
                else f"Criterion: {criterion}"
            )
            self.log_text.info(print_text)
            # Using a dictionary for the criterion, so that we can have multiple loss functions if needed
            if isinstance(criterion, torch.nn.ModuleDict):
                self.criterion = criterion
            elif isinstance(criterion, dict):
                if any(isinstance(v, torch.nn.Module) for v in criterion.values()):
                    self.criterion = torch.nn.ModuleDict(criterion)
                else:
                    self.criterion = criterion
            elif isinstance(criterion, torch.nn.Module):
                self.criterion = torch.nn.ModuleDict({"preds": criterion})
            else:
                self.log_text.debug("Criterion is not a torch.nn.Module! This may cause issues if it contain weights.")
                self.criterion = {"preds": criterion}

        self.ema_scope = None  # EMA scope for the model. May be set by the BaseExperiment instance
        # self._parent_module = None    # BaseExperiment instance (only needed for edge cases)

    @property
    def short_description(self) -> str:
        return self.name if self.name else self.__class__.__name__

    def get_parameters(self) -> list:
        """Return the parameters for the optimizer."""
        return list(self.parameters())

    def _get_loss_callable_from_name_or_config(self, loss_function: str, **kwargs):
        """Return the loss function"""
        if isinstance(loss_function, str):
            loss = get_loss(loss_function, **kwargs)
        elif isinstance(loss_function, dict):
            loss = {k: get_loss(v, **kwargs) for k, v in loss_function.items()}
        else:
            loss = hydra.utils.instantiate(loss_function)
        return loss

    def get_loss_callable(self, reduction: str = "mean", **kwargs):
        """Return the loss function"""
        loss_function = self.hparams.loss_function
        if self.hparams.learned_channel_variance_loss:
            lvdn_to_i_n = kwargs.get("learned_var_dim_name_to_idx_and_n_dims", {})
            lvdn_to_i_n["channels"] = (self.channel_dim, self.num_output_channels_raw)  # Tuple: (idx, n_dims)
            kwargs["learned_var_dim_name_to_idx_and_n_dims"] = lvdn_to_i_n
        if self.hparams.learned_spatial_variance_loss:
            lvdn_to_i_n = kwargs.get("learned_var_dim_name_to_idx_and_n_dims", {})
            n_spatial_dims = len(self.spatial_shape_out)
            for i, dim in enumerate(self.spatial_shape_out):
                lvdn_to_i_n[f"spatial_{i}"] = (-n_spatial_dims + i, dim)  # if n_spatial_dims=2 start from -2, -1
            kwargs["learned_var_dim_name_to_idx_and_n_dims"] = lvdn_to_i_n
        if "learned_var_dim_name_to_idx_and_n_dims" in kwargs.keys():
            kwargs["reduce_op"] = self.hparams.log_vars_reduce_op
            kwargs["learn_per_dim"] = self.hparams.log_vars_learn_per_dim

        loss = self._get_loss_callable_from_name_or_config(loss_function, reduction=reduction, **kwargs)

        if self.predict_non_spatial_condition:
            assert not isinstance(
                loss, dict
            ), "Multiple loss functions are not supported for conditional_non_spatial_preds"
            # Don't use the same loss for condition_non_spatial_preds (e.g. no weighted MSE)
            loss = dict(preds=loss, condition_non_spatial=torch.nn.MSELoss())  # todo: balance loss
        return loss

    @property
    def num_params(self):
        """Returns the number of parameters in the model"""
        return sum(p.numel() for p in self.get_parameters() if p.requires_grad)

    @property
    def channel_dim(self):
        return self._channel_dim

    def evaluation_results_to_xarray(self, results: Dict[str, np.ndarray], **kwargs) -> Dict[str, xr.DataArray]:
        """Convert the evaluation results to a xarray dataset"""
        raise NotImplementedError(f"Please implement ``evaluation_results_to_xarray`` for {self.__class__.__name__}")

    @log_timing(name="model_forward", log_every_n=50)
    def forward(self, X: Tensor, condition: Tensor = None, member: int = None, **kwargs):
        r"""Standard ML model forward pass (to be implemented by the specific ML model).

        Args:
            X (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`
            member (int): If an int, 0 indicated 1st ens. member, > 1 additional members.
                Models that can cache part of the forward pass may use this for improved efficiency.
        Shapes:
            - Input: :math:`(B, *, C_{in})`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{in}` is the number of input features/channels.
        """
        raise NotImplementedError("Base model is an abstract class!")

    def initialize_non_spatial_conditioning(
        self,
        non_spatial_conditioning_mode,
        non_spatial_cond_hdim: int,
        null_embedding_for_non_spatial_cond: str = None,
    ):
        valid_cond_modes = ["cross_attn", "adaLN", None]
        raise_error_if_invalid_value(non_spatial_conditioning_mode, valid_cond_modes, "non_spatial_conditioning_mode")
        if non_spatial_conditioning_mode == "adaLN" and self.hparams.with_time_emb:
            raise ValueError("adaLN is not compatible with with_time_emb=True")
        if self.num_conditional_channels_non_spatial == 0:
            assert non_spatial_conditioning_mode is None, "non_spatial_conditioning_mode is not None"
            self.non_spatial_conditioning_mode = None
            self.non_spatial_cond_hdim = None
            assert (
                non_spatial_cond_hdim is None
            ), "non_spatial_cond_hdim is not None but non_spatial_conditioning_mode is None"
        else:
            assert non_spatial_conditioning_mode is not None, "non_spatial_conditioning_mode is None"
            self.log_text.info(f"Using non_spatial_conditioning_mode: ``{non_spatial_conditioning_mode}``")
            self.non_spatial_conditioning_mode = non_spatial_conditioning_mode

            if non_spatial_cond_hdim is not None:
                assert non_spatial_cond_hdim > 0, "non_spatial_cond_hdim is not greater than 0"
                self.non_spatial_cond_hdim = non_spatial_cond_hdim
                non_spatial_cond_hdim_int = non_spatial_cond_hdim
                if self.num_conditional_channels_non_spatial > 2 * non_spatial_cond_hdim:
                    non_spatial_cond_hdim_int *= 2
                self.non_spatial_cond_preprocessing = nn.Sequential(
                    nn.Linear(self.num_conditional_channels_non_spatial, non_spatial_cond_hdim_int),
                    nn.GELU(),
                    nn.Linear(non_spatial_cond_hdim_int, non_spatial_cond_hdim),
                )
            else:
                self.log_text.info("non_spatial_cond_hdim is None, using Identity for non_spatial_cond_preprocessing")
                self.non_spatial_cond_preprocessing = nn.Identity()
                self.non_spatial_cond_hdim = self.num_conditional_channels_non_spatial

            if self.non_spatial_cond_hdim is not None and self.non_spatial_cond_hdim > 0:
                if null_embedding_for_non_spatial_cond == "learn":
                    _non_spatial_cond_null_emb = torch.zeros(self.non_spatial_cond_hdim)
                    self._non_spatial_cond_null_emb = torch.nn.Parameter(
                        _non_spatial_cond_null_emb, requires_grad=True
                    )
                elif null_embedding_for_non_spatial_cond == "zeros":  # not trainable
                    self.register_buffer("_non_spatial_cond_null_emb", torch.zeros(self.non_spatial_cond_hdim))
                else:
                    self._non_spatial_cond_null_emb = None
                    assert null_embedding_for_non_spatial_cond is None

    def preprocess_non_spatial_conditioning(self, condition_non_spatial: Tensor):
        if self.non_spatial_conditioning_mode == "adaLN" and self._non_spatial_cond_null_emb is not None:
            norms: torch.Tensor = torch.norm(condition_non_spatial, p=2, dim=1)
            # Create mask for zero embeddings (null)
            null_mask: torch.Tensor = norms == 0
            if null_mask.any():  # replace with null embedding (can be None, if no-op: skip AdaLN)
                condition_non_spatial[null_mask] = self._non_spatial_cond_null_emb
        if condition_non_spatial.shape[-1] != self.non_spatial_cond_hdim:
            condition_non_spatial = self.non_spatial_cond_preprocessing(condition_non_spatial)
        return condition_non_spatial

    def concat_condition_if_needed(
        self,
        inputs: Tensor,
        condition: Tensor = None,
        dynamical_condition: Tensor = None,
        static_condition: Tensor = None,
    ):
        if self.num_conditional_channels > 0:
            # condition NAND dynamical_condition must hold
            if condition is not None and dynamical_condition is not None:
                condition = torch.cat((condition, dynamical_condition), dim=1)
            elif condition is None and dynamical_condition is not None:
                condition = dynamical_condition
            else:
                assert dynamical_condition is None, "oops"
            del dynamical_condition

            # exactly one of condition or static_condition should be not None
            if condition is None and static_condition is None:
                raise ValueError(
                    f"condition and static_condition are both None but num_conditional_channels is {self.num_conditional_channels}"
                )
            elif condition is not None and static_condition is not None:
                # condition.shape = torch.Size([32, 3, 1, 221, 42]), static_condition.shape = torch.Size([32, 2, 3, 221, 42])
                condition = torch.cat((condition, static_condition), dim=1)
            elif condition is None:
                assert static_condition is not None, "condition and static_condition are both None"
                condition = static_condition
            else:
                assert static_condition is None, "condition and static_condition are both not None"

            if hasattr(self, "upsample_condition"):
                condition = self.upsample_condition(condition)

            if condition.shape[1] != self.num_conditional_channels:
                raise ValueError(
                    "Conditional channel mismatch before concatenation: "
                    f"condition has {condition.shape[1]} channels but model expects "
                    f"num_conditional_channels={self.num_conditional_channels}. "
                    "If using residual diffusion with residual_condition_mode='concat', "
                    "configure model.num_conditional_channels to match the concatenated condition channels."
                )
            try:
                # log.info(f"{inputs.shape=}, {condition.shape=}")
                x = torch.cat((inputs, condition), dim=1)
            except RuntimeError as e:
                static_cond_shape = static_condition.shape if static_condition is not None else None
                devices = [
                    str(x.device) if torch.is_tensor(x) else None for x in [inputs, condition, static_condition]
                ]
                raise RuntimeError(f"{inputs.shape=}, {condition.shape=}, {static_cond_shape=} ({devices=})") from e
        else:
            x = inputs
            assert condition is None, "condition is not None but num_conditional_channels is 0"
            assert static_condition is None, "static_condition is not None but num_conditional_channels is 0"
        return x

    @log_timing(name="get_loss", log_every_n=50)
    def get_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
        raw_targets: Tensor = None,
        metadata: Any = None,
        predictions_mask: Optional[Tensor] = None,
        # targets_mask: Optional[Tensor] = None,
        return_predictions: bool = False,
        predictions_post_process: Optional[Callable] = None,
        targets_pre_process: Optional[Callable] = None,
        criterion_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Get the loss for the given inputs and targets.

        Args:
            inputs (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`
            targets (Tensor): Target data tensor of shape :math:`(B, *, C_{out})`
            raw_targets (Tensor): Raw target data tensor of shape :math:`(B, *, C_{out})`
            metadata (Any): Optional metadata
            predictions_mask (Tensor): Mask for the predictions, before computing the loss. Default: None (no mask)
            return_predictions (bool): Whether to return the predictions or not. Default: False.
                                    Note: this will return all the predictions, not just the masked ones (if any).
        """

        def mask_data(data):
            if predictions_mask is not None:
                return data[..., predictions_mask]
            return data

        def postprocess(predictions_raw) -> Tensor:
            if predictions_post_process is not None:
                if torch.is_tensor(predictions_raw):
                    predictions_raw = predictions_post_process(predictions_raw)
                else:
                    # Do post-processing of the predictions (but not other outputs of the model)
                    predictions_raw["preds"] = predictions_post_process(predictions_raw["preds"])
            return predictions_raw

        # Predict
        if torch.is_tensor(inputs):
            inputs = [inputs]

        if getattr(self.criterion["preds"], "require_ensembling", False):
            n_ens_mems = self.hparams.num_training_ensemble_members
            predictions_raw = self.ensemble_forward(postprocess, *inputs, n_ens_mems=n_ens_mems, **kwargs)
        else:
            predictions_raw = postprocess(self(*inputs, **kwargs))

        criterion_kwargs = criterion_kwargs if criterion_kwargs is not None else {}
        if torch.is_tensor(predictions_raw):
            predictions = mask_data(predictions_raw)
            targets = mask_data(targets)
            # assert (
            #     predictions.shape == targets.shape
            # ), f"Be careful: Predictions shape {predictions.shape} != targets shape {targets.shape}. Missing singleton dimensions after batch dim. can be fatal."
            loss = self.criterion["preds"](predictions, targets, **criterion_kwargs)
            assert len(self.loss_function_weights) == 0, "Loss function weights are not supported for this case"
            loss_dict = dict(loss=loss) if torch.is_tensor(loss) else loss
        else:
            loss = 0.0
            loss_dict = dict()
            # For example, base_keys = ["preds", "condition_non_spatial_preds"]
            # With corresponding preds & targets shapes: (B, *, C_out, H, W) & (B, C_non_spatial)
            for k in targets.keys():
                base_key = k.replace("inputs", "preds")
                loss_weight_k = self.loss_function_weights.get(base_key, 1.0)
                predictions_k = mask_data(predictions_raw[base_key])
                targets_k = mask_data(targets[k])
                loss_k = self.criterion[base_key](predictions_k, targets_k, **criterion_kwargs)
                loss_k = {"loss": loss_k} if torch.is_tensor(loss_k) else loss_k
                loss += loss_weight_k * loss_k["loss"]
                for key, value in loss_k.items():
                    loss_dict[f"{key}/{base_key}"] = value.item()

            loss_dict["loss"] = loss  # total loss, used to backpropagate

        if return_predictions:
            return loss_dict, predictions_raw
        return loss_dict

    def ensemble_forward(self, postprocess_func, *inputs, n_ens_mems: int = None, **kwargs):
        return self._ensemble_forward(postprocess_func, *inputs, n_ens_mems=n_ens_mems, **kwargs)

    def _ensemble_forward(self, postprocess_func, *inputs, n_ens_mems: int = None, **kwargs):
        if n_ens_mems is None:
            raise ValueError("n_ens_mems must be provided for ensemble forward")
        if self.hparams.train_ensemble_type == "loop":
            predictions_raw = []
            for member in range(n_ens_mems):
                predictions_raw += [postprocess_func(self(*inputs, member=member, **kwargs))]
            predictions_raw = torch.stack(predictions_raw, dim=0)  # (ens, batch, 1)
        elif self.hparams.train_ensemble_type == "batched":
            inputs_ens = []
            for inp in inputs:
                if not torch.is_tensor(inp):
                    raise TypeError(f"Expected tensor ensemble input, got {type(inp)}")
                inp_ens = inp.unsqueeze(0).expand(n_ens_mems, *inp.shape).reshape(-1, *inp.shape[1:])
                inputs_ens.append(inp_ens)
            # same for kwargs:
            kwargs_ens = {}
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    v_ens = v.unsqueeze(0).expand(n_ens_mems, *v.shape).reshape(-1, *v.shape[1:])
                    kwargs_ens[k] = v_ens
                else:
                    kwargs_ens[k] = v
            predictions_raw = postprocess_func(self(*inputs_ens, **kwargs_ens))
            predictions_raw = predictions_raw.reshape(n_ens_mems, -1, *predictions_raw.shape[1:])  # (ens, batch, 1)
        else:
            raise ValueError(f"Invalid train_ensemble_type: {self.hparams.train_ensemble_type}")
        return predictions_raw

    def predict_forward(self, *inputs: Tensor, metadata: Any = None, **kwargs):
        """Forward pass for prediction. Usually the same as the forward pass,
        but can be different for some models (e.g. sampling in probabilistic models).
        """
        y = self(*inputs, **kwargs)
        return y

    # Auxiliary methods
    @contextmanager
    def inference_dropout_scope(self, condition: bool, context=None):
        assert isinstance(condition, bool), f"Condition must be a boolean, got {condition}"
        if condition:
            enable_inference_dropout(self)
            if context is not None:
                self.log_text.info(f"{context}: Switched to enabled inference dropout")
        try:
            yield None
        finally:
            if condition:
                disable_inference_dropout(self)
                if context is not None:
                    self.log_text.info(f"{context}: Switched to disabled inference dropout")

    def enable_inference_dropout(self):
        """Set all dropout layers to training mode"""
        enable_inference_dropout(self)

    def disable_inference_dropout(self):
        """Set all dropout layers to eval mode"""
        disable_inference_dropout(self)

    def register_buffer_dummy(self, name, tensor, **kwargs):
        try:
            self.register_buffer(name, tensor, **kwargs)
        except TypeError:  # old pytorch versions do not have the arg 'persistent'
            self.register_buffer(name, tensor)
