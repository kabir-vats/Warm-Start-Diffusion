from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

import pytorch_lightning as pl
import torch
from tensordict import TensorDictBase

from src.evaluation.torchmetrics import Metric
from src.utilities.utils import ellipsis_torch_dict_boolean_tensor, get_logger, to_tensordict


class AbstractAggregator(Metric):
    def __init__(
        self,
        is_ensemble: bool = False,
        area_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        name: str | None = None,
        verbose: bool = True,
        coords: Optional[Dict[str, Any]] = None,
        lit_module_handle: Optional[pl.LightningModule] = None,
        preprocess_fn: Callable = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.log_text = get_logger(name=self.__class__.__name__)

        self.mask = mask
        if mask is not None:
            self.log_text.info(f"{name}: Using mask for evaluation of shape {mask.shape}") if verbose else None
            if area_weights is not None:
                area_weights = area_weights[mask]

        if area_weights is not None and verbose:
            prefix = f"{name}: " if name is not None else ""
            self.log_text.info(f"{prefix}Using area weights for evaluation of shape {area_weights.shape}")
        self._area_weights = area_weights
        self._is_ensemble = is_ensemble
        assert name is None or isinstance(name, str), f"Name must be a string, got {name} ({type(name)=})"
        self.name = name
        self.prefix_name = name
        self.coords = coords
        self.preprocess_fn = preprocess_fn
        self.lit_module_handle = lit_module_handle

    @abstractmethod
    def _record_batch(self, **kwargs) -> None: ...

    def update(self, predictions_mask: Optional[torch.Tensor] = None, **kwargs) -> None:
        assert predictions_mask is None, f"Deprecated predictions_mask {predictions_mask}"
        if self.preprocess_fn is not None:
            for key, data in kwargs.items():
                if torch.is_tensor(data):
                    kwargs[key] = self.preprocess_fn(data, key)
                elif hasattr(data, "items"):  # Check if it's a tensordict-like object
                    kwargs[key] = to_tensordict(
                        {k: self.preprocess_fn(v, key) for k, v in data.items()}, find_batch_size_max=True
                    )
                else:
                    kwargs[key] = data

        if self.mask is not None:
            # Apply mask to all tensors
            for key, data in kwargs.items():
                # print(f"{key} Shape before ellipsis_torch_dict_boolean_tensor: {data.shape}")
                if torch.is_tensor(data):
                    kwargs[key] = data[..., self.mask]
                elif isinstance(data, TensorDictBase):
                    kwargs[key] = to_tensordict(
                        {k: ellipsis_torch_dict_boolean_tensor(v, self.mask) for k, v in data.items()},
                        find_batch_size_max=True,
                    )
                else:
                    raise ValueError(f"Unsupported data type {type(data)}")
                # print(f"{key} Shape after ellipsis_torch_dict_boolean_tensor: {kwargs[key].shape}")

        return self._record_batch(**kwargs)

    @torch.inference_mode()
    def compute(self, prefix: str = None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        prefix = "" if prefix is None else prefix
        if "label" in kwargs.keys():
            prefix_prefix = kwargs.pop("label")
            prefix = f"{prefix_prefix}/{prefix}" if prefix_prefix not in prefix else prefix
        if self.prefix_name is not None and self.prefix_name not in prefix:
            prefix = f"{prefix}/{self.prefix_name}"
        prefix = prefix.replace("//", "/").rstrip("/").lstrip("/")
        logs_values, logs_media, logs_own_xaxis = self._get_logs(prefix=prefix, **kwargs)
        return logs_values, logs_media, logs_own_xaxis

    @abstractmethod
    def _get_logs(
        self, prefix: str = "", epoch: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: ...


class _Aggregator(Protocol):
    def compute(self, prefix: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: ...

    """
    Returns a tuple of three dictionaries:
    1. logs_values: A dictionary of scalar values that can be reported to WandB.
    2. logs_media: A dictionary of media that can be reported to WandB.
    3. logs_own_xaxis: A dictionary of media that can be reported to WandB with its own x-axis.
    
    For the latter, return a dict following this structure:
    {
        "<primary_x_axis_name>": {  // E.g. "lead_time", "wavelength", etc.
            "x_axes": <dict-of-all-x-axis-names-to-their-values>,  // Should be at least {<primary_x_axis_name>: [<x_axis_values>]}
            
            <x_axis_value_1>: {     // Specific value of the primary x-axis
                "<metric_name_1>": <metric_value_1>,  // Log metric for this x-axis value
                "<metric_name_2>": <metric_value_2>,  // Log another metric for this x-axis value
                "<primary_x_axis_name>": <x_axis_value_1>,  // This is redundant, but needed for now
                // ... potentially more metrics and x_axes values such as:
                // "<secondary_x_axis_name>": <secondary_x_axis_value>,  // E.g. "lead_time": 24
            },
            
            <x_axis_value_2>: {
                // ... similar structure for the next x-axis value
            },
            // etc.. for all x-axis values
        },
    }
    """

    def update(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any] = None,
        predictions_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None: ...
