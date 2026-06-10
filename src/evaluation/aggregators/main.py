from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.nn import Module

from src.evaluation.aggregators._abstract_aggregator import AbstractAggregator, _Aggregator
from src.evaluation.aggregators.snapshot import SnapshotAggregator
from src.evaluation.aggregators.spectra import SpectraAggregator
from src.evaluation.aggregators.temporal_metrics import TemporalMetricsAggregator
from src.evaluation.aggregators.timestepwise import MetricAggregator
from src.evaluation.torchmetrics import Metric
from src.utilities.utils import get_logger


log = get_logger(__name__)


class ListAggregator(AbstractAggregator, ABC):
    def __init__(
        self,
        aggregators: List[AbstractAggregator],
        **kwargs,
    ):
        super().__init__(**kwargs)
        agg_names = [agg.name for agg in aggregators]
        # Set to self.name if all aggregators have the same name
        if self.name is None:
            self.name = agg_names[0] if all(name == agg_names[0] for name in agg_names) else None
        self.prefix_name = None  # Do not use prefix_name for ListAggregator,
        assert self._area_weights is None, f"ListAggregator {self.name} should not have area weights"

        self._aggregators = aggregators
        for i, aggregator in enumerate(self._aggregators):
            assert isinstance(aggregator, AbstractAggregator), f"Aggregator {i} is not an AbstractAggregator"
            assert aggregator.name is not None, f"Aggregator {i}: {aggregator} has no name"

    def update(self, **kwargs) -> None:
        for aggregator in self._aggregators:
            aggregator.update(**kwargs)

    def _apply(self, fn: Callable, exclude_state: Sequence[str] = "") -> Module:
        for aggregator in self._aggregators:
            aggregator._apply(fn, exclude_state=exclude_state)
        return self

    def _record_batch(self, **kwargs) -> None:
        raise NotImplementedError("ListAggregator should not be called directly")

    def _get_logs(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        logs_values = {}
        logs_media = {}
        logs_own_xaxis = {}
        for aggregator in self._aggregators:
            logs_values_i, logs_media_i, logs_own_xaxis = aggregator.compute(**kwargs)
            logs_values.update(logs_values_i)
            logs_media.update(logs_media_i)
            logs_own_xaxis.update(logs_own_xaxis)
        return logs_values, logs_media, logs_own_xaxis


class LossAggregator(Metric):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.name = "loss_aggregator"
        self._losses = set()
        self._x_axes = None
        self.add_state("_n_batches", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @torch.inference_mode()
    def update(
        self,
        loss: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ):
        loss_dict = loss if isinstance(loss, dict) else {"loss": loss}  # make sure it's a dict
        for k, v in loss_dict.items():
            # If x-axes, check it's always the same
            if k == "x_axes":
                if self._x_axes is None:
                    self._x_axes = v
                else:
                    assert self._x_axes == v, f"x axes differ: {self._x_axes} != {v}"
                continue
            if isinstance(v, Sequence):
                raise ValueError("LossAggregator does not support sequences")
                v = np.array(v)
            if k not in self._losses:
                self.add_state(f"_{k}_sum", default=torch.tensor(0.0, device=self.device), dist_reduce_fx="sum")
                self._losses.add(k)
            self.__dict__[f"_{k}_sum"] += torch.tensor(v, device=self.device)

        self._n_batches += torch.tensor(1.0, device=self.device)

    @torch.inference_mode()
    def compute(self, prefix: str = "", epoch: Optional[int] = None) -> Dict[str, float]:
        prefix = prefix + "/" if prefix else ""
        logs = {f"{prefix}{k}": float(self.__dict__[f"_{k}_sum"] / self._n_batches) for k in self._losses}
        # for k, v in logs.items():
        #     if v < 0 and "after" not in k and k != "val/loss":
        #         kk = k.replace(prefix, '')
        #         raise ValueError(f"LossAggregator: {k=}, {v=}, {self.__dict__[f'_{kk}_sum']=}, {self._n_batches=}")
        logs_own_xaxis = {}
        # If x-axes, log them separately too
        if self._x_axes is not None:
            logs_own_xaxis = dict(x_axes=self._x_axes)
            first_xaxis = list(self._x_axes.keys())[0]
            for k in list(logs.keys()):
                x_axis_names_in_k = [x_axis_name for x_axis_name in self._x_axes if x_axis_name in k]
                if len(x_axis_names_in_k) == 0:
                    continue  # Skip if x-axis is not in the key (e.g. "loss")
                # This could be something like loss/sigma0.1 -> Need to split it into logs_own_xaxis[0.1]["loss"] = v
                k_split = k.split("/")
                # Remove the x-axis from the key and extract its value
                x_axis_in_k = [x_axis_str for x_axis_str in x_axis_names_in_k if x_axis_str in k]
                assert len(x_axis_in_k) == 1, f"Expected one x-axis name, got {x_axis_in_k=}, {k=}"
                x_axis_in_k = x_axis_in_k[0]
                k_name_with_xaxis = [k_name for k_name in k_split if x_axis_in_k in k_name]
                assert len(k_name_with_xaxis) == 1, f"Expected one x-axis name, got {k_name_with_xaxis=}"
                k_name_with_xaxis = k_name_with_xaxis[0]
                k_axis_value = k_name_with_xaxis
                # LossAggregator: k='loss_per_noise_level/sigma800.000', k_name_with_xaxis='sigma800.000', k_axis_value='sigma800.000', k_split=['loss_per_noise_level', 'sigma800.000']
                # LossAggregator 2: k_axis_value=800.0, k_split=['loss_per_noise_level'], k_no_xaxis='loss_per_noise_level'
                for x_axis_name in x_axis_names_in_k:
                    k_axis_value = k_axis_value.replace(x_axis_name, "")
                # Convert to float (if possible)
                if k_axis_value.replace(".", "").isdigit():
                    k_axis_value = float(k_axis_value)
                k_split.remove(k_name_with_xaxis)
                k_no_xaxis = "/".join(k_split)
                # print(f"LossAggregator 2: {k_axis_value=}, {k_split=}, {k_no_xaxis=}")
                if k_axis_value not in logs_own_xaxis.keys():
                    logs_own_xaxis[k_axis_value] = {}
                assert (
                    k_no_xaxis not in logs_own_xaxis[k_axis_value]
                ), f"Key {k_no_xaxis} already exists in {logs_own_xaxis[k_axis_value]=}"
                #  Remove the original key as we will log it separately
                logs_own_xaxis[k_axis_value][k_no_xaxis] = logs.pop(k)
                if x_axis_in_k not in logs_own_xaxis:
                    # Add x-axes values to logs_own_xaxis
                    logs_own_xaxis[k_axis_value][x_axis_in_k] = k_axis_value
            logs_own_xaxis = {first_xaxis: logs_own_xaxis}
        return logs, {}, logs_own_xaxis


class OneStepAggregator(AbstractAggregator):
    """
    Aggregates statistics for the timestep pairs.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        record_metrics: bool = True,
        record_normed: bool = False,
        record_rmse: bool = True,
        record_rmse_quantiles: Optional[Sequence[float]] = None,
        record_abs_values: bool = False,  # logs absolutes mean and std of preds and targets
        use_snapshot_aggregator: bool = True,
        record_spectra: bool = False,
        metrics_kwargs: dict = None,
        snapshot_kwargs: dict = None,
        spectra_kwargs: dict = None,
        temporal_kwargs: dict = None,
        save_to_path: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        snapshot_agg = mean_agg = spectra_agg = temporal_agg = None

        # record_spectra = "targets"

        if record_spectra == "targets":
            # Do not record anything if we only want to record spectra of targets
            log.info("Recording spectra of targets only, disabling all other metrics and image logging.")
            record_metrics = use_snapshot_aggregator = False

        if record_metrics:
            mean_agg = MetricAggregator(
                area_weights=self._area_weights,
                is_ensemble=self._is_ensemble,
                record_normed=record_normed,
                record_rmse=record_rmse,
                record_rmse_quantiles=record_rmse_quantiles,
                record_abs_values=record_abs_values,
                **(metrics_kwargs or {}),
            )

        if use_snapshot_aggregator:
            snapshot_agg = SnapshotAggregator(
                is_ensemble=self._is_ensemble,
                **(snapshot_kwargs or {}),
            )

        if record_spectra:
            spectra_agg = SpectraAggregator(
                is_ensemble=self._is_ensemble,
                coords=self.coords,
                data_to_log="targets" if record_spectra == "targets" else "preds",
                **(spectra_kwargs or {}),
            )

        if temporal_kwargs is not None and temporal_kwargs.get("run", False):
            temporal_agg = TemporalMetricsAggregator(
                area_weights=self._area_weights, is_ensemble=self._is_ensemble, coords=self.coords, **temporal_kwargs
            )

        self._aggregators: Dict[str, _Aggregator] = {
            "snapshot": snapshot_agg,
            "mean": mean_agg,
            "spectra": spectra_agg,
            "temporal": temporal_agg,
        }

    @torch.inference_mode()
    def _record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any] = None,
    ):
        if len(target_data) == 0:
            raise ValueError("No data in target_data")
        if len(gen_data) == 0:
            raise ValueError("No data in gen_data")

        gen_data = self.preprocess_gen_data(gen_data)
        gen_data_norm = self.preprocess_gen_data(gen_data_norm)
        for k, aggregator in self._aggregators.items():
            if aggregator is None:
                continue
            aggregator.update(
                target_data=target_data,
                gen_data=gen_data,
                target_data_norm=target_data_norm,
                gen_data_norm=gen_data_norm,
                metadata=metadata,
            )

    def preprocess_gen_data(self, gen_data: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        # This function is only there for debugging purposes with the DebugDataModule and DebugAggregator
        return gen_data

    @torch.inference_mode()
    def _get_logs(self, **kwargs) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Returns logs as can be reported to WandB.
        """
        logs, logs_media, logs_own_xaxis = {}, {}, {}
        for agg_type, agg in self._aggregators.items():
            if agg is None:
                continue
            try:
                logs_i_all = agg.compute(**kwargs)
            except ValueError as e:
                log.error(
                    f"Aggregator ``{self.name}/{agg_type}`` has problems.\n" f"Did you forget to record any batches?"
                )
                raise e
            if not isinstance(logs_i_all, tuple):
                assert logs_i_all is None or len(logs_i_all) == 0, f"Expected one dict, got {logs_i_all=}"
                continue

            logs_i, logs_media_i, logs_own_xaxis_i = logs_i_all
            logs.update(logs_i)
            logs_media.update(logs_media_i)
            logs_own_xaxis.update(logs_own_xaxis_i)

        return logs, logs_media, logs_own_xaxis

    def _apply(self, fn: Callable, exclude_state: Sequence[str] = "") -> Module:
        """Apply a function to all metrics in the aggregator (e.g. .to(device))"""
        for agg in self._aggregators.values():
            if isinstance(agg, Metric):
                agg._apply(fn, exclude_state=exclude_state)
        return self


class DebugOneStepAggregator(OneStepAggregator):
    def preprocess_gen_data(self, gen_data: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.zeros_like(gen_data)
