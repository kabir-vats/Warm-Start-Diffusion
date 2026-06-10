from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import xarray as xr

from src.evaluation.torchmetrics import (
    ContinuousRankedProbabilityScore,
    GradientMagnitudePercentDifference,
    MeanAbsoluteError,
    MeanError,
    MeanSquaredError,
    Metric,
    SpreadSkillRatio,
)
from src.evaluation.torchmetrics.regression import Average
from src.evaluation.torchmetrics.regression.variance import StdDeviation


class MetricAggregator(Metric):
    """
    Aggregator for mean-reduced metrics.

    These are metrics such as means which reduce to a single float for each batch,
    and then can be averaged across batches to get a single float for the
    entire dataset. This is important because the aggregator uses the mean to combine
    metrics across batches and processors.
    """

    def __init__(
        self,
        area_weights: torch.Tensor,
        is_ensemble: bool,
        record_normed: bool = False,
        record_rmse: bool = True,
        record_rmse_quantiles: Optional[Sequence[float]] = None,
        record_abs_values: bool = False,
        record_ssr_square_dist: bool = False,
        preprocess_fn: Optional[callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._area_weights = area_weights
        self._variable_metrics: Optional[Dict[str, Dict[str, Metric]]] = None
        self.is_ensemble = is_ensemble
        self.record_normed = record_normed
        self.record_rmse = record_rmse
        self.record_rmse_quantiles = tuple(record_rmse_quantiles or ())
        self.record_abs_values = record_abs_values
        self.record_ssr_square_dist = record_ssr_square_dist
        self.preprocess_fn = preprocess_fn

    def _get_variable_metrics(self, gen_data: Mapping[str, torch.Tensor]):
        if self._variable_metrics is None:
            self._variable_metrics = defaultdict(dict)
            device = gen_data[list(gen_data.keys())[0]].device  # any key will do
            gen_data_keys = list(gen_data.keys())
            assert self.device == device, f"Device mismatch: {self.device=} vs {device=}"

            area_weights = None if self._area_weights is None else self._area_weights.to(self.device)

            mse_name = "rmse" if self.record_rmse else "mse"
            mse_squared = not self.record_rmse
            suffixes = ["", "_normed"] if self.record_normed else [""]
            for suffix in suffixes:
                for i, var_name in enumerate(gen_data_keys):
                    self._variable_metrics[f"l1{suffix}"][var_name] = MeanAbsoluteError(weights=area_weights)
                    self._variable_metrics[f"{mse_name}{suffix}"][var_name] = MeanSquaredError(
                        weights=area_weights, squared=mse_squared
                    )

                    if suffix == "_normed" and self.record_rmse:
                        for q in self.record_rmse_quantiles:
                            q_name = f"rmse_normed_p{int(round(q * 100))}"
                            self._variable_metrics[q_name][var_name] = QuantileRMSE(
                                weights=area_weights,
                                quantile=q,
                            )
                    self._variable_metrics[f"bias{suffix}"][var_name] = MeanError(weights=area_weights)
                    self._variable_metrics[f"grad_mag_percent_diff{suffix}"][var_name] = (
                        GradientMagnitudePercentDifference(
                            weights=area_weights, ensemble_dim=0 if self.is_ensemble else None
                        )
                    )
                    pred_batch_dim = 1 if self.is_ensemble else 0
                    if self.is_ensemble:
                        self._variable_metrics[f"ssr{suffix}"][var_name] = SpreadSkillRatio(
                            weights=area_weights, ensemble_dim=0
                        )
                        self._variable_metrics[f"crps{suffix}"][var_name] = ContinuousRankedProbabilityScore(
                            weights=area_weights
                        )

                    if self.record_abs_values:
                        # todo: Implement weighted std
                        self._variable_metrics[f"mean_gen{suffix}"][var_name] = Average(
                            weights=area_weights, source="pred", batch_dim=pred_batch_dim
                        )
                        self._variable_metrics[f"std_gen{suffix}"][var_name] = StdDeviation(
                            # weights=area_weights,
                            source="pred",
                            dim="except_1",
                        )
                        self._variable_metrics[f"mean_target{suffix}"][var_name] = Average(
                            weights=area_weights, source="target", batch_dim=0
                        )
                        self._variable_metrics[f"std_target{suffix}"][var_name] = StdDeviation(
                            # weights=area_weights,
                            source="target",
                            dim=None,
                        )
            # Move to device
            for metric in self._variable_metrics.values():
                for v in metric.values():
                    v.to(self.device)
        return self._variable_metrics

    @torch.inference_mode()
    def update(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor] = None,
        gen_data_norm: Mapping[str, torch.Tensor] = None,
        metadata: Mapping[str, Any] = None,
    ):
        # Get rank of process
        # import os
        # rank = os.environ.get("RANK", None) or os.environ.get("LOCAL_RANK", None)
        # print(f"Rank: {rank}, {target_data.mean()=}")
        # Be cautious when DDP and eval dataset size not divisible by world size!! (some items will be duplicated)
        if self.preprocess_fn is not None:
            gen_data = self.preprocess_fn(gen_data)
            target_data = self.preprocess_fn(target_data)
            gen_data_norm = self.preprocess_fn(gen_data_norm) if gen_data_norm is not None else None
            target_data_norm = self.preprocess_fn(target_data_norm) if target_data_norm is not None else None

        is_tensor = torch.is_tensor(gen_data)
        if is_tensor:  # add dummy key
            gen_data = {"": gen_data}
            target_data = {"": target_data}
            gen_data_norm = {"": gen_data_norm}
            target_data_norm = {"": target_data_norm}

        variable_metrics = self._get_variable_metrics(gen_data)
        record_normed_list = [True, False] if self.record_normed else [False]
        for is_normed in record_normed_list:
            if is_normed:
                # Use normalized data to compute metrics
                preds_data = gen_data_norm
                truth_data = target_data_norm
                var_metrics_here = {metric: v for metric, v in variable_metrics.items() if "normed" in metric}
            else:
                preds_data = gen_data
                truth_data = target_data
                var_metrics_here = {metric: v for metric, v in variable_metrics.items() if "normed" not in metric}

            for metric in var_metrics_here.keys():
                for var_name, var_preds in preds_data.items():  # e.g. temperature, precipitation, etc
                    if "ssr" in metric or "crps" in metric or "grad_mag" in metric or not self.is_ensemble:
                        preds = var_preds
                    else:
                        preds = var_preds.mean(dim=0)

                    # time_s = time.time()
                    try:
                        variable_metrics[metric][var_name].update(preds, truth_data[var_name])
                    except AssertionError as e:
                        raise AssertionError(f"Error with {metric=}. {var_name=}, {self.is_ensemble=}") from e
                    # time.time() - time_s
                    # print(f"Time taken for {metric} {name} in s: {time_taken:.5f}")

    @torch.inference_mode()
    def compute(self, prefix: str = "", epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Returns logs as can be reported to WandB.

        Args:
            prefix: Label to prepend to all log keys.
            epoch: Current epoch number.
        """
        if self._variable_metrics is None:
            raise ValueError("No batches have been recorded.")
        logs = {}
        prefix = prefix + "/" if prefix else ""
        for i, metric in enumerate(self._variable_metrics):
            for variable, metric_value in self._variable_metrics[metric].items():
                metric_value = metric_value.compute()
                if metric_value is None:
                    raise ValueError(f"{metric=} hasn't been computed for {variable=}. ({prefix=},  {i=})")
                log_key = f"{prefix}{metric}/{variable}".rstrip("/")
                logs[log_key] = float(metric_value.detach().item())
                if metric == "ssr" and self.record_ssr_square_dist:
                    log_key_ssr_sq = f"{prefix}ssr_squared_dist/{variable}".rstrip("/")
                    logs[log_key_ssr_sq] = (1 - logs[log_key]) ** 2  # Distance from 1

        # for key in sorted(logs.keys()):
        # logs[key] = float(logs[key].cpu())  # .numpy()

        return logs, {}, {}

    @torch.inference_mode()
    def get_dataset(self, label: str) -> xr.Dataset:
        logs = self.compute(prefix=label)
        logs = {key.replace("/", "-"): logs[key] for key in logs}
        data_vars = {}
        for key, value in logs.items():
            data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars=data_vars)


class QuantileRMSE(Metric):
    """
    Stores per-sample RMSE values over the epoch and returns a chosen quantile.
    Intended for one-step validation summaries like p95(normed RMSE).
    """
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        quantile: float = 0.95,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights = weights
        self.quantile = quantile
        self.add_state(
            "values",
            default=torch.empty(0, dtype=torch.float32),
            dist_reduce_fx="cat",
        )

    @property
    def percentile_name(self) -> str:
        return f"p{int(round(self.quantile * 100))}"

    def _reduce_over_non_batch_dims(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, ...)
        returns shape: (B,)
        """
        if x.ndim == 1:
            return x

        if self.weights is None:
            return x.reshape(x.shape[0], -1).mean(dim=1)

        w = self.weights.to(device=x.device, dtype=x.dtype)

        # Broadcast area weights onto the trailing dims.
        # Typical case: x=(B,H,W), w=(H,W)
        # Also works if x has extra non-batch dims before spatial dims.
        view_shape = [1] * (x.ndim - w.ndim) + list(w.shape)
        w = w.view(*view_shape)
        w = w.expand_as(x)

        weighted_sum = (x * w).reshape(x.shape[0], -1).sum(dim=1)
        weight_sum = w.reshape(x.shape[0], -1).sum(dim=1).clamp_min(1e-12)
        return weighted_sum / weight_sum

    @torch.inference_mode()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        se = (preds - target).pow(2)
        rmse_per_sample = torch.sqrt(self._reduce_over_non_batch_dims(se).clamp_min(0.0))
        rmse_per_sample = rmse_per_sample.detach().to(
            device=self.values.device,
            dtype=self.values.dtype,
        )
        self.values = torch.cat([self.values, rmse_per_sample], dim=0)

    @torch.inference_mode()
    def compute(self):
        if self.values.numel() == 0:
            raise ValueError("No batches have been recorded.")
        vals = self.values[torch.isfinite(self.values)]
        if vals.numel() == 0:
            raise ValueError("All stored RMSE values are non-finite.")
        return torch.quantile(vals, self.quantile)