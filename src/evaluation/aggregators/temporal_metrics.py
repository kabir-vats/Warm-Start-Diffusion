import datetime
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from torch import Tensor

from src.evaluation.aggregators.save_data import SaveToDiskAggregator
from src.evaluation.metrics import (
    compute_metric_on,
    mean_absolute_error,
    root_mean_squared_error,
    spread_skill_ratio,
    weighted_mean,
    weighted_mean_bias,
    weighted_std,
)
from src.optimization.losses import crps_ensemble
from src.utilities.plotting import create_wandb_figures
from src.utilities.utils import get_logger, rrearrange


log = get_logger(__name__)

metric_functions = {
    "bias": weighted_mean_bias,
    "bias_member_avg": weighted_mean_bias,
    "rmse": root_mean_squared_error,
    "rmse_member_avg": root_mean_squared_error,
    "ssr": spread_skill_ratio,
    "crps": crps_ensemble,
    #
    "mae_variance": mean_absolute_error,
    "mean_variance_target": compute_metric_on(source="target", metric=weighted_mean),
    "mean_variance_gen": compute_metric_on(source="gen", metric=weighted_mean),
    #
    "mean_target": compute_metric_on(source="target", metric=weighted_mean),
    "mean_gen": compute_metric_on(source="gen", metric=weighted_mean),
    "mean_gen_member_avg": compute_metric_on(source="gen", metric=weighted_mean),
    #
    "std_target": compute_metric_on(source="target", metric=weighted_std),
    "std_gen": compute_metric_on(source="gen", metric=weighted_std),
    "std_gen_member_avg": compute_metric_on(source="gen", metric=weighted_std),
}


class TemporalMetricsAggregator:
    """
    Aggregator for temporal metrics
    """

    def __init__(
        self,
        is_ensemble: bool,
        area_weights: Optional[Tensor] = None,
        normalization: str = "raw",
        metrics: List[str] = ["rmse", "crps", "rmse_member_avg"],
        temporal_scale: str = "monthly",
        var_names: Optional[List[str]] = None,
        coords: Optional[Dict[str, np.ndarray]] = None,  # Xarray coordinates
        save_data: bool = False,
        save_to_wandb: bool = False,
        **kwargs,
    ):
        """
        Args:
            metrics: List of metrics to aggregate (e.g. ['rmse', 'mae', 'crps])
            temporal_scale: Temporal scale of the aggregation (e.g. 'monthly', 'yearly')
        """
        self.is_ensemble = is_ensemble
        self.normalization = normalization
        self.temporal_scale = temporal_scale  # 'monthly' or 'yearly'
        self.var_names = var_names
        self.area_weights = None if area_weights is None else area_weights.cpu()
        self.save_data = save_data
        self.save_to_wandb = save_to_wandb
        if metrics == "all":
            metrics = list(metric_functions.keys())
        self.metrics = metrics

        # Build dictionarys to store the aggregated data (i.e, when record_batch is called we only get a batch of data, so we need to store it)
        self._aggregated_generated_data = {}
        self._aggregated_target_data = {}
        self._aggregated_generated_data_norm = {}
        self._aggregated_target_data_norm = {}

        # log parameters
        self._log_dict = {
            "date": {"x_axes": {"date": []}},
        }

        self.coords = coords
        self._device = None

    @torch.inference_mode()
    def update(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record batch of data to be aggregated
        - Data to be generated for all of available years in validation/predict set. (data to be years long)
        """

        def to_tensor(x):
            return x
            # return {k: v.cpu() for k, v in x.items()} if isinstance(x, dict) else x.cpu()

        _target_data = to_tensor(target_data)
        _gen_data = to_tensor(gen_data)
        _target_data_norm = to_tensor(target_data_norm)
        _gen_data_norm = to_tensor(gen_data_norm)
        _metadata = metadata if metadata is not None else {}
        batch_size = target_data.batch_size[0]
        self.var_names = list(_target_data.keys()) if self.var_names is None else self.var_names

        # get the size of the data (excluding the batch size)
        any_var_name = self.var_names[0]
        any_target = _target_data[any_var_name]
        target_shape = any_target.shape[1:]
        gen_shape = _gen_data[any_var_name].shape[0:1] + _gen_data[any_var_name].shape[2:]
        self._device = any_target.device

        for i in range(batch_size):
            # get idx date for batch element
            date = self._time_formatting_function(_metadata["datetime"][i])

            data_collections = {
                "target": (self._aggregated_target_data, _target_data),
                "target_norm": (self._aggregated_target_data_norm, _target_data_norm),
                "generated": (self._aggregated_generated_data, _gen_data),
                "generated_norm": (self._aggregated_generated_data_norm, _gen_data_norm),
            }
            for key, (agg_dict, data) in data_collections.items():
                if date not in agg_dict:
                    zero_tensor = (
                        torch.zeros(target_shape[0], target_shape[1], device=self._device)
                        if "target" in key
                        else torch.zeros(gen_shape[0], gen_shape[1], gen_shape[2], device=self._device)
                    )
                    agg_dict[date] = {
                        # "data_sum": {k: zero_tensor.clone() for k in data.keys()},
                        # 'data_sum2': {k: zero_tensor.clone() for k in data.keys()},
                        "data_mean": {k: zero_tensor.clone() for k in data.keys()},
                        "data_sum2_difference": {k: zero_tensor.clone() for k in data.keys()},
                        "count": 0,
                    }

                # update the count
                agg_dict[date]["count"] += 1

                # calculate mean and variance + sum data for each output variable
                for k, v in data.items():
                    # loop for each output variable - only relevant when using output variables = ['tas', 'pr']
                    batch_slice = v[i, :, :] if "target" in key else v[:, i, :, :]
                    # temp to have old methods of calculating mean to compare results with welford's algorithm
                    # agg_dict[date]["data_sum"][k] += batch_slice
                    # agg_dict[date]["data_sum2"][k] += batch_slice * batch_slice

                    # perform welford's algorithm to calculate mean and variance
                    delta = batch_slice - agg_dict[date]["data_mean"][k]  # get delta
                    agg_dict[date]["data_mean"][k] += delta / agg_dict[date]["count"]  # update mean
                    delta2 = batch_slice - agg_dict[date]["data_mean"][k]  # get delta2
                    agg_dict[date]["data_sum2_difference"][k] += delta * delta2  # update sum of squared differences

    @torch.inference_mode()
    def compute(self, prefix: str = "", epoch: int = None):
        log.info("Getting logs for temporal metrics(extended validation) --- May take a while")
        # asset data to ensure trustworthy evaluation
        # self._assert_data()

        # mean the aggregated data
        data_dict = {
            "generated_mean": {k: v["data_mean"] for k, v in self._aggregated_generated_data.items()},
            "target_mean": {k: v["data_mean"] for k, v in self._aggregated_target_data.items()},
            # "generated_norm": self._mean_aggregated_data(self._aggregated_generated_data_norm),
            # "target_norm": self._mean_aggregated_data(self._aggregated_target_data_norm),
            "generated_variance": self._variance_aggregated_data(self._aggregated_generated_data),
            "target_variance": self._variance_aggregated_data(self._aggregated_target_data),
        }
        # calculate metrics for log
        dates = list(data_dict["generated_mean"].keys())

        logs = self._log_dict.copy()
        image_logs = {}
        ssp = prefix if prefix else ""
        prefix = f"extended_{prefix}/" if prefix else ""
        gen_ens_mean = None
        for date in dates:  # loop for each date separating metrics by the date
            date_as_str = self._timestamp_to_str(date)
            # get date from date
            gen = data_dict["generated_mean"][date]
            target = data_dict["target_mean"][date]
            gen_variance = data_dict["generated_variance"][date]
            target_variance = data_dict["target_variance"][date]
            if self.is_ensemble:
                # get ens means
                gen_ens_mean = {k: v.mean(dim=0) for k, v in gen.items()}

            # get the values for the x-axis
            logs["date"]["x_axes"]["date"].append(date)
            # get metrics for each date
            logs["date"][date] = self._get_metrics(
                target,
                gen,
                gen_ens_mean,
                target_variance,
                gen_variance,
                date,
                self.var_names,
                self.metrics,
                label=prefix,
            )

            # Don't save snapshots for daily data (too many)
            if self.temporal_scale != "daily":
                # get mean snapshots
                image_logs.update(
                    {
                        f"{prefix}snapshots-{self.temporal_scale}-mean/{key}": val
                        for key, val in self._timeAggSnapshots(
                            target, gen, date_as_str, is_ensemble=self.is_ensemble, ssp=ssp
                        ).items()
                    }
                )
                # get variance snapshots
                image_logs.update(
                    {
                        f"{prefix}snapshots-{self.temporal_scale}-variance/{key}": val
                        for key, val in self._timeAggSnapshots(
                            target_variance,
                            gen_variance,
                            date_as_str,
                            is_ensemble=self.is_ensemble,
                            ssp=ssp,
                            show_log_precip=False,
                        ).items()
                    }
                )

        # if timescale is yearly save the data to wandb
        if self.save_to_wandb and self.temporal_scale == "yearly":
            to_save = ["mean", "variance"]
            save_to_disk = SaveToDiskAggregator(
                is_ensemble=self.is_ensemble,
                final_dims_of_data=["latitude", "longitude"],
                var_names=self.var_names,
                max_ensemble_members=None,  # save all ensemble members
                batch_dim_name="datetime",
                save_to_wandb=self.save_to_wandb,
                coords=self.coords,
            )
            for key in to_save:
                # get the dates
                dates_as_str = [self._timestamp_to_str(date) for date in data_dict[f"target_{key}"].keys()]
                # call the record_batch method
                target_tensor = {
                    var: torch.stack([data_dict[f"target_{key}"][date][var] for date in dates])
                    for var in self.var_names
                }
                gen_tensor = {
                    var: torch.stack([data_dict[f"generated_{key}"][date][var] for date in dates])
                    for var in self.var_names
                }
                # fix gen_tensor to have the same shape as what 'save_to_disk' expects
                gen_tensor = rrearrange(gen_tensor, "date ensem lat lon -> ensem date lat lon")
                # call the update method
                save_to_disk.update(
                    target_data=target_tensor, gen_data=gen_tensor, metadata={"datetime": dates_as_str}
                )

                # save the data to wandb
                save_to_disk.compute(
                    label=key,
                )

        return {}, image_logs, logs

    def _get_metrics(self, target, pred, pred_ens_mean, target_variance, pred_variance, date, vars, metrics, label=""):
        computed_metrics = {}
        weights = self.area_weights.to(self._device)
        for var in vars:
            for metric in metrics:
                metric_key = f"{label}/{metric}/{var}".strip("/").replace("//", "/")
                if "member_avg" in metric:
                    computed_metrics[metric_key] = np.mean(
                        [
                            metric_functions[metric](predicted=pred[var][i], truth=target[var], weights=weights).cpu()
                            for i in range(pred[var].shape[0])
                        ]
                    )
                elif "variance" in metric:
                    computed_metrics[metric_key] = float(
                        metric_functions[metric](
                            predicted=pred_variance[var], truth=target_variance[var], weights=weights
                        ).cpu()
                    )
                else:
                    # Do we use the full ensemble or the mean of the ensemble?
                    pred_to_use = pred[var] if metric in ["crps", "ssr"] else pred_ens_mean[var]
                    computed_metrics[metric_key] = float(
                        metric_functions[metric](predicted=pred_to_use, truth=target[var], weights=weights).cpu()
                    )
        # add date with key 'date' for wandb logging purposes
        computed_metrics["date"] = date
        return computed_metrics

    def _timeAggSnapshots(self, target, gen, date, is_ensemble=True, ssp="", **kwargs):
        """
        Generate Snapshots of the data aggregated in time
        Args:
            data_dict: Dictionary with the aggregated data (e.g. self._aggregated_generated_data)
        Returns:
            dictionary with the snapshots of the data aggregated in time (keys: Dates, Values: snapshots figs)
        """
        if not is_ensemble:
            raise NotImplementedError("Only implemented for ensemble data")

        ssp = ssp + "-" if ssp else ""

        snapshots = {}
        for var in self.var_names:
            snapshots_var = create_wandb_figures(target, gen, var, date, self.coords, **kwargs)
            snapshots.update(snapshots_var)

        return snapshots

    def _mean_aggregated_data(self, data_dict):
        """
        Mean the aggregated data dictionary

        Args:
            data_dict: Dictionary with the aggregated data (e.g. self._aggregated_generated_data)
        Returns:
            Dictionary with the aggregated data meaned (Keys: Dates, Values: Mean of the data)
        """
        meaned_data = {}
        for date, agg_data in data_dict.items():
            count = agg_data["count"]
            meaned_data[date] = {k: v / count for k, v in agg_data["data_sum"].items()}
        return meaned_data

    def _variance_aggregated_data(self, data_dict):
        """
        Variance the aggregated data dictionary

        Args:
            data_dict: Dictionary with the aggregated data (e.g. self._aggregated_generated_data)
        Returns:
            Dictionary with the aggregated data variance (Keys: Dates, Values: Variance of the data)
        """
        variance_data = {}
        for date, agg_data in data_dict.items():
            count = agg_data["count"]

            # loop for each output variable
            for var in self.var_names:
                # calculate the variance using welford's algorithm
                variance_welford = agg_data["data_sum2_difference"][var] / count

                # add to the variance data with the var key
                variance_data.setdefault(date, {})[var] = variance_welford

        return variance_data

    def _assert_data(self):
        # assert dates in generated and target data are the same
        assert (
            self._aggregated_generated_data.keys() == self._aggregated_target_data.keys()
        ), "Dates in generated and target data are not the same!"
        # assert that the number of data points for each date corresponds to what we expect (i.e jan - 31 values / feb - 28 values for monthly) // (365 values for yearly)
        month_31 = ["01", "03", "05", "07", "08", "10", "12"]
        month_30 = ["04", "06", "09", "11"]
        month_28 = ["02"]
        if self.temporal_scale == "monthly":
            # check jan and dec have 31 values, where we can infer the year from the keys
            date_keys = list(self._aggregated_generated_data.keys())
            date_counts = [self._aggregated_generated_data[date]["count"] for date in date_keys]
            for i, date in enumerate(date_keys):
                month = date.split("-")[1]
                if month in month_31:
                    assert date_counts[i] == 31, f"Month {month} has {date_counts[i]} values, expected 31"
                elif month in month_30:
                    assert date_counts[i] == 30, f"Month {month} has {date_counts[i]} values, expected 30"
                elif month in month_28:
                    assert date_counts[i] == 28, f"Month {month} has {date_counts[i]} values, expected 28"
                else:
                    raise ValueError(f"Month {month} is not valid!")
        elif self.temporal_scale == "yearly":
            # check that each year has 365 values
            date_keys = list(self._aggregated_generated_data.keys())
            date_counts = [self._aggregated_generated_data[date]["count"] for date in date_keys]
            for i, date in enumerate(date_keys):
                assert date_counts[i] == 365, f"Year {date} has {date_counts[i]} values, expected 365"
        else:
            raise ValueError(f"Temporal scale {self.temporal_scale} is not supported!")

    def _time_formatting_function(self, datetime_obj):
        """
        Helper function to get the function to format datetime object to the desired temporal scale
            returns - Function
        """
        # We use day 15 for monthly and yearly data to ensure the datetime object is in the middle of the month/year
        if self.temporal_scale == "yearly":
            datetime_obj = datetime.datetime(int(datetime_obj.split("-")[0]), 6, 15)
            # datetime_obj = cftime.DatetimeNoLeap(int(datetime_obj.split("-")[0]), 6, 15)
        elif self.temporal_scale == "monthly":
            datetime_obj = datetime.datetime(*[int(i) for i in datetime_obj.split("-")[:2] + ["15"]])
            # datetime_obj = cftime.DatetimeNoLeap(*[int(i) for i in datetime_obj.split("-")[:2] + ["15"]])
        elif self.temporal_scale == "daily":
            datetime_obj = datetime.datetime(*[int(i) for i in datetime_obj.split("-")])
        else:
            raise ValueError(f"Temporal scale {self.temporal_scale} is not supported!")

        # datetime_obj = datetime.datetime(datetime_obj.year, datetime_obj.month, datetime_obj.day)
        # Map to timestamp for visualization on Weights&Biases with corresponding datetime format
        datetime_obj = datetime_obj.timestamp()
        return datetime_obj

    def _timestamp_to_str(self, timestamp):
        datetime_obj = datetime.datetime.fromtimestamp(timestamp)
        if self.temporal_scale == "yearly":
            return datetime_obj.strftime("%Y")
        elif self.temporal_scale == "monthly":
            return datetime_obj.strftime("%Y-%m")
        elif self.temporal_scale == "daily":
            return datetime_obj.strftime("%Y-%m-%d")
        else:
            raise ValueError(f"Temporal scale {self.temporal_scale} is not supported!")
