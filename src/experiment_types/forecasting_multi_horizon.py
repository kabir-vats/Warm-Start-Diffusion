from __future__ import annotations

import inspect
import math
from abc import ABC
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

# This ERA5-MINI transplant keeps only the EDM/ADM paths. These names are used
# by optional Dyffusion/PDERefiner classes below, which are intentionally not
# part of the submitted training setup.
class BaseDYffusion:
    pass


class DYffusionEnd2End:
    pass


class DYffusionEnd2EndSeparateModels:
    pass


class DYffusionMarkov:
    pass


class PDERefiner:
    pass
from src.evaluation.aggregators._abstract_aggregator import _Aggregator
from src.experiment_types._base_experiment import BaseExperiment
from src.interface import NoTorchModuleWrapper
from src.models.ema.ema import LitEma
from src.utilities.checkpointing import reload_checkpoint
from src.utilities.utils import (
    freeze_model,
    log_peak_memory,
    multiply_by_scalar,
    rrearrange,
    run_func_in_sub_batches_and_aggregate,
    split3d_and_merge_variables,
    subtract_if_present,
    to_tensordict,
    torch_select,
    torch_to_numpy,
)


class AbstractMultiHorizonForecastingExperiment(BaseExperiment, ABC):
    PASS_METADATA_TO_MODEL = True

    def __init__(
        self,
        autoregressive_steps: int = 0,
        prediction_timesteps: Optional[Sequence[float]] = None,
        empty_cache_at_autoregressive_step: bool = False,
        inference_val_every_n_epochs: int = 1,
        return_outputs_at_evaluation: str | bool = "auto",
        dyn_cond_from_inputs: bool = True,
        **kwargs,
    ):
        assert autoregressive_steps >= 0, f"Autoregressive steps must be >= 0, but is {autoregressive_steps}"
        assert autoregressive_steps == 0, "Autoregressive steps are not yet supported for this experiment type."
        super().__init__(**kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.autoregressive_steps
        self.save_hyperparameters(ignore=["model"])
        self.USE_TIME_AS_EXTRA_INPUT = False
        self._test_metrics_aggregate = defaultdict(list)
        self._prediction_timesteps = prediction_timesteps
        self.hparams.pop("prediction_timesteps", None)
        if prediction_timesteps is not None:
            self.log_text.info(f"Using prediction timesteps {prediction_timesteps}")

        val_time_range = self.valid_time_range_for_backbone_model
        if hasattr(self.model, "set_min_max_time"):
            self.model.set_min_max_time(min_time=val_time_range[0], max_time=val_time_range[-1])
        elif hasattr(self.model, "model") and hasattr(self.model.model, "set_min_max_time"):
            # For diffusion models
            self.model.model.set_min_max_time(min_time=val_time_range[0], max_time=val_time_range[-1])

    @property
    def horizon_range(self) -> List[int]:
        return list(np.arange(1, self.horizon + 1))

    @property
    def valid_time_range_for_backbone_model(self) -> List[int]:
        return self.horizon_range

    @property
    def true_horizon(self) -> int:
        return self.horizon

    @property
    def horizon_name(self) -> str:
        s = f"{self.true_horizon}h"
        return s

    @property
    def prediction_timesteps(self) -> List[float]:
        """By default, we predict the timesteps in the horizon range (i.e. at data resolution)"""
        return self._prediction_timesteps or self.horizon_range

    @prediction_timesteps.setter
    def prediction_timesteps(self, value: List[float]):
        assert max(value) <= self.horizon_range[-1], f"Prediction range {value} exceeds {self.horizon_range=}"
        self._prediction_timesteps = value

    @property
    def num_autoregressive_steps(self) -> int:
        n_autoregressive_steps = self.hparams.autoregressive_steps
        if n_autoregressive_steps == 0 and self.prediction_horizon is not None:
            n_autoregressive_steps = self.num_autoregressive_steps_for_horizon(self.prediction_horizon)
        return n_autoregressive_steps

    def num_autoregressive_steps_for_horizon(self, horizon: int) -> int:
        return max(1, math.ceil(horizon / self.true_horizon)) - 1

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (h={self.horizon_name})"
        return name

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        # if we use the inputs as conditioning, and use an output-shaped input (e.g. for DDPM),
        # we need to use the output channels here!
        is_dyffusion = self.is_diffusion_model and "dyffusion" in self.diffusion_config._target_.lower()
        if self.is_standard_diffusion:
            return self.actual_num_output_channels(self.dims["output"])
        elif is_dyffusion:
            return num_input_channels  # window is used as conditioning
        if self.stack_window_to_channel_dim:
            return multiply_by_scalar(num_input_channels, self.window)
        return num_input_channels

    @property
    def num_temporal_channels(self) -> Optional[int]:
        """The number of temporal dimensions."""
        if self.stack_window_to_channel_dim:
            return None
        elif self.is_standard_diffusion:
            if self.diffusion_config.get("when_3d_concat_condition_to") != "channel":
                return self.window + self.horizon  # Condition prompt + sequence to denoise
            else:
                return self.horizon  # Sequence to denoise only
        else:
            return self.window

    def get_horizon(self, split: str, dataloader_idx: int = 0) -> int:
        if self.datamodule is not None and hasattr(self.datamodule, "get_horizon"):
            return self.datamodule.get_horizon(split, dataloader_idx=dataloader_idx)
        self.log_text.warning(f"Using default horizon {self.horizon} for split ``{split}``.")
        return self.horizon

    @property
    def prediction_horizon(self) -> int:
        if hasattr(self.datamodule_config, "prediction_horizon") and self.datamodule_config.prediction_horizon:
            return self.datamodule_config.prediction_horizon
        return self.horizon * (self.hparams.autoregressive_steps + 1)

    # def on_train_start(self) -> None:
    # def on_fit_start(self) -> None:
    def on_any_start(self, stage: str = None) -> None:
        super().on_any_start(stage)
        horizon = self.get_horizon(stage)
        ar_steps = self.num_autoregressive_steps_for_horizon(horizon)
        # max_horizon = horizon * (ar_steps + 1)
        if "val" not in stage and ar_steps > 0:
            self.log_text.info(f"Using {ar_steps} autoregressive steps for stage ``{stage}`` with horizon={horizon}.")

    # --------------------------------- Metrics
    def get_epoch_aggregators(self, split: str, dataloader_idx: int = None) -> dict:
        assert split in ["val", "test", "predict"], f"Invalid split {split}"
        is_inference_val = split == "val" and dataloader_idx == 1
        if is_inference_val and self.current_epoch % self.hparams.inference_val_every_n_epochs != 0:
            # Skip inference on validation set for this epoch (for efficiency)
            return {}

        return super().get_epoch_aggregators(split, dataloader_idx)

    @torch.inference_mode()  # torch.no_grad()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        return_outputs: bool | str = None,
        # "auto",  #  True = -> "preds" + "targets". False: None "all": all outputs
        boundary_conditions: Callable = None,
        t0: float = 0.0,
        dt: float = 1.0,
        aggregators: Dict[str, _Aggregator] = None,
        verbose: bool = True,
        prediction_horizon: int = None,
    ):
        return_dict = dict()
        if prediction_horizon is not None:
            assert split == "predict", "Prediction horizon only to be used for split='predict'"
        else:
            prediction_horizon = self.get_horizon(split, dataloader_idx=dataloader_idx)

        return_outputs = return_outputs or self.hparams.return_outputs_at_evaluation
        if return_outputs == "auto":
            return_outputs = "all" if split == "predict" and prediction_horizon < 1500 else False
        no_aggregators = aggregators is None or len(aggregators.keys()) == 0
        if not no_aggregators:
            split3d_and_merge_variables_p = (
                partial(split3d_and_merge_variables, level_names=self.datamodule.hparams.pressure_levels)
                if hasattr(self.datamodule.hparams, "pressure_levels")
                else lambda x: x
            )

        # Get predictions mask if available (applied to preds and targets, e.g. for spatially masked predictions)
        predictions_mask = batch.pop("predictions_mask", None)  # pop to ensure that it's not used in model
        if predictions_mask is not None:
            predictions_mask = predictions_mask[0, ...]  # e.g. (2, 40, 80) -> (40, 80)

        target_dynamics_raw = batch.pop("raw_dynamics", None)  # Unnormalized (raw scale) data, used to compute targets
        if "target_dynamics" in batch.keys():
            target_dynamics_raw = batch.pop("target_dynamics")  # Overwrite if "target_dynamics" is present
            target_dynamics_raw = to_tensordict(target_dynamics_raw, find_batch_size_max=True)

        dynamic_conds = batch.pop("dynamical_condition", None)  # will be added back to batch later, piece by piece
        if prediction_horizon > 20:  # Move to CPU to save GPU memory, if long horizon.
            target_dynamics_raw = target_dynamics_raw.to("cpu") if target_dynamics_raw is not None else None
            dynamic_conds = dynamic_conds.to("cpu") if dynamic_conds is not None else None
        # main_batch = batch.copy()
        # Compute how many autoregressive steps to complete
        if dataloader_idx is not None and dataloader_idx > 0 and no_aggregators:
            self.log_text.info(f"No aggregators for {split=} {dataloader_idx=} {self.current_epoch=}")
            return {}
        else:
            assert split in ["val", "test", "predict"] + self.test_set_names, f"Invalid split {split}"
            n_outer_loops = self.num_autoregressive_steps_for_horizon(prediction_horizon) + 1
            dyn_any = target_dynamics_raw if target_dynamics_raw is not None else batch["dynamics"]
            if dyn_any.shape[1] < prediction_horizon:
                raise ValueError(f"Prediction horizon {prediction_horizon} is larger than {dyn_any.shape}[1]")

        # Remove the last part of the dynamics that is not needed for prediction inside the module/model
        # dynamics = batch["dynamics"].clone()
        batch["dynamics"] = batch["dynamics"][:, : self.window + self.true_horizon, ...]
        if dataloader_idx in [0, None] and not no_aggregators:
            # self._set_loss_weights()  # Set the loss weights (might be needed if only doing validation)
            # log validation loss
            if dynamic_conds is not None:
                # first window of dyn. condition
                batch["dynamical_condition"] = dynamic_conds[:, : self.window + self.true_horizon].to(self.device)
            with log_peak_memory(
                log_mem=self.hparams.log_mem, logger_instance=self, name="forward_peak_gb", prefix=f"memory/{split}/"
            ):
                loss = self.get_loss(batch)
            aggregators["loss"].update(loss=loss)
            # if isinstance(loss, dict):
            #     # add split/ prefix if not already there
            #     log_dict = {f"{split}/{k}" if not k.startswith(split) else k: float(v) for k, v in loss.items()}
            # elif torch.is_tensor(loss):
            #     log_dict = {f"{split}/loss": float(loss)}
            # self.log_dict(log_dict, on_step=False, on_epoch=True)

        # Initialize autoregressive loop
        total_t = t0
        predicted_range_last = [0.0] + self.prediction_timesteps[:-1]
        pbar = tqdm(
            range(n_outer_loops),
            desc="Autoregressive Step",
            position=0,
            leave=True,
            disable=not verbose or n_outer_loops <= 1 or self.global_rank != 0,
        )
        # Loop over autoregressive steps (to cover timesteps beyond training horizon)
        for ar_step in pbar:
            self.print_gpu_memory_usage(tqdm_bar=pbar, empty_cache=self.hparams.empty_cache_at_autoregressive_step)
            ar_window_steps = []
            if self.window <= len(self.prediction_timesteps):
                ar_input_steps = self.prediction_timesteps[-self.window :]
            else:  # window is >= 2; we need to take some ar inputs from current input window
                ar_input_steps = self.prediction_timesteps
                n_ar_from_inputs = self.window - len(self.prediction_timesteps)
                for i in reversed(range(n_ar_from_inputs)):
                    # print(f"Adding AR input from input window at position {i}, {self.window-i-1}")
                    ar_win_i = batch["dynamics"][:, self.window - i - 1, ...]
                    if ar_step == 0:
                        ar_win_i = self.get_ensemble_inputs(ar_win_i, split=split)
                    ar_window_steps += [ar_win_i]

            if dynamic_conds is not None:  # self.true_horizon=1
                # ar_step = 0 --> slice(0, H+1), ar_step = 1 --> slice(H, 2H+1), etc.
                current_slice = slice(ar_step * self.true_horizon, (ar_step + 1) * self.true_horizon + self.window)
                batch["dynamical_condition"] = dynamic_conds[:, current_slice].to(self.device)

            # Loop over training horizon
            for t_step_last, t_step in zip(predicted_range_last, self.prediction_timesteps):
                total_horizon = ar_step * self.true_horizon + t_step
                if total_horizon > prediction_horizon:
                    # May happen if we have a prediction horizon that is not a multiple of the true horizon
                    break
                PREFIX_t = f"t{t_step}_"
                PREDS_NORMED_K = f"{PREFIX_t}preds_normed"
                PREDS_RAW_K = f"{PREFIX_t}preds"
                # When autoregressive, don't predict more predictions than needed (already have an ensemble).
                pr_kwargs = {}  # todo: need to fix this?: if autoregressive_inputs is None else {"num_predictions": 1}
                # If we uncomment the above, need to fix base experiment predict() when N_preds_in_mem < n_predictions

                if float(total_horizon).is_integer() and target_dynamics_raw is not None:
                    target_time = self.window + int(total_horizon) - 1
                    targets_tensor_t = target_dynamics_raw[:, target_time, ...].to(self.device)
                    targets = self.get_target_variants(targets_tensor_t, is_normalized=False)
                else:
                    targets = None

                targets_normed = targets["targets_normed"] if targets is not None else None
                warm_start_target_normed = targets_normed
                if targets_normed is not None and bool(getattr(self.hparams, "residual_pred", False)):
                    # In residual-prediction mode, model outputs are residual-normalized.
                    # Derive the previous state from ground-truth trajectory (not from batch['dynamics'],
                    # which can be ensemble-flattened in autoregressive rollout).
                    prev_time = target_time - 1
                    prev_tensor_t = target_dynamics_raw[:, prev_time, ...].to(self.device)
                    prev_normed = self.get_target_variants(prev_tensor_t, is_normalized=False)["targets_normed"]
                    target_residual_normed = subtract_if_present(targets_normed, prev_normed)
                    warm_start_target_normed = self.get_normalizer("dynamics").normalized_to_residual_normalized(
                        target_residual_normed
                    )

                results = self.get_preds_at_t_for_batch(
                    batch,
                    t_step,
                    split,
                    is_autoregressive=ar_step > 0,
                    ensemble=True,
                    total_horizon=total_horizon,
                    warm_start_target_normed=warm_start_target_normed,
                    **pr_kwargs,
                )
                total_t += dt * (t_step - t_step_last)  # update time, by default this is == dt
                if "condition_non_spatial" in results.keys() and results["condition_non_spatial"] is not None:
                    # Overwrite the batch["condition_non_spatial"] with the autoregressive predictions for next step
                    if batch_idx == 0 and self.current_epoch == 0:
                        shape_old = batch["condition_non_spatial"].shape
                        shape_new = (results["condition_non_spatial"].shape,)
                        if shape_old != shape_new:
                            self.log_text.info(f"AR step: ``condition_non_spatial`` shape: {shape_old} -> {shape_new}")
                    batch["condition_non_spatial"] = results["condition_non_spatial"]

                targets_raw = targets["targets"] if targets is not None else None
                # Apply boundary conditions to predictions, if any
                if boundary_conditions is not None:
                    data_t = target_dynamics_raw[:, target_time, ...]
                    for k in [PREDS_NORMED_K, "preds_autoregressive_init_normed"]:
                        if k in results:
                            results[k] = boundary_conditions(
                                preds=results[k],
                                targets=targets_normed,
                                metadata=batch.get("metadata", None),
                                data=data_t,
                                time=total_t,
                            )
                preds_normed = results.pop(PREDS_NORMED_K)
                if not no_aggregators and "save_to_disk" in aggregators.keys():
                    aggregators["save_to_disk"].update(
                        target_data=targets_raw,
                        gen_data=results[PREDS_RAW_K],
                        target_data_norm=targets_normed,
                        gen_data_norm=preds_normed,
                        concat_dim_key=total_horizon,
                        metadata=batch.get("metadata", None),
                        **{k.replace(PREFIX_t, ""): v for k, v in results.items() if k != PREDS_RAW_K},
                    )
                if return_outputs in [True, "all"]:
                    return_dict[f"t{total_horizon}_targets_normed"] = torch_to_numpy(targets_normed)
                    return_dict[f"t{total_horizon}_preds_normed"] = torch_to_numpy(preds_normed)
                elif return_outputs == "preds_only":
                    return_dict[f"t{total_horizon}_preds_normed"] = torch_to_numpy(preds_normed)

                if return_outputs == "all":
                    return_dict[f"t{total_horizon}_targets"] = torch_to_numpy(targets_raw)
                    return_dict.update(
                        {k.replace(f"t{t_step}", f"t{total_horizon}"): torch_to_numpy(v) for k, v in results.items()}
                    )  # update keys to total horizon (instead of relative horizon of autoregressive step)

                if t_step in ar_input_steps:
                    # if predicted_range == self.horizon_range and window == 1, then this is just the last step :)
                    # Need to keep the last window steps that are INTEGER steps!
                    ar_init = results.pop("preds_autoregressive_init_normed", preds_normed)
                    if self.use_ensemble_predictions(split):
                        ar_init = rrearrange(ar_init, "N B ... -> (N B) ...")  # flatten ensemble dimension
                    ar_window_steps += [ar_init]  # keep t,c,z,h,w

                if not float(total_horizon).is_integer():
                    self.log_text.info(f"Skipping non-integer total horizon {total_horizon}")
                    continue

                if no_aggregators:
                    continue

                with self.timing_scope(context=f"aggregators_{split}", no_op=True):
                    assert predictions_mask is None, "Predictions mask not yet supported for aggregators"
                    pred_data = split3d_and_merge_variables_p(results[PREDS_RAW_K])
                    target_data = split3d_and_merge_variables_p(targets_raw)
                    aggregators[f"t{total_horizon}"].update(
                        target_data=target_data,
                        gen_data=pred_data,
                        target_data_norm=split3d_and_merge_variables_p(targets_normed),
                        gen_data_norm=split3d_and_merge_variables_p(preds_normed),
                        predictions_mask=predictions_mask,
                    )
                    if "time_mean" in aggregators:
                        aggregators["time_mean"].update(
                            target_data=target_data,
                            gen_data=pred_data,
                            predictions_mask=predictions_mask,
                            target_data_norm=None,
                            gen_data_norm=None,
                        )
                del results, targets

            if ar_step < n_outer_loops - 1:  # if not last step, then update dynamics
                autoregressive_inputs = torch.stack(ar_window_steps, dim=1)  # shape (b, window, c, h, w)
                if not torch.is_tensor(autoregressive_inputs):
                    # Rename keys to make clear that these are treated as inputs now
                    for k in list(autoregressive_inputs.keys()):
                        autoregressive_inputs[k.replace("preds", "inputs")] = autoregressive_inputs.pop(k)
                batch["dynamics"] = autoregressive_inputs

            del ar_window_steps

        self.on_autoregressive_loop_end(split, dataloader_idx=dataloader_idx)
        return return_dict

    def on_autoregressive_loop_end(self, split: str, dataloader_idx: int = None, **kwargs):
        pass

    def _prepare_warm_start_target_for_inputs(
        self,
        warm_start_target_normed,
        expected_batch_size: int | None = None,
    ):
        if warm_start_target_normed is None:
            raise ValueError(
                "warm_start_sanity=True requires per-step normalized target (warm_start_target_normed)."
            )
        warm_start_target_in = warm_start_target_normed
        if torch.is_tensor(warm_start_target_in):
            warm_start_target_in = warm_start_target_in.to(self.device)
        else:
            if not hasattr(warm_start_target_in, "to"):
                warm_start_target_in = to_tensordict(warm_start_target_in, find_batch_size_max=True)
            warm_start_target_in = warm_start_target_in.to(self.device)

        warm_start_target = self.pack_data(warm_start_target_in, input_or_output="output")

        if expected_batch_size is not None and warm_start_target.shape[0] != expected_batch_size:
            if expected_batch_size % warm_start_target.shape[0] != 0:
                raise ValueError(
                    "warm_start_sanity target batch mismatch: "
                    f"target_batch={warm_start_target.shape[0]}, expected_batch={expected_batch_size}."
                )
            n_rep = expected_batch_size // warm_start_target.shape[0]
            warm_start_target = warm_start_target.unsqueeze(0).repeat(
                n_rep, *([1] * warm_start_target.ndim)
            )
            warm_start_target = rrearrange(warm_start_target, "n b ... -> (n b) ...")

        return warm_start_target

    def get_preds_at_t_for_batch(
        self,
        batch: Dict[str, Tensor],
        horizon: int | float,
        split: str,
        ensemble: bool = False,
        is_autoregressive: bool = False,
        prepare_inputs: bool = True,
        total_horizon=None,
        warm_start_target_normed=None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        b, t = batch["dynamics"].shape[0:2]  # batch size, time steps
        assert 0 < horizon <= self.true_horizon, f"horizon={horizon} must be in [1, {self.true_horizon}]"

        isi1 = isinstance(self, MHDYffusionAbstract)
        isi2 = isinstance(self, SimultaneousMultiHorizonForecasting)
        isi3 = isinstance(self, MultiHorizonForecastingTimeConditioned)
        isi4 = isinstance(self, PDERefinerModule)
        cache_preds = isi1 or isi2 or isi4
        if not cache_preds or horizon == self.prediction_timesteps[0]:
            if self.prediction_timesteps != self.horizon_range:
                if isi1:
                    self.model.hparams.prediction_timesteps = [p_h for p_h in self.prediction_timesteps]
            # create time tensor full of t_step, with batch size shape
            if prepare_inputs:
                inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(
                    batch, time=horizon, split=split, is_autoregressive=is_autoregressive, ensemble=ensemble
                )
            else:
                inputs = batch.pop(self.inputs_data_key)
                extra_kwargs = batch
                if isi3:
                    extra_kwargs["time"] = torch.full((b,), horizon, device=self.device, dtype=torch.float)

            if bool(getattr(self.model.hparams, "warm_start_sanity", False)):
                warm_start_target = self._prepare_warm_start_target_for_inputs(
                    warm_start_target_normed=warm_start_target_normed,
                    expected_batch_size=int(inputs.shape[0]),
                )
                extra_kwargs["warm_start_target"] = warm_start_target

            # inputs may be a repeated version of batch["dynamics"] for ensemble predictions
            with torch.inference_mode():
                with log_peak_memory(
                    log_mem=self.hparams.log_mem and total_horizon <= 10,
                    logger_instance=self,
                    name=f"predict_peak_gb_t{total_horizon}",
                    prefix=f"memory/{split}",
                ):
                    self._current_preds = self.predict(inputs, **extra_kwargs, **kwargs)
                # for k, v, in {**self._current_preds, "dynamics": batch["dynamics"]}.items():
                # log.info(f"key={k}, shape={v.shape}, min={v.min()}, max={v.max()}, mean={v.mean()}, std={v.std()}")

        if cache_preds:
            # for this model, we can cache the multi-horizon predictions
            preds_key = f"t{horizon}_preds"  # key for this horizon's predictions
            results = {k: self._current_preds.pop(k) for k in list(self._current_preds.keys()) if preds_key in k}
            if horizon == self.horizon_range[-1]:
                assert all(
                    ["preds" not in k or "preds_autoregressive_init" in k for k in self._current_preds.keys()]
                ), (
                    f'{preds_key=} must be the only key containing "preds" in last prediction. '
                    f"Got: {list(self._current_preds.keys())}"
                )
                results = {**results, **self._current_preds}  # add the rest of the results, if any
                del self._current_preds
        else:
            results = {f"t{horizon}_{k}": v for k, v in self._current_preds.items()}
        return results

    def get_inputs_from_dynamics(self, dynamics: Tensor | Dict[str, Tensor]) -> Tensor | Dict[str, Tensor]:
        return dynamics[:, : self.window, ...]  # (b, window, c, lat, lon) at time 0

    def get_condition_from_dynamica_cond(
        self, dynamics: Tensor | Dict[str, Tensor], time: Union[float, Tensor], **kwargs
    ) -> Tensor | Dict[str, Tensor]:
        assert dynamics.shape[1] == self.window + self.horizon, f"{dynamics.shape=}, {self.window=}, {self.horizon=}"
        if self.hparams.dyn_cond_from_inputs:
            dynamics_cond = dynamics[:, self.window - 1 : self.window, ...]
        else:
            # time is 1-based index for horizon (e.g. time=1 -> first horizon step)
            dynamics_cond = dynamics[:, self.window - 1 + time].unsqueeze(1)
        dynamics_cond = self.transform_inputs(dynamics_cond, **kwargs)
        return dynamics_cond

    def transform_inputs(
        self,
        inputs: Tensor,
        time: Tensor = None,
        ensemble: bool = True,
        stack_window_to_channel_dim: bool = None,
        **kwargs,
    ) -> Tensor:
        if stack_window_to_channel_dim is None:
            stack_window_to_channel_dim = self.stack_window_to_channel_dim
        if stack_window_to_channel_dim:
            inputs = rrearrange(inputs, "b window c ... -> b (window c) ...")
        else:
            inputs = rrearrange(inputs, "b window c ... -> b c window ...")  # channels first
        if ensemble:
            inputs = self.get_ensemble_inputs(inputs, **kwargs)
        return inputs

    def get_extra_model_kwargs(
        self,
        batch: Dict[str, Tensor],
        split: str,
        time: Union[Tensor, float] = None,
        ensemble: bool = False,
        is_autoregressive: bool = False,
    ) -> Dict[str, Any]:
        extra_kwargs = dict()
        ensemble_k = ensemble and not is_autoregressive
        if self.USE_TIME_AS_EXTRA_INPUT:
            b = batch["dynamics"].shape[0]
            batch["time"] = (
                time if torch.is_tensor(time) else torch.full((b,), time, device=self.device, dtype=torch.float)
            )

        for k, v in batch.items():
            if k == "dynamics":
                continue
            elif k == "metadata":
                if self.PASS_METADATA_TO_MODEL:
                    extra_kwargs[k] = self.get_ensemble_inputs(v, split=split, add_noise=False) if ensemble_k else v
            elif k == "predictions_mask":
                extra_kwargs[k] = v[0, ...]  # e.g. (2, 40, 80) -> (40, 80)
            elif k in ["static_condition", "time", "lookback"]:
                # Static features or time: simply add ensemble dimension and done
                extra_kwargs[k] = self.get_ensemble_inputs(v, split=split, add_noise=False) if ensemble else v
            elif k == "condition_non_spatial":  # and split not in ["train", "fit"]
                extra_kwargs[k] = self.get_ensemble_inputs(v, split=split, add_noise=False) if ensemble else v
            elif "dynamical_condition" == k:  # k in ["condition", "time_varying_condition"]:
                # Time-varying features
                extra_kwargs[k] = self.get_condition_from_dynamica_cond(
                    v, split=split, time=time, ensemble=ensemble, add_noise=False
                )
            elif k == "x_prev":
                pass  # extra_kwargs[k] = v
            else:
                raise ValueError(f"Unsupported key {k} in batch")
        return extra_kwargs

    def get_inputs(
        self,
        batch: Dict[str, Tensor],
        split: str = None,
        ensemble: bool = False,
        is_autoregressive: bool = False,
    ) -> Tensor:
        inputs = self.get_inputs_from_dynamics(batch["dynamics"])
        ensemble_inputs = ensemble and not is_autoregressive
        inputs = self.pack_data(inputs, input_or_output="input")
        inputs = self.transform_inputs(inputs, split=split, ensemble=ensemble_inputs)
        return inputs

    def get_inputs_and_extra_kwargs(
        self,
        batch: Dict[str, Tensor],
        time: Union[float, Tensor] = None,
        split: str = None,
        ensemble: bool = False,
        is_autoregressive: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        inputs = self.get_inputs(batch, split, ensemble, is_autoregressive)
        extra_kwargs = self.get_extra_model_kwargs(
            batch, split=split, time=time, ensemble=ensemble, is_autoregressive=is_autoregressive
        )
        return inputs, extra_kwargs


class MHDYffusionAbstract(AbstractMultiHorizonForecastingExperiment):
    PASS_METADATA_TO_MODEL = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.diffusion_config is not None, "diffusion config must be set. Use ``diffusion=<dyffusion>``!"
        assert self.diffusion_config.timesteps == self.horizon, "diffusion timesteps must be equal to horizon"


# This class is a subclass of MHDYffusionAbstract for multi-horizon forecasting using diffusion
# models.
class MultiHorizonForecastingDYffusion(MHDYffusionAbstract):
    model: BaseDYffusion

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Problematic when module.torch_compile="model":
        # assert isinstance(
        #     self.model, BaseDYffusion
        # ), f"Model must be an instance of BaseDYffusion, but got {type(self.model)}"
        if hasattr(self.model, "interpolator"):
            # self.log_text.info(f"------------------- Setting num_predictions={self.hparams.num_predictions}")
            self.model.interpolator.hparams.num_predictions = self.hparams.num_predictions
            self.model.interpolator.num_predictions_in_mem = self.num_predictions_in_mem

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if hasattr(self.model, "interpolator"):
            self.model.interpolator._datamodule = self.datamodule

    @property
    def valid_time_range_for_backbone_model(self) -> List[int]:
        return self.model.valid_time_range_for_backbone_model

    def get_condition_from_dynamica_cond(
        self, dynamics: Tensor | Dict[str, Tensor], **kwargs
    ) -> Tensor | Dict[str, Tensor]:
        # selection of times will be handled inside src.diffusion.dyffusion
        return self.transform_inputs(dynamics, stack_window_to_channel_dim=False, **kwargs)

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        split = "train" if self.training else "val"
        dynamics = batch["dynamics"]
        x_last = dynamics[:, -1, ...]
        x_last = self.pack_data(x_last, input_or_output="output")
        inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(batch, split=split, ensemble=False)

        loss = self.model.p_losses(input_dynamics=inputs, xt_last=x_last, **extra_kwargs)
        return loss

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        #  Skip loading the interpolator state_dict, as its weights are loaded in src.diffusion.dyffusion.__init__
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("model.interpolator")}
        return super().load_state_dict(state_dict, strict=False)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        # Pop the interpolator state_dict from the checkpoint, as it is not needed
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if "model.interpolator" not in k}


class DYffusionE2E(MHDYffusionAbstract):
    model: DYffusionEnd2End

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack_window_to_channel_dim = False
        if self.hparams.torch_compile != "model":  # todo: add property that returns the model regardless of compile
            assert isinstance(
                self.model, DYffusionEnd2End
            ), f"Model must be an instance of BaseDYffusion2, but got {type(self.model)}"

    @property
    def valid_time_range_for_backbone_model(self) -> List[int]:
        return self.model.valid_time_range_for_backbone_model

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        c_in = num_input_channels
        if hasattr(self.diffusion_config, "condition_on_x_last") and self.diffusion_config.condition_on_x_last:
            c_in *= 2
        return c_in

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        dynamics = batch["dynamics"]
        x_th = self.pack_data(dynamics[:, -1, ...], input_or_output="output")
        dynamics = self.pack_data(dynamics, input_or_output="input")
        split = "train" if self.training else "val"
        extra_kwargs = self.get_extra_model_kwargs(batch, split=split, ensemble=False, is_autoregressive=False)
        return self.model.p_losses(dynamics=dynamics, x_th=x_th, **extra_kwargs)


class DYffusionMarkovModule(MHDYffusionAbstract):
    model: DYffusionMarkov

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack_window_to_channel_dim = False
        if self.hparams.torch_compile != "model":  # todo: add property that returns the model regardless of compile
            assert isinstance(
                self.model, DYffusionMarkov
            ), f"Model must be an instance of BaseDYffusion2, but got {type(self.model)}"

    @property
    def valid_time_range_for_backbone_model(self) -> List[int]:
        return self.model.valid_time_range_for_backbone_model

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        dynamics = batch["dynamics"]
        dynamics = self.pack_data(dynamics, input_or_output="output")
        split = "train" if self.training else "val"
        extra_kwargs = self.get_extra_model_kwargs(batch, split=split, ensemble=False, is_autoregressive=False)
        return self.model.p_losses(dynamics=dynamics, **extra_kwargs)


class DYffusionE2ESeparateModels(MHDYffusionAbstract):
    model: DYffusionEnd2EndSeparateModels

    def __init__(self, use_ema_interpolator: bool = False, interpolator_ema_decay: float = 0.9999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.model, DYffusionEnd2EndSeparateModels
        ), f"Model must be an instance of DYffusionEnd2EndSeparateModels, but got {type(self.model)}"
        self.stack_window_to_channel_dim = False
        # Initialize the EMA model, if needed
        self.use_ema_interpolator = use_ema_interpolator
        if self.use_ema_interpolator:
            self.interpolator_ema = LitEma(self.interpolator, decay=interpolator_ema_decay)

    @property
    def valid_time_range_for_backbone_model(self) -> List[int]:
        return self.model.valid_time_range_for_backbone_model

    @property
    def model_handle_for_ema(self) -> torch.nn.Module:
        return self.model.model

    @property
    def interpolator(self):
        return self.model.interpolator

    @contextmanager
    def interpolator_ema_scope(self, context=None, force_non_ema: bool = False, condition: bool = None):
        condition = self.use_ema_interpolator if condition is None else condition
        if condition and not force_non_ema:
            self.interpolator_ema.store(self.interpolator.parameters())
            self.interpolator_ema.copy_to(self.interpolator)
            if context is not None:
                self.log_text.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if condition and not force_non_ema:
                self.interpolator_ema.restore(self.interpolator.parameters())
                if context is not None:
                    self.log_text.info(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        # very important to call the parent method, otherwise the EMA of model.model will not be updated
        super().on_train_batch_end(*args, **kwargs)
        if self.use_ema_interpolator:
            self.interpolator_ema(self.interpolator)  # update the interpolator EMA

    def evaluation_step(self, batch: Any, batch_idx: int, split: str, **kwargs) -> Dict[str, Tensor]:
        with self.interpolator_ema_scope():
            return super().evaluation_step(batch, batch_idx, split, **kwargs)

    def get_inputs_from_dynamics(self, dynamics: Tensor | Dict[str, Tensor]) -> Tensor | Dict[str, Tensor]:
        return super().get_inputs_from_dynamics(dynamics).squeeze(1)

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        dynamics = batch["dynamics"]
        split = "train" if self.training else "val"
        extra_kwargs = self.get_extra_model_kwargs(batch, split=split, ensemble=False, is_autoregressive=False)
        return self.model.p_losses(dynamics=dynamics, **extra_kwargs)


class MultiHorizonForecastingTimeConditioned(AbstractMultiHorizonForecastingExperiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.hidden_dims
        self.save_hyperparameters(ignore=["model"])
        self.USE_TIME_AS_EXTRA_INPUT = True

    def get_forward_kwargs(self, batch: Dict[str, Tensor]) -> dict:
        split = "train" if self.training else "val"
        dynamics = batch["dynamics"]
        b, t, c = dynamics.shape[:3]
        # b, t, c, h, w = dynamics.shape  # this may not work when masking the spatial dimensions
        time = torch.randint(1, self.horizon + 1, (b,), device=self.device, dtype=torch.long)
        assert (time >= 1).all() and (
            time <= self.true_horizon
        ).all(), f"Train time must be in [1, {self.true_horizon}], but got {time}"
        # Don't ensemble for validation of forward function losses
        # t0_data = self.get_inputs_from_dynamics(dynamics, split=split, ensemble=False)  # (b, c, h, w) at time 0
        shifted_t = self.window + time - 1  # window = past timesteps we use as input, so we shift by that
        targets = dynamics[torch.arange(b), shifted_t.long(), ...]
        targets = self.pack_data(targets, input_or_output="output")

        inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(batch, time=time, ensemble=False, split=split)
        kwargs = {**extra_kwargs, "inputs": inputs, "targets": targets}
        return kwargs

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        forward_kwargs = self.get_forward_kwargs(batch)
        loss, predictions = self.model.get_loss(return_predictions=True, **forward_kwargs)
        return loss


class AbstractSimultaneousMultiHorizonForecastingModule(AbstractMultiHorizonForecastingExperiment):
    SUPPORT_RESIDUAL_PREDICTION = True
    _horizon_at_once: int = None

    def __init__(self, horizon_at_once: int = None, autoregressive_loss_weights: Sequence[float] = None, **kwargs):
        """Simultaneous multi-horizon forecasting module.

        Args:
            horizon_at_once (int, optional): Number of time steps to forecast at once. Defaults to None.
                If None, then the full horizon is forecasted at once.
                Otherwise, only ``horizon_at_once`` time steps are forecasted at once and trained autoregressively until the full horizon is reached.
        """
        super().__init__(**kwargs)
        self.autoregressive_train_steps = self.horizon // self.horizon_at_once
        if self.autoregressive_train_steps > 1:
            self.log_text.info(f"Training with {self.autoregressive_train_steps=} steps with {self.horizon_at_once=}")

        if autoregressive_loss_weights is None:
            autoregressive_loss_weights = [
                1.0 / self.autoregressive_train_steps for _ in range(self.autoregressive_train_steps)
            ]
        elif autoregressive_loss_weights == "logvar":
            self._ar_logvars = torch.nn.Parameter(
                torch.randn(self.autoregressive_train_steps, requires_grad=True) * 0.01
            )
        else:
            assert len(autoregressive_loss_weights) == self.autoregressive_train_steps
        self.autoregressive_loss_weights = autoregressive_loss_weights

        if self.stack_window_to_channel_dim:
            # Need to reshape the noisy inputs to (b, (t c), h, w), where t = num_time_steps predicted
            # if self.horizon_at_once > 1:
            self.targets_pre_process = partial(rrearrange, pattern="b t c ... -> b (t c) ...", t=self.horizon_at_once)
            # else:
            #     self.targets_pre_process = lambda x: x
            self.predictions_post_process = partial(
                rrearrange, pattern="b (t c) ... -> b t c ...", t=self.horizon_at_once
            )
        else:
            self.targets_pre_process = partial(rrearrange, pattern="b t c ... -> b c t ...", t=self.horizon_at_once)
            self.predictions_post_process = partial(
                rrearrange, pattern="b c t ... -> b t c ...", t=self.horizon_at_once
            )

    @property
    def horizon_at_once(self) -> int:
        if self._horizon_at_once is None:
            self._horizon_at_once = self.hparams.horizon_at_once or self.horizon
            assert self.horizon % self.horizon_at_once == 0, "horizon must be divisible by horizon_at_once"
        return self._horizon_at_once

    @property
    def true_horizon(self) -> int:
        return self.horizon_at_once

    @property
    def horizon_range(self) -> List[int]:
        return list(range(1, self.horizon_at_once + 1))

    def actual_num_output_channels(self, num_output_channels: int) -> int:
        num_output_channels = super().actual_num_output_channels(num_output_channels)
        if self.stack_window_to_channel_dim:
            return multiply_by_scalar(num_output_channels, self.horizon_at_once)
        return num_output_channels

    def unpack_predictions(self, results: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Unpack the predictions from the model. This modifies the input dictionary in-place.
        Args:
           results (Dict[str, Tensor]): The model outputs. Access the predictions via results['preds'].
        """
        # reshape predictions to (b, t, c, h, w), where t = num_time_steps predicted
        # ``b`` corresponds to the batch dimension and potentially the ensemble dimension
        horizon_dim = 1
        preds = self.predictions_post_process(results.pop("preds"))
        assert (
            preds.shape[horizon_dim] == self.horizon_at_once
        ), f"Expected {preds.shape=} with dim {horizon_dim}={self.horizon_at_once}"
        for h in self.horizon_range:
            results[f"t{h}_preds"] = torch_select(preds, dim=horizon_dim, index=h - 1)
            # th_pred.shape = (E*B, C, H, W); E = ensemble, B = batch, C = channels, H = height, W = width

        # Use the following code if you want to keep the original keys and have multiple "preds"
        # results_new = defaultdict(dict)
        # for k, v in results.items():
        #     if "preds" not in k:
        #         continue
        #     assert (
        #         v.shape[horizon_dim] == self.horizon
        #     ), f"Expected {k}={v.shape} with dim-{horizon_dim}={self.horizon}"
        #     for h in self.horizon_range:
        #         results_new[f"t{h}_preds"][k] = torch_select(v, dim=horizon_dim, index=h - 1)

        # new_shape = [d for i, d in enumerate(v.shape) if i != horizon_dim]
        # results_new = TensorDict(results_new, batch_size=new_shape)
        return super().unpack_predictions(results)

    def weigh_ar_loss(self, loss_raw, ar_step: int):
        if self.autoregressive_loss_weights == "logvar":
            return loss_raw / self._ar_logvars[ar_step].exp() + self._ar_logvars[ar_step]
        return loss_raw * self.autoregressive_loss_weights[ar_step]


class SimultaneousMultiHorizonForecasting(AbstractSimultaneousMultiHorizonForecastingModule):
    def __init__(self, timestep_loss_weights: Sequence[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model", "timestep_loss_weights"])

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        dynamics = batch["dynamics"]
        split = "train" if self.training else "val"
        inputs = self.get_inputs(batch, split=split, ensemble=False)

        losses = dict(loss=0.0, raw_loss=0.0)
        for ar_step in range(self.autoregressive_train_steps):
            offset_left = self.window + self.horizon_at_once * ar_step
            offset_right = self.window + self.horizon_at_once * (ar_step + 1)
            targets = dynamics[:, offset_left:offset_right, ...]  # e.g. window: window+1
            extra_kwargs = self.get_extra_model_kwargs(
                batch, split=split, time=self.horizon_at_once * ar_step + 1, ensemble=False
            )
            if self.hparams.residual_pred:
                prev_input_dict = dynamics[:, offset_left - 1 : offset_right - 1, ...]
                # Map normalized target to normalized residual space (taking the last input as reference)
                targets = subtract_if_present(targets, prev_input_dict)
                targets = self.get_normalizer("dynamics").normalized_to_residual_normalized(targets)
            targets = self.pack_data(targets, input_or_output="output")
            # if self.stack_window_to_channel_dim:
            # =========== THE BELOW GIVES TERRIBLE LOSS CURVES ==========================
            # DO NOT DO THIS: targets = rrearrange(targets, "b t c ... -> b (t c) ...") |
            # ===========================================================================
            # targets = self.targets_pre_process(targets)  # This will still do it, but only if t > 1
            if self.model.predict_non_spatial_condition:
                # Forecasting condition_non_spatial too (only last time step)
                targets_non_spatial = batch["condition_non_spatial"][:, offset_right - 1 : offset_right, ...]
                targets_non_spatial = self.model.non_spatial_cond_preprocessing(targets_non_spatial.squeeze(1))
                targets = dict(preds=targets, condition_non_spatial=targets_non_spatial)

            loss_ar_i, preds = self.model.get_loss(
                inputs=inputs,
                targets=targets,
                return_predictions=True,
                # todo: Possible to just reshape targets above and not pass the below to the model?
                predictions_post_process=self.predictions_post_process,
                targets_pre_process=self.targets_pre_process,
                **extra_kwargs,
            )
            loss_ar_i_dict = {"loss": loss_ar_i} if not isinstance(loss_ar_i, dict) else loss_ar_i
            loss_ar_i = loss_ar_i_dict.pop("loss")
            losses["raw_loss"] += float(loss_ar_i_dict.get("raw_loss", loss_ar_i))
            for k, v in loss_ar_i_dict.items():
                k_ar = f"{k}_ar{ar_step}" if ar_step > 0 else k
                losses[k_ar] = float(v) if not isinstance(v, dict) else v

            loss_ar_i = self.weigh_ar_loss(loss_ar_i, ar_step)
            losses["loss"] += loss_ar_i
            losses[f"loss_ar{ar_step}"] = float(loss_ar_i)

            if ar_step < self.autoregressive_train_steps - 1:
                if isinstance(preds, dict):
                    # log.info(f"inputs.shape={inputs.shape}, preds.shape={preds['preds'].shape}")
                    inputs = preds.pop("preds")  # use the predictions as inputs for the next autoregressive step
                    for k, v in preds.items():
                        # log.info(f"Adding {k} to loss_ar_i, shape={v.shape}, before: {extra_kwargs.get(k).shape}")
                        extra_kwargs[k] = v  # overwrite the condition_non_spatial etc. for the next step
                else:
                    inputs = preds
                inputs = inputs[:, -self.window :, ...]  # keep only the last window steps
                inputs = self.transform_inputs(inputs, ensemble=False)

        return losses


class ForecastingStormer(SimultaneousMultiHorizonForecasting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extra_model_kwargs(self) -> dict:
        variables = list(self.datamodule_config.input_vars)
        static_fields = list(self.datamodule_config.static_fields) if self.datamodule_config.static_fields else []
        # if "lat_lon_embeddings" in static_vars, we need to replace it with x, y, z since it adds 3 extra channels
        for vari in static_fields:
            assert vari not in variables, f"Variable {vari} is in both input_vars and static_vars"
            if vari == "lat_lon_embeddings":
                variables.extend(["coord_x", "coord_y", "coord_z"])
            else:
                variables.append(vari)
        return dict(list_variables=variables)


class PDERefinerModule(AbstractSimultaneousMultiHorizonForecastingModule):
    model: PDERefiner

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model"])
        assert isinstance(self.model, PDERefiner), f"model must be an instance of PDERefiner, got {type(self.model)}"

    @property
    def valid_time_range_for_backbone_model(self) -> List[int]:
        return list(
            range(0, int((self.model.scheduler.config.num_train_timesteps - 1) * self.model.time_multiplier) + 1)
        )

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        dynamics = batch[self.target_key]
        split = "train" if self.training else "val"
        inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(batch, split=split, ensemble=False)
        targets = dynamics[:, self.window :, ...]
        if self.stack_window_to_channel_dim:
            targets = rrearrange(targets, "b t c h w -> b (t c) h w")  # (t_step = 1, ..., horizon)

        _ = extra_kwargs.pop("metadata", None)
        # assert len(extra_kwargs.keys()) == 0, f"Extra kwargs must be empty, got {list(extra_kwargs.keys())}"
        pde_refiner_batch = (inputs, targets)
        loss, _, _ = self.model.train_step(pde_refiner_batch, **extra_kwargs)
        return loss


class RollingDiffusion(AbstractMultiHorizonForecastingExperiment):

    def __init__(
        self,
        initialize_window: str = "regression",
        init_truth_shift: int = 0,
        regression_model: Optional[torch.nn.Module] = None,
        regression_run_id: Optional[str] = None,
        regression_local_checkpoint_path: Optional[str] = True,  # automatic search in local dirs
        regression_ckpt_filename: Optional[str] = "last.ckpt",
        regression_use_ema: bool = False,
        regression_inference_dropout: bool = False,
        regression_overrides: Optional[List[str]] = None,  # a dot list, e.g. ["diffusion.num_steps=16"]
        return_noisy_window: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.horizon >= 1, "horizon must be >= 1 for RollingDiffusion"
        assert init_truth_shift >= 0, "Oops"
        self.stack_window_to_channel_dim = False  # do internally in the model
        self._pretrained_is_conditional = False
        if initialize_window == "regression":
            regression_overrides = list(regression_overrides) if regression_overrides is not None else []
            regression_overrides.append("model.verbose=False")
            regression_overrides.append("module.from_pretrained_checkpoint_run_id=null")
            self.log_text.info(f"Regression model overrides: {regression_overrides}")
            regr_model_reloaded = reload_checkpoint(
                run_id=regression_run_id,
                local_checkpoint_path=regression_local_checkpoint_path,
                epoch="last",
                ckpt_filename=regression_ckpt_filename,
                use_ema_weights_only=False,  # Use regression_use_ema to set the EMA weights
                override_key_value=regression_overrides,
                print_name="Init. window regression model",
            )

            self.skip_every_n_reg_step = 1
            time_res_keys = ["hourly_resolution"]
            # Check if any time_res key exists in datamodule and check if different to the reloaded model
            for k in time_res_keys:
                v_here = self.datamodule_config.get(k)
                v_regr = regr_model_reloaded["config"]["datamodule"].get(k)
                if v_here is not None and v_regr is not None and v_here != v_regr:
                    # Time res. must be lower in reloaded model, so that we can skip some time steps for initialization
                    assert v_regr < v_here, f"Time resolution {k} must be lower in reloaded model"
                    assert v_here % v_regr == 0, f"Time resolution {k} must be divisible by {v_regr}"
                    self.skip_every_n_reg_step = v_here // v_regr
                    self.log_text.info(
                        f"Time resolution {k} is different in datamodule ({v_here}) and reloaded model ({v_regr})"
                        f", so we will skip every {self.skip_every_n_reg_step} regression step for initialization."
                    )

            self._regression_model = NoTorchModuleWrapper(regr_model_reloaded["model"].cpu())
            freeze_model(self.regression_model)
            self.regression_model.num_predictions = self.num_predictions
            self.regression_model.num_predictions_in_mem = self.num_predictions_in_mem
        else:
            self._regression_model = None

        self.sampler = None

    @property
    def regression_model(self) -> Optional[torch.nn.Module]:
        if self._regression_model is None:
            return None
        return self._regression_model.module  # unwrap the NoTorchModuleWrapper

    def reload_weights_or_freeze_some(self):
        """Reload weights from a pretrained model, potentially freezing some layers."""
        reloaded_pretrained_ckpt = super().reload_weights_or_freeze_some()
        if reloaded_pretrained_ckpt is not None:
            # Check that reloaded model is consistent with HPs of the current model
            pretrained_unconditional = reloaded_pretrained_ckpt["config"]["diffusion"].get(
                "force_unconditional", False
            )
            if pretrained_unconditional == self.model.hparams.conditional:
                # If the reloaded model is unconditional, the current model must be unconditional
                raise ValueError(f"{pretrained_unconditional=} mustn't be equal to {self.model.hparams.conditional=}")
            self._pretrained_is_conditional = not pretrained_unconditional
        return reloaded_pretrained_ckpt

    @property
    def lr_groups(self):
        if self.hparams.from_pretrained_lr_multiplier is not None:
            assert not self.hparams.from_pretrained_frozen, "If frozen, do not change LR"
            # Use a negative match to train the pretrained model with a different (lower) LR
            return {"!_temporal": self.hparams.from_pretrained_lr_multiplier}
        return None

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        if self.diffusion_config.time_to_channels:
            self.log_text.info(f"Using time_to_channels, {num_input_channels=}, horizon={self.horizon}")
            num_input_channels *= self.horizon
        return num_input_channels

    def actual_num_output_channels(self, num_output_channels: int) -> int:
        if self.diffusion_config.time_to_channels:
            num_output_channels *= self.horizon
        return num_output_channels

    @property
    def num_conditional_channels(self) -> int:
        if self.diffusion_config.conditional:
            return super().num_conditional_channels
        return self.dims.get("conditional", 0)

    @property
    def num_temporal_channels(self) -> int:
        if self.diffusion_config.time_to_channels:
            return 0
        return self.horizon  # check if add + 1 if conditional

    def on_any_start(self, stage: str = None) -> None:
        super().on_any_start(stage)
        if self.regression_model is not None:
            if hasattr(self.regression_model, "sigma_data") and self.regression_model.sigma_data is None:
                self.regression_model.sigma_data = self.datamodule.sigma_data
            if hasattr(self.regression_model.model, "sigma_data") and self.regression_model.model.sigma_data is None:
                self.regression_model.model.sigma_data = self.datamodule.sigma_data
            self.regression_model._datamodule = self.datamodule
        if self.hparams.initialize_window == "climatology":
            assert hasattr(self.datamodule, "climatology"), "datamodule must have a climatology attribute"
            self.climatology = self.datamodule.climatology

    def _reshape_loss_weights(self, loss_weights: Tensor) -> Tensor:
        if len(loss_weights.shape) == 3 and loss_weights.shape[0] == self.model.num_output_channels_raw:
            loss_weights = loss_weights.unsqueeze(1)  # add time dimension
        if len(loss_weights.shape) < 5:
            loss_weights = loss_weights.unsqueeze(0)  # add batch dimension
        return loss_weights

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        split = "train" if self.training else "val"
        targets = batch["dynamics"][:, self.window : self.window + self.horizon, ...]
        targets = self.pack_data(targets, input_or_output="output")
        inputs, extra_kwargs = self.get_inputs_and_extra_kwargs(batch, split=split, ensemble=False)
        assert len(targets.shape) == 5, f"targets.shape={targets.shape}"  # (B, T, C, H, W)

        if self.model.hparams.conditional:  # self.diffusion_config.conditional:
            if self._pretrained_is_conditional:
                cond = batch["dynamics"][:, self.window - 1 : self.window + self.horizon - 1, ...]
                extra_kwargs["condition"] = self.pack_data(cond, input_or_output="input")
            else:
                extra_kwargs["condition"] = inputs
        else:
            del inputs

        if self.model.time_dim == 2:
            # (B, T, C, H, W) -> (B, C, T, H, W)
            targets = rrearrange(targets, "b t c ... -> b c t ...", t=self.horizon)
            for k, v in extra_kwargs.items():
                if torch.is_tensor(v) and len(v.shape) == 5:
                    if k != "condition":
                        self.log_text.warning(
                            f"Permuting {k} from (B, T, C, H, W) to (B, C, T, H, W). shape={v.shape}"
                        )
                    extra_kwargs[k] = rrearrange(v, "b t c ... -> b c t ...", t=self.horizon)
        extra_kwargs["loss_kwargs"] = {"return_intermediate": split == "val"}
        loss = self.model.get_loss(inputs=None, targets=targets, **extra_kwargs)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        to_log = {}
        if self.model.hparams.learnable_schedule:
            to_log["model/rho_proxy"] = self.model.exp.detach().item()
            to_log["model/rho"] = self.model.get_exp().detach().item()
            if self.model.hparams.learnable_schedule == "v2":
                to_log["model/sigma_min"] = self.model.get_tmin().detach().item()
                to_log["model/sigma_max"] = self.model.get_tmax().detach().item()
        if self.model.hparams.variance_loss:
            times_vars = self.get_logvar("times").exp().detach()
            times_vars = {f"train/learned_var/time_{i}": float(v) for i, v in enumerate(times_vars)}
            to_log.update(times_vars)
        if len(to_log) > 0:
            self.log_dict(to_log, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return super().on_train_batch_end(*args, **kwargs)

    def get_preds_at_t_for_batch(
        self,
        batch: Dict[str, Tensor],
        horizon: int | float,
        split: str,
        ensemble: bool = False,
        is_autoregressive: bool = False,
        total_horizon: int = None,
        warm_start_target_normed=None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        b, t = batch["dynamics"].shape[0:2]  # batch size, time steps (channel, height, width)
        assert 0 < horizon <= self.true_horizon, f"horizon={horizon} must be in [1, {self.true_horizon}]"
        window_size = self.horizon

        warm_start_sanity = bool(getattr(self.model.hparams, "warm_start_sanity", False))
        if warm_start_sanity and self.sampler is not None:
            for sampler in self.sampler:
                sampler.close()
            self.sampler = None

        if self.sampler is None:
            n_discard = 0
            # Generate initial window predictions
            if self.hparams.initialize_window == "naive":
                # Copy the initial window from the batch
                init_window = batch["dynamics"][:, self.window - 1, ...].clone()
                # Copy the initial window to the full horizon
                init_window = torch.stack([init_window for _ in range(window_size)], dim=self.model.time_dim)
                init_window = self.pack_data(init_window, input_or_output="output")
                if ensemble:
                    # Add ensemble dimension to batch dimension (B, T, C, H, W) -> (B*E, T, C, H, W)
                    init_window = self.get_ensemble_inputs(init_window, split=split, add_noise=False)

            elif self.hparams.initialize_window == "truth":
                # Note: This is cheating, but interesting to see how well the model can do
                if "lookback" in batch.keys():
                    init_window = batch["lookback"]
                    assert init_window.shape[1] == window_size, f"init_window.shape={init_window.shape}"
                    lookback_start_t = self.datamodule.hparams.lookback_window[0]  # negative value
                    assert lookback_start_t < 0, f"lookback_start_t={lookback_start_t} must be negative"
                    assert self.datamodule.hparams.lookback_window[1] in [
                        0,
                        1,
                    ], f"lookback[1]={self.datamodule.hparams.lookback_window[1]} must be 0 or 1"
                    n_discard = -lookback_start_t + 1  # Get sampler to timestep 0
                else:
                    left_t = self.window  # - self.hparams.truth_shift
                    init_window = batch["dynamics"][:, left_t : left_t + window_size, ...].clone()
                init_window = self.pack_data(init_window, input_or_output="output")
                if self.model.time_dim == 2:
                    init_window = rrearrange(init_window, "b t c ... -> b c t ...")
                else:
                    assert self.model.time_dim == 1, f"Unexpected time_dim={self.model.time_dim}"
                if ensemble:
                    init_window = self.get_ensemble_inputs(init_window, split=split, add_noise=False)

            elif self.hparams.initialize_window == "climatology":
                assert hasattr(self, "climatology"), "climatology attribute must be set"
                init_window = (
                    self.climatology[None, ...].repeat(b, 1, 1, 1).to(device=self.device, dtype=self.dtype)
                )  # (1, C, H, W) -> (B, C, H, W)
                init_window = torch.stack([init_window for _ in range(window_size)], dim=self.model.time_dim)
                if ensemble:
                    init_window = self.get_ensemble_inputs(init_window, split=split, add_noise=False)

            elif self.hparams.initialize_window == "regression":
                # dyn_copy = batch["dynamics"].clone()
                assert not is_autoregressive, f"is_autoregressive={is_autoregressive} unexpected if sampler is None"
                # log.info(f"Generating initial window predictions for horizon={horizon}. batch['dynamics'].shape={dyn_copy.shape}")
                self.regression_model.to(self.device)
                N = self.skip_every_n_reg_step
                if N > 1:
                    # stack batch["dynamics"] to have N time steps in the time dimension
                    batch["dynamics"] = torch.cat([batch["dynamics"] for _ in range(N)], dim=1)
                # Enable dropout during inference
                with self.regression_model.inference_dropout_scope(
                    condition=self.hparams.regression_inference_dropout
                ):
                    # Use the EMA parameters for the validation step (if using EMA)
                    with self.regression_model.ema_scope(condition=self.hparams.regression_use_ema):
                        deterministic_preds = self.regression_model._evaluation_step(
                            batch=batch,
                            batch_idx=0,
                            split="predict",
                            return_outputs="preds_only",
                            aggregators=None,
                            verbose=self.global_rank == 0,
                            prediction_horizon=window_size * self.skip_every_n_reg_step,
                        )
                self.regression_model.cpu()  # move to CPU to save GPU memory
                # log.info(f"deterministic_preds.shape={deterministic_preds[f't{1}_preds_normed'].shape}. keys={deterministic_preds.keys()}")
                # This doesnt work yet when boundary conditions are used # todo: fix
                # Now, stack them into a single tensor
                dim = self.model.time_dim + 1 if ensemble else self.model.time_dim  # time_dim of ERDM class
                # pack_data will map a dict of vars to a dense tensor (if needed for the dataset)
                # if N=1: t=1, 2, 3, 4 == 6h,12h,18h,24h
                # if N=2: t=2, 4, 6, 8 == 12h,24h,36h,48h
                init_window = torch.stack(
                    [
                        self.pack_data(to_tensordict(deterministic_preds[f"t{t}_preds_normed"]), "input")
                        for t in range(N, window_size * N + 1, N)
                    ],
                    dim=dim,
                ).to(device=self.device, dtype=self.dtype)
                # Previously:
                # init_window = torch.stack(
                #     [torch.from_numpy(deterministic_preds[f"t{t}_preds_normed"]) for t in range(1, window_size + 1)],
                #     dim=dim,
                # ).to(device=self.device, dtype=self.dtype)
                # log.info(f"{init_window.shape=}") # init_window.shape = torch.Size([10, 8, 8, 69, 240, 121])
                if ensemble and self.num_predictions > 1:
                    init_window = rrearrange(init_window, "b ens t c h w -> (b ens) t c h w")
                    # log.info(f"B. {init_window.shape=}") # torch.Size([80, 8, 69, 240, 121])
                #  Visualize preds, first to xr dataarray
                # ds = xr.DataArray(deterministic_preds.squeeze(), dims=["b", "t", "h", "w"])
                # log.info(f"ds.shape={ds.shape}, unique={np.unique(ds.values)}")
                # ds.isel(b=[0, 1, -1]).plot(x="w", y="h", col="t", row="b")
                # log.info(f"batch['dynamics'].shape={batch['dynamics'].shape} vs {dyn_copy.shape}")
                # dyns = rrearrange(dyn_copy, "b ens t c h w -> (b ens) t c h w").squeeze() if len(dyn_copy.shape) == 6 else dyn_copy.squeeze()
                # ds_dynas = xr.DataArray(dyns, dims=["b", "t", "h", "w"])
                # ds_dynas.isel(b=[0, 1, -1]).plot(x="w", y="h", col="t", row="b")
                # plt.show()
            else:
                raise ValueError(f"Unsupported initialize_window={self.hparams.initialize_window}")

            dynamics_cond = None
            if self.model.hparams.conditional:
                # Condition on last clean samples (if conditional=True). Note that dynamics are normalized already>
                dynamics_cond = batch["dynamics"][:, : self.window, ...].to(dtype=self.dtype)
                dynamics_cond = self.pack_data(dynamics_cond, input_or_output="input")
                # Now, we can use the diffusion model to refine the predictions
                if dynamics_cond.shape[0] != init_window.shape[0]:
                    assert (
                        dynamics_cond.shape[0] * self.num_predictions == init_window.shape[0]
                    ), f"init_window.shape={init_window.shape}, dynamics_cond.shape={dynamics_cond.shape}"
                    dynamics_cond = dynamics_cond.repeat(self.num_predictions, *[1] * (len(dynamics_cond.shape) - 1))

                if self._pretrained_is_conditional:
                    assert self.model.time_dim == 2, f"Unexpected time_dim={self.model.time_dim}"
                    dynamics_cond = rrearrange(dynamics_cond, "b t c ... -> b c t ...")
                    # Concat the first part of the init window to the condition
                    dynamics_cond = torch.cat([dynamics_cond, init_window[:, :, :-1, ...]], dim=self.model.time_dim)

            extra_kwargs = self.get_extra_model_kwargs(
                batch, split=split, ensemble=ensemble, is_autoregressive=is_autoregressive
            )

            if warm_start_sanity:
                warm_start_target = self._prepare_warm_start_target_for_inputs(
                    warm_start_target_normed=warm_start_target_normed,
                    expected_batch_size=int(init_window.shape[0]),
                )
                extra_kwargs["warm_start_target"] = warm_start_target

            del batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            with self.model.guidance_scope():
                self.sampler = run_func_in_sub_batches_and_aggregate(
                    self.model.sample,
                    init_window,
                    num_prediction_loops=self.num_prediction_loops,
                    condition=dynamics_cond,
                    **extra_kwargs,
                )
            if len(self.sampler) > 1:
                self.log_text.info(f"Using {len(self.sampler)} samplers ({self.num_prediction_loops=}).")

            if n_discard > 0:
                # no-op sampling iterations to arrive at t=0 (and t=1 after if statement below)
                self.log_text.info(f"Discarding first {n_discard} 'predictions'")
                for _ in range(n_discard):
                    for sampler in self.sampler:
                        _ = next(sampler)
                assert self.model.curr_frame + lookback_start_t == 0, f"{self.model.curr_frame=}, {lookback_start_t=}"

        with log_peak_memory(
            log_mem=self.hparams.log_mem and total_horizon <= 10,
            logger_instance=self,
            name=f"predict_peak_gb_t{total_horizon}",
            prefix=f"memory/{split}",
        ):
            if len(self.sampler) == 1:
                results = next(self.sampler[0])  # shape (b, c, t, h, w)
            else:
                results = defaultdict(list)
                for sampler in self.sampler:
                    results_i = next(sampler)
                    for k, v in results_i.items():
                        results[k].append(v)
                results = {k: torch.cat(v, dim=0) for k, v in results.items()}

        plot = False
        if plot:
            import matplotlib.pyplot as plt
            import xarray as xr

            for k, v in results.items():
                self.log_text.info(
                    f"{horizon}, results[{k}].shape={v.shape}, min, mean, max={v.min().item(), v.mean().item(), v.max().item()}"
                )
                if k != "frames":
                    self.log_text.info(
                        f"{horizon}, min, mean, max per timestep = {[(v[:, :, i].min().item(), v[:, :, i].mean().item(), v[:, :, i].max().item()) for i in range(v.shape[2])]}"
                    )
            keys = ["noisy_future", "denoised_future", "frames"]
            keys = ["noisy_future"]
            for k in keys:
                dims = ["b", "h", "w"] if k == "frames" else ["b", "t", "h", "w"]
                col, row = ("b", None) if k == "frames" else ("t", "b")
                ds_res = xr.DataArray(results[k].squeeze(), dims=dims, name=k)
                ds_res.isel(b=[0, 1, -1], t=slice(0, 6)).plot(x="w", y="h", col=col, row=row)
            plt.title(f"horizon={horizon}")
            plt.show()
        # remove singleton time dimension, shape (b, c, h, w)
        if self.hparams.return_noisy_window:
            # If set, also set ++module.save_prediction_batches=all ++module.save_predictions_filename=zarr
            results["preds_window_noisy"] = results.pop("noisy_future")
            results["preds_window_denoised"] = results.pop("denoised_future")
            results["preds"] = results.pop("frames").squeeze(self.model.time_dim)
        else:
            results = {"preds": results["frames"].squeeze(self.model.time_dim)}
        # denormalize, reshape into batch and ensemble dimensions, etc. (e, b', c, h, w)
        results = self.postprocess_predictions(results)
        return {f"t{horizon}_{k}": v for k, v in results.items()}

    def predict_forward(self, *inputs, condition=None, metadata: Any = None, **kwargs):
        return next(self.sampler)

    def on_autoregressive_loop_end(self, split: str, dataloader_idx: int = None, **kwargs):
        self.log("diffusion/nfe", self.model.nfe)
        for sampler in self.sampler:
            sampler.close()
        self.sampler = None

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("regression_model.")}
        return super().load_state_dict(state_dict, strict=False)


class ClimatologyBaseline(AbstractMultiHorizonForecastingExperiment):
    def __init__(self, **kwargs):
        kwargs["use_ema"] = False
        super().__init__(**kwargs)

    def on_any_start(self, stage: str = None) -> None:
        super().on_any_start(stage)
        assert hasattr(self.datamodule, "climatology"), "datamodule must have a climatology attribute"
        self.climatology = self.datamodule.climatology

    def instantiate_model(self) -> None:
        pass

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        raise NotImplementedError(
            "ClimatologyBaseline does not have a loss function. It is only meant for prediction."
        )

    def get_preds_at_t_for_batch(
        self,
        batch: Dict[str, Tensor],
        horizon: int | float,
        split: str,
        ensemble: bool = False,
        is_autoregressive: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        clim = self.climatology.to(self.device)  # shape (c, h, w)
        b = batch["dynamics"].shape[0]
        # Copy the climatology to the batch size
        if ensemble and not is_autoregressive:
            b *= self.num_predictions  # Need to simulate an ensemble of predictions
        clim = clim.unsqueeze(0).expand(b, -1, -1, -1)

        results = {"preds": clim}  # remove singleton time dimension, shape (b, c, h, w)
        # denormalize, reshape into batch and ensemble dimensions, etc. (e, b', c, h, w)
        results = self.postprocess_predictions(results)
        return {f"t{horizon}_{k}": v for k, v in results.items()}


def infer_class_from_ckpt(ckpt_path: str, state=None) -> Type[AbstractMultiHorizonForecastingExperiment]:
    """Infer the experiment class from the checkpoint path."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu") if state is None else state
    module_config = ckpt["hyper_parameters"]
    abstract_kwargs = inspect.signature(AbstractMultiHorizonForecastingExperiment).parameters
    base_kwargs = {k: v for k, v in module_config.items() if k not in abstract_kwargs}
    diffusion_cfg = module_config["diffusion_config"]
    if diffusion_cfg is not None:
        if "dyffusion" in diffusion_cfg.get("_target_", ""):
            return MultiHorizonForecastingDYffusion
        elif "pderefiner" in diffusion_cfg.get("_target_", ""):
            return PDERefinerModule
        return SimultaneousMultiHorizonForecasting  # DDPM/MCVD
    elif "timestep_loss_weights" in base_kwargs.keys():
        return SimultaneousMultiHorizonForecasting
    else:
        return MultiHorizonForecastingTimeConditioned


# dummy simplification of model and regression model above:
# class DummyModel(torch.nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.model = ...
#         self.regresion_model = load_checkpoint("path/to/regression_model.ckpt").cpu()
#     def on_training_epoch_start(self):
#         self.regression_model = self.regression_model.cpu()
#     def forward(self, x):
#         return self.model(x)
#     def on_validation_epoch_start(self):
#         self.regression_model = self.regression_model.to(self.device)
#         y_reg = self.regression_model(x)
#         self.model.update(y_reg)
#         self.regression_model = self.regression_model.cpu()
