from abc import abstractmethod
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from scipy.stats import truncnorm                

# from src.utilities.torch_utils import persistence
from src.diffusion._base_diffusion import BaseDiffusion
from src.optimization.losses import AbstractWeightedLoss, crps_ensemble
from src.utilities.random_control import StackedRandomGenerator
from src.utilities.utils import get_logger, rrearrange
import contextlib


log = get_logger(__name__)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


# @persistence.persistent_class
class VPPrecond(BaseDiffusion):
    def __init__(
        self,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        beta_d=19.9,  # Extent of the noise level schedule.
        beta_min=0.1,  # Initial slope of the noise level schedule.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        epsilon_t=1e-5,  # Minimum t-value used during training.
        **kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__(**kwargs)
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.criterion = VPLoss(beta_d=beta_d, beta_min=beta_min, epsilon_t=epsilon_t)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def get_loss(self, targets, labels=None, augment_pipe=None):
        return self.criterion(self, targets, labels, augment_pipe)

    def sigma(self, t):
        return self.loss.sigma(t)

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


# @persistence.persistent_class
class VEPrecond(BaseDiffusion):
    def __init__(
        self,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0.02,  # Minimum supported noise level.
        sigma_max=100,  # Maximum supported noise level.
        **kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__(**kwargs)
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.criterion = VELoss(sigma_min=sigma_min, sigma_max=sigma_max)

    def forward(self, x, sigma, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def get_loss(self, targets, labels=None, augment_pipe=None):
        return self.criterion(self, targets, labels, augment_pipe)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".


class iDDPMPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        C_1=0.001,  # Timestep adjustment at low noise levels.
        C_2=0.008,  # Timestep adjustment at high noise levels.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        model_type="DhariwalUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels * 2,
            label_dim=label_dim,
            **model_kwargs,
        )

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer("u", u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, : self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(
            sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)
        ).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


# @persistence.persistent_class
class EDMPrecond(BaseDiffusion):
    def __init__(
        self,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=None,  # Maximum supported noise level.
        sigma_max_inf=80,  # Maximum supported noise level.
        sigma_min_train=None,
        P_mean=-1.2,  # Mean of the noise level distribution.
        P_std=1.2,  # Standard deviation of the noise level distribution.
        channel_noise_mult=None, # len num channels
        learn_noise_mult: bool = False,  # Learn one noise multiplier per channel.
        noise_mult_min: float | None = None,  # Optional lower bound for learned multipliers.
        noise_mult_max: float | None = None,  # Optional upper bound for learned multipliers.
        noise_mult_reg_weight: float = 0.0, # regularize multipliers close to 1
        noise_mult_gain: float = 1.0, # scale the noise multiplier by this factor
        noise_mult_grad_scale: float = 1.0, # scale the gradient for the noise multiplier by this factor
        noise_distribution: str = "lognormal",  # Distribution of the noise level.
        use_noise_logvar: bool = False,
        when_3d_concat_condition_to: str = None,  # When using 3D model: Concat to 'time' or 'channel' dimension?
        force_unconditional=False,  # Ignore conditioning information?
        # Sampling parameters.
        num_steps=18,  # Number of steps in the sampling loop.
        rho=7,  # Exponent of the time step discretization.
        rho_train=None, # Separate rho just for training
        S_churn=0,  # Maximum noise increase per step.
        S_min=0,  # Minimum noise level for increased noise.
        S_max=float("inf"),  # Maximum noise level for increased noise.
        S_noise=1,  # Noise level for increased noise.
        heun: bool = True,  # Use Heun's method for the sampling loop.
        compute_loss_per_sigma: bool = False,  # Compute loss for each sigma in the range.
        dtype="double",  # double or float
                # Residual diffusion around guidance / CRPS forecast.
        residual_diffusion: bool = False,  # Train diffusion on residuals y - guidance(x)?
        residual_condition_mode: str = "none",  # "none" | "replace" | "concat"
        residual_use_guidance_ema: bool = True,  # Use EMA weights when querying guidance model.
        residual_guidance_dropout: float = 0.0,  # Optional dropout for guidance inference during training/sampling.
        residual_detach_guidance: bool = True,  # Detach guidance prediction before forming residual targets.
        residual_rms_ema_decay: float = 0.999,  # Very long EMA for per-channel residual RMS.
        residual_rms_eps: float = 1e-6,  # Numerical floor for residual RMS normalization.
        residual_rms_log_every_n_steps: int = 1000,  # Logging cadence for residual RMS buffer.
        **kwargs,  # Keyword arguments for the underlying model.
    ):
        kwargs["timesteps"] = num_steps
        super().__init__(**kwargs)
        self._USE_SIGMA_DATA = True
        self.use_fp16 = use_fp16
        self.sigma_min_train = sigma_min_train or sigma_min
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max or float("inf")
        self.sigma_max_inf = sigma_max_inf or self.sigma_max
        assert (self.hparams.warm_start or self.sigma_min < self.sigma_max_inf <= self.sigma_max), (
            f"Invalid sigma range: {self.sigma_min=} < {self.sigma_max_inf=} <= {self.sigma_max=}"
        )
        if self.hparams.warm_start_train:
            if self.guidance_model is None:
                raise ValueError("warm_start_train=True requires a guidance model (guidance_run_id).")
            if self.hparams.warm_start_min is None or self.hparams.warm_start_max is None:
                raise ValueError("warm_start_train=True requires warm_start_min and warm_start_max to be set.")
            if not (self.hparams.warm_start_min < self.hparams.warm_start_max):
                raise ValueError("Require warm_start_min < warm_start_max when warm_start_train=True.")
        self.channel_noise_mult = channel_noise_mult


        self.learn_noise_mult = learn_noise_mult
        self.noise_mult_min = noise_mult_min
        self.noise_mult_max = noise_mult_max
        self.noise_mult_reg_weight = noise_mult_reg_weight
        self.noise_mult_gain = noise_mult_gain
        self.noise_mult_grad_scale = noise_mult_grad_scale
        if self.learn_noise_mult:
            num_channels = self.num_input_channels
            self.channel_noise_mult_logits = torch.nn.Parameter(torch.zeros(num_channels))
            if self.noise_mult_grad_scale != 1.0:
                self.channel_noise_mult_logits.register_hook(
                    lambda grad: grad * self.noise_mult_grad_scale
                )
        else:
            self.channel_noise_mult_logits = None

        if self.noise_mult_min is not None and self.noise_mult_max is not None:
            if not (self.noise_mult_min > 0 and self.noise_mult_max > 0):
                raise ValueError("noise_mult_min and noise_mult_max must be > 0")
            if not (self.noise_mult_min <= 1.0 <= self.noise_mult_max):
                raise ValueError("Bounds must satisfy noise_mult_min <= 1 <= noise_mult_max")
            if not (self.noise_mult_min < self.noise_mult_max):
                raise ValueError("Require noise_mult_min < noise_mult_max")
        elif (self.noise_mult_min is None) ^ (self.noise_mult_max is None):
            raise ValueError("Specify both noise_mult_min and noise_mult_max, or neither.")
        self.heun = heun
        self.label_dim = 0
        self.log_text.info(
            f"EDM: {sigma_min=}, {self.sigma_max_inf=}, {num_steps=}, {rho=}, {S_churn=}, {S_min=}, {S_max=}"
        )

        if self.hparams.residual_diffusion and self.guidance_model is None:
            raise ValueError("residual_diffusion=True requires a guidance model (guidance_run_id).")

        if self.hparams.residual_diffusion:
            self.log_text.info(
                "Residual diffusion enabled: target = y - guidance(x). "
                "Legacy warm_start / warm_start_train branches are ignored in residual mode."
            )

        if self._uses_residual_normalized_space():
            if not (0.0 < residual_rms_ema_decay < 1.0):
                raise ValueError("residual_rms_ema_decay must be in (0, 1).")
            if residual_rms_eps <= 0:
                raise ValueError("residual_rms_eps must be > 0.")
            if residual_rms_log_every_n_steps <= 0:
                raise ValueError("residual_rms_log_every_n_steps must be > 0.")

            self.residual_rms_ema_decay = float(residual_rms_ema_decay)
            self.residual_rms_eps = float(residual_rms_eps)
            self.residual_rms_log_every_n_steps = int(residual_rms_log_every_n_steps)
            self.register_buffer("residual_rms_ema", torch.ones(self.num_input_channels, dtype=torch.float32))
            self.register_buffer("residual_rms_initialized", torch.tensor(False, dtype=torch.bool))

    def _uses_residual_normalized_space(self) -> bool:
        return bool(getattr(self.hparams, "residual_diffusion", False))

    def _effective_sigma_data(self) -> float:
        # Residual space is explicitly normalized to RMS~1 per channel.
        if self._uses_residual_normalized_space():
            return 1.0
        # Keep legacy behavior: sigma_data may be initialized later by the training stack.
        return self.sigma_data

    def _residual_channel_rms_vector(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim < 2:
            raise ValueError(f"Expected residual to have at least 2 dims [B, C, ...], got {residual.shape}")
        n_channels = self.residual_rms_ema.numel()
        candidate_dims = [d for d in range(1, residual.ndim) if residual.shape[d] == n_channels]
        if len(candidate_dims) == 0:
            raise ValueError(
                f"Residual channels mismatch: could not find a channel dim of size {n_channels} in shape {residual.shape}"
            )
        channel_dim = 1 if 1 in candidate_dims else candidate_dims[0]
        reduce_dims = tuple(d for d in range(residual.ndim) if d != channel_dim)
        rms = residual.to(torch.float32).pow(2).mean(dim=reduce_dims).sqrt()
        return rms.clamp_min(self.residual_rms_eps)

    @torch.no_grad()
    def _update_residual_rms_ema(self, residual: torch.Tensor):
        if not self._uses_residual_normalized_space():
            return
        channel_rms = self._residual_channel_rms_vector(residual)
        if not bool(self.residual_rms_initialized.item()):
            self.residual_rms_ema.copy_(channel_rms.to(device=self.residual_rms_ema.device))
            self.residual_rms_initialized.fill_(True)
            return
        decay = self.residual_rms_ema_decay
        self.residual_rms_ema.mul_(decay).add_(
            channel_rms.to(device=self.residual_rms_ema.device), alpha=1.0 - decay
        )

    def _residual_rms_broadcast(self, like: torch.Tensor) -> torch.Tensor:
        if like.ndim < 2:
            raise ValueError(f"Expected tensor with shape [B, C, ...], got {like.shape}")
        n_channels = self.residual_rms_ema.numel()
        candidate_dims = [d for d in range(1, like.ndim) if like.shape[d] == n_channels]
        if len(candidate_dims) == 0:
            raise ValueError(
                f"Residual channels mismatch: could not find a channel dim of size {n_channels} in shape {like.shape}"
            )
        channel_dim = 1 if 1 in candidate_dims else candidate_dims[0]
        if bool(self.residual_rms_initialized.item()):
            rms = self.residual_rms_ema.to(device=like.device, dtype=like.dtype)
        else:
            rms = torch.ones_like(self.residual_rms_ema, device=like.device, dtype=like.dtype)
        rms = rms.clamp_min(self.residual_rms_eps)
        view_shape = [1] * like.ndim
        view_shape[channel_dim] = n_channels
        return rms.reshape(*view_shape)

    def _normalize_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if not self._uses_residual_normalized_space():
            return residual
        return residual / self._residual_rms_broadcast(residual)

    def _denormalize_residual(self, residual_norm: torch.Tensor) -> torch.Tensor:
        if not self._uses_residual_normalized_space():
            return residual_norm
        return residual_norm * self._residual_rms_broadcast(residual_norm)

    def _maybe_log_residual_rms_buffer(self):
        if not self._uses_residual_normalized_space():
            return
        if not bool(self.residual_rms_initialized.item()):
            return

        step = getattr(self, "global_step", None)
        if step is None:
            return

        every_n = self.residual_rms_log_every_n_steps
        if every_n <= 0 or (step % every_n) != 0:
            return

        rms = self.residual_rms_ema.detach()
        if not torch.isfinite(rms).all():
            log.warning("residual_rms_ema contains non-finite values at step=%s: %s", step, rms.cpu().tolist())
            return

        rms_cpu = rms.cpu()
        log.info(
            "residual_rms_ema(step=%s): mean=%.6g min=%.6g max=%.6g values=%s",
            step,
            float(rms_cpu.mean().item()),
            float(rms_cpu.min().item()),
            float(rms_cpu.max().item()),
            rms_cpu.tolist(),
        )

    @staticmethod
    def _strip_non_guidance_kwargs(kwargs):
        """
        Remove kwargs that belong to the diffusion/loss stack and should not be forwarded
        to the deterministic guidance model.
        """
        blocked = {
            "condition",
            "guidance_condition",
            "predictions_post_process",
            "targets_pre_process",
            "criterion_kwargs",
            "return_predictions",
            "return_loss_telemetry",
            "return_intermediate",
            "raw_targets",
            "predictions_mask",
            "sigma",
        }
        return {k: v for k, v in kwargs.items() if k not in blocked}

    @staticmethod
    def _extract_guidance_prediction(output):
        """
        Guidance model may return a tensor directly or a dict with tensor predictions.
        """
        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            for key in ("preds", "prediction", "mean", "mu", "output"):
                if key in output and torch.is_tensor(output[key]):
                    return output[key]

        raise TypeError(
            "Guidance model output must be a Tensor or a dict containing a Tensor under "
            "one of: 'preds', 'prediction', 'mean', 'mu', 'output'."
        )

    @staticmethod
    def _align_guidance_to_targets_shape(guidance_pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Align guidance prediction shape to target shape for residual subtraction.

        This only applies singleton-dimension adjustments and raises on non-trivial mismatches.
        """
        if guidance_pred.shape == targets.shape:
            return guidance_pred

        # Common case: targets carry an explicit singleton time axis (e.g., [B, 1, C, H, W])
        # while guidance is [B, C, H, W]. Insert singleton dims where targets has size=1.
        if guidance_pred.ndim < targets.ndim:
            candidate = guidance_pred
            for dim in range(1, targets.ndim):
                if candidate.ndim == targets.ndim:
                    break
                if targets.shape[dim] == 1:
                    candidate = candidate.unsqueeze(dim)
            if candidate.shape == targets.shape:
                return candidate

        # Opposite case: guidance has extra singleton axes that can be removed.
        if guidance_pred.ndim > targets.ndim:
            candidate = guidance_pred
            for dim in range(guidance_pred.ndim - 1, 0, -1):
                if candidate.ndim == targets.ndim:
                    break
                if candidate.shape[dim] == 1:
                    candidate = candidate.squeeze(dim)
            if candidate.shape == targets.shape:
                return candidate

        raise ValueError(
            "Guidance prediction shape must match targets in residual diffusion after singleton alignment. "
            f"guidance_pred.shape={guidance_pred.shape}, targets.shape={targets.shape}"
        )

    def _predict_guidance(self, condition, **kwargs):
        """
        Run the frozen deterministic / CRPS model and return its tensor prediction.
        """
        if self.guidance_model is None:
            raise ValueError("Guidance model is required for residual diffusion.")

        guidance_model = self.guidance_model.to(condition.device)
        guidance_kwargs = self._strip_non_guidance_kwargs(kwargs)

        use_ema = bool(getattr(self.hparams, "residual_use_guidance_ema", True))
        guidance_dropout = float(getattr(self.hparams, "residual_guidance_dropout", 0.0))

        ema_ctx = (
            guidance_model.ema_scope(condition=True)
            if use_ema and hasattr(guidance_model, "ema_scope")
            else contextlib.nullcontext()
        )
        dropout_ctx = (
            guidance_model.inference_dropout_scope(condition=guidance_dropout)
            if hasattr(guidance_model, "inference_dropout_scope")
            else contextlib.nullcontext()
        )

        with torch.no_grad():
            with ema_ctx:
                with dropout_ctx:
                    output = guidance_model(condition, **guidance_kwargs)

        pred = self._extract_guidance_prediction(output)

        if not torch.is_tensor(pred):
            raise TypeError(f"Guidance prediction must be a Tensor, got {type(pred)}")

        return pred

    def _merge_residual_condition(self, base_condition, guidance_pred):
        """
        Build the condition that is passed to the diffusion model in residual mode.

        Modes:
        - "none":    keep existing condition unchanged
        - "replace": use only guidance_pred as condition
        - "concat":  concatenate [base_condition, guidance_pred] along channel dim
        """
        mode = getattr(self.hparams, "residual_condition_mode", "none")

        if mode == "none":
            return base_condition

        if mode == "replace":
            return guidance_pred

        if mode == "concat":
            if base_condition is None:
                return guidance_pred
            if base_condition.ndim != guidance_pred.ndim:
                raise ValueError(
                    "Cannot concat residual condition: "
                    f"base_condition.shape={base_condition.shape}, "
                    f"guidance_pred.shape={guidance_pred.shape}"
                )
            if base_condition.shape[0] != guidance_pred.shape[0]:
                raise ValueError(
                    "Batch mismatch for residual condition concat: "
                    f"{base_condition.shape[0]} vs {guidance_pred.shape[0]}"
                )
            if base_condition.shape[2:] != guidance_pred.shape[2:]:
                raise ValueError(
                    "Non-channel shape mismatch for residual condition concat: "
                    f"base_condition.shape={base_condition.shape}, "
                    f"guidance_pred.shape={guidance_pred.shape}"
                )
            return torch.cat([base_condition, guidance_pred], dim=1)

        raise ValueError(
            f"Unknown residual_condition_mode={mode!r}. "
            "Expected one of: 'none', 'replace', 'concat'."
        )

    @staticmethod
    def _extra_spatial_condition_channels(kwargs) -> int:
        total = 0
        for key in ("dynamical_condition", "static_condition"):
            value = kwargs.get(key, None)
            if value is None:
                continue
            if not torch.is_tensor(value) or value.ndim < 2:
                raise ValueError(f"Expected tensor `{key}` with shape [B, C, ...], got {type(value)}")
            total += int(value.shape[1])
        return total

    def _validate_total_residual_condition_channels(self, kwargs):
        expected_cond_channels = getattr(self, "num_conditional_channels", None)
        if expected_cond_channels is None:
            return

        condition = kwargs.get("condition", None)
        condition_channels = 0 if condition is None else int(condition.shape[1])
        total_channels = condition_channels + self._extra_spatial_condition_channels(kwargs)

        if total_channels != expected_cond_channels:
            dyn = kwargs.get("dynamical_condition", None)
            sta = kwargs.get("static_condition", None)
            raise ValueError(
                "Residual conditioning channel mismatch: "
                f"total spatial condition channels={total_channels} "
                f"(condition={condition_channels}, "
                f"dynamical={0 if dyn is None else int(dyn.shape[1])}, "
                f"static={0 if sta is None else int(sta.shape[1])}) "
                f"but model expects num_conditional_channels={expected_cond_channels}."
            )

    def _prepare_residual_targets_and_kwargs(self, inputs, targets, kwargs):
        """
        Residual-diffusion training branch:
            target_residual = targets - guidance(inputs)

        Returns:
            residual_targets, updated_kwargs, guidance_pred
        """
        kwargs = dict(kwargs)
        base_condition = kwargs.get("condition", inputs)
        guidance_kwargs = {k: v for k, v in kwargs.items() if k != "condition"}

        guidance_pred = self._predict_guidance(base_condition, **guidance_kwargs)
        guidance_pred_for_targets = self._align_guidance_to_targets_shape(guidance_pred, targets)

        guidance_ref_for_targets = (
            guidance_pred_for_targets.detach() if self.hparams.residual_detach_guidance else guidance_pred_for_targets
        )
        guidance_ref_for_condition = guidance_pred.detach() if self.hparams.residual_detach_guidance else guidance_pred

        residual_targets = targets - guidance_ref_for_targets

        kwargs["condition"] = self._merge_residual_condition(base_condition, guidance_ref_for_condition)
        self._validate_total_residual_condition_channels(kwargs)
        return residual_targets, kwargs, guidance_ref_for_condition

    def _prepare_residual_sampling_kwargs(self, condition, kwargs):
        """
        Residual-diffusion inference branch:
            sample residual around the guidance forecast and add the guidance back afterwards.

        Returns:
            updated_kwargs, guidance_pred
        """
        kwargs = dict(kwargs)
        base_condition = kwargs.get("condition", condition)
        guidance_kwargs = {k: v for k, v in kwargs.items() if k != "condition"}

        guidance_pred = self._predict_guidance(base_condition, **guidance_kwargs)
        guidance_ref = guidance_pred.detach() if self.hparams.residual_detach_guidance else guidance_pred

        kwargs["condition"] = self._merge_residual_condition(base_condition, guidance_ref)
        self._validate_total_residual_condition_channels(kwargs)
        return kwargs, guidance_ref

    @staticmethod
    def _align_tensor_to_reference_shape(tensor: torch.Tensor, reference: torch.Tensor, tensor_name: str) -> torch.Tensor:
        """
        Align tensor shape to reference shape by inserting/removing singleton dimensions only.
        """
        if tensor.shape == reference.shape:
            return tensor

        if tensor.ndim < reference.ndim:
            candidate = tensor
            for dim in range(1, reference.ndim):
                if candidate.ndim == reference.ndim:
                    break
                if reference.shape[dim] == 1:
                    candidate = candidate.unsqueeze(dim)
            if candidate.shape == reference.shape:
                return candidate

        if tensor.ndim > reference.ndim:
            candidate = tensor
            for dim in range(tensor.ndim - 1, 0, -1):
                if candidate.ndim == reference.ndim:
                    break
                if candidate.shape[dim] == 1:
                    candidate = candidate.squeeze(dim)
            if candidate.shape == reference.shape:
                return candidate

        raise ValueError(
            f"{tensor_name} shape must match reference after singleton alignment. "
            f"{tensor_name}.shape={tensor.shape}, reference.shape={reference.shape}"
        )

    def _resolve_warm_start_sanity_target(self, kwargs, reference: torch.Tensor) -> torch.Tensor:
        """
        Extract the ground-truth tensor used for warm-start sanity initialization.
        """
        value = kwargs.get("warm_start_target", None)
        if torch.is_tensor(value):
            target = value.to(device=reference.device, dtype=reference.dtype)
            return self._align_tensor_to_reference_shape(target, reference, tensor_name="warm_start_target")

        raise ValueError(
            "warm_start_sanity=True requires a ground-truth tensor in sampling kwargs under "
            "'warm_start_target'."
        )


    def _get_loss_callable_from_name_or_config(self, loss_function: str, **kwargs):
        """Return the loss function used for training.
        Function will be called when needed by the BaseModel class.
        Better to do it here in case self.* parameters are changed."""
        loss_kwargs = dict(
            sigma_data=self._effective_sigma_data(),
            use_logvar=self.hparams.use_noise_logvar,
            sigma_min = (
                self.hparams.sigma_min_train
                if self.hparams.sigma_min_train is not None
                else self.hparams.sigma_min
            ),
            sigma_max=self.hparams.sigma_max,
            noise_distribution=self.hparams.noise_distribution,
            **kwargs,
        )
        if self.hparams.noise_distribution == "lognormal" or self.hparams.noise_distribution == "truncated_lognormal":
            loss_kwargs.update({"P_mean": self.hparams.P_mean, "P_std": self.hparams.P_std})
        elif self.hparams.noise_distribution == "uniform":
            loss_kwargs.update({"P_mean": None, "P_std": None})
        else:
            raise ValueError(f"Unknown noise distribution: {self.hparams.noise_distribution}")

        log.info(f"Using EDM loss function: {loss_function}")
        if loss_function in ["mse", "l1"]:
            loss_kwargs.pop("reduction", None)
            ignored_kwargs = ["learned_var_dim_name_to_idx_and_n_dims", "reduce_op", "learn_per_dim"]
            dropped = {k: loss_kwargs.pop(k) for k in ignored_kwargs if k in loss_kwargs}
            if len(dropped) > 0:
                log.warning(
                    "Ignoring learned-variance kwargs for loss_function=%s: %s. "
                    "Use a weighted loss (e.g. 'wmse'/'wmae') to enable learned variance weighting.",
                    loss_function,
                    sorted(dropped.keys()),
                )
            return EDMLoss(**loss_kwargs) if loss_function == "mse" else EDMLossMAE(**loss_kwargs)
        elif loss_function == "wmse":
            return WeightedEDMLoss(**loss_kwargs, loss_type="L2")
        elif loss_function == "wmae":
            return WeightedEDMLoss(**loss_kwargs, loss_type="L1")
        elif loss_function == "wcrps":
            return WeightedEDMLossCRPS(**loss_kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_function}")

    def _sigma_base_to_broadcast(self, sigma, x):
        """
        Convert sigma to shape [B, 1, 1, 1] for 2D or [B, 1, 1, 1, 1] for 3D,
        matching the input tensor x.
        """
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32)
        batch_size = x.shape[0]

        if sigma.ndim == 0:
            sigma = sigma.expand(batch_size)
        elif sigma.ndim == 1:
            if sigma.shape[0] == 1 and batch_size != 1:
                sigma = sigma.expand(batch_size)
            elif sigma.shape[0] != batch_size:
                raise ValueError(
                    f"Sigma batch mismatch: sigma.shape={sigma.shape}, expected batch_size={batch_size}"
                )

        if sigma.ndim == 1:
            sigma = sigma.reshape(batch_size, *([1] * (x.ndim - 1)))
        elif sigma.ndim != x.ndim:
            raise ValueError(
                f"Unexpected sigma shape {sigma.shape} for input shape {x.shape}"
            )

        return sigma

    def _channel_noise_mult_tensor(self, x):
        """
        Return per-channel multipliers as shape [1, C, 1, 1] or [1, C, 1, 1, 1].

        Cases:
        - fixed channel_noise_mult: use provided values
        - learn_noise_mult: learn positive per-channel multipliers with arithmetic mean 1
        """
        if self.learn_noise_mult:
            logits = self.channel_noise_mult_logits.to(device=x.device, dtype=torch.float32)

            if logits.ndim != 1:
                raise ValueError(
                    f"channel_noise_mult_logits must be 1D, got shape {logits.shape}"
                )
            if logits.numel() != x.shape[1]:
                raise ValueError(
                    f"Learned channel multiplier length mismatch: got {logits.numel()}, expected {x.shape[1]}"
                )

            if self.noise_mult_min is not None and self.noise_mult_max is not None:
                # Bounded positive raw values, then renormalize to mean 1.
                mult = self.noise_mult_min + (
                    self.noise_mult_max - self.noise_mult_min
                ) * torch.sigmoid(logits * self.noise_mult_gain)
                mult = mult
                mult = mult / mult.mean()
            else:
                # Positive multipliers with exact arithmetic mean 1.
                mult = torch.softmax(logits, dim=0) * logits.numel()

        elif self.channel_noise_mult is not None:
            mult = torch.as_tensor(self.channel_noise_mult, device=x.device, dtype=torch.float32)

            if mult.ndim != 1:
                raise ValueError(
                    f"channel_noise_mult must be 1D, got shape {mult.shape}"
                )
            if mult.numel() != x.shape[1]:
                raise ValueError(
                    f"channel_noise_mult length mismatch: got {mult.numel()}, expected {x.shape[1]}"
                )
            if (mult <= 0).any():
                raise ValueError("All entries in channel_noise_mult must be > 0")

        else:
            return None

        return mult.reshape(1, x.shape[1], *([1] * (x.ndim - 2)))

    def get_channel_noise_mult(self, device=None):
        """
        Returns the current per-channel multipliers as shape [C].
        Useful for logging/debugging.
        """
        if self.learn_noise_mult:
            logits = self.channel_noise_mult_logits
            if device is not None:
                logits = logits.to(device)

            if self.noise_mult_min is not None and self.noise_mult_max is not None:
                mult = self.noise_mult_min + (
                    self.noise_mult_max - self.noise_mult_min
                ) * torch.sigmoid(logits)
                mult = mult / mult.mean()
            else:
                mult = torch.softmax(logits, dim=0) * logits.numel()
            return mult

        if self.channel_noise_mult is not None:
            mult = torch.as_tensor(self.channel_noise_mult, dtype=torch.float32)
            if device is not None:
                mult = mult.to(device)
            return mult

        return None

    def _get_sigma_eff(self, sigma, x):
        """
        Effective sigma used for actual corruption/preconditioning:
            sigma_eff[b, c, ...] = sigma_base[b] * channel_noise_mult[c]
        """
        sigma_base = self._sigma_base_to_broadcast(sigma, x)
        mult = self._channel_noise_mult_tensor(x)
        if mult is None:
            return sigma_base
        return sigma_base * mult

    def _sigma_for_model(self, sigma, batch_size, device):
        """
        Extract the scalar/base sigma per sample for the model noise embedding.
        Output shape: [B]
        """
        sigma = torch.as_tensor(sigma, device=device, dtype=torch.float32)

        if sigma.ndim == 0:
            return sigma.expand(batch_size)

        if sigma.ndim == 1:
            if sigma.shape[0] == 1 and batch_size != 1:
                return sigma.expand(batch_size)
            if sigma.shape[0] != batch_size:
                raise ValueError(
                    f"Sigma batch mismatch: sigma.shape={sigma.shape}, expected batch_size={batch_size}"
                )
            return sigma

        return sigma.reshape(batch_size, -1)[:, 0]

    def forward(self, x, sigma, force_fp32=False, **model_kwargs):
        if self.hparams.force_unconditional:
            if isinstance(self.hparams.force_unconditional, float):
                # Implement unconditional sampling by setting the condition to 0 with probability force_unconditional
                if self.training:
                    mask = torch.rand(x.shape[0], device=x.device) < self.hparams.force_unconditional
                    x[mask] = 0
                else:
                    pass
            else:
                _ = model_kwargs.pop("condition", None)

        x = x.to(torch.float32)

        # Base sigma for model conditioning, shape [B]
        sigma_model = self._sigma_for_model(sigma, batch_size=x.shape[0], device=x.device)

        # Effective sigma for actual noising/preconditioning, shape [B, 1/C, ...]
        sigma_eff = self._get_sigma_eff(sigma_model, x)

        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        sigma_data = self._effective_sigma_data()
        c_skip = sigma_data**2 / (sigma_eff**2 + sigma_data**2)
        c_out = sigma_eff * sigma_data / (sigma_eff**2 + sigma_data**2).sqrt()
        c_in = 1 / (sigma_data**2 + sigma_eff**2).sqrt()

        # Keep the model noise embedding scalar/base-sigma only.
        c_noise = sigma_model.log() / 4

        x_in = (c_in * x).to(dtype)

        if self.model.is_3d:
            assert x_in.ndim == 5, f"Expected 5D input for 3D model, got {x_in.ndim}D"  # (B, C, T, H, W)
            condition = model_kwargs.pop("condition", None)  # (B, C, T_cond, H, W), x_in: (B, C, T_gen, H, W)
            if condition is not None:
                if self.hparams.when_3d_concat_condition_to == "time":
                    x_in = torch.cat([condition, x_in], dim=2)  # (B, C, T_cond+T_gen, H, W)
                elif self.hparams.when_3d_concat_condition_to == "channel":
                    condition = rrearrange(condition, "b c tcond ... -> b (c tcond) ...")
                    condition = repeat(condition, "b ctcond ... -> b ctcond tgen ...", tgen=x_in.shape[2])
                    model_kwargs["condition"] = condition
                else:
                    raise ValueError(f"Invalid {self.hparams.when_3d_concat_condition_to=}")

        F_x = self.model(x_in, c_noise, **model_kwargs)

        if self.model.is_3d and self.hparams.when_3d_concat_condition_to == "time":
            F_x = F_x[:, :, -x.shape[2] :, ...]  # (B, C, T_gen, H, W)

        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        if self.learn_noise_mult:
            mult = self.get_channel_noise_mult(device=self.device).detach().cpu()
            log.info(f"learned channel_noise_mult={mult.tolist()}")

        return D_x


    def get_loss(self, inputs, targets, return_predictions=False, **kwargs):
        loss_kwargs = dict(kwargs)
        loss_kwargs.setdefault("condition", inputs)

        # Optional custom sigma sampling branch already present in your code.
        if self.hparams.noise_distribution == "uniform":
            steps = torch.rand(inputs.shape[0], device=inputs.device)
            sigmas = self.edm_discretization(
                steps=steps,
                sigma_min=self.sigma_min_train,
                sigma_max=self.sigma_max,
                rho=self.hparams["rho_train"],
            )
            loss_kwargs["sigma"] = sigmas

        # Residual-diffusion training branch.
        loss_targets = targets
        if getattr(self.hparams, "residual_diffusion", False):
            loss_targets, loss_kwargs, _guidance_pred = self._prepare_residual_targets_and_kwargs(
                inputs=inputs,
                targets=targets,
                kwargs=loss_kwargs,
            )
            if self.training:
                self._update_residual_rms_ema(loss_targets.detach())
                self._maybe_log_residual_rms_buffer()
            loss_targets = self._normalize_residual(loss_targets)

        loss = self.criterion["preds"](self, targets=loss_targets, **loss_kwargs)

        if self.learn_noise_mult and self.noise_mult_reg_weight > 0:
            mult = self.get_channel_noise_mult(device=inputs.device)
            reg = (mult.log() ** 2).mean()
            if torch.is_tensor(loss):
                loss = loss + self.noise_mult_reg_weight * reg
            elif isinstance(loss, dict):
                if "loss" in loss:
                    loss["loss"] = loss["loss"] + self.noise_mult_reg_weight * reg
                loss["noise_mult_reg"] = reg.detach()

        if self.hparams.compute_loss_per_sigma and not self.training:
            loss = {"loss": loss} if torch.is_tensor(loss) else loss
            loss_kwargs_no_sigma = dict(loss_kwargs)
            _ = loss_kwargs_no_sigma.pop("sigma", None)
            loss.update(
                self.get_loss_vs_sigmas(
                    inputs=inputs,
                    targets=loss_targets,
                    targets_are_residual=getattr(self.hparams, "residual_diffusion", False),
                    **loss_kwargs_no_sigma,
                )
            )

        if return_predictions:
            # Keep current framework behavior unchanged.
            return loss, None

        return loss

    def get_loss_vs_sigmas(self, inputs, targets, **kwargs) -> Dict[str, float]:
        def _scalarize(value) -> float:
            if torch.is_tensor(value):
                return float(value.detach().mean().cpu())
            return float(value)

        sigma_kwargs = dict(kwargs)
        sigma_kwargs.setdefault("condition", inputs)

        sigmas = self.edm_discretization(steps=200)
        losses = dict()  # defaultdict(list)
        collect_pre_logvar_telemetry = isinstance(self.criterion["preds"], AbstractWeightedLoss)
        for sigma in sigmas:
            sigma_proj = sigma.broadcast_to(inputs.shape[0])
            loss_sigma = self.criterion["preds"](
                self,
                targets=targets,
                sigma=sigma_proj,
                return_loss_telemetry=collect_pre_logvar_telemetry,
                **sigma_kwargs,
            )
            loss_sigma = {"loss": loss_sigma} if torch.is_tensor(loss_sigma) else loss_sigma

            if collect_pre_logvar_telemetry:
                # Telemetry should report pre-logvar losses; reduce any remaining dimensions to scalars.
                if "weighted_loss_before_vars" in loss_sigma:
                    loss_sigma["weighted_loss"] = _scalarize(loss_sigma.pop("weighted_loss_before_vars"))
                # Keep logs concise: this is only used as an internal intermediate.
                loss_sigma.pop("weighted_loss_after_vars", None)

                if "unweighted_loss" in loss_sigma:
                    loss_sigma["unweighted_loss"] = _scalarize(loss_sigma["unweighted_loss"])

            if "raw_loss" not in loss_sigma:
                loss_sigma["raw_loss"] = loss_sigma["loss"]
            for k, v in loss_sigma.items():
                # losses[f"{k}_per_noise_level"].append(float(v))
                losses[f"{k}_per_noise_level_epoch_{self.current_epoch}/sigma{sigma:.3f}"] = _scalarize(v)

        # return_dict = {"x_axes": {"sigma": list(sigmas.cpu())}}
        losses["x_axes"] = {"sigma": list(sigmas.cpu())}
        return losses

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def edm_discretization(self, steps, sigma_min: float = None, sigma_max: float = None, rho: float = None):
        sigma_min = sigma_min or self.sigma_min  # max(sigma_min, self.sigma_min)
        sigma_max = sigma_max or self.sigma_max_inf  # min(sigma_max, self.sigma_max)
        rho = rho or self.hparams.rho
        if isinstance(steps, int):
            step_indices = torch.arange(steps, dtype=self.dtype, device=self.device)
            steps = step_indices / (steps - 1)
        return (sigma_max ** (1 / rho) + steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    def _warm_start_sanity_lowpass(self, x: torch.Tensor, factor: int = 4) -> torch.Tensor:
        """
        Assumes x is shaped like:
        [B, C, L]      for 1D signals
        [B, C, H, W]   for 2D signals
        [B, C, D, H, W] for 3D signals

        It low-passes by shrinking spatial dims, then resizing back.
        """
        if factor <= 1:
            return x

        spatial_ndim = x.ndim - 2
        if spatial_ndim not in (1, 2, 3):
            raise ValueError(
                f"Expected [B, C, ...] with 1/2/3 spatial dims, got shape {tuple(x.shape)}"
            )

        orig_dtype = x.dtype
        x_f = x.float()

        spatial_shape = x.shape[-spatial_ndim:]
        small_shape = tuple(max(1, s // factor) for s in spatial_shape)

        if spatial_ndim == 1:
            mode = "linear"
        elif spatial_ndim == 2:
            mode = "bilinear"
        else:
            mode = "trilinear"

        y = F.interpolate(x_f, size=small_shape, mode=mode, align_corners=False)
        y = F.interpolate(y, size=spatial_shape, mode=mode, align_corners=False)

        return y.to(orig_dtype)

    def edm_sampler(
        self,
        noise,
        randn_like=torch.randn_like,
        **kwargs,
    ):
        dtype = torch.float64 if self.hparams.dtype == "double" else torch.float32
        dtype = torch.float32

        # Control-only kwargs for warm-start sanity mode must never reach model.forward.
        warm_start_target = kwargs.pop("warm_start_target", None)
        warm_start_target_kwargs = {"warm_start_target": warm_start_target}

        def denoise(x, t):
            denoised = self(x, t, **kwargs).to(dtype)
            if self.hparams.guidance == 1:
                return denoised
            elif self.hparams.guidance_interval is not None and (
                t < self.hparams.guidance_interval[0] or t > self.hparams.guidance_interval[1]
            ):
                return denoised

            kwargs_g = kwargs
            if self.guidance_model.model.hparams.force_unconditional:
                kwargs_g = {k: v for k, v in kwargs.items() if k != "dynamical_condition"}
            ref_Dx = self.guidance_model(x, t, **kwargs_g).to(dtype)
            denoised = ref_Dx.lerp(denoised, self.hparams.guidance)
            return denoised

        S_churn = self.hparams.S_churn
        S_min = self.hparams.S_min
        S_max = self.hparams.S_max
        S_noise = self.hparams.S_noise
        num_steps = self.hparams.num_steps

        # Base scalar schedule
        t_steps = self.edm_discretization(num_steps)
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        use_legacy_warm_start = self.hparams.warm_start and not getattr(self.hparams, "residual_diffusion", False)
        warm_start_sanity = bool(getattr(self.hparams, "warm_start_sanity", False))

        if use_legacy_warm_start:
            t_steps = t_steps[-self.hparams.warm_start_steps:]

            if warm_start_sanity:
                # Sanity mode: start from the provided ground truth and add the same initial EDM noise.
                init_pred_raw = self._resolve_warm_start_sanity_target(
                    kwargs=warm_start_target_kwargs,
                    reference=noise.to(dtype),
                )
                if getattr(self.hparams, "warm_start_sanity_lowpass", False):
                    init_pred = self._warm_start_sanity_lowpass(
                        init_pred_raw,
                        factor=getattr(self.hparams, "warm_start_sanity_lowpass_factor", 4),
                    )
                else:
                    init_pred = init_pred_raw
            elif self.hparams.condition_warm_start:
                init_pred = kwargs["condition"].to(dtype)
            else:
                blocked_guidance_kwargs = {"condition", "warm_start_target"}
                init_kwargs = {k: v for k, v in kwargs.items() if k not in blocked_guidance_kwargs}
                with self.guidance_model.inference_dropout_scope(condition=self.hparams.warm_start_dropout):
                    init_pred = self.guidance_model(kwargs["condition"], **init_kwargs).to(dtype)

            sigma0_eff = self._get_sigma_eff(t_steps[0], init_pred)
            n = torch.randn_like(init_pred) * sigma0_eff
            x_next = init_pred + n
        else:
            sigma0_eff = self._get_sigma_eff(t_steps[0], noise)
            x_next = noise.to(dtype) * sigma0_eff

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            t_cur_eff = self._get_sigma_eff(t_cur, x_cur)
            t_next_eff = self._get_sigma_eff(t_next, x_cur)

            # Increase noise temporarily.
            if S_churn > 0 and S_min <= t_cur <= S_max:
                gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
                t_hat = t_cur + gamma * t_cur  # scalar/base sigma
                t_hat_eff = self._get_sigma_eff(t_hat, x_cur)
                noise_scale = (t_hat_eff.square() - t_cur_eff.square()).clamp_min(0).sqrt()
                x_hat = x_cur + noise_scale * S_noise * randn_like(x_cur)
            else:
                t_hat = t_cur
                t_hat_eff = t_cur_eff
                x_hat = x_cur

            # Euler step.
            denoised = denoise(x_hat, t_hat).to(dtype)
            d_cur = (x_hat - denoised) / t_hat_eff.clamp_min(1e-12)
            x_next = x_hat + (t_next_eff - t_hat_eff) * d_cur

            # Apply 2nd order correction.
            if self.heun and i < len(t_steps) - 2:
                denoised = denoise(x_next, t_next).to(dtype)
                d_prime = (x_next - denoised) / t_next_eff.clamp_min(1e-12)
                x_next = x_hat + (t_next_eff - t_hat_eff) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.to(self.dtype)

    @torch.inference_mode()
    def sample(self, condition, batch_seeds=None, **kwargs):
        batch_size = condition.shape[0]
        batch_seeds = batch_seeds or torch.randint(0, 2**32, (batch_size,), device=condition.device)

        sample_kwargs = dict(kwargs)
        sample_kwargs.setdefault("condition", condition)

        guidance_pred = None
        if getattr(self.hparams, "residual_diffusion", False):
            sample_kwargs, guidance_pred = self._prepare_residual_sampling_kwargs(
                condition=condition,
                kwargs=sample_kwargs,
            )

        rnd = StackedRandomGenerator(self.device, batch_seeds)

        if self.model.is_3d:
            nt_gen = self.num_temporal_channels
            if self.hparams.when_3d_concat_condition_to != "channel":
                nt_gen -= condition.shape[2]
            init_latents_shape = (batch_size, self.num_input_channels, nt_gen, *self.spatial_shape_out)
        else:
            init_latents_shape = (batch_size, self.num_input_channels, *self.spatial_shape_out)

        latents = rnd.randn(
            init_latents_shape,
            dtype=condition.dtype,
            layout=condition.layout,
            device=condition.device,
        )

        sample_out = self.edm_sampler(latents, **sample_kwargs)

        if getattr(self.hparams, "residual_diffusion", False):
            if guidance_pred is None:
                raise RuntimeError("Expected guidance prediction in residual diffusion sampling.")
            residual_sample = self._denormalize_residual(sample_out)
            return guidance_pred.to(sample_out.dtype) + residual_sample

        return sample_out


# ----------------------------------------------------------------------------

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


# @persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, targets, labels, augment_pipe=None):
        rnd_uniform = torch.rand([targets.shape[0], 1, 1, 1], device=targets.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = augment_pipe(targets) if augment_pipe is not None else (targets, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return {"loss": loss}

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


# @persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, targets, labels, augment_pipe=None):
        rnd_uniform = torch.rand([targets.shape[0], 1, 1, 1], device=targets.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = augment_pipe(targets) if augment_pipe is not None else (targets, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return {"loss": loss}


# ----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, zero_init=False):
        super().__init__()
        self.out_channels = out_channels
        if zero_init:
            self.weight = torch.nn.Parameter(torch.zeros(out_channels, in_channels, *kernel))
        else:
            self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


# @persistence.persistent_class
class EDMLossAbstract:  # For some reason this cannot inherit (torch.nn.Module) when using wmse/wmae loss - why?
    def __init__(self, P_mean, P_std, sigma_data, sigma_min, sigma_max, noise_distribution, use_logvar: bool = False):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.use_logvar = use_logvar
        self.noise_distribution = noise_distribution
        if use_logvar:
            logvar_channels = 128  # Intermediate dimensionality for uncertainty estimation.
            log.info(f"Using log-variance with {logvar_channels} intermediate channels for noise weighting.")
            self.logvar_fourier = MPFourier(logvar_channels)
            self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    @abstractmethod
    def loss(self, preds, targets, sigma_weights, **kwargs):
        pass

    @staticmethod
    def _strip_non_guidance_kwargs(kwargs):
        """
        Remove kwargs that belong to the diffusion/loss stack and should not be forwarded
        to the deterministic guidance model.
        """
        blocked = {
            "condition",
            "guidance_condition",
            "predictions_post_process",
            "targets_pre_process",
            "criterion_kwargs",
            "return_predictions",
            "return_loss_telemetry",
            "return_intermediate",
            "raw_targets",
            "predictions_mask",
            "sigma",
        }
        return {k: v for k, v in kwargs.items() if k not in blocked}

    @staticmethod
    def _extract_guidance_prediction(output):
        """
        Guidance model may return a tensor directly or a dict with tensor predictions.
        """
        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            for key in ("preds", "prediction", "mean", "mu", "output"):
                if key in output and torch.is_tensor(output[key]):
                    return output[key]

        raise TypeError(
            "Guidance model output must be a Tensor or a dict containing a Tensor under "
            "one of: 'preds', 'prediction', 'mean', 'mu', 'output'."
        )

    def _predict_guidance(self, condition, **kwargs):
        """
        Run the frozen deterministic / CRPS model and return its tensor prediction.
        """
        if self.guidance_model is None:
            raise ValueError("Guidance model is required for residual diffusion.")

        guidance_model = self.guidance_model.to(condition.device)
        guidance_kwargs = self._strip_non_guidance_kwargs(kwargs)

        use_ema = bool(getattr(self.hparams, "residual_use_guidance_ema", True))
        guidance_dropout = float(getattr(self.hparams, "residual_guidance_dropout", 0.0))

        ema_ctx = (
            guidance_model.ema_scope(condition=True)
            if use_ema and hasattr(guidance_model, "ema_scope")
            else contextlib.nullcontext()
        )
        dropout_ctx = (
            guidance_model.inference_dropout_scope(condition=guidance_dropout)
            if hasattr(guidance_model, "inference_dropout_scope")
            else contextlib.nullcontext()
        )

        with torch.no_grad():
            with ema_ctx:
                with dropout_ctx:
                    output = guidance_model(condition, **guidance_kwargs)

        pred = self._extract_guidance_prediction(output)

        if not torch.is_tensor(pred):
            raise TypeError(f"Guidance prediction must be a Tensor, got {type(pred)}")

        return pred

    def _merge_residual_condition(self, base_condition, guidance_pred):
        """
        Build the condition that is passed to the diffusion model in residual mode.

        Modes:
        - "none":    keep existing condition unchanged
        - "replace": use only guidance_pred as condition
        - "concat":  concatenate [base_condition, guidance_pred] along channel dim
        """
        mode = getattr(self.hparams, "residual_condition_mode", "none")

        if mode == "none":
            return base_condition

        if mode == "replace":
            return guidance_pred

        if mode == "concat":
            if base_condition is None:
                return guidance_pred
            if base_condition.ndim != guidance_pred.ndim:
                raise ValueError(
                    "Cannot concat residual condition: "
                    f"base_condition.shape={base_condition.shape}, "
                    f"guidance_pred.shape={guidance_pred.shape}"
                )
            if base_condition.shape[0] != guidance_pred.shape[0]:
                raise ValueError(
                    "Batch mismatch for residual condition concat: "
                    f"{base_condition.shape[0]} vs {guidance_pred.shape[0]}"
                )
            if base_condition.shape[2:] != guidance_pred.shape[2:]:
                raise ValueError(
                    "Non-channel shape mismatch for residual condition concat: "
                    f"base_condition.shape={base_condition.shape}, "
                    f"guidance_pred.shape={guidance_pred.shape}"
                )
            # Concatenate along channels.
            return torch.cat([base_condition, guidance_pred], dim=1)

        raise ValueError(
            f"Unknown residual_condition_mode={mode!r}. "
            "Expected one of: 'none', 'replace', 'concat'."
        )

    @staticmethod
    def _extra_spatial_condition_channels(kwargs) -> int:
        total = 0
        for key in ("dynamical_condition", "static_condition"):
            value = kwargs.get(key, None)
            if value is None:
                continue
            if not torch.is_tensor(value) or value.ndim < 2:
                raise ValueError(f"Expected tensor `{key}` with shape [B, C, ...], got {type(value)}")
            total += int(value.shape[1])
        return total

    def _validate_total_residual_condition_channels(self, kwargs):
        expected_cond_channels = getattr(self, "num_conditional_channels", None)
        if expected_cond_channels is None:
            return

        condition = kwargs.get("condition", None)
        condition_channels = 0 if condition is None else int(condition.shape[1])
        total_channels = condition_channels + self._extra_spatial_condition_channels(kwargs)

        if total_channels != expected_cond_channels:
            dyn = kwargs.get("dynamical_condition", None)
            sta = kwargs.get("static_condition", None)
            raise ValueError(
                "Residual conditioning channel mismatch: "
                f"total spatial condition channels={total_channels} "
                f"(condition={condition_channels}, "
                f"dynamical={0 if dyn is None else int(dyn.shape[1])}, "
                f"static={0 if sta is None else int(sta.shape[1])}) "
                f"but model expects num_conditional_channels={expected_cond_channels}."
            )

    def _prepare_residual_targets_and_kwargs(self, inputs, targets, kwargs):
        """
        Residual-diffusion training branch:
            target_residual = targets - guidance(inputs)

        Returns:
            residual_targets, updated_kwargs, guidance_pred
        """
        kwargs = dict(kwargs)
        base_condition = kwargs.get("condition", inputs)
        guidance_kwargs = {k: v for k, v in kwargs.items() if k != "condition"}

        guidance_pred = self._predict_guidance(base_condition, **guidance_kwargs)
        guidance_pred_for_targets = self._align_guidance_to_targets_shape(guidance_pred, targets)

        guidance_ref_for_targets = (
            guidance_pred_for_targets.detach() if self.hparams.residual_detach_guidance else guidance_pred_for_targets
        )
        guidance_ref_for_condition = guidance_pred.detach() if self.hparams.residual_detach_guidance else guidance_pred

        residual_targets = targets - guidance_ref_for_targets

        kwargs["condition"] = self._merge_residual_condition(base_condition, guidance_ref_for_condition)
        self._validate_total_residual_condition_channels(kwargs)
        return residual_targets, kwargs, guidance_ref_for_condition

    def _prepare_residual_sampling_kwargs(self, condition, kwargs):
        """
        Residual-diffusion inference branch:
            sample residual around the guidance forecast and add the guidance back afterwards.

        Returns:
            updated_kwargs, guidance_pred
        """
        kwargs = dict(kwargs)
        base_condition = kwargs.get("condition", condition)
        guidance_kwargs = {k: v for k, v in kwargs.items() if k != "condition"}

        guidance_pred = self._predict_guidance(base_condition, **guidance_kwargs)
        guidance_ref = guidance_pred.detach() if self.hparams.residual_detach_guidance else guidance_pred

        kwargs["condition"] = self._merge_residual_condition(base_condition, guidance_ref)
        self._validate_total_residual_condition_channels(kwargs)
        return kwargs, guidance_ref

    @staticmethod
    def _strip_non_forward_kwargs(kwargs):
        # These kwargs are consumed by the training/loss stack, not by model.forward.
        blocked = {
            "predictions_post_process",
            "targets_pre_process",
            "criterion_kwargs",
            "return_predictions",
            "return_loss_telemetry",
            "return_intermediate",
            "raw_targets",
            "predictions_mask",
            "sigma",
        }
        return {k: v for k, v in kwargs.items() if k not in blocked}

    @staticmethod
    def _extract_guidance_prediction(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            for key in ("preds", "prediction", "mean", "mu", "output"):
                if key in output and torch.is_tensor(output[key]):
                    return output[key]
        raise TypeError(
            "Guidance output must be a Tensor or a dict containing a Tensor under "
            "one of: 'preds', 'prediction', 'mean', 'mu', 'output'."
        )

    @staticmethod
    def _align_guidance_to_targets_shape(guidance_pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if guidance_pred.shape == targets.shape:
            return guidance_pred

        if guidance_pred.ndim < targets.ndim:
            candidate = guidance_pred
            for dim in range(1, targets.ndim):
                if candidate.ndim == targets.ndim:
                    break
                if targets.shape[dim] == 1:
                    candidate = candidate.unsqueeze(dim)
            if candidate.shape == targets.shape:
                return candidate

        if guidance_pred.ndim > targets.ndim:
            candidate = guidance_pred
            for dim in range(guidance_pred.ndim - 1, 0, -1):
                if candidate.ndim == targets.ndim:
                    break
                if candidate.shape[dim] == 1:
                    candidate = candidate.squeeze(dim)
            if candidate.shape == targets.shape:
                return candidate

        raise ValueError(
            "Guidance prediction shape must match targets in residual diffusion after singleton alignment. "
            f"guidance_pred.shape={guidance_pred.shape}, targets.shape={targets.shape}"
        )

    def _build_warm_start_training_clean_target(self, net, y, sigma_base, kwargs):
        # This mode is intended only for training; keep eval/validation behavior unchanged.
        if getattr(net.hparams, "residual_diffusion", False):
            return y
        if not net.hparams.warm_start_train or not net.training:
            return y

        condition = kwargs.get("condition", None)
        if condition is None:
            raise ValueError("warm_start_train=True requires `condition` to be provided in loss kwargs.")

        guidance_model = net.guidance_model
        if guidance_model is None:
            raise ValueError("warm_start_train=True requires a guidance model (guidance_run_id).")

        sigma_scalar = sigma_base.reshape(sigma_base.shape[0], -1)[:, 0]
        ws_min = float(net.hparams.warm_start_min)
        ws_max = float(net.hparams.warm_start_max)
        # alpha=0 -> pure ground truth, alpha=1 -> pure warm-start prediction.
        # we want to square the difference between the scalar and min / max
        alpha = ((sigma_scalar - ws_min)**2 / (ws_max - ws_min)**2).clamp(0.0, 1.0)
        alpha = alpha.reshape(sigma_base.shape[0], *([1] * (y.ndim - 1)))

        guidance_kwargs = self._strip_non_forward_kwargs({k: v for k, v in kwargs.items() if k != "condition"})
        # Keep guidance resident on the active device to avoid per-step CPU/GPU thrashing.
        guidance_model.to(y.device)
        with torch.no_grad():
            with guidance_model.ema_scope(condition=True):
                with guidance_model.inference_dropout_scope(condition=net.hparams.warm_start_dropout):
                    warm_pred = self._extract_guidance_prediction(guidance_model(condition, **guidance_kwargs))
        if warm_pred.shape != y.shape:
            raise ValueError(
                f"warm_start_train shape mismatch: warm_pred={warm_pred.shape}, y={y.shape}. "
                "Ensure guidance model output matches diffusion target shape."
            )

        return (1 - alpha) * y + alpha * warm_pred

    def __call__(
        self,
        net,
        targets,
        predictions_post_process=None,
        targets_pre_process=None,
        sigma=None,
        return_loss_telemetry: bool = False,
        **kwargs,
    ):
        y = targets_pre_process(targets) if targets_pre_process is not None else targets
        n_dims1 = (1,) * (y.ndim - 1)

        if sigma is None:
            # Sample base sigma per sample. Channel scaling is applied below via net._get_sigma_eff(...)
            if self.noise_distribution == "lognormal":
                rnd_normal = torch.randn([targets.shape[0], *n_dims1], device=targets.device)
            elif self.noise_distribution == "truncated_lognormal":
                if self.sigma_min <= 0:
                    lower_bound = -np.inf
                else:
                    lower_bound = (np.log(self.sigma_min) - self.P_mean) / self.P_std
                upper_bound = (np.log(self.sigma_max) - self.P_mean) / self.P_std
                trunc_normal_samples = truncnorm.rvs(
                    lower_bound,
                    upper_bound,
                    loc=0,
                    scale=1,
                    size=(targets.shape[0], *n_dims1),
                )
                rnd_normal = torch.as_tensor(trunc_normal_samples, device=targets.device, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown noise distribution: {self.noise_distribution}")

            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        else:
            sigma = sigma.reshape(-1, *n_dims1)

        sigma_base = sigma
        warm_clean = self._build_warm_start_training_clean_target(net=net, y=y, sigma_base=sigma_base, kwargs=kwargs)
        sigma_eff = net._get_sigma_eff(sigma_base, y)

        # Standard EDM weighting, but now based on the actual per-channel corruption level.
        weight = (sigma_eff**2 + self.sigma_data**2) / (sigma_eff * self.sigma_data) ** 2

        try:
            n = torch.randn_like(y) * sigma_eff
        except RuntimeError as e:
            raise RuntimeError(
                f"Shape mismatch: y={y.shape}, sigma_base={sigma_base.shape}, sigma_eff={sigma_eff.shape}"
            ) from e

        # Important: keep passing only base sigma into the network.
        D_yn = net(warm_clean + n, sigma_base, **kwargs)

        if predictions_post_process is not None:
            D_yn = predictions_post_process(D_yn)
            diff_shape = len(D_yn.shape) - len(weight.shape)
            if diff_shape != 0:
                assert diff_shape == 1, f"Shape mismatch: {D_yn.shape=} and {weight.shape=}"
                weight = weight.unsqueeze(1)  # add missing dimension (e.g. time)

        loss_kwargs = {}
        if self.use_logvar:
            # Keep logvar conditioning on the base sigma, same philosophy as c_noise above.
            sigma_for_logvar = sigma_base.reshape(sigma_base.shape[0], -1)[:, 0]
            loss_kwargs["batch_logvars"] = self.logvar_linear(
                self.logvar_fourier(sigma_for_logvar.log() / 4)
            )

        if return_loss_telemetry:
            # Telemetry-only: include intermediates from weighted losses.
            loss_kwargs["return_intermediate"] = True

        loss = self.loss(D_yn, targets, weight, **loss_kwargs)
        return {"loss": loss} if torch.is_tensor(loss) else loss


class EDMLoss(EDMLossAbstract):
    def loss(self, preds, targets, sigma_weights, **kwargs):
        assert len(kwargs) == 0, f"Unknown kwargs: {kwargs}. Consider using a weighted loss like 'wmse' instead?"
        # loss y, , n, and D_yn have the same shape (B, C, H, W). weight has shape (B, 1, 1, 1)
        assert preds.shape == targets.shape, f"Shape mismatch: {preds.shape} and {targets.shape}"
        return (sigma_weights * ((preds - targets) ** 2)).mean()


class EDMLossMAE(EDMLossAbstract):
    def loss(self, preds, targets, sigma_weights, **kwargs):
        assert len(kwargs) == 0, f"Unknown kwargs: {kwargs}. Consider using a weighted loss like 'wmse' instead?"
        return (sigma_weights * ((preds - targets).abs())).mean()


class WeightedEDMLossAbstract(AbstractWeightedLoss, EDMLossAbstract):
    def __init__(self, P_mean, P_std, sigma_data, sigma_min, sigma_max, noise_distribution, use_logvar: bool = False, **kwargs):
        AbstractWeightedLoss.__init__(self, use_batch_logvars=use_logvar, **kwargs)
        EDMLossAbstract.__init__(self, P_mean, P_std, sigma_data, sigma_min, sigma_max, noise_distribution, use_logvar=use_logvar)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        if weights is not None:
            # Need to add batch dimension to weights since we will multiply the lambda(\sigma) weights with it
            if weights.ndim == 3:
                weights = weights.unsqueeze(0)  # Add batch dimension
            elif weights.ndim == 2:
                weights = weights.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        self._weights = weights

    def forward(self, *args, **kwargs):
        return EDMLossAbstract.__call__(self, *args, **kwargs)


class WeightedEDMLoss(WeightedEDMLossAbstract):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__(**kwargs)
        if loss_type == "L2":
            self.loss_func = lambda x: x**2
        elif loss_type == "L1":
            self.loss_func = lambda x: x.abs()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def loss(self, preds, targets, sigma_weights, **kwargs):
        assert preds.shape == targets.shape, f"Shape mismatch: {preds.shape} and {targets.shape}"
        return self.weigh_loss(self.loss_func(preds - targets), multiply_weight=sigma_weights, **kwargs)


class WeightedEDMLossCRPS(WeightedEDMLossAbstract):
    def __call__(self, net, targets, predictions_post_process=None, targets_pre_process=None, **kwargs):
        return_loss_telemetry = kwargs.pop("return_loss_telemetry", False)
        y = targets_pre_process(targets) if targets_pre_process is not None else targets
        n_dims1 = (1,) * (y.ndim - 1)
        # Sample noise level from the prior distribution. Only specify sigma for analysis of loss vs. sigma.
        rnd_normal1 = torch.randn([targets.shape[0], *n_dims1], device=targets.device)
        #rnd_normal2 = torch.randn([targets.shape[0], *n_dims1], device=targets.device)

        if kwargs.get("sigma") is None:
            # Sample noise level from the prior distribution. Only specify sigma for analysis of loss vs. sigma.
            rnd_normal = torch.randn([targets.shape[0], *n_dims1], device=targets.device)
            sigma1 = (rnd_normal1 * self.P_std + self.P_mean).exp()
        else:
            sigma1 = kwargs.pop("sigma").reshape(-1, *n_dims1)
        #sigma2 = (rnd_normal2 * self.P_std + self.P_mean).exp()
        weight1 = (sigma1**2 + self.sigma_data**2) / (sigma1 * self.sigma_data) ** 2
        weight2 = (sigma1**2 + self.sigma_data**2) / (sigma1 * self.sigma_data) ** 2
        n1 = torch.randn_like(y) * sigma1
        n2 = torch.randn_like(y) * sigma1
        
        kwargs_no_sigma = {k: v for k, v in kwargs.items() if k != "sigma"}
        D_yn1 = net(y + n1, sigma1, **kwargs_no_sigma)
        D_yn2 = net(y + n2, sigma1, **kwargs_no_sigma)

        if predictions_post_process is not None:
            D_yn1 = predictions_post_process(D_yn1)
            D_yn2 = predictions_post_process(D_yn2)
            diff_shape = len(D_yn1.shape) - len(weight1.shape)
            if diff_shape != 0:
                assert diff_shape == 1, f"Shape mismatch: {D_yn1.shape=} and {weight1.shape=}"
                weight1 = weight1.unsqueeze(1)  # add missing dimension (e.g. time)
                weight2 = weight2.unsqueeze(1)  # add missing dimension (e.g. time)

        D_yn = torch.stack([D_yn1, D_yn2], dim=0)  # (2, B, C, H, W), where 2 is the ensemble size
        # collapse dimension that has size 1 as long as it's not the first dimension
        D_yn = D_yn.squeeze([i for i in range(1, D_yn.ndim) if D_yn.shape[i] == 1])

        # Take the min of the two weights
        weight_lam = torch.min(weight1, weight2)

        crps = crps_ensemble(predicted=D_yn, truth=y, reduction="none")

        loss_kwargs = {}
        if self.use_logvar:
            loss_kwargs["batch_logvars"] = self.logvar_linear(self.logvar_fourier(sigma1.flatten().log() / 4))
        if return_loss_telemetry:
            loss_kwargs["return_intermediate"] = True

        return self.weigh_loss(
            crps, multiply_weight=weight_lam, **loss_kwargs
        )
        # Copilot suggestion:
        # CRPS = E[|D_yn1 - D_yn2|] - 0.5 * E[|D_yn1 - y|] - 0.5 * E[|D_yn2 - y|]
        # crps = (D_yn1 - D_yn2).abs().mean() - 0.5 * (D_yn1 - y).abs().mean() - 0.5 * (D_yn2 - y).abs().mean()
