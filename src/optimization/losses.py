import operator
from functools import partial, reduce
from typing import Dict, Iterable, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.evaluation.metrics import weighted_mean
from src.utilities.utils import get_logger


log = get_logger(__name__)


class LpLoss(torch.nn.Module):
    def __init__(
        self,
        p=2,
        relative: bool = True,
        weights: Optional[Tensor] = None,
        weighted_dims: Union[int, Iterable[int]] = (),
    ):
        """
        Args:
            p: Lp-norm type. For example, p=1 for L1-norm, p=2 for L2-norm.
            relative: If True, compute the relative Lp-norm, i.e. ||x - y||_p / ||y||_p.
        """
        super(LpLoss, self).__init__()

        if p <= 0:
            raise ValueError("Lp-norm type should be positive")

        self.p = p
        self.loss_func = self.rel if relative else self.abs
        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
        if weights is not None:
            self.mean_func = partial(weighted_mean, weights=weights)
        else:
            self.mean_func = torch.mean

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        # print(diff_norms.shape, y_norms.shape, self.mean_func)
        return self.mean_func(diff_norms / y_norms)

    def abs(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        return self.mean_func(diff_norms)

    def __call__(self, x, y):
        return self.loss_func(x, y)


class CRPSLoss(torch.nn.Module):
    require_ensembling: bool = True

    def forward(self, inputs, targets):
        return crps_ensemble(
            truth=targets,
            predicted=inputs,
        )


def crps_ensemble(
    truth: Tensor,  # TRUTH
    predicted: Tensor,  # FORECAST
    weights: Tensor = None,
    dim: Union[int, Iterable[int]] = (),
    reduction="mean",
) -> Tensor:
    """
    .. Author: Salva Rühling Cachay

    pytorch adaptation of https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/_crps.py#L187
    but implementing the fair, unbiased CRPS as in Zamo & Naveau (2018; https://doi.org/10.1007/s11004-017-9709-7)

    This implementation is based on the identity:
    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|
    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    We use the fair, unbiased formulation of the ensemble CRPS, which is particularly important for small ensembles.
    Anecdotically, the unbiased CRPS leads to slightly smaller (i.e. "better") values than the biased version.
    Basically, we use n_members * (n_members - 1) instead of n_members**2 to average over the ensemble spread.
    See Zamo & Naveau (2018; https://doi.org/10.1007/s11004-017-9709-7) for details.

    Alternative implementation: https://github.com/NVIDIA/modulus/pull/577/files
    """
    assert truth.ndim == predicted.ndim - 1, f"{truth.shape=}, {predicted.shape=}"
    assert truth.shape == predicted.shape[1:]  # ensemble ~ first axis
    n_members = predicted.shape[0]
    skill = (predicted - truth).abs().mean(dim=0)
    # insert new axes so forecasts_diff expands with the array broadcasting
    # torch.unsqueeze(predictions, 0) has shape (1, E, ...)
    # torch.unsqueeze(predictions, 1) has shape (E, 1, ...)
    forecasts_diff = torch.unsqueeze(predicted, 0) - torch.unsqueeze(predicted, 1)
    # Forecasts_diff has shape (E, E, ...)
    # Old version: score += - 0.5 * forecasts_diff.abs().mean(dim=(0, 1))
    # Using n_members * (n_members - 1) instead of n_members**2 is the fair, unbiased CRPS. Better for small ensembles.
    spread = forecasts_diff.abs().sum(dim=(0, 1)) / (n_members * (n_members - 1))
    crps = skill - 0.5 * spread
    # score has shape (...)  (same as observations)
    if reduction == "none":
        return crps
    assert reduction == "mean", f"Unknown reduction {reduction}"
    if weights is not None:  # weighted mean
        crps = (crps * weights).sum(dim=dim) / weights.expand(crps.shape).sum(dim=dim)
    else:
        crps = crps.mean(dim=dim)
    return crps


class AbstractWeightedLoss(torch.nn.Module):
    def __init__(
        self,
        weights: Tensor = None,
        reduction: str = "mean",
        learned_var_dim_name_to_idx_and_n_dims: Dict[int, int] = None,
        use_batch_logvars: bool = False,
        learn_per_dim: bool = True,
        reduce_op: str = "add",  # "add" or "mul"
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights = weights
        self.reduction = reduction
        self.learn_per_dim = learn_per_dim
        learned_var_dim_name_to_idx_and_n_dims = learned_var_dim_name_to_idx_and_n_dims or {}
        # If more than 1, sort by dim index (the first element of value, a tuple)
        # if len(learned_var_dim_name_to_idx_and_n_dims) >= 1:
        # learned_var_dim_name_to_idx_and_n_dims = dict(sorted(learned_var_dim_name_to_idx_and_n_dims.items(), key=lambda x: x[1][0]))
        # dim0_idx = next(iter(learned_var_dim_name_to_idx_and_n_dims.values()))[0]
        # assert all(dim0_idx <= idx for idx, _ in learned_var_dim_name_to_idx_and_n_dims.values()), f"Learned variance dimensions must be sorted by index. Got {learned_var_dim_name_to_idx_and_n_dims=}"
        # log.info(f">>>> {learned_var_dim_name_to_idx_and_n_dims=}")
        self.learned_var_dim_name_to_idx_and_n_dims = learned_var_dim_name_to_idx_and_n_dims
        if isinstance(use_batch_logvars, Sequence) or use_batch_logvars is True:
            b_log_var_dims = use_batch_logvars if isinstance(use_batch_logvars, Sequence) else [0]
            use_batch_logvars = True
        else:
            b_log_var_dims = []
        self.n_logvar_dims = len(learned_var_dim_name_to_idx_and_n_dims) + int(use_batch_logvars)
        self._spatial_logvar_dim_names = [k for k in learned_var_dim_name_to_idx_and_n_dims if k.startswith("spatial")]
        # Assert all dim idxs are unique
        self.use_batch_logvars = use_batch_logvars
        dim_idxs = b_log_var_dims
        dim_idxs += [idx for idx, _ in self.learned_var_dim_name_to_idx_and_n_dims.values()]
        dim_sizes = [n_dims for _, n_dims in self.learned_var_dim_name_to_idx_and_n_dims.values()]
        self.n_logvar_dims = len(dim_idxs)
        assert len(dim_idxs) == len(set(dim_idxs)), f"Dimension indices must be unique. {dim_idxs=}"

        # Save tuple of the dimension idxs for the log vars
        self._dim_idxs_from = tuple(dim_idxs)
        self._dim_idxs_to = tuple(range(self.n_logvar_dims))

        if learn_per_dim in [True, "except_spatial"]:
            log_var_names = []
            self.reduce_op = getattr(operator, reduce_op)  # add or mul
            if verbose and self.n_logvar_dims > 1:
                log.info(f"Using loss with learned variance for {self.n_logvar_dims} dimensions with {reduce_op=}.")
            for i, (dim_name, (dim_idx, n_dims)) in enumerate(self.learned_var_dim_name_to_idx_and_n_dims.items()):
                i = i + 1 if use_batch_logvars else i
                assert n_dims > 0, f"Number of dimensions must be positive. {n_dims=}"
                # Register in __init__ to make it part of the model's parameters
                # We implement the variance-weighted loss by simply learning a list of scalars, one per frame
                if learn_per_dim == "except_spatial" and dim_name in self._spatial_logvar_dim_names:
                    # Skip spatial dimensions
                    continue
                if self.n_logvar_dims > 1:
                    shape = [1] * self.n_logvar_dims
                    shape[i] = n_dims
                else:
                    shape = (n_dims,)
                dim_lv_weight = torch.randn(*shape, requires_grad=True) * 0.01
                logvar_param_name = f"{dim_name}_logvar"
                setattr(self, logvar_param_name, nn.Parameter(dim_lv_weight, requires_grad=True))
                log_var_names.append(logvar_param_name)
                # Double check it's just as good as:
                # setattr(self, f"{dim_name}_logvar", nn.Parameter(torch.zeros(n_dims, requires_grad=True)))
                if verbose:
                    log.info(f"Using loss with learned ``{dim_name}`` logvar with {n_dims=}, {dim_idx=}.")
            if learn_per_dim == "except_spatial":
                # Learn a scalar for each spatial_i x spatial_j x ... x spatial_n dimension
                assert (
                    len(self._spatial_logvar_dim_names) > 0
                ), f"Number of spatial dimensions must be positive. {self._spatial_logvar_dim_names=}"
                spatial_lv_dims = []
                for dim_name, (dim_idx, n_dims) in self.learned_var_dim_name_to_idx_and_n_dims.items():
                    if dim_name in self._spatial_logvar_dim_names:
                        spatial_lv_dims.append(n_dims)
                    else:
                        spatial_lv_dims.append(1)
                self._spatial_logvar = nn.Parameter(torch.zeros(*spatial_lv_dims, requires_grad=True))
                log_var_names.append("_spatial_logvar")
                if verbose:
                    log.info(f"Using loss with learned spatial logvar with dimensions {spatial_lv_dims}.")
            else:
                assert len(log_var_names) == self.n_logvar_dims - int(
                    use_batch_logvars
                ), f"{log_var_names=}, {self.n_logvar_dims=}"
            self.log_var_names = log_var_names
        else:
            assert learn_per_dim is False, f"Unknown {learn_per_dim=}"
            # Learn a scalar for each entry in dim_0 x dim_1 x ... x dim_n
            assert len(dim_sizes) > 0, f"Number of dimensions must be positive. {dim_sizes=}"
            self.logvars = nn.Parameter(torch.zeros(*dim_sizes, requires_grad=True))
            self._dim_to_logvar_dim = {
                dim_k: dim_idx for dim_idx, dim_k in enumerate(learned_var_dim_name_to_idx_and_n_dims)
            }
            if verbose:
                log.info(f"Using loss function with learned logvar with dimensions {dim_sizes}.")

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def channels_logvar_vector(self):
        return self.logvar_vector("channels")

    @property
    def spatial_logvar(self):
        if hasattr(self, "_spatial_logvar"):
            return self._spatial_logvar.squeeze()
        return self.logvar_vector(self._spatial_logvar_dim_names)

    def logvar_vector(self, dim_name):
        # Generally, the lower the variance, the higher the weight
        if isinstance(dim_name, str) and dim_name not in self.learned_var_dim_name_to_idx_and_n_dims:
            raise AttributeError(f"{dim_name=} not found in {self.learned_var_dim_name_to_idx_and_n_dims=}.")
        if self.learn_per_dim:
            if isinstance(dim_name, str):
                return getattr(self, f"{dim_name}_logvar").squeeze()
            else:
                return reduce(self.reduce_op, [getattr(self, f"{dim_name}_logvar") for dim_name in dim_name]).squeeze()
        else:
            dim_names = [dim_name] if isinstance(dim_name, str) else dim_name
            dim_idxs = [self._dim_to_logvar_dim[dim_n] for dim_n in dim_names]
            # Take mean over non-dim_name dimensions
            return self.logvars.mean(dim=tuple([idx for idx in range(self.logvars.ndim) if idx not in dim_idxs]))

    def weigh_loss(
        self, loss, add_weight=None, multiply_weight=None, batch_logvars=None, return_intermediate: bool = False
    ) -> Dict[str, Tensor]:
        losses = dict()
        if self.weights is not None:
            weights = self.weights
        else:
            assert (
                add_weight is not None
                or multiply_weight is not None
                    or batch_logvars is not None
                    or self.n_logvar_dims > 0
                ), "No weights given. Please provide/set - for example, as `loss_weights_tensor` in your training dataset."
            weights = torch.ones_like(loss)

        # Shapes are e.g.: inputs: (B, C, H, W), targets: (B, C, H, W), weights: (H, W) or (C, H, W)
        # Similarly, inputs, targets and weights can have shapes (B, C, S), (B, C, S), (S) or (C, S),
        # where S is the sequence length
        if add_weight is not None:
            # Don't do += because it may fail when broadcasting weights for different shapes
            weights = weights + add_weight

        if multiply_weight is not None:
            diff_shape = len(multiply_weight.shape) - len(weights.shape)
            if diff_shape != 0:
                # Add singleton dimensions to multiply_weight to match weights (based on diff_shape)
                for _ in range(diff_shape):
                    weights = weights.unsqueeze(0)  # Add batch dimension to weights # todo: do somewhere else

                # if len(weights.shape) < len(multiply_weight.shape):
                #     # Check if dim=1 is channel or time dimension
                #     if self.channel_dim is not None and weights.shape[self.channel_dim] == self.num_channels:
                #         # Add singleton channel dimension to weights at dim=self.channel_dim
                #         weights = weights.unsqueeze(2) #self.time_dim)
                #     # elif self.time_dim is not None and weights.shape[self.time_dim] == self.num_times:
                #         # Add singleton time dimension to weights at dim=self.time_dim
                #         # weights = weights.unsqueeze(self.time_dim)
                # # if self.time_dim is not None and weights.shape[self.time_dim] != self.num_times:
                #     # Add singleton time dimension to weights at dim=self.time_dim
                #     # weights = weights.unsqueeze(self.time_dim)
                self.weights = weights  # Update weights with new singleton dimensions to not repeat this step

            # print(f"{weights.shape=}, {multiply_weight.shape=}")
            # Don't do *= because it may fail when broadcasting weights for different shapes
            try:
                weights = weights * multiply_weight
            except RuntimeError as e:
                raise RuntimeError(f"Failed to compute {weights.shape=} * {multiply_weight.shape=}.") from e

        if return_intermediate:
            if self.n_logvar_dims == 0:
                if batch_logvars is None or not isinstance(batch_logvars, tuple):
                    losses["unweighted_loss"] = loss.detach().cpu()
                else:
                    # Special logvars on multiple dimensions
                    dims = tuple([idx for idx in range(loss.ndim) if idx not in batch_logvars[1]])
                    losses["unweighted_loss"] = loss.mean(dim=dims).detach().cpu()
            else:
                # Bring logvar dimensions to the front. note that the dims are sorted by index (increasing)
                loss_temp = torch.movedim(loss, self._dim_idxs_from, self._dim_idxs_to)
                # Take mean over non-logvar dimensions
                loss_temp = loss_temp.mean(dim=tuple(range(self.n_logvar_dims, loss_temp.ndim)))
                losses["unweighted_loss"] = loss_temp.detach().cpu()

        try:
            loss = weights * loss
        except RuntimeError as e:
            raise RuntimeError(f"Failed to multiply {weights.shape=} by {loss.shape=}.") from e

        losses["raw_loss"] = float(loss.mean().detach().cpu())
        if self.n_logvar_dims == 0:
            assert batch_logvars is None, "Please set use_batch_logvars of loss function to True to use batch_logvars."
            loss_final = loss
        else:
            if batch_logvars is not None:
                assert self.use_batch_logvars, "Please set use_batch_logvars to True."
            else:
                assert not self.use_batch_logvars, "Please set use_batch_logvars to False."

            # Bring logvar dimensions to the front. note that the dims are sorted by index (increasing)
            loss = torch.movedim(loss, self._dim_idxs_from, self._dim_idxs_to)
            # Take mean over non-logvar dimensions
            loss = loss.mean(dim=tuple(range(self.n_logvar_dims, loss.ndim)))

            if self.n_logvar_dims == 1:
                # Only one learned variance dimension
                log_vars = getattr(self, self.log_var_names[0]) if batch_logvars is None else batch_logvars.squeeze(-1)
            elif self.learn_per_dim:
                # Multiple learned variance dimensions
                log_vars = [batch_logvars] if batch_logvars is not None else []
                log_vars += [getattr(self, log_var_name) for log_var_name in self.log_var_names]
                log_vars = reduce(self.reduce_op, log_vars)  # Will properly broadcast (69, 1, 1) * (1, 240, 1) * etc.
                # log_vars = torch.outer(*log_vars)
            else:
                log_vars = self.logvars  # Learned variance for all dimensions (>=2)

            # Important check below.
            #   If specified logvar dims are misaligned (e.g. due to unexpected singleton dimensions),
            #   the loss will be broadcasted incorrectly (e.g. loss of shape (1,) will be broadcasted to log_vars shape)
            #   rather than correctly applying on a per-dimension basis.
            assert loss.shape == log_vars.shape, f"{loss.shape=}, {log_vars.shape=}, {weights.shape=}"
            # Apply the learned variance to the loss
            loss_final = loss / torch.exp(log_vars) + log_vars
        if return_intermediate:
            losses["weighted_loss_before_vars"] = loss.detach().cpu()
            losses["weighted_loss_after_vars"] = loss_final.detach().cpu()

        if self.reduction == "mean":
            loss_final = loss_final.mean()
        elif self.reduction == "sum":
            loss_final = loss_final.sum()
        else:
            raise ValueError(f"Unknown reduction {self.reduction}")
        losses["loss"] = loss_final
        return losses

    def forward(self, preds, targets):
        raise NotImplementedError("Subclasses must implement this method.")


class WeightedMSE(AbstractWeightedLoss):
    def forward(self, preds, targets, **kwargs):
        error = self.weigh_loss((preds - targets) ** 2, **kwargs)
        return error


class WeightedMAE(AbstractWeightedLoss):
    def forward(self, preds, targets, **kwargs):
        error = self.weigh_loss((preds - targets).abs(), **kwargs)
        return error


class WeightedCRPS(AbstractWeightedLoss):
    require_ensembling: bool = True

    def forward(self, preds, targets):
        error = self.weigh_loss(crps_ensemble(predicted=preds, truth=targets, reduction="none"))
        return error


def get_loss(name, reduction="mean", **kwargs):
    """Returns the loss function with the given name."""
    name = name.lower().strip().replace("-", "_")
    if name in ["l1", "mae", "mean_absolute_error"]:
        loss = nn.L1Loss(reduction=reduction, **kwargs)
    elif name in ["l2", "mse", "mean_squared_error"]:
        loss = nn.MSELoss(reduction=reduction, **kwargs)
    elif name in ["l2_rel"]:
        loss = LpLoss(p=2, relative=True, **kwargs)
    elif name in ["l1_rel"]:
        loss = LpLoss(p=1, relative=True, **kwargs)
    elif name in ["smoothl1", "smooth"]:
        loss = nn.SmoothL1Loss(reduction=reduction, **kwargs)
    elif name in ["wmse", "weighted_mse"]:
        loss = WeightedMSE(**kwargs)
    elif name in ["wmseold2"]:
        raise ValueError("wmseold2 is not included in this ERA5-MINI transplant.")
    elif name in ["wmae", "weighted_mae"]:
        loss = WeightedMAE(**kwargs)
    elif name in ["wcrps", "weighted_crps"]:
        loss = WeightedCRPS(**kwargs)
    # elif name in ["crps_gaussian"]:
    #     loss = CRPSGaussianLoss(reduction=reduction)
    elif name in ["crps"]:
        assert reduction == "mean", "CRPS loss only supports mean reduction"
        loss = CRPSLoss(**kwargs)
    # elif name in ["nll", "negative_log_likelihood"]:
    #     loss = NLLLoss(reduction=reduction)
    else:
        raise ValueError(f"Unknown loss function {name}")
    return loss
