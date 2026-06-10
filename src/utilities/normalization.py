from functools import partial
from typing import Any, Dict, List

import torch
import os
import xarray as xr

from src.utilities.utils import get_logger


log = get_logger(__name__)


class StandardNormalizer(torch.nn.Module):
    """
    Responsible for normalizing tensors.
    """

    def __init__(
        self,
        means: Dict[str, torch.Tensor],
        stds: Dict[str, torch.Tensor],
        names=None,
        var_to_transform_name=None,
        std_residual=None,
        input_names: List[str] = None,
    ):
        super().__init__()
        self.means = means
        self.stds = stds
        self.std_residual = std_residual
        self.var_to_transform_name = var_to_transform_name

        if torch.is_tensor(means) or isinstance(means, float):
            assert (
                var_to_transform_name is None
            ), f"{var_to_transform_name=} must be None if means and stds are floats!"
            assert names is None, f"{names=} must be None if means and stds are floats!"
            self.names = None
            self._normalize = _normalize
            self._denormalize = _denormalize
        else:
            self.names = names if names is not None else list(means.keys())
            assert isinstance(means, dict), "Means and stds must be either both tensors, floats, or dictionaries!"
            assert all(name in means for name in self.names), "All names must be keys in the means dictionary!"
            assert all(name in stds for name in self.names), "All names must be keys in the stds dictionary!"
            if var_to_transform_name is None or len(var_to_transform_name) == 0:
                self._normalize = _normalize_dict
                self._denormalize = _denormalize_dict
            else:
                assert isinstance(var_to_transform_name, dict), "var_to_transform_name must be a dict!"
                transforms, inverse_transforms = dict(), dict()
                for name in self.names:
                    transforms_name = var_to_transform_name.get(name, "null")
                    transforms[name] = TRANSFORMS[transforms_name]["transform"]
                    inverse_transforms[name] = TRANSFORMS[transforms_name]["inverse"]
                self._normalize = partial(_normalize_dict_with_transform, transforms=transforms)
                self._denormalize = partial(_denormalize_dict_with_transform, inverse_transforms=inverse_transforms)

        if self.std_residual is not None:
            scale_normed_to_residual_normed = dict()
            scale_normed_residual_to_normed = dict()
            for k in self.names:
                if k not in input_names:
                    scale_normed_to_residual_normed[k] = torch.tensor(1.0)
                    scale_normed_residual_to_normed[k] = torch.tensor(1.0)
                    log.warning(f"Variable {k} not in input_names; setting residual scaling factors to 1.0")
                else:
                    scale_normed_to_residual_normed[k] = self.stds[k] / self.std_residual[k]
                    scale_normed_residual_to_normed[k] = self.std_residual[k] / self.stds[k]
            self.scale_normed_to_residual_normed = scale_normed_to_residual_normed
            self.scale_normed_residual_to_normed = scale_normed_residual_to_normed

    def _apply(self, fn, recurse=True):
        super()._apply(fn)  # , recurse=recurse)
        if isinstance(self.means, dict):
            self.means = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.means.items()}
            self.stds = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.stds.items()}
            if self.std_residual is not None:
                self.std_residual = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.std_residual.items()}
        else:
            self.means = fn(self.means) if torch.is_tensor(self.means) else self.means
            self.stds = fn(self.stds) if torch.is_tensor(self.stds) else self.stds
            if self.std_residual is not None:
                self.std_residual = fn(self.std_residual) if torch.is_tensor(self.std_residual) else self.std_residual

    def normalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._normalize(tensors, means=self.means, stds=self.stds)

    def denormalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.names is not None:  # todo: remove this check
            assert (
                len(set(tensors.keys()) - set(self.names)) == 0
            ), f"Some keys would not be denormalized: {set(tensors.keys()) - set(self.names)}!"
        return self._denormalize(tensors, means=self.means, stds=self.stds)

    def normalized_to_residual_normalized(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return scale_dict(tensors, scales=self.scale_normed_to_residual_normed)

    def normalized_residual_to_normalized(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return scale_dict(tensors, scales=self.scale_normed_residual_to_normed)

    def __copy__(self):
        return StandardNormalizer(self.means, self.stds, self.names, self.var_to_transform_name)

    def clone(self):
        return self.__copy__()


@torch.jit.script
def _normalize_dict(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: (t - means[k]) / stds[k] for k, t in tensors.items()}


# @torch.jit.script
def _normalize_dict_with_transform(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
    transforms,  # e.g. precip: lambda x: torch.log(x + 1), temperature: lambda x: x
) -> Dict[str, torch.Tensor]:
    return {k: (transforms[k](t) - means[k]) / stds[k] for k, t in tensors.items()}


@torch.jit.script
def _denormalize_dict(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: t * stds[k] + means[k] for k, t in tensors.items()}


# @torch.jit.script
def _denormalize_dict_with_transform(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
    inverse_transforms,  # e.g. precip: lambda x: torch.exp(x) - 1, temperature: lambda x: x
) -> Dict[str, torch.Tensor]:
    return {k: inverse_transforms[k](t * stds[k] + means[k]) for k, t in tensors.items()}


@torch.jit.script
def scale_dict(
    tensors: Dict[str, torch.Tensor],
    scales: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: t * scales[k] for k, t in tensors.items()}


@torch.jit.script
def _normalize(tensor: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    return (tensor - means) / stds


@torch.jit.script
def _denormalize(tensor: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    return tensor * stds + means


@torch.jit.script
def _normalized_to_residual_normalized(
    tensor: torch.Tensor, stds: torch.Tensor, std_res: torch.Tensor
) -> torch.Tensor:
    return tensor * (stds / std_res)


@torch.jit.script
def _normalized_residual_to_normalized(
    tensor: torch.Tensor, stds: torch.Tensor, std_res: torch.Tensor
) -> torch.Tensor:
    return tensor * (std_res / stds)


def to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.as_tensor(x.values, dtype=torch.float)


def _extract_variables(ds: xr.Dataset, names: List[str], is_2d_flattened: bool) -> Dict[str, torch.Tensor]:
    """Helper to extract specific variables or pressure levels from a dataset."""
    extracted = {}
    for name in names:
        # Case 1: Simple extraction (Direct match or not flattened mode)
        if not is_2d_flattened or name in ds:
            extracted[name] = to_tensor(ds[name])
            continue

        # Case 2: Flattened mode logic (parsing <var_name>_<pressure_level>)
        parts = name.split("_")
        var_name = "_".join(parts[:-1])
        pressure_level = parts[-1]

        if not pressure_level.isdigit():
            raise ValueError(f"{name} is not in format <var>_<level>. Available keys: {list(ds.keys())}")

        level = int(pressure_level)
        try:
            # Select specific level from the 3D variable
            data = ds[var_name].sel(level=level)
            extracted[name] = to_tensor(data)
        except KeyError as e:
            print(f"Available coords: {ds.coords.values}")
            raise KeyError(f"Variable {name} (var: {var_name}, level: {level}) not found in dataset.") from e

    return extracted


def get_normalizer(
    global_means_path: str,
    global_stds_path: str,
    names: List[str],
    global_stds_res_path: str = None,
    sel: Dict[str, Any] = None,
    is_2d_flattened: bool = False,
    **kwargs,
) -> StandardNormalizer:
    # 1. Load Data
    mean_ds = xr.open_dataset(global_means_path)
    std_ds = xr.open_dataset(global_stds_path)
    std_res_ds = xr.open_dataset(global_stds_res_path) if (global_stds_res_path is not None and
        os.path.exists(global_stds_res_path)) else None

    # 2. Apply global selection if provided
    if sel is not None:
        mean_ds = mean_ds.sel(**sel)
        std_ds = std_ds.sel(**sel)
        if std_res_ds is not None:
            std_res_ds = std_res_ds.sel(**sel)

    # 3. Extract tensors using the new helper function
    means = _extract_variables(mean_ds, names, is_2d_flattened)
    stds = _extract_variables(std_ds, names, is_2d_flattened)
    if std_res_ds is not None:
        std_res_ds = _extract_variables(std_res_ds, names, is_2d_flattened)

    return StandardNormalizer(means=means, stds=stds, names=names, std_residual=std_res_ds, **kwargs)


@torch.jit.script
def log1p_transform(x):
    return torch.log(x + 1)


@torch.jit.script
def log1p_transform_inverse(x):
    return torch.exp(x) - 1


@torch.jit.script
def log_transform(x):
    return torch.log(x + 1e-8)


@torch.jit.script
def log_transform_inverse(x):
    return torch.exp(x) - 1e-8


@torch.jit.script
def log_transform_general(x, factor: float, offset: float):
    return torch.log(x * factor + offset)


@torch.jit.script
def log_transform_general_inverse(x, factor: float, offset: float):
    return (torch.exp(x) - offset) / factor


TRANSFORMS = {
    "log1p": {"transform": log1p_transform, "inverse": log1p_transform_inverse},
    "log": {"transform": log_transform, "inverse": log_transform_inverse},
    "log_mm_day_1": {
        "transform": partial(log_transform_general, factor=86400, offset=1),
        "inverse": partial(log_transform_general_inverse, factor=86400, offset=1),
    },
    "log_mm_day_001": {
        "transform": partial(log_transform_general, factor=86400, offset=0.01),
        "inverse": partial(log_transform_general_inverse, factor=86400, offset=0.01),
    },
    "null": {"transform": lambda x: x, "inverse": lambda x: x},
}
TRANSFORMS["log_1"] = TRANSFORMS["log1p"]
TRANSFORMS["log_1e-8"] = TRANSFORMS["log"]

# for n in np.linspace(0, 10, 500):
#     assert log_transform_inverse(log_transform(torch.tensor(n))) == torch.tensor(n)
