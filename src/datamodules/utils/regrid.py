from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


# Type hints
Tensor = torch.Tensor
Coords = Dict[str, Union[np.ndarray, Tensor, list]]


class ConservativeRegridder(nn.Module):
    """
    PyTorch implementation of linear conservative regridding.
    Preserves the integral of the field (e.g. mass, energy) across the regridding.

    Expects input tensors of shape (..., lat, lon).
    """

    def __init__(
        self,
        coords_source: Coords,
        coords_dest: Coords,
        periodic: bool = True,
        include_poles: bool = True,
        which_substrings: Optional[List[str]] = None,  # If set, used to filter which tensors to regrid in forward
        two_step_einsum: bool = False,
    ):
        super().__init__()

        self.periodic = periodic
        self.include_poles = include_poles
        self.two_step_einsum = two_step_einsum
        self.which_substrings = which_substrings

        # Extract coordinates and ensure they are torch tensors
        src_lat = self._to_tensor(coords_source.get("latitude", coords_source.get("lat")))
        src_lon = self._to_tensor(coords_source.get("longitude", coords_source.get("lon")))
        dst_lat = self._to_tensor(coords_dest.get("latitude", coords_dest.get("lat")))
        dst_lon = self._to_tensor(coords_dest.get("longitude", coords_dest.get("lon")))

        # Ensure coordinates are increasing
        self._check_increasing(src_lat, "Source Latitude")
        self._check_increasing(src_lon, "Source Longitude")
        self._check_increasing(dst_lat, "Target Latitude")
        self._check_increasing(dst_lon, "Target Longitude")

        # Precompute weights
        # We perform weight calculation in float64 for precision, then cast to float32 (or default)
        # to match the module's floating point standard.
        with torch.no_grad():
            lat_weights = self._compute_latitude_weights(
                src_lat.double(), dst_lat.double(), include_poles, include_poles
            )
            lon_weights = self._compute_longitude_weights(src_lon.double(), dst_lon.double(), periodic, periodic)

        # Register as buffers so they are saved with state_dict and move to device automatically
        self.register_buffer("lat_weights", lat_weights.to(torch.get_default_dtype()))
        self.register_buffer("lon_weights", lon_weights.to(torch.get_default_dtype()))

    def _to_tensor(self, x) -> Tensor:
        if x is None:
            raise ValueError("Coordinates must contain 'Lat'/'lat' and 'Lon'/'lon' keys.")
        if torch.is_tensor(x):
            return x.detach()
        elif hasattr(x, "values"):  # e.g. xarray.DataArray
            return torch.from_numpy(x.values)
        return torch.tensor(x)

    def _check_increasing(self, x: Tensor, name: str):
        diffs = torch.diff(x)
        if not torch.all(diffs > 0):
            raise ValueError(f"{name} array is not strictly increasing.")

    def forward(self, field: Tensor, name: str = None) -> Tensor:
        """
        Regrid tensor of shape (..., lat, lon).
        Handles NaNs using the same logic as the JAX implementation (renormalization).
        """
        if self.which_substrings is not None and name is not None:
            if not any(sub in name for sub in self.which_substrings):
                # print(f"{name} does not contain any of {self.which_substrings}, skipping regridding. {field.shape=}")
                return field
        # Ensure input is on the same device as weights
        if field.device != self.lat_weights.device:
            self.lat_weights = self.lat_weights.to(field.device)
            self.lon_weights = self.lon_weights.to(field.device)

        if torch.isnan(field).any():
            out = self._nanmean(field)
        else:
            out = self._apply_weights(field, name)
        if torch.isnan(out).any():
            assert torch.isnan(field).any(), f"{torch.isnan(out).sum().item()=}, {torch.isnan(field).sum().item()=}"

        return out

    def _apply_weights(self, field: Tensor, name=None) -> Tensor:
        """
        Applies weights using Einsum.
        Field: (..., source_lat, source_lon)
        Lat_weights: (target_lat, source_lat)
        Lon_weights: (target_lon, source_lon)

        Equation: source_lat(i), target_lat(j), source_lon(k), target_lon(l)
        W_lat[j, i] * W_lon[l, k] * Field[..., i, k] -> Output[..., j, l]
        """
        if len(field.shape) < 2:
            raise ValueError(f"{field.shape=}, expected at least 2D tensor with lat and lon dimensions. {name=}")
        # 1. Force float32 to prevent float16 overflow/NaNs
        # 2. Disable autocast to prevent PyTorch from downcasting back to float16 inside the block
        with torch.amp.autocast("cuda", enabled=False):
            field_f32 = field.float()
            lat_w_f32 = self.lat_weights.float()
            lon_w_f32 = self.lon_weights.float()

            if self.two_step_einsum:
                # Two-step matmul to avoid materializing the full outer product
                # (saves memory for large grids like GLORYS12):
                # Equivalent to: einsum("ji,lk,...ik->...jl", lat_w, lon_w, field)
                # Step 1: contract over source latitude → (..., target_lat, source_lon)
                tmp = torch.einsum("ji,...ik->...jk", lat_w_f32, field_f32)
                # Step 2: contract over source longitude → (..., target_lat, target_lon)
                out = torch.einsum("lk,...jk->...jl", lon_w_f32, tmp)
            else:
                if field.shape[-2] < field.shape[-1]:
                    out = torch.einsum("ji,lk,...ik->...jl", lat_w_f32, lon_w_f32, field_f32)
                else:
                    out = torch.einsum("ji,lk,...ki->...lj", lat_w_f32, lon_w_f32, field_f32)

            # Cast back to original dtype if needed
            return out.to(field.dtype)

    def _nanmean(self, field: Tensor) -> Tensor:
        """Compute cell-averages skipping NaNs."""
        nulls = torch.isnan(field)
        # Replace NaN with 0 for the sum
        filled_field = torch.where(nulls, torch.tensor(0.0, device=field.device, dtype=field.dtype), field)

        total = self._apply_weights(filled_field)

        # Calculate the weight of non-nan values
        valid_mask = torch.logical_not(nulls).to(field.dtype)
        count = self._apply_weights(valid_mask)

        # Avoid division by zero
        return total / count

    # --------------------------------------------------------------------------
    # Latitude Logic
    # --------------------------------------------------------------------------

    def _compute_latitude_weights(self, src: Tensor, dst: Tensor, src_poles: bool, dst_poles: bool) -> Tensor:
        overlap = self._latitude_overlap(src, dst, src_poles, dst_poles)
        coverage = torch.sum(overlap, dim=1, keepdim=True)

        weights = overlap / coverage

        # If source doesn't include poles, handle coverage gaps (set to NaN if target not fully covered)
        if not src_poles:
            # Double check that code below is correct before enabling this
            raise NotImplementedError("Source grids without poles are not yet supported.")
            target_areas = self._latitude_area(dst, dst_poles).unsqueeze(1)
            is_covered = torch.isclose(coverage, target_areas, rtol=1e-3)
            weights = torch.where(
                is_covered, weights, torch.tensor(float("nan"), device=weights.device, dtype=weights.dtype)
            )

        return weights

    def _latitude_cell_bounds(self, x: Tensor, include_poles: bool) -> Tensor:
        if include_poles:
            initial = torch.tensor([-90.0], device=x.device, dtype=x.dtype)
            final = torch.tensor([90.0], device=x.device, dtype=x.dtype)
        else:
            initial = x[:1] - (x[1] - x[0]) / 2
            final = x[-1:] + (x[-1] - x[-2]) / 2

        midpoints = (x[:-1] + x[1:]) / 2
        return torch.cat([initial, midpoints, final])

    def _latitude_area_from_bounds(self, lower: Tensor, upper: Tensor) -> Tensor:
        return torch.sin(torch.deg2rad(upper)) - torch.sin(torch.deg2rad(lower))

    def _latitude_area(self, points: Tensor, include_poles: bool) -> Tensor:
        bounds = self._latitude_cell_bounds(points, include_poles)
        return self._latitude_area_from_bounds(bounds[:-1], bounds[1:])

    def _latitude_overlap(self, src: Tensor, dst: Tensor, src_poles: bool, dst_poles: bool) -> Tensor:
        src_bounds = self._latitude_cell_bounds(src, src_poles)
        dst_bounds = self._latitude_cell_bounds(dst, dst_poles)

        # Broadcast comparison: dst (rows) vs src (cols)
        # dst_bounds: (M+1), src_bounds: (N+1)
        # Create intervals
        src_lower = src_bounds[:-1][None, :]  # (1, N)
        src_upper = src_bounds[1:][None, :]  # (1, N)
        dst_lower = dst_bounds[:-1][:, None]  # (M, 1)
        dst_upper = dst_bounds[1:][:, None]  # (M, 1)

        upper = torch.minimum(dst_upper, src_upper)
        lower = torch.maximum(dst_lower, src_lower)

        mask = upper > lower
        overlap = self._latitude_area_from_bounds(lower, upper)
        return torch.where(mask, overlap, torch.tensor(0.0, device=overlap.device, dtype=overlap.dtype))

    # --------------------------------------------------------------------------
    # Longitude Logic
    # --------------------------------------------------------------------------

    def _compute_longitude_weights(self, src: Tensor, dst: Tensor, src_periodic: bool, dst_periodic: bool) -> Tensor:
        if len(dst) < 3 and dst_periodic:
            raise ValueError("Need 3 or more target points for periodic boundaries.")

        overlap = self._longitude_overlap(dst, src, dst_periodic, src_periodic)
        coverage = torch.sum(overlap, dim=1, keepdim=True)

        weights = overlap / coverage

        if not src_periodic:
            target_lengths = self._longitude_length(dst, dst_periodic).unsqueeze(1)
            is_covered = torch.isclose(coverage, target_lengths, rtol=1e-3)
            weights = torch.where(
                is_covered, weights, torch.tensor(float("nan"), device=weights.device, dtype=weights.dtype)
            )

        return weights

    def _align_phase_with(self, x: Tensor, target: Tensor, period: float = 360.0) -> Tensor:
        """
        Aligns phase of x to be near target.
        Supports broadcasting: x(1, N) and target(M, 1) -> result(M, N)
        """
        # shifts needed
        shift_down = x > target + period / 2
        shift_up = x < target - period / 2
        return x + period * shift_up.double() - period * shift_down.double()

    def _periodic_upper_lower_bounds(self, x: Tensor, period: Optional[float]):
        if period is None:
            # Extrapolation logic
            x_minus = torch.cat([x[:1] - (x[1] - x[0]), x[:-1]])
            x_plus = torch.cat([x[1:], x[-1:] + (x[-1] - x[-2])])
            upper = (x + x_plus) / 2
            lower = (x_minus + x) / 2
            return upper, lower
        else:
            x = x % period
            # Midpoint of x and roll(x, -1)
            x_rolled_neg = torch.roll(x, -1, dims=0)
            x_plus = self._align_phase_with(x_rolled_neg, x, period)
            upper = (x + x_plus) / 2

            # Midpoint of x and roll(x, 1)
            x_rolled_pos = torch.roll(x, 1, dims=0)
            x_minus = self._align_phase_with(x_rolled_pos, x, period)
            lower = (x_minus + x) / 2
            return upper, lower

    def _longitude_length(self, points: Tensor, periodic: bool) -> Tensor:
        period = 360.0 if periodic else None
        upper, lower = self._periodic_upper_lower_bounds(points, period)
        return upper - lower

    def _longitude_overlap(
        self, target_pts: Tensor, src_pts: Tensor, target_periodic: bool, src_periodic: bool
    ) -> Tensor:

        src_period = 360.0 if src_periodic else None
        target_period = 360.0 if target_periodic else None

        # 1. Get bounds for both grids
        # src_upper/lower: (N,)
        src_upper, src_lower = self._periodic_upper_lower_bounds(src_pts, src_period)
        # target_upper/lower: (M,)
        target_upper, target_lower = self._periodic_upper_lower_bounds(target_pts, target_period)

        # 2. Prepare for broadcasting (M, N)
        # Source becomes row vectors (1, N)
        src_l = src_lower.unsqueeze(0)
        src_u = src_upper.unsqueeze(0)

        # Target becomes col vectors (M, 1)
        tgt_l = target_lower.unsqueeze(1)
        tgt_u = target_upper.unsqueeze(1)

        # 3. Compute overlap with phase alignment (period = 360)
        period = 360.0

        # Align source bounds (y) to match target bounds (x)
        # We align the intervals.
        # Note: _align_phase_with(val, target, period)
        y0 = self._align_phase_with(src_l, tgt_l, period)
        y1 = self._align_phase_with(src_u, tgt_l, period)

        # Intersection of [tgt_l, tgt_u] and [y0, y1]
        upper = torch.minimum(tgt_u, y1)
        lower = torch.maximum(tgt_l, y0)

        return torch.maximum(upper - lower, torch.tensor(0.0, device=upper.device, dtype=upper.dtype))


def latitude_values(num: int, latitude_spacing: str = "with_poles") -> np.ndarray:
    """Latitude node values given spacing and number of nodes."""
    if latitude_spacing == "with_poles":
        lat_start = -90
        lat_stop = 90
    elif latitude_spacing == "without_poles":
        lat_start = -90 + 0.5 * 180 / num
        lat_stop = 90 - 0.5 * 180 / num
    else:
        raise ValueError(f"Unhandled {latitude_spacing=}")
    return np.linspace(lat_start, lat_stop, num=num)


def longitude_values(num: int, longitude_scheme: str = "start_at_0") -> np.ndarray:
    """Longitude node values given scheme and number of nodes."""
    lon_delta = 360 / num
    if longitude_scheme == "start_at_0":
        lon_start = 0
        lon_stop = 360 - lon_delta
    elif longitude_scheme == "centered_at_0":
        lon_start = -180 + lon_delta / 2
        lon_stop = 180 - lon_delta / 2
    else:
        raise ValueError(f"Unhandled {longitude_scheme=}")
    return np.linspace(lon_start, lon_stop, num=num)


if __name__ == "__main__":
    # Create dummy grids
    n_lat, n_lon = 181, 360
    lat_src = torch.linspace(-90, 90, n_lat)
    lon_src = torch.linspace(0, 359, n_lon)  # coarse
    lat_dst = torch.linspace(-90, 90, 121)
    lon_dst = torch.linspace(0, 359, 240)  # fine

    coords_src = {"lat": lat_src, "lon": lon_src}
    coords_dst = {"lat": lat_dst, "lon": lon_dst}

    # Initialize Regridder
    regridder = ConservativeRegridder(coords_src, coords_dst)

    # Create dummy data (Batch, Time, Lat, Lon)
    data = torch.rand(2, 5, n_lat, n_lon)

    # Regrid
    out = regridder(data)

    print(f"Input shape: {data.shape}")
    print(f"Output shape: {out.shape}")
    # Input shape: torch.Size([2, 5, 44, 36])
    # Output shape: torch.Size([2, 5, 90, 180])
