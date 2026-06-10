# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset utilities."""

from typing import Any, Mapping, Sequence, Union

import numpy as np
import xarray


TimedeltaLike = Any  # Something convertible to pd.Timedelta.
TimedeltaStr = str  # A string convertible to pd.Timedelta.

TargetLeadTimes = Union[TimedeltaLike, Sequence[TimedeltaLike], slice]  # with TimedeltaLike as its start and stop.

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

DAY_PROGRESS = "day_progress"
YEAR_PROGRESS = "year_progress"
_DERIVED_VARS = {
    DAY_PROGRESS,
    f"{DAY_PROGRESS}_sin",
    f"{DAY_PROGRESS}_cos",
    YEAR_PROGRESS,
    f"{YEAR_PROGRESS}_sin",
    f"{YEAR_PROGRESS}_cos",
}
TISR = "toa_incident_solar_radiation"


def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
    """Computes year progress for times in seconds.

    Args:
      seconds_since_epoch: Times in seconds since the "epoch" (the point at which
        UNIX time starts).

    Returns:
      Year progress normalized to be in the [0, 1) interval for each time point.
    """

    # Start with the pure integer division, and then float at the very end.
    # We will try to keep as much precision as possible.
    years_since_epoch = seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR)
    # Note depending on how these ops are down, we may end up with a "weak_type"
    # which can cause issues in subtle ways, and hard to track here.
    # In any case, casting to float32 should get rid of the weak type.
    # [0, 1.) Interval.
    return np.mod(years_since_epoch, 1.0).astype(np.float32)


def get_day_progress(
    seconds_since_epoch: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """Computes day progress for times in seconds at each longitude.

    Args:
      seconds_since_epoch: 1D array of times in seconds since the 'epoch' (the
        point at which UNIX time starts).
      longitude: 1D array of longitudes at which day progress is computed.

    Returns:
      2D array of day progress values normalized to be in the [0, 1) inverval
        for each time point at each longitude.
    """

    # [0.0, 1.0) Interval.
    day_progress_greenwich = np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY

    # Offset the day progress to the longitude of each point on Earth.
    longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
    day_progress = np.mod(day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0)
    return day_progress.astype(np.float32)


def featurize_progress(name: str, dims: Sequence[str], progress: np.ndarray) -> Mapping[str, xarray.Variable]:
    """Derives features used by ML models from the `progress` variable.

    Args:
      name: Base variable name from which features are derived.
      dims: List of the output feature dimensions, e.g. ("day", "longitude").
      progress: Progress variable values.

    Returns:
      Dictionary of xarray variables derived from the `progress` values. It
      includes the original `progress` variable along with its sin and cos
      transformations.

    Raises:
      ValueError if the number of feature dimensions is not equal to the number
        of data dimensions.
    """
    if len(dims) != progress.ndim:
        raise ValueError(
            f"Number of feature dimensions ({len(dims)}) must be equal to the"
            f" number of data dimensions: {progress.ndim}."
        )
    progress_phase = progress * (2 * np.pi)
    return {
        name: xarray.Variable(dims, progress),
        name + "_sin": xarray.Variable(dims, np.sin(progress_phase)),
        name + "_cos": xarray.Variable(dims, np.cos(progress_phase)),
    }


def get_seconds_since_epoch(time_sequence: xarray.DataArray) -> np.ndarray:
    """Computes seconds since epoch from `data` in place if missing."""
    # Note `time_sequence.astype("datetime64[s]").astype(np.int64)`
    # does not work as xarrays always cast dates into nanoseconds!
    return time_sequence.data.astype("datetime64[s]").astype(np.int64)


def add_derived_vars(data: xarray.Dataset) -> None:
    """Adds year and day progress features to `data` in place if missing.

    Args:
      data: Xarray dataset to which derived features will be added.

    Raises:
      ValueError if `time` or `lon` are not in `data` coordinates.
    """

    for coord in ("time", "longitude"):
        if coord not in data.coords:
            raise ValueError(f"'{coord}' must be in `data` coordinates.")

    # Compute seconds since epoch.
    seconds_since_epoch = get_seconds_since_epoch(data.coords["time"])
    batch_dim = ("batch",) if "batch" in data.dims else ()

    # Add year progress features if missing.
    if YEAR_PROGRESS not in data.data_vars:
        year_progress = get_year_progress(seconds_since_epoch)
        data.update(
            featurize_progress(
                name=YEAR_PROGRESS,
                dims=batch_dim + ("time",),
                progress=year_progress,
            )
        )

    # Add day progress features if missing.
    if DAY_PROGRESS not in data.data_vars:
        longitude_coord = data.coords["longitude"]
        day_progress = get_day_progress(seconds_since_epoch, longitude_coord.data)
        data.update(
            featurize_progress(
                name=DAY_PROGRESS,
                dims=batch_dim + ("time",) + longitude_coord.dims,
                progress=day_progress,
            )
        )


def add_tisr_var(data: xarray.Dataset) -> None:
    """Adds TISR feature to `data` in place if missing.

    Args:
      data: Xarray dataset to which TISR feature will be added.

    Raises:
      ValueError if `time`, 'latitude', or `lon` are not in `data` coordinates.
    """
    from src.datamodules.utils import solar_radiation

    if TISR in data.data_vars:
        return

    for coord in ("time", "latitude", "longitude"):
        if coord not in data.coords:
            raise ValueError(f"'{coord}' must be in `data` coordinates.")

    # Remove `batch` dimension of size one if present. An error will be raised if
    # the `batch` dimension exists and has size greater than one.
    data_no_batch = data.squeeze("batch") if "batch" in data.dims else data

    tisr = solar_radiation.get_toa_incident_solar_radiation_for_xarray(data_no_batch, use_jit=True)

    if "batch" in data.dims:
        tisr = tisr.expand_dims("batch", axis=0)

    data.update({TISR: tisr})


def compute_forcings_numpy(
    time_vals: np.ndarray,  # Shape (T,) datetime64[ns]
    lon_vals: np.ndarray,  # Shape (Lon,)
    lat_vals: np.ndarray,  # Shape (Lat,) - needed for TISR if you use it
    forcing_fields: list[str],
) -> dict:
    """
    Numpy-only replacement for add_derived_vars.
    Output shapes will broadcast to (Time, Lon, Lat) to match your Zarr structure.
    """
    results = {}
    if not forcing_fields:
        return results

    T = len(time_vals)
    Lon = len(lon_vals)
    Lat = len(lat_vals)

    # 1. Compute Seconds Since Epoch (NumPy version of get_seconds_since_epoch)
    # Cast to seconds, then to int64
    seconds_since_epoch = time_vals.astype("datetime64[s]").astype(np.int64)

    # --- YEAR PROGRESS ---
    # Check if any year_progress var is requested
    if any("year_progress" in f for f in forcing_fields):
        # Result shape: (T,)
        yp = get_year_progress(seconds_since_epoch)

        # Pre-calculate sin/cos if needed
        yp_phase = yp * (2 * np.pi)
        yp_sin = np.sin(yp_phase)
        yp_cos = np.cos(yp_phase)

        # Broadcast (T) -> (T, Lon, Lat)
        # We use stride_tricks or tiling. Tiling is safer for now.
        if "year_progress" in forcing_fields:
            results["year_progress"] = np.tile(yp[:, None, None], (1, Lon, Lat))
        if "year_progress_sin" in forcing_fields:
            results["year_progress_sin"] = np.tile(yp_sin[:, None, None], (1, Lon, Lat))
        if "year_progress_cos" in forcing_fields:
            results["year_progress_cos"] = np.tile(yp_cos[:, None, None], (1, Lon, Lat))

    # --- DAY PROGRESS ---
    if any("day_progress" in f for f in forcing_fields):
        # Result shape: (T, Lon)
        dp = get_day_progress(seconds_since_epoch, lon_vals)

        dp_phase = dp * (2 * np.pi)
        dp_sin = np.sin(dp_phase)
        dp_cos = np.cos(dp_phase)

        # Broadcast (T, Lon) -> (T, Lon, Lat)
        if "day_progress" in forcing_fields:
            results["day_progress"] = np.repeat(dp[:, :, None], Lat, axis=2)
        if "day_progress_sin" in forcing_fields:
            results["day_progress_sin"] = np.repeat(dp_sin[:, :, None], Lat, axis=2)
        if "day_progress_cos" in forcing_fields:
            results["day_progress_cos"] = np.repeat(dp_cos[:, :, None], Lat, axis=2)

    # --- TISR (Solar Radiation) ---
    if TISR in forcing_fields:
        # Note: Your TISR code relies on `solar_radiation.get_toa...` which likely takes Xarray.
        # You have two choices:
        # 1. Port that function to numpy (recommended)
        # 2. Wrap the numpy arrays in a dummy DataArray just for this calculation (slower but easier)

        import xarray as xr

        from src.datamodules.utils import solar_radiation

        # Temporary wrapper to satisfy the existing library
        dummy_ds = xr.Dataset(coords={"time": time_vals, "longitude": lon_vals, "latitude": lat_vals})
        # This will be slower than pure numpy but happens only on small coordinate arrays
        tisr_xr = solar_radiation.get_toa_incident_solar_radiation_for_xarray(dummy_ds, use_jit=True)

        # Extract numpy values. Result should be (Time, Lat, Lon) or similar.
        # Ensure it matches (Time, Lon, Lat). TISR usually comes out as (Time, Lat, Lon).
        tisr_np = tisr_xr.values

        # If your data is (Time, Lon, Lat) but TISR is (Time, Lat, Lon), transpose it:
        if tisr_np.shape == (T, Lat, Lon):
            tisr_np = tisr_np.transpose(0, 2, 1)

        results[TISR] = tisr_np

    return results
