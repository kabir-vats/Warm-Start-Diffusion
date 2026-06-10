# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pyformat: mode=pyink
"""Classes for computing derived variables dynamically for evaluation."""
import dataclasses
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import xarray as xr


@dataclasses.dataclass
class DerivedVariable:
    """Derived variable base class.

    Attributes:
      variable_name: Name of variable to compute.
    """

    variable_name: str

    @property
    def base_variables(self) -> List[str]:
        """Return a list of base variables."""
        return []

    def compute(self, dataset: xr.Dataset) -> xr.DataArray:
        """Compute derived variable, returning it in a new DataArray."""
        raise NotImplementedError


@dataclasses.dataclass
class ZonalEnergySpectrum(DerivedVariable):
    """Energy spectrum along the zonal direction.

    Given dataset with longitude dimension, this class computes spectral energy as
    a function of wavenumber (as a dim). wavelength and frequency are also present
    as coords with units "1 / m" and "m" respectively. Only non-negative
    frequencies are included.

    Let f[l], l = 0,..., L - 1, be dataset values along a zonal circle of constant
    latitude, with circumference C (m).  The DFT is
      F[k] = (1 / L) Σₗ f[l] exp(-i2πkl/L)
    The energy spectrum is then set to
      S[0] = C |F[0]|²,
      S[k] = 2 C |F[k]|², k > 0, to account for positive and negative frequencies.

    With C₀ the equatorial circumference, the ith zonal circle has circumference
      C(i) = C₀ Cos(π latitude[i] / 180).
    Since data points occur at longitudes longitude[l], l = 0, ..., L - 1, the DFT
    will measure spectra at zonal sampling frequencies
      f(k, i) = longitude[k] / (C(i) 360), k = 0, ..., L // 2,
    and corresponding wavelengths
      λ(k, i) = 1 / f(k, i).

    This choice of normalization ensures Parseval's relation for energy holds:
    Supposing f[l] are sampled values of f(ℓ), where 0 < ℓ < C (meters) is a
    coordinate on the circle. Then (C / L) is the spacing of longitudinal samples,
    whence
      ∫|f(ℓ)|² dℓ ≈ (C / L) Σₗ |f[l]|² = Σₖ S[k].

    If f has units β, then S has units of m β². For example, if f is
    `u_component_of_wind`, with units (m / s), then S has units (m³ / s²). In
    air with mass density ρ (kg / m³), this gives energy density at wavenumber k
      ρ S[k] ~ (kg / m³) (m³ / s²) = kg / s²,
    which is energy density (per unit area).
    """

    variable_name: str
    lon_name: str = "longitude"
    lat_name: str = "latitude"
    EARTH_RADIUS_M = 1000 * (6357 + 6378) / 2

    @property
    def base_variables(self) -> List[str]:
        return [self.variable_name]

    def _circumference(self, dataset: xr.Dataset) -> xr.DataArray:
        """Earth's circumference as a function of latitude."""
        circum_at_equator = 2 * np.pi * self.EARTH_RADIUS_M
        return np.cos(getattr(dataset, self.lat_name) * np.pi / 180) * circum_at_equator

    def lon_spacing_m(self, dataset: xr.Dataset) -> xr.DataArray:
        """Spacing (meters) between longitudinal values in `dataset`."""
        diffs = getattr(dataset, self.lon_name).diff(self.lon_name)
        if np.max(np.abs(diffs - diffs[0])) > 1e-3:
            raise ValueError(f"Expected uniform longitude spacing. {dataset.longitude.values=}")
        return self._circumference(dataset) * diffs[0].data / 360

    def compute(self, dataset: xr.Dataset) -> xr.DataArray:
        """Computes zonal power at wavenumber and frequency."""
        is_earth = "lon" in self.lon_name
        if is_earth:
            wavenumber_name = "zonal_wavenumber"
        else:
            wavenumber_name = "wavenumber"

        def simple_power(f_x):
            f_k = np.fft.rfft(f_x, axis=-1, norm="forward")
            # freq > 0 should be counted twice in power since it accounts for both
            # positive and negative complex values.
            one_and_many_twos = np.concatenate(([1], [2] * (f_k.shape[-1] - 1)))
            return np.real(f_k * np.conj(f_k)) * one_and_many_twos

        spectrum = xr.apply_ufunc(
            simple_power,
            dataset,
            input_core_dims=[[self.lon_name]],
            output_core_dims=[[self.lon_name]],
            exclude_dims={self.lon_name},
        ).rename_dims({self.lon_name: wavenumber_name})[self.variable_name]
        spectrum = spectrum.assign_coords({wavenumber_name: (wavenumber_name, spectrum[wavenumber_name].data)})
        base_frequency = xr.DataArray(
            np.fft.rfftfreq(len(getattr(dataset, self.lon_name))),
            dims=wavenumber_name,
            coords={wavenumber_name: spectrum[wavenumber_name]},
        )
        if is_earth:
            spacing = self.lon_spacing_m(dataset)
            spectrum = spectrum.assign_coords(frequency=base_frequency / spacing)
            spectrum["frequency"] = spectrum.frequency.assign_attrs(units="1 / m")

            spectrum = spectrum.assign_coords(wavelength=1 / spectrum.frequency)
            spectrum["wavelength"] = spectrum.wavelength.assign_attrs(units="m")

            # This last step ensures the sum of spectral components is equal to the
            # (discrete) integral of data around a line of latitude.
            spectrum = spectrum * self._circumference(spectrum)
        else:
            spectrum = spectrum.assign_coords(frequency=base_frequency)
            spectrum = spectrum.assign_coords(wavelength=1 / spectrum.frequency)
        return spectrum


def interpolate_spectral_frequencies(
    spectrum: xr.DataArray,
    wavenumber_dim: str,
    frequencies: Optional[Sequence[float]] = None,
    method: str = "linear",
    **interp_kwargs: Optional[Dict[str, Any]],
) -> xr.DataArray:
    """Interpolate frequencies in `spectrum` to common values.

    Args:
      spectrum: Data as produced by ZonalEnergySpectrum.compute.
      wavenumber_dim: Dimension that indexes wavenumber, e.g. 'zonal_wavenumber'
        if `spectrum` is produced by ZonalEnergySpectrum.
      frequencies: Optional 1-D sequence of frequencies to interpolate to. By
        default, use the most narrow range of frequencies in `spectrum`.
      method: Interpolation method passed on to DataArray.interp.
      **interp_kwargs: Additional kwargs passed on to DataArray.interp.

    Returns:
      New DataArray with dimension "frequency" replacing the "wavenumber" dim in
        `spectrum`.
    """

    if set(spectrum.frequency.dims) != set((wavenumber_dim, "latitude")):
        raise ValueError(f"{spectrum.frequency.dims=} was not a permutation of " f'("{wavenumber_dim}", "latitude")')

    if frequencies is None:
        freq_min = spectrum.frequency.max("latitude").min(wavenumber_dim).data
        freq_max = spectrum.frequency.min("latitude").max(wavenumber_dim).data
        frequencies = np.linspace(freq_min, freq_max, num=spectrum.sizes[wavenumber_dim])
    frequencies = np.asarray(frequencies)
    if frequencies.ndim != 1:
        raise ValueError(f"Expected 1-D frequencies, found {frequencies.shape=}")

    def interp_at_one_lat(da: xr.DataArray) -> xr.DataArray:
        da = (
            da.swap_dims({wavenumber_dim: "frequency"})  # pytype: disable=wrong-arg-types
            .drop_vars(wavenumber_dim)
            .interp(frequency=frequencies, method=method, **interp_kwargs)
        )
        # Interp didn't deal well with the infinite wavelength, so just reset λ as..
        da["wavelength"] = 1 / da.frequency
        da["wavelength"] = da["wavelength"].assign_attrs(units="m")
        return da

    return spectrum.groupby("latitude").apply(interp_at_one_lat)


def _output_dims(source: xr.Dataset, include_averaging_dims: bool, lon_name: str = "longitude") -> List[str]:
    """Dimensions in the output, in canonical order."""
    assert include_averaging_dims, "Not implemented"
    dims = []
    for d in source.dims:
        if d == lon_name:
            dims.append("zonal_wavenumber")
        elif include_averaging_dims:  # or d not in AVERAGING_DIMS.value:
            dims.append(d)
    return dims


def _make_derived_variables_ds(
    source: xr.Dataset,
    derived_variables: Sequence[ZonalEnergySpectrum],
) -> xr.Dataset:
    """Dataset with power spectrum for BASE_VARIABLES before averaging."""
    arrays = []
    for dv in derived_variables:
        arrays.append({dv.variable_name: dv.compute(source[dv.base_variables])})
    return xr.merge(arrays).transpose(*_output_dims(source, include_averaging_dims=True))
