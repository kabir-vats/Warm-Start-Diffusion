from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import xarray as xr

from src.evaluation.torchmetrics import Metric
from src.utilities.spectra import ZonalEnergySpectrum
from src.utilities.utils import extract_xarray_metadata, get_logger, reconstruct_xarray, torch_to_numpy


log = get_logger(__name__)


class SpectraAggregator(Metric):
    """
    Aggregator for spectra metrics.
    """

    def __init__(
        self,
        is_ensemble: bool,
        spectra_type: str = "zonal_60_90",  # "zonal" or "meridional" or "basic" (for non-earth data)
        var_names: Optional[List[str]] = None,
        coords: Optional[Dict[str, np.ndarray]] = None,
        spatial_dims: Sequence[str] = None,  # Need to be in same order as in data
        data_to_log: str = "preds",
        preprocess_fn: Optional[callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert data_to_log in ["preds", "targets"]
        assert spatial_dims is not None, "spatial_dims must be provided to spectra aggregator."
        self.name = "spectra"
        self.is_ensemble = is_ensemble
        self.var_names = var_names
        self.data_to_log = data_to_log
        self._data_coords = coords
        self.preprocess_fn = preprocess_fn
        self.dims = ("batch",) + tuple(spatial_dims)
        if self.is_ensemble and data_to_log == "preds":
            self.dims = ("ensemble",) + self.dims
        # Infer names of longitude and latitude dimensions (if exist)
        self.latitude_dim = self.longitude_dim = None
        for dim in self.dims:
            if dim.startswith("lat"):
                self.latitude_dim = dim
            elif dim.startswith("lon"):
                self.longitude_dim = dim
        if "x" in self.dims and self.longitude_dim is None:
            self.longitude_dim = "x"

        if self.longitude_dim is None:
            log.info(f"Power spectra will be computed over last dimension {spatial_dims[-1]}.")
            self.longitude_dim = spatial_dims[-1]
        if self.latitude_dim is None:
            self.latitude_dim = [d for d in spatial_dims if d != self.longitude_dim][0]

        self.mean_dims = [d for d in self.dims if d not in ["batch", self.longitude_dim]]
        self._subsel_spectra = None
        if "zonal" in spectra_type or "meridional" in spectra_type:
            assert "lat" in self.latitude_dim, f"Latitude dimension not found in {self.latitude_dim=}"
            assert "lon" in self.longitude_dim, f"Longitude dimension not found in {self.longitude_dim=}"
            _, subsel_spectra_l, subsel_spectra_r = spectra_type.split("_")
            dim_subset = self.latitude_dim if "zonal" in spectra_type else self.longitude_dim
            self._subsel_spectra = {dim_subset: slice(int(subsel_spectra_l), int(subsel_spectra_r))}

        self._spectra_vars_to_xr_metadata = dict()
        self.add_state("_n_batches", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update_running_spectra(self, spectra: xr.DataArray, var_name: str):
        to_add_spectrum = spectra.sum(dim="batch")
        to_add_spectrum_tensor = torch.tensor(to_add_spectrum.values, device=self.device)
        if var_name not in self._spectra_vars_to_xr_metadata.keys():
            xr_metadata = extract_xarray_metadata(to_add_spectrum)
            self._spectra_vars_to_xr_metadata[var_name] = xr_metadata
            # Add state
            self.add_state(
                f"_running_spectra_{var_name}",
                default=torch.zeros_like(to_add_spectrum_tensor, device=self.device),
                dist_reduce_fx="sum",
            )
        # Update state
        self.__dict__[f"_running_spectra_{var_name}"] += to_add_spectrum_tensor

    def get_aggregated_spectrum(self, var_name: str) -> xr.DataArray:
        # Compute mean spectrum, should be called from compute() to get a proper DDP reduction
        running_spectrum = self.__dict__[f"_running_spectra_{var_name}"]
        running_spectrum = running_spectrum / self._n_batches
        running_spectrum = reconstruct_xarray(running_spectrum, self._spectra_vars_to_xr_metadata[var_name])
        return running_spectrum

    @torch.inference_mode()
    def update(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor] = None,
        gen_data_norm: Mapping[str, torch.Tensor] = None,
        metadata: Mapping[str, Any] = None,
    ):
        if self.data_to_log == "preds":
            data = gen_data_norm
        elif self.data_to_log == "targets":
            data = target_data_norm
        else:
            raise ValueError(f"Unknown data_to_log: {self.data_to_log}")
        data = self.preprocess_fn(data) if self.preprocess_fn is not None else data

        if torch.is_tensor(data):  # add dummy key
            raise ValueError("data must be a dict, not a tensor for spectra aggregator")
        data = torch_to_numpy(data)

        names = self.var_names if self.var_names is not None else data.keys()
        for i, name in enumerate(names):
            spectra_compute_class = ZonalEnergySpectrum(
                variable_name=name, lon_name=self.longitude_dim, lat_name=self.latitude_dim
            )
            # Map gen_data to xarray
            data_xr = xr.Dataset({name: xr.DataArray(data[name], dims=self.dims, coords=self._data_coords)})

            # Compute spectra
            spectra = spectra_compute_class.compute(data_xr.load())
            if self._subsel_spectra is not None:
                spectra = spectra.sel(**self._subsel_spectra)

            self.update_running_spectra(spectra, name)

        self._n_batches += torch.tensor(spectra.sizes["batch"])

    @torch.inference_mode()
    def compute(self, prefix: str = "", epoch: Optional[int] = None):
        """
        Returns logs as can be reported to WandB.

        Args:
            prefix: Label to prepend to all log keys.
            epoch: Current epoch number.
        """
        prefix = prefix + "/" if prefix else ""
        wavenumber_dim = "zonal_wavenumber" if "lon" in self.longitude_dim else "wavenumber"

        logs = dict()
        for name in self._spectra_vars_to_xr_metadata.keys():
            log_key = f"{prefix}{name}/spectrum".rstrip("/")
            spectra_v = self.get_aggregated_spectrum(name)
            if wavenumber_dim == "zonal_wavenumber":
                wavelength_mean = spectra_v.wavelength.mean(dim=self.latitude_dim).values
            else:
                wavelength_mean = spectra_v.wavelength.values
            mean_spectra = spectra_v.mean(dim=self.mean_dims)
            # Log the mean spectra for each wavelength and wavenumber separately (so that we can plot them as x-axis)
            for i, wavenumber in enumerate(spectra_v[wavenumber_dim]):
                wavenumber_float = float(wavenumber)
                if wavenumber_float not in logs.keys():
                    logs[wavenumber_float] = dict()
                    logs[wavenumber_float]["wavelength"] = float(wavelength_mean[i])
                    logs[wavenumber_float]["wavenumber"] = wavenumber_float
                logs[wavenumber_float][log_key] = float(mean_spectra.sel({wavenumber_dim: wavenumber}).values)
        assert len(logs) > 0, f"No spectra to log. {self._spectra_vars_to_xr_metadata=}"
        logs["x_axes"] = {
            "wavenumber": spectra_v[wavenumber_dim].values,
            "wavelength": wavelength_mean,
        }
        return {}, {}, {"wavenumber": logs}
