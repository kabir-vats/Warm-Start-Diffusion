from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from tensordict import TensorDictBase

from src.evaluation.aggregators._abstract_aggregator import AbstractAggregator
from src.utilities.utils import get_logger, rrearrange, to_tensordict, torch_to_numpy


log = get_logger(__name__)


class SaveToDiskAggregator(AbstractAggregator):
    """
    Aggregator for spectra metrics.
    """

    def __init__(
        self,
        final_dims_of_data: List[str],  # e.g. ["channel", "latitude", "longitude"], or ["latitude", "longitude"]
        var_names: Optional[List[str]] = None,
        coords: Optional[Dict[str, np.ndarray]] = None,  # Xarray coordinates
        concat_dim_name: Optional[str] = None,
        batch_dim_name: Optional[str] = "batch",
        max_ensemble_members: Optional[int] = 5,  # Number of ensemble members to save (if applicable)
        save_to_path: Optional[str] = None,
        save_to_wandb: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.var_names = var_names
        self.final_dims_of_data = final_dims_of_data
        self._running_data = None
        self._metadatas = []
        self._concat_keys = []
        self._data_coords = coords
        self.concat_dim_name = concat_dim_name
        self.batch_dim_name = batch_dim_name
        self.max_ensemble_members = max_ensemble_members
        self.save_to_path = save_to_path
        self.save_to_wandb = save_to_wandb
        self.dims = None
        if coords is not None:
            for k in coords.keys():
                assert k in final_dims_of_data, f"coord {k} must be in final_dims_of_data ({final_dims_of_data=})"

    @property
    def rank(self):
        if self.lit_module_handle is not None:
            return self.lit_module_handle.trainer.global_rank
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    @property
    def world_size(self):
        if self.lit_module_handle is not None:
            return self.lit_module_handle.trainer.world_size
        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def all_gather(self, data: Mapping[str, torch.Tensor]):
        if self.world_size > 1:
            if self.lit_module_handle is None:
                raise ValueError("lit_module_handle is not set. Cannot gather data.")
            else:
                if torch.is_tensor(data):
                    data = self.lit_module_handle.all_gather(data)
                elif isinstance(data, (dict, TensorDictBase)):
                    new_data = {}
                    # Iterate over keys in SAME order regardless of rank (VERY, VERY IMPORTANT!!)
                    for k in sorted(data.keys()):
                        v = data[k]
                        if torch.is_tensor(v):
                            new_data[k] = self.lit_module_handle.all_gather(v)
                        else:
                            raise ValueError(f"Unsupported data type {type(v)}")
                        # if "10m" in k:
                        #     assert v.max() < 140, f"Wind data {k} has values > 100: {v.max()}"
                        # log.info(f"Min, max wind data1 {k}: {v.min().item()}, {v.max().item()}")
                        # log.info(f"Min, max wind data2 {k}: {new_data[k].min().item()}, {new_data[k].max().item()}")
                    data = new_data
                else:
                    raise ValueError(f"Unsupported data type {type(data)}")
                # Reshaping from w,b to (b,w) reorders the data correctly (only necessary when batchsize>1)
                data = rrearrange(data, "w b ... -> (b w) ...")
        return data

    @torch.inference_mode()
    def _record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor] = None,
        gen_data_norm: Mapping[str, torch.Tensor] = None,
        concat_dim_key: str = None,
        metadata: Mapping[str, Any] = None,
    ):
        batch_dim = 0
        if concat_dim_key is not None and concat_dim_key not in self._concat_keys:
            self._concat_keys.append(concat_dim_key)

        if metadata is not None and (concat_dim_key is None or concat_dim_key == self._concat_keys[0]):
            # Only insert metadata if it's for the first concat_dim_key (this assumes that metadata is the same for all concat_dim_keys!)
            if self.world_size > 1:
                # Gather metadata from all ranks
                metadata = self.all_gather(metadata)
                # self.log_text.info(f"[rank: {self.rank}] Metadata: {metadata}")
                # Metadata: {'datetime': tensor([1609459200, 1617753600, 1611532800, 1619827200], device='cuda:0')}
            if self.rank == 0:
                self._metadatas.append(torch_to_numpy(metadata))

        if self._is_ensemble:
            if self.max_ensemble_members is not None:
                gen_data = gen_data[: self.max_ensemble_members, ...]
            # Re-arrange ensemble dim (e, b, h, w) from the front to (b, e, h, w)
            gen_data = rrearrange(gen_data, "e b ... -> b e ...")
        # batch_size_orig = gen_data.shape[0]

        if self.world_size > 1:
            # print(f"[{self.rank=}] before gather: {gen_data['2m_temperature'].shape=}, {self.world_size=}")
            target_data = self.all_gather(target_data)
            gen_data = self.all_gather(gen_data)
            if self.rank != 0:
                # Only rank 0 should save data
                return

        if torch.is_tensor(target_data):  # add dummy key
            data = {"targets": target_data, "preds": gen_data}
            batch_size = target_data.shape[0]
        else:
            data = {
                **{f"{k}_targets": v for k, v in target_data.items()},
                **{f"{k}_preds": v for k, v in gen_data.items()},
            }
            batch_size = target_data[list(target_data.keys())[0]].shape[0]

        data = to_tensordict(data, device="cpu", batch_size=[batch_size]).to("cpu")
        if concat_dim_key is None:
            if self._running_data is None:
                self._running_data = data
            else:
                # Simply concatenate the data along the batch dimension
                self._running_data = torch.cat([self._running_data, data], dim=batch_dim)
        else:
            # E.g. concat_dim_key = "t1", "t4", "t8" etc.
            if self._running_data is None:
                self._running_data = dict()
            if concat_dim_key not in self._running_data.keys():
                # Initialize the running data with the new data
                self._running_data[concat_dim_key] = data
            else:
                # Concatenate the data into specific dimension for values with the same concat_dim_key
                self._running_data[concat_dim_key] = torch.cat(
                    [self._running_data[concat_dim_key], data], dim=batch_dim
                )
            # if "2m_temperature" in gen_data.keys():
            #     self.log_text.info(
            #         f"[rank: {self.rank}] {batch_size_orig=}, {batch_size=}, {batch_size_orig*self.world_size=} "
            #         f"{self._running_data[concat_dim_key].shape=}"
            #         f"\n{gen_data['2m_temperature'].shape=}"
            #         f"{self._running_data[concat_dim_key]['2m_temperature_preds'].shape=}"
            #         f""
            #     )

    @torch.inference_mode()
    def _get_logs(self, prefix: str = "", epoch: Optional[int] = None, metadata=None) -> Dict[str, float]:
        """Converts running data to xarray dataset."""
        if self._running_data is None or self.rank != 0:
            self.log_text.info(f"[rank: {self.rank}] No data to log.")
            return {}, {}, {}
        log.info(f"Saving data to {self.save_to_path} with prefix {prefix} and epoch {epoch}...")
        metadata = metadata or {}
        metadata["label"] = prefix
        if epoch is not None:
            metadata["epoch"] = epoch  # Add epoch information if provided

        if self._metadatas:  # Process metadata only on rank 0
            if self.batch_dim_name in self._metadatas[0]:
                batch_coords_list = []
                for m in self._metadatas:
                    # {'datetime': tensor([1609459200, 1617753600, 1611532800, 1619827200], device='cuda:0')}
                    # {'datetime': tensor([1626048000, 1634342400, 1628121600, 1636416000], device='cuda:0')}
                    # print(f"Metadata: {m}, {self._metadatas=}")
                    # Metadata: {'datetime': array([1609459200, 1617753600, 1611532800, 1619827200])},
                    # self._metadatas=[{}, {}, {}, {}, {}, {}, {},
                    # {'datetime': array([1609459200, 1617753600, 1611532800, 1619827200])},
                    # {'datetime': array([1626048000, 1634342400, 1628121600, 1636416000])},
                    # {'datetime': array([1626048000, 1634342400, 1628121600, 1636416000])},
                    # {'datetime': array([1626048000, 1634342400, 1628121600, 1636416000])},
                    # {'datetime': array([1626048000, 1634342400, 1628121600, 1636416000])},
                    # {'datetime': array([1626048000, 1634342400, 1628121600, 1636416000])},
                    # {'datetime': array([1626048000, 1634342400, 1628121600, 1636416000])},
                    # {'datetime': array([1626048000, 1634342400, 1628121600, 1636416000])},
                    # {'datetime': array([1626048000, 1634342400, 1628121600, 1636416000])}]
                    v = m.pop(self.batch_dim_name)
                    if isinstance(v, np.ndarray) or self.batch_dim_name == "datetime":
                        for vi in v:
                            vi = np.datetime64(int(vi.item()), "s").astype("datetime64[h]")
                            batch_coords_list.append(vi)
                    # Check if datetime was converted to .astype('datetime64[s]').astype('int64') => convert back
                    elif isinstance(v, np.int64):
                        v = int(v)  # For some reason, the below throws an error if v is a numpy int64
                        v_dt = np.datetime64(v, "s")
                        v_dt = v_dt.astype("datetime64[h]")
                        batch_coords_list.append(v_dt)
                    elif torch.is_tensor(v):
                        batch_coords_list.append(v.cpu().item())  # Assuming scalar tensor
                    elif isinstance(v, list):
                        batch_coords_list.extend(v)  # if v is already a list of items
                    else:
                        batch_coords_list.append(v)
                self.log_text.info(f"Batch coords list: {batch_coords_list}")
                # [numpy.datetime64('2021-01-01T00','h'), numpy.datetime64('2021-04-07T00','h'),
                # numpy.datetime64('2021-01-25T00','h'), numpy.datetime64('2021-05-01T00','h'),
                # numpy.datetime64('2021-07-12T00','h'), numpy.datetime64('2021-10-16T00','h'),
                # numpy.datetime64('2021-08-05T00','h'), numpy.datetime64('2021-11-09T00','h')]
                self._data_coords[self.batch_dim_name] = np.array(batch_coords_list)

        # Handle case where data is stored with concat dimensions
        if isinstance(self._running_data, dict):  # Assuming dict of tensors (concat_dim case)
            # First concatenate along the concat dimension
            concat_dim = self.concat_dim_name or "concat_dim"
            data = torch.stack(list(self._running_data.values()), dim=1)
            final_ds = self._tensordict_to_dataset(data, {concat_dim: list(self._running_data.keys())})

        else:
            # Direct conversion for data without concat dimensions
            final_ds = self._tensordict_to_dataset(self._running_data)

        log.info(f"Final dataset shape: {final_ds.sizes}")
        # Add metadata if available
        if self._metadatas:
            for key, value in self._metadatas[0].items():
                # TEMPORARY: Dont add 'ssp' metadata
                if key == "ssp":
                    continue

                if isinstance(value, (np.ndarray, dict)):
                    log.info(f"Adding {type(value)} to metadata is not supported. Skipping {key}")
                    continue

                final_ds.attrs[key] = [m[key] for m in self._metadatas if m[key] is not None]
                log.info(f"Added {key} to metadata: {final_ds.attrs[key]}")

        for key, value in metadata.items():
            final_ds.attrs[key] = value

        # Save to file if path is provided
        log.info(f"Done! Saving results to {self.save_to_path}")
        # predictions/6h-1AR_Attn23_ADM_EMA_256x1-2-3-4d_WMSE_54lr_LC5:200_15wd_fLV_11seed_19h03mOct18_3423514-5214396-hor30-TAG-ENS=5-max_val_samples=1-val_slice=20210329_20210430-possible_initial_times=12-prediction_horizon=30-TAG-epoch199.nc
        if self.save_to_path.endswith(".nc"):
            final_ds.to_netcdf(self.save_to_path)
        elif self.save_to_path.endswith(".zarr"):
            chunks = {}
            for dim in final_ds.dims:
                # All other dimensions except for batch are chunked to full size
                chunks[dim] = final_ds.sizes[dim] if dim not in [self.batch_dim_name, self.concat_dim_name] else 4

            final_ds.chunk(chunks).to_zarr(self.save_to_path, consolidated=True, mode="w", zarr_format=2)
        else:
            raise ValueError(f"Unsupported file format: {self.save_to_path}. Supported formats are .zarr and .nc")
        if self.save_to_wandb:
            import wandb

            wandb.save(self.save_to_path)

        # Reset running data
        self._running_data = None
        self._metadatas = []

        return {}, {}, {}

    def _tensordict_to_dataset(
        self, tensordict: TensorDictBase, concat_coord: Optional[Dict[str, Any]] = None
    ) -> xr.Dataset:
        """
        Convert a tensor dictionary to xarray dataset with proper coordinates.

        Args:
            tensordict: Dictionary of tensors or TensorDictBase
            concat_coord: Optional coordinate to add for concat dimension

        Returns:
            xarray Dataset with proper coordinates
        """
        coords = {}
        dims = [self.batch_dim_name]

        # Set up basic coordinates if provided
        if self._data_coords is not None:
            coords.update(self._data_coords)

        # Add concat coordinate if provided
        if concat_coord is not None:
            assert len(concat_coord) == 1, "Only one concat dimension is supported."
            coords.update(concat_coord)
            dims.extend(list(concat_coord.keys()))

        data_vars = {}
        # Convert each tensor to data variable
        for name, tensor in tensordict.items():
            dims_here = dims.copy()
            # Move tensor to CPU and convert to numpy
            tensor_np = torch_to_numpy(tensor)
            # Determine dimensions based on tensor shape
            if self._is_ensemble and "preds" in name:
                dims_here.append("ensemble")
            dims_here.extend(self.final_dims_of_data)

            # Ensure dims match tensor shape
            assert len(tensor_np.shape) == len(dims_here), f"{tensor_np.shape=} does not match dims {dims_here}"
            # dims = dims[:len(tensor_np.shape)]
            data_vars[name] = (dims_here, tensor_np)

        # Create dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        return ds
