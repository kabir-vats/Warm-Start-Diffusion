import os
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
import zarr
from tensordict import TensorDictBase

from src.evaluation.aggregators._abstract_aggregator import AbstractAggregator
from src.utilities.utils import get_logger, rrearrange, to_tensordict, torch_to_numpy


log = get_logger(__name__)


class MemEfficientSaveToDiskAggregator(AbstractAggregator):
    """
    Memory-efficient aggregator for saving data to zarr format.
    Saves data to zarr as soon as concat_dim_key repeats to avoid accumulating data in memory.
    Only supports zarr format for efficient incremental saving and appending.
    The saved file location will be stored in wandb.summary['predictions_outputs_filepath']
    Open it with xr.open_zarr(filepath) to load the data.
    If you get weird AttributeErrors, try upgrading xarray and zarr (pip install --upgrade xarray zarr).
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
        self._metadatas = []
        self._data_coords = coords
        self.concat_dim_name = concat_dim_name
        self.batch_dim_name = batch_dim_name
        self.max_ensemble_members = max_ensemble_members
        self.save_to_path = save_to_path
        self.save_to_wandb = save_to_wandb
        self.dims = None

        # Initialize dictionaries to store data per concat_dim_key
        self._current_data = {}  # Stores current batch data for each concat_dim_key
        self._concat_counts = {}  # Tracks counts for each concat_dim_key (and also serves as the list of concat keys)
        self._last_concat_key = None  # Track the last concat key seen to detect cycles
        self._zarr_initialized = False
        self._batch_coords_list = []  # Store all batch coordinates
        self._current_batch_coords = []  # Store coordinates for the current batch only
        self._batch_count = 0  # Track the current batch number

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
                    data = new_data
                else:
                    raise ValueError(f"Unsupported data type {type(data)}")
                # Reshaping from w,b to (b,w) reorders the data correctly (only necessary when batchsize>1)
                data = rrearrange(data, "w b ... -> (b w) ...")
        return data

    def _initialize_zarr_store(self, data, concat_dim_key=None, prefix="", epoch=None):
        """Initialize zarr store with the first batch of data"""
        if self.save_to_path is None:
            return

        save_to_path = self.save_to_path
        if not save_to_path.endswith(".zarr"):
            log.warning(f"Path {save_to_path} does not end with .zarr extension. Adding .zarr extension.")
            save_to_path += ".zarr"

        # Add epoch to path if not already there
        if epoch is not None and f"epoch{epoch}" not in save_to_path:
            save_to_path = save_to_path.rstrip(".zarr")  # Remove existing extensions
            save_to_path += f"{prefix}-epoch{epoch}-results.zarr"

        self.save_to_path = save_to_path
        log.info(f"Initializing data store at {save_to_path}")

        # Create zarr store
        if os.path.exists(save_to_path):
            log.info(f"Removing existing zarr store at {save_to_path}")
            import shutil

            shutil.rmtree(save_to_path)

        self._zarr_initialized = True

    def _process_batch_coordinates(self, metadata):
        """Process batch coordinates from metadata"""
        if self.batch_dim_name not in metadata:
            return

        # Clear current batch coordinates when we get new metadata
        self._current_batch_coords = []

        v = metadata.pop(self.batch_dim_name)

        # Process based on the type of the batch coordinate
        coords = []
        if isinstance(v, np.ndarray) or self.batch_dim_name == "datetime":
            for vi in v:
                vi = np.datetime64(int(vi.item()), "s").astype("datetime64[h]")
                coords.append(vi)
        # Check if datetime was converted to .astype('datetime64[s]').astype('int64') => convert back
        elif isinstance(v, np.int64):
            v = int(v)  # For some reason, the below throws an error if v is a numpy int64
            v_dt = np.datetime64(v, "s")
            v_dt = v_dt.astype("datetime64[h]")
            coords.append(v_dt)
        elif torch.is_tensor(v):
            coords.append(v.cpu().item())  # Assuming scalar tensor
        elif isinstance(v, list):
            coords.extend(v)  # if v is already a list of items
        else:
            coords.append(v)

        # Update both coordinate lists
        self._batch_coords_list.extend(coords)
        self._current_batch_coords.extend(coords)
        log.info(f"Added {len(coords)} batch coordinates. Total now: {len(self._batch_coords_list)}")

    def _stack_and_save_concat_data(self):
        """
        Stack all data in _current_data along concat dimension and save to zarr.

        Returns:
            None
        """
        if not self._current_data or not self._concat_counts:
            return

        # Stack all data along a new concat dimension
        concat_dim = self.concat_dim_name or "concat_dim"

        # Stack in key order to ensure consistent ordering
        ordered_keys = sorted(self._concat_counts.keys())
        log.info(f"Stacking data for keys: {ordered_keys}") if self._batch_count == 0 else None
        stacked_data = torch.stack([self._current_data[k] for k in ordered_keys], dim=1)

        # Save as a single dataset with concat dimension
        concat_coord = {concat_dim: ordered_keys}

        # Let _save_batch_to_zarr handle the append logic
        self._save_batch_to_zarr(stacked_data, None, concat_coord=concat_coord)

        # Clear all data after saving
        self._current_data = dict()

    def _save_batch_to_zarr(self, data, concat_dim_key=None, append=None, concat_coord=None):
        """Save a batch of data to zarr store

        Args:
            data: The data to save
            concat_dim_key: If provided, uses this as the single concat dimension key
            append: Whether to append to an existing store. If None, determined by batch_count
            concat_coord: Optional dict with concat dimension name and values
        """
        if self.save_to_path is None or not self._zarr_initialized:
            return

        # Determine append mode based on batch_count if not explicitly specified
        if append is None:
            append = self._batch_count > 0

        # Convert tensordict to xarray dataset
        if concat_coord is not None:
            # Use the provided concat coord (for stacked data)
            ds = self._tensordict_to_dataset(data, concat_coord)
        elif concat_dim_key is not None:
            # Create a concat coord from the single key
            single_key_coord = {self.concat_dim_name or "concat_dim": [concat_dim_key]}
            ds = self._tensordict_to_dataset(data, single_key_coord)
        else:
            # No concat dimension
            ds = self._tensordict_to_dataset(data)

        # Set up chunking
        chunks = {}
        for dim in ds.dims:
            # All other dimensions except for batch are chunked to full size
            if dim in [self.batch_dim_name, self.concat_dim_name]:
                chunks[dim] = min(4, ds.sizes[dim])
            else:
                chunks[dim] = ds.sizes[dim]
        ds = ds.chunk(chunks)

        # Save to zarr with appropriate mode
        mode = "a" if append else "w"
        append_dim = self.batch_dim_name if append else None

        # Only support zarr format
        log.info(f"Saving data with {mode=}, {ds.sizes=}")
        # Remove _FillValue attribute if it exists, from each coord to avoid a weird zarr ValueError when saving
        for var in ds.coords:
            if "_FillValue" in ds[var].attrs:
                if self._batch_count == 0:
                    log.warning(f"Removing {ds[var].attrs['_FillValue']=} attribute from coordinate {var}")
                del ds[var].attrs["_FillValue"]
        ds.drop_encoding().to_zarr(self.save_to_path, mode=mode, append_dim=append_dim, zarr_format=3)

        return ds

    @torch.inference_mode()
    def _record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor] = None,
        gen_data_norm: Mapping[str, torch.Tensor] = None,
        concat_dim_key: str = None,
        metadata: Mapping[str, Any] = None,
        **kwargs,
    ):
        # Update concat counts (and track keys)
        if concat_dim_key is not None:
            if concat_dim_key not in self._concat_counts:
                self._concat_counts[concat_dim_key] = 1
            else:
                self._concat_counts[concat_dim_key] += 1
        first_concat_key = next(iter(self._concat_counts.keys())) if self._concat_counts else None
        # Detect cycle in concat keys (b0c0, b0c1, b0c2, b1c0, b1c1, b1c2...)
        # A cycle is when we return to the first concat key in the sequence
        cycle_detected = (
            concat_dim_key == first_concat_key
            and self._last_concat_key is not None
            and self._last_concat_key != first_concat_key
        )

        # If we detect a cycle, save all accumulated data from the previous batch
        if cycle_detected and self.rank == 0 and self._current_data:
            log.info(f"Cycle detected at concat_key={concat_dim_key}, saving accumulated data ({self._batch_count=})")
            # Stack and save data from the current batch
            self._stack_and_save_concat_data()
            self._batch_count += 1
            self._current_batch_coords = []

        # Track this key for cycle detection
        self._last_concat_key = concat_dim_key

        # Process metadata
        if metadata is not None and (concat_dim_key is None or concat_dim_key == first_concat_key):
            if self.world_size > 1:
                metadata = self.all_gather(metadata)
            if self.rank == 0:
                metadata_copied = metadata.copy()  # Copy before processing
                self._metadatas.append(torch_to_numpy(metadata))
                self._process_batch_coordinates(metadata_copied)

        # Process ensemble data
        if self._is_ensemble:
            if self.max_ensemble_members is not None:
                gen_data = gen_data[: self.max_ensemble_members, ...]
            # Re-arrange ensemble dim (e, b, h, w) from the front to (b, e, h, w)
            gen_data = rrearrange(gen_data, "e b ... -> b e ...")
            for k in list(kwargs.keys()):
                if "preds" in k:
                    v = kwargs.pop(k)
                    if self.max_ensemble_members is not None:
                        v = v[: self.max_ensemble_members, ...]
                    v = rrearrange(v, "e b ... -> b e ...")
                    if hasattr(v, "keys"):  # TensorDictBase
                        # Flatten nested dict
                        for vk, vv in v.items():
                            kwargs[f"{k}_{vk}"] = vv
                    else:
                        kwargs[k] = v

        # Handle distributed training
        if self.world_size > 1:
            target_data = self.all_gather(target_data)
            gen_data = self.all_gather(gen_data)
            if self.rank != 0:
                # Only rank 0 should save data
                return

        # Prepare data dictionary
        if torch.is_tensor(target_data):  # add dummy key
            data = {"targets": target_data, "preds": gen_data, **kwargs}
            batch_size = target_data.shape[0]
        else:
            data = {
                **{f"{k}_targets": v for k, v in target_data.items()},
                **{f"{k}_preds": v for k, v in gen_data.items()},
                **kwargs,
            }
            batch_size = target_data[list(target_data.keys())[0]].shape[0]

        # Move data to CPU
        data = to_tensordict(data, device="cpu", batch_size=[batch_size]).to("cpu")

        # Initialize zarr store if not already done
        if not self._zarr_initialized:
            self._initialize_zarr_store(data, concat_dim_key)

        # Store this batch's data
        if concat_dim_key is None:
            # No concat dimension case, let save_batch determine append mode
            self._save_batch_to_zarr(data, None)
        else:
            assert concat_dim_key not in self._current_data, f"Duplicate concat_dim_key {concat_dim_key} detected"
            # For concat dimension case, store the current data
            # It will be saved when the cycle completes or in _get_logs
            self._current_data[concat_dim_key] = data

    @torch.inference_mode()
    def _get_logs(self, prefix: str = "", epoch: Optional[int] = None, metadata=None) -> Dict[str, float]:
        """Finalize zarr dataset and add metadata"""
        if self.rank != 0 or not self._zarr_initialized:
            return {}, {}, {}

        # Save any remaining data that hasn't been saved yet
        if self._current_data and all(data is not None for data in self._current_data.values()):
            self._stack_and_save_concat_data()

        # Add attributes (metadata) to zarr store
        try:
            store = zarr.open(self.save_to_path, mode="a")
            # Add prefix/epoch metadata
            store.attrs["label"] = prefix
            if epoch is not None:
                store.attrs["epoch"] = epoch
            store.attrs["batch_count"] = self._batch_count

            # Add other metadata from self._metadatas
            if self._metadatas:
                for key, value in self._metadatas[0].items():
                    if key == "ssp":  # TEMPORARY: Don't add 'ssp' metadata
                        continue
                    if isinstance(value, (np.ndarray, dict)):
                        log.info(f"Adding {type(value)} to metadata is not supported. Skipping {key}")
                        continue
                    store.attrs[key] = [m[key] for m in self._metadatas if key in m and m[key] is not None]

            # Add additional metadata passed to _get_logs
            if metadata:
                for key, value in metadata.items():
                    store.attrs[key] = value

            zarr.consolidate_metadata(store)
        except Exception as e:
            log.error(f"Error adding metadata to zarr store: {e}")

        # Save to wandb if requested
        if self.save_to_wandb:
            import wandb

            wandb.save(self.save_to_path)

        # Reset for next use
        self._current_data = {}
        self._concat_counts = {}
        self._batch_coords_list = []  # Reset full history
        self._current_batch_coords = []  # Reset current batch
        self._metadatas = []
        self._zarr_initialized = False
        self._last_concat_key = None
        self._batch_count = 0

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

        # Add batch coordinate if available
        if self.batch_dim_name not in coords:
            # First try to use current batch coordinates if available
            coords[self.batch_dim_name] = np.array(self._current_batch_coords)
            log.info(f"Using current batch coordinates ({self._current_batch_coords}) for {self.batch_dim_name}")

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

            # Handle extra dimensions (diagnostic data that doesn't end with 'preds' or 'targets')
            is_diagnostic = not (name.endswith("preds") or name.endswith("targets"))
            expected_ndim = len(dims_here)
            actual_ndim = len(tensor_np.shape)

            if is_diagnostic and actual_ndim > expected_ndim:
                # Insert extra dimensions after base dims but before final_dims_of_data
                n_extra = actual_ndim - expected_ndim
                extra_dim_names = [f"dim_{i}" for i in range(n_extra)]
                # Insert after batch/concat dims but before final dims
                insert_pos = len(dims_here) - len(self.final_dims_of_data)
                dims_here = dims_here[:insert_pos] + extra_dim_names + dims_here[insert_pos:]

            # Ensure dims match tensor shape
            assert len(tensor_np.shape) == len(
                dims_here
            ), f"Shape mismatch for {name}: {tensor_np.shape=} vs {dims_here=}"
            data_vars[name] = (dims_here, tensor_np)

        # Create dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        return ds
