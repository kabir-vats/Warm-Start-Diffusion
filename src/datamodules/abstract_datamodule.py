from __future__ import annotations

import multiprocessing
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.evaluation.aggregators._abstract_aggregator import _Aggregator
from src.utilities.utils import get_logger, raise_error_if_invalid_value


log = get_logger(__name__)


class BaseDataModule(pl.LightningDataModule):
    """
    ----------------------------------------------------------------------------------------------------------
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    _data_train: Dataset
    _data_val: Union[Dataset, Sequence[Dataset]]
    _data_test: Dataset
    _data_predict: Dataset

    def __init__(
        self,
        data_dir: str,
        model_config: DictConfig = None,
        batch_size: int = 2,
        eval_batch_size: int = 64,
        num_workers: int = -1,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        multiprocessing_context: Optional[str] = None,
        drop_last: bool = False,
        shuffle_train_data: bool = True,
        debug_mode: bool = False,
        verbose: bool = True,
        seed_data: int = 43,
        batch_size_per_gpu=None,  # should be none and be handled outside (directly set in batch_size)
    ):
        """
        Args:
            data_dir (str):  A path to the data folder that contains the input and output files.
            batch_size (int): Batch size for the training dataloader
            eval_batch_size (int): Batch size for the test and validation dataloader's
            num_workers (int): Dataloader arg for higher efficiency (usually set to # of CPU cores).
                                Default: Set to -1 to use all available cores.
            pin_memory (bool): Dataloader arg for higher efficiency. Default: True
            drop_last (bool): Only for training data loading: Drop the last incomplete batch
                                when the dataset size is not divisible by the batch size. Default: False
            shuffle_train_data (bool): Only for training data loading: Shuffle the training data.
                                Default: True
            verbose (bool): Print the dataset sizes. Default: True
        """
        super().__init__()
        # The following makes all args available as, e.g., self.hparams.batch_size
        self.save_hyperparameters(ignore=["model_config", "verbose"])
        self.model_config = model_config
        self.test_batch_size = eval_batch_size  # just for testing
        assert (
            batch_size_per_gpu is None or batch_size_per_gpu == batch_size
        ), f"batch_size_per_gpu should be None, but got {batch_size_per_gpu}. (batch_size={batch_size})"
        self._data_train = self._data_val = self._data_test = self._data_predict = None
        self._experiment_class_name = None
        self._check_args()

    def _check_args(self):
        """Check if the arguments are valid."""
        if self.hparams.debug_mode is True:
            self.hparams.num_workers = 0
            self.hparams.batch_size = 8
            self.hparams.eval_batch_size = 8

        if self.hparams.num_workers == 0:
            if self.hparams.persistent_workers is True:
                log.warning(
                    "persistent_workers can only be set to True if num_workers > 0. "
                    "Setting persistent_workers to False."
                )
                self.hparams.persistent_workers = False

    @property
    def sigma_data(self) -> float:
        raise NotImplementedError("Please specify the standard deviation of the training data in the subclass.")

    @property
    def experiment_class_name(self) -> str:
        if self._experiment_class_name is None:
            if self.trainer is not None and hasattr(self.trainer, "lightning_module"):
                self._experiment_class_name = self.trainer.lightning_module.__class__.__name__
        return self._experiment_class_name or "unknown"

    def _concat_variables_into_channel_dim(self, data: xr.Dataset, variables: List[str], filename=None) -> np.ndarray:
        """Concatenate xarray variables into numpy channel dimension (last)."""
        data_all = []
        for var in variables:
            # Get the variable from the dataset (as numpy array, by selecting .values)
            var_data = data[var].values
            # add feature dimension (channel)
            var_data = np.expand_dims(var_data, axis=-1)
            # add to list of all variables
            data_all.append(var_data)

        # Concatenate all the variables into a single array along the last (channel/feature) dimension
        dataset = np.concatenate(data_all, axis=-1)
        assert dataset.shape[-1] == len(variables), "Number of variables does not match number of channels."
        return dataset

    # @rank_zero_only
    def print_data_sizes(self, stage: str = None):
        """Print the sizes of the data."""

        if stage in ["fit", None]:
            val_size = [len(dv) for dv in self._data_val] if isinstance(self._data_val, list) else len(self._data_val)
            log.info(f"Dataset sizes train: {len(self._data_train)}, val: {val_size}")
        elif stage == "validate":
            val_size = [len(dv) for dv in self._data_val] if isinstance(self._data_val, list) else len(self._data_val)
            log.info(f"Dataset validation size: {val_size}")
        elif stage in ["test", None]:
            log.info(f"Dataset test size: {len(self._data_test)}")
        elif stage == "predict":
            log.info(f"Dataset predict size: {len(self._data_predict)}")

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        raise_error_if_invalid_value(stage, ["fit", "validate", "test", "predict", None], "stage")

        if stage == "fit" or stage is None:
            self._data_train = ...  # get_tensor_dataset_from_numpy(X_train, Y_train, dataset_id='train')
        if stage in ["fit", "validate", None]:
            self._data_val = ...  # get_tensor_dataset_from_numpy(X_val, Y_val, dataset_id='val')
        if stage in ["test", None]:
            self._data_test = ...  # get_tensor_dataset_from_numpy(X_test, Y_test, dataset_id='test')
        if stage in ["predict"]:
            self._data_predict = ...
        raise NotImplementedError("This class is not implemented yet.")

    @abstractmethod
    def get_horizon(self, split: str, dataloader_idx: int = 0) -> int:
        """Return the horizon for the given split."""
        return self.hparams.get("horizon", 1)

    def get_horizon_range(self, split: str, dataloader_idx: int = 0) -> List[int]:
        """Return the horizon range for the given split."""
        return list(np.arange(1, self.get_horizon(split, dataloader_idx) + 1))

    @property
    def valid_time_range_for_backbone_model(self) -> List[int]:
        return self.get_horizon_range("fit")

    def get_epoch_aggregators(
        self,
        split: str,
        is_ensemble: bool,
        dataloader_idx: int = 0,
        experiment_type: str = None,
        device: torch.device = None,
        verbose: bool = True,
        save_to_path: str = None,
        **kwargs,
    ) -> Dict[str, _Aggregator]:
        """Return the epoch aggregators for the given split.
        Make sure to pass the kwargs to the SaveToDiskAggregator!
        """
        return {}

    @property
    def num_workers(self) -> int:
        if self.hparams.num_workers == -1:
            return multiprocessing.cpu_count()
        return int(self.hparams.num_workers)

    def _shared_dataloader_kwargs(self) -> dict:
        shared_kwargs = dict(
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )
        if self.hparams.prefetch_factor is not None:
            shared_kwargs["prefetch_factor"] = self.hparams.prefetch_factor
        if self.hparams.multiprocessing_context is not None:
            mp_context = self.hparams.multiprocessing_context
            if mp_context == "fork":
                mp_context = multiprocessing.get_context("fork")
            shared_kwargs["multiprocessing_context"] = mp_context
        return shared_kwargs

    def train_dataloader(self):
        assert self.hparams.shuffle_train_data, "Why not?"
        return (
            DataLoader(
                dataset=self._data_train,
                batch_size=self.hparams.batch_size,
                drop_last=self.hparams.drop_last,  # drop last incomplete batch (only for training)
                shuffle=self.hparams.shuffle_train_data,
                **self._shared_dataloader_kwargs(),
            )
            if self._data_train is not None
            else None
        )

    def _shared_eval_dataloader_kwargs(self) -> dict:
        return dict(**self._shared_dataloader_kwargs(), shuffle=False)

    def adjust_eval_batch_size_for_dataset(self, dataset: Dataset, default_batch_size: int, name: str) -> int:
        """Adjust the eval batch size based on the dataset size when using DDP."""
        if self.trainer is None:
            return default_batch_size
        world_size = self.trainer.world_size
        data_points_per_gpu = len(dataset) // world_size
        batch_size = min(default_batch_size, data_points_per_gpu)
        if batch_size != default_batch_size:
            log.info(
                f"Dataset `{name}`: Adjusting eval batch size to {batch_size} based on"
                f" dataset size ({len(dataset)}) and world size ({world_size})."
            )
        return batch_size

    def val_dataloader(self):
        if self._data_val is None:
            return None
        ds_val = [self._data_val] if isinstance(self._data_val, Dataset) else self._data_val
        dataloaders = []
        for i, ds in enumerate(ds_val):
            bs_here = self.adjust_eval_batch_size_for_dataset(ds, self.hparams.eval_batch_size, f"val_{i}")
            dataloaders.append(DataLoader(dataset=ds, batch_size=bs_here, **self._shared_eval_dataloader_kwargs()))
        return dataloaders

    def test_dataloader(self) -> DataLoader:
        if self._data_test is None:
            return None
        return DataLoader(
            dataset=self._data_test,
            batch_size=self.adjust_eval_batch_size_for_dataset(self._data_test, self.test_batch_size, "test"),
            **self._shared_eval_dataloader_kwargs(),
        )

    def predict_dataloader(self) -> DataLoader:
        if self._data_predict is None:
            return None
        ebs_here = self.adjust_eval_batch_size_for_dataset(self._data_predict, self.hparams.eval_batch_size, "predict")
        return DataLoader(
            dataset=self._data_predict,
            batch_size=ebs_here,
            **self._shared_eval_dataloader_kwargs(),
        )

    def boundary_conditions(
        self,
        preds: Union[Tensor, TensorDict],
        targets: Union[Tensor, TensorDict],
        data: Any = None,
        metadata: Any = None,
        time: float = None,
    ) -> Union[Tensor, TensorDict]:
        """Return predictions that satisfy the boundary conditions for a given item (batch element)."""
        return preds

    def get_boundary_condition_kwargs(self, batch: Any, batch_idx: int, split: str) -> dict:
        return dict(t0=0.0, dt=1.0)

    @property
    def validation_set_names(self) -> List[str] | None:
        """Use for using specific prefix for logging validation metrics."""
        return None
