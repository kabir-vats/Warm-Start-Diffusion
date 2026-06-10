from __future__ import annotations

import copy
import json
import os
import time
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dask
import numpy as np
import pandas as pd
import tensorstore as ts
import torch
import xarray as xr
import xbatcher
import zarr
from dask.distributed import Client, LocalCluster
from omegaconf import ListConfig
from torch import multiprocessing

from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.utils import regrid
from src.datamodules.utils.forcings import TISR, add_derived_vars, add_tisr_var, compute_forcings_numpy
from src.evaluation.aggregators.main import ListAggregator, OneStepAggregator
from src.evaluation.aggregators.save_data_mem_efficient import MemEfficientSaveToDiskAggregator
from src.evaluation.metrics_wb import get_lat_weights
from src.utilities.normalization import get_normalizer
from src.utilities.packer import Packer
from src.utilities.timing import log_timing
from src.utilities.utils import (
    get_logger,
    raise_error_if_invalid_type,
    raise_error_if_invalid_value,
    subsample_preselected_indices,
    to_torch_and_device,
)


log = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_zarr_dataset(zarr_path, **kwargs):
    try:
        ds = xr.open_zarr(zarr_path, **kwargs)
    except Exception:
        try:
            ds = xr.open_zarr(zarr_path, zarr_format=3, **kwargs)
        except Exception:
            kwargs["consolidated"] = False
            # zarr_format=3,
            ds = xr.open_zarr(zarr_path, **kwargs)
    return ds


def find_path_from_dir_opts(data_dirs: List[str], dataset_name: str):
    for data_dir in data_dirs:
        potential_path = join(data_dir, dataset_name)
        if os.path.isfile(potential_path) or os.path.isdir(potential_path):
            return potential_path
    raise FileNotFoundError(f"Could not find {dataset_name} in any of the following data directories: {data_dirs}")


# should be moved to dataset_utils
def extract_date(date_info, shift_text_date):
    # date info can be in format a) "2020-01-01" or b) "2020-01-01 00:00:00" or c) datetime object
    if isinstance(date_info, datetime):
        date = date_info
    else:
        # read the date
        date = date_info.split(" ")[0]  # "2020-01-01 00:00:00" -> "2020-01-01"

    # Remove hours from the date
    date = np.datetime64(date, "D")

    if shift_text_date is not None and shift_text_date != 0:
        # print(f"Shifting {date=} to {date + np.timedelta64(shift_text_date, 'D')}")
        date += np.timedelta64(shift_text_date, "D")

    return date


def get_date(date_str: str):
    if ":" in date_str and "T" in date_str:
        fmt = "%Y-%m-%dT%H:%M:%S"
    elif ":" in date_str:
        fmt = "%Y-%m-%d %H:%M:%S"
    else:
        fmt = "%Y-%m-%d"
    return datetime.strptime(date_str, fmt)


def extract_time_subsample(dataset: str, hourly_resolution: int) -> None:
    # Infer hourly resolution of dataset
    if "-6h-" in dataset:
        if hourly_resolution == 6:
            time_subsample = 1
        elif hourly_resolution == 12:
            time_subsample = 2
        else:
            raise ValueError(f"Invalid hourly resolution: {hourly_resolution} for dataset: {dataset}")
    elif "-12h-" in dataset:
        if hourly_resolution == 12:
            time_subsample = 1
        else:
            raise ValueError(f"Invalid hourly resolution: {hourly_resolution} for dataset: {dataset}")
    elif "-1h-" in dataset:
        time_subsample = hourly_resolution
    else:
        raise ValueError(f"Could not infer hourly resolution from dataset: {dataset}")

    if time_subsample > 1:
        log.info(f"Setting slice subsample to {time_subsample} due to {hourly_resolution=} for {dataset=}.")
    return time_subsample


def get_slice(slice_, split: str, time_subsample: int) -> slice:
    if isinstance(slice_, Sequence) and len(slice_) == 2:
        slice_ = slice(*slice_)
    assert isinstance(slice_, slice), f"Invalid slice for {split}: {slice_}"
    # Convert start and end to dates, if only years are given
    if isinstance(slice_.start, int):
        slice_ = slice(f"{slice_.start}-01-01", slice_.stop, slice_.step)
    if isinstance(slice_.stop, int):
        slice_ = slice(slice_.start, f"{slice_.stop}-12-31", slice_.step)
    # If it does not have a step, set the step to time_subsample
    if slice_.step is None:
        slice_ = slice(slice_.start, slice_.stop, time_subsample)  # e.g. slice(2014, 2020, 12)
    # To datetime
    slice_ = slice(get_date(str(slice_.start)), get_date(str(slice_.stop)), slice_.step)
    if split != "predict":
        assert slice_.step == time_subsample, f"Invalid step for {split=}: {slice_.step}"
    return slice_


class ERA5DataModuleBase(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        data_dir_stats: Optional[str] = None,
        text_data_path: Optional[str] = None,
        dataset: str = "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
        training_datasets: Sequence[str] = None,  # If none, use the dataset name only
        target_dataset: Optional[str] = None,
        train_slice: Optional[slice] = slice("2015-01-01", "2018-12-31"),
        val_slice: Optional[slice] = slice("2019-01-01", "2019-12-31"),
        test_slice: Optional[slice] = slice("2020-01-01", "2020-12-31"),
        predict_slice: Optional[slice] = slice("2020-03-01", "2020-12-31", 96),
        hourly_resolution: int = 1,
        possible_initial_times: Optional[List[str]] = None,
        possible_initial_times_eval: Optional[List[str]] = None,
        subsample_valid: int = 1,
        window: int = 1,  # Number of time steps to use in the input
        horizon: int = 1,  # Number of time steps to predict into the future
        prediction_horizon: int = None,  # None means use horizon and no auto-regressive prediction
        prediction_horizon_long: int = None,  # None means use horizon and no auto-regressive prediction
        static_fields: Sequence[str] = (
            "land_sea_mask",
            "geopotential_at_surface",
            # "soil_type",
            # "lat_lon_embeddings",
        ),
        forcing_fields: Sequence[str] = None,
        spatial_crop_inputs: Optional[Dict[str, slice]] = None,
        spatial_crop_outputs: Optional[Dict[str, slice]] = None,
        spatial_crop_during_training: bool = False,  # only valid if spatial_crop_outputs is not None
        output_mask_area: Optional[str] = None,
        loss_latitude_weighting: bool = True,
        loss_pressure_weighting: bool = False,
        loss_pressure_weighting_levels: Union[str, List[int]] = "era5",  # can be "era5", "wb", or a list of levels
        loss_pressure_weighting_divide_by: str = "mean",  # can be "mean" or "sum"
        loss_surface_vars_weighting: (
            str | None
        ) = None,  # todo: implement a 1/IFS ENS performance weighting (GenCast reports success with this)
        loss_multipliers: Optional[Dict[str, float]] = None,
        text_period_start: Optional[str] = None,  # If None, use all text data
        text_period_end: Optional[str] = None,  # If None, use all text data. The end date is inclusive.
        shift_text_date: int = 0,
        text_history: int = 0,
        text_conditioning: str = "time",
        text_skip_missing_dates: bool = False,  # If false, use null embeddings for missing dates
        return_future_date_for_training: bool = False,
        # set to true if self.model.predict_non_spatial_condition=True
        normalize_std_fname: str = "std",  # use std_rescaled for residual prediction, std for direct prediction
        use_dask: bool = False,
        num_dask_workers: int = 16,
        dask_scheduler: str = "threads",  # can be "threads", "processes", "synchronous", "distributed"
        dask_cache_size: str = "10GB",
        dask: Optional[Dict] = None,
        # Advanced dask config: {multinode, scheduler_port, threads_per_worker, memory_limit, worker_saturation, chunk_time, chunk_lat, chunk_lon, chunk_level, optimize_lustre}
        load_type: str = "xbatcher",  # xarray, xbatcher, tstore
        lat_lon_format: str = "lon_lat",
        text_type: str = "tf-idf",  # can be tf-idf, bert, bow
        log_metrics: bool = True,
        log_normed: bool = True,
        log_abs_values: bool = True,
        log_images: bool = True,
        log_spectra: bool = False,
        every_nth_epoch_snapshot: int = 8,
        max_val_samples: int = None,
        eval_resolution: Tuple[int, int] = None,  # e.g. (64, 32) to regrid targets/preds to this res for eval
        **kwargs,
    ):
        """

        Args:
            data_dir (str): Path to the directory containing the zarr dataset (or a ``dataset`` subdirectory)
            data_dir_stats (str): Path to the directory containing the normalization statistics
            text_data_path (str): Path to the text data file (if using text embeddings)
            dataset (str): Name of the weatherbench2 dataset
            training_datasets: A list of dataset names to use for training. If None, uses the dataset specified in `dataset` for training.
            target_dataset: The dataset to use as target. If None, uses the dataset specified in `dataset` as target. This can be used to e.g. use a higher-resolution dataset for the inputs and a lower-resolution dataset for the targets.
            train_slice: slice for the training period
            val_slice: slice for the validation period
            test_slice:  slice for the test period
            predict_slice:  slice for the prediction period
            hourly_resolution:  1 for hourly, 6 for 6-hourly etc.
            possible_initial_times: Possible initial times for the prediction (e.g. ["00:00", "06:00", "12:00", "18:00"]), only use if hourly_resolution = 1
            subsample_valid: Subsample the validation set by this factor (for faster validation).
            window: The number of time steps to use in the input
            horizon: The number of time steps to predict into the future during training
            prediction_horizon: The number of time steps to predict into the future during validation
            prediction_horizon_long: The number of time steps to predict into the future during inference/testing
            static_fields: The names of the static fields to include as conditional inputs
            spatial_crop_inputs: A dictionary of slices to crop the input fields by, if desired
            spatial_crop_outputs: A dictionary of slices to crop the output/target fields by, if desired
            spatial_crop_during_training: Only applies if spatial_crop_outputs is not None.
                If True, the spatial crop is applied during training, otherwise only during validation and testing.
            output_mask_area: Only applies if spatial_crop_outputs is not None. Used to use the spatial_crop_outputs
                mask over specific areas (e.g. "land" or "ocean").
            loss_latitude_weighting: Whether to weight the loss by cosine of latitude
            loss_pressure_weighting: Whether to weight the loss proportionally to the pressure levels (more weight to near-surface vars which have larger pressure levels)
            loss_surface_vars_weighting: How to weight the loss for surface variables. Can be "graphcast" or None
            loss_pressure_weighting_levels: Only applies if loss_pressure_weighting=True.
                Can be "era5", "wb", or a list of levels. If "era5", uses the ERA5 pressure levels. If "wb", uses the WeatherBench pressure levels.
                These levels are used to compute the normalization weights for the pressure levels.
            loss_pressure_weighting_divide_by: Only applies if loss_pressure_weighting=True. Can be "mean" or "sum".
                The pressure weighting is divided by the mean or sum of the pressure levels.
            shift_text_date:  Number of days to shift the text dates by (if positive, this uses future information!)
            text_conditioning: How to condition the text embeddings. Can be "time" or "cross_attn".
            normalize_std_fname: Which standard deviation file to use for normalization
            use_dask:
            lat_lon_format:
            num_dask_workers:
            text_type: Which type of text embeddings to use. Can be "tf-idf", "bert", or "bow"
            log_metrics: Whether to log metrics (e.g. RMSE, CRPS, etc.)
            log_images: Whether to log images (e.g. global predictions, targets, bias)
            log_spectra: Whether to log power spectra. If "targets", logs the target spectra. If true, logs predictions spectra.
            **kwargs:

        Note:
            For Autoregressive training you need to make sure that:
                - input_vars == output_vars
                - You may want to set spatial_crop_during_training=False
        """
        raise_error_if_invalid_type(data_dir, possible_types=[str, List, ListConfig], name="data_dir")
        raise_error_if_invalid_value(lat_lon_format, possible_values=["lat_lon", "lon_lat"], name="lat_lon_format")
        possible_text_conds = ["time"]  # cross_attn
        raise_error_if_invalid_value(text_conditioning, possible_values=possible_text_conds, name="text_conditioning")
        assert hourly_resolution >= 1, f"Invalid hourly_resolution: {hourly_resolution}"
        assert dataset.endswith(".zarr"), f"dataset should be a .zarr file. Invalid dataset: {dataset}"
        if not isinstance(data_dir, (ListConfig, List)):
            data_dir = [data_dir]
        data_dir = list(data_dir)  # in case it's a ListConfig
        for i, data_dir_i in enumerate(data_dir):
            assert ".zarr" not in data_dir_i, "data_dir should not include the .zarr file. Specify in `dataset`"
            if "weatherbench2" not in data_dir_i:
                if os.path.isdir(join(data_dir_i, "weatherbench2")):
                    data_dir[i] = join(data_dir_i, "weatherbench2")

        self.zarr_path = find_path_from_dir_opts(data_dir, dataset)
        if isinstance(predict_slice, str) and "slice" in predict_slice:
            predict_slice = eval(predict_slice)
        # Process dask config dict and set defaults
        if dask is None:
            dask = {}
        self.dask_config = {
            "multinode": dask.get("multinode", False),
            "scheduler_port": dask.get("scheduler_port", 8786),
            "dashboard_port": dask.get("dashboard_port", 8787),
            "threads_per_worker": dask.get("threads_per_worker", 2),
            "memory_limit": dask.get("memory_limit", "auto"),
            "worker_saturation": dask.get("worker_saturation", 1.0),
            "chunk_time": dask.get("chunk_time", None),
            "chunk_lat": dask.get("chunk_lat", None),
            "chunk_lon": dask.get("chunk_lon", None),
            "chunk_level": dask.get("chunk_level", None),
            "optimize_lustre": dask.get("optimize_lustre", False),
        }

        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        self.hparams.data_dir = data_dir
        if self.hparams.debug_mode:
            log.info("------------------ Running in debug mode -------------------")
            self.hparams.train_slice = slice("2015-01-01", "2015-01-05")
            self.hparams.val_slice = slice("2015-02-01", "2015-02-10")
            self.hparams.subsample_valid = 6
        if use_dask and False:
            from dask.cache import Cache as dask_Cache

            # comment these the next two lines out to disable Dask's cache
            log.info(f"Registering Dask cache with size: {dask_cache_size}")
            cache = dask_Cache(dask_cache_size)  # dask_Cache(1e10)  # 10gb cache
            cache.register()

        if possible_initial_times is not None and possible_initial_times_eval is None:
            self.hparams.possible_initial_times_eval = possible_initial_times_eval = possible_initial_times
        # Set the temporal slices for the train, val, and test sets

        time_subsample = extract_time_subsample(dataset, hourly_resolution)
        if isinstance(train_slice, ListConfig):
            train_slice = list(train_slice)
        if isinstance(train_slice, List) and isinstance(train_slice[0], ListConfig):
            train_slice = [list(tslice) if isinstance(tslice, ListConfig) else tslice for tslice in train_slice]
        if not (isinstance(train_slice, List) and isinstance(train_slice[0], (List, Tuple, slice))):
            train_slice = [train_slice]  # Should be a list of slices
        for i, tslice in enumerate(train_slice):
            train_slice[i] = get_slice(tslice, "train", time_subsample)
        self.train_slice = train_slice
        eval_slices = dict(val=val_slice, test=test_slice, predict=predict_slice)
        for split, slice_ in eval_slices.items():
            setattr(self, f"{split}_slice", get_slice(slice_, split, time_subsample))  # e.g. self.val_slice = ...

        # Check that train and test slices are not overlapping
        train_slice_end_date = extract_date(self.train_slice[0].stop, 0)
        test_slice_start_date = extract_date(self.test_slice.start, 0)
        assert train_slice_end_date <= test_slice_start_date, f"train_slice: {train_slice}, test_slice: {test_slice}"

        # Normalization
        if data_dir_stats is None:
            opts = data_dir + [os.path.dirname(data_dir_i) for data_dir_i in data_dir]  # also check parent directories
            data_dir_stats = find_path_from_dir_opts(opts, "statistics")
            log.info(f"data_dir_stats is not specified. Found data_dir_stats at: {data_dir_stats}")

        if "era5" not in normalize_std_fname:
            normalize_std_fname = f"era5_{normalize_std_fname}"
        normalize_std_fname += ".nc" if not normalize_std_fname.endswith(".nc") else ""

        data_dir_stats = Path(data_dir_stats)
        path_mean = data_dir_stats / "era5_mean.nc"
        path_std = data_dir_stats / normalize_std_fname
        path_std_res = data_dir_stats / "era5_residual_std.nc"
        path_min = data_dir_stats / "era5_min.nc"
        if not path_mean.exists() or not path_std.exists():
            raise FileNotFoundError(f"Could not find normalization files at ``{path_mean}`` and/or ``{path_std}``")
        self._normalizer_files = dict(mean=path_mean, std=path_std, std_residual=path_std_res, min=path_min)

        self._latitude, self._longitude, self._split_to_time = None, None, dict()
        if spatial_crop_inputs is not None:
            spatial_crop_inputs = dict(**spatial_crop_inputs)
            for k, v in spatial_crop_inputs.items():
                if isinstance(v, Sequence) and len(v) == 2:
                    spatial_crop_inputs[k] = slice(*[int(x) for x in v])

        if spatial_crop_outputs is not None:
            spatial_crop_outputs = dict(**spatial_crop_outputs)
            for k, v in spatial_crop_outputs.items():
                if isinstance(v, Sequence) and len(v) == 2:
                    spatial_crop_outputs[k] = slice(*[int(x) for x in v])
            crop_lats, crop_lons = spatial_crop_outputs.get("latitude"), spatial_crop_outputs.get("longitude")
            if crop_lats == slice(10, 70) and crop_lons == slice(190, 310):
                self.crop_name = "NA"  # "north_america"
            elif crop_lats == slice(24, 50) and crop_lons == slice(235, 295):
                self.crop_name = "ConUS"  # "united_states"
            else:
                raise ValueError(f"Plese give your crop a name. Current crop: {spatial_crop_inputs}")
        else:
            self.crop_name = ""  # "global"

        self.spatial_crop_inputs = spatial_crop_inputs
        self.spatial_crop_outputs = spatial_crop_outputs

        if self.spatial_crop_inputs is not None and self.spatial_crop_outputs is not None:
            # Check that output crop is a subset of input crop
            for k, v in self.spatial_crop_outputs.items():
                crop_inputs_k = self.spatial_crop_inputs.get(k)
                if isinstance(v, slice):
                    assert v.start >= crop_inputs_k.start, f"Invalid crop for {k}: {v}"
                    assert v.stop <= crop_inputs_k.stop, f"Invalid crop for {k}: {v}"

        self.text_data = self.text_emb_dim = None

        if text_data_path is None:
            assert text_type is None, f"Invalid text_type: {text_type} without text_data_path"
        else:
            if text_type is None:
                text_type = "tf-idf"
                log.info(f"Text type is not specified. Using default: {text_type}")
            if not os.path.isfile(text_data_path):
                opts = data_dir + [os.path.dirname(data_dir_i) for data_dir_i in data_dir]
                text_data_path = find_path_from_dir_opts(opts, text_data_path)
            # text data loading
            df = pd.read_csv(text_data_path)
            corpus = df["output"] if "output" in df.columns else df["text"]
            dates = df["date"]
            # Convert from str to datetime
            dates = pd.to_datetime(dates)
            assert len(corpus) == len(dates), f"Corpus and dates have different lengths: {len(corpus)} vs {len(dates)}"
            if text_period_start is not None:
                text_period_start = np.datetime64(text_period_start, "D")
                corpus = corpus[dates >= text_period_start]
                dates = dates[dates >= text_period_start]
            if text_period_end is not None:
                text_period_end = np.datetime64(text_period_end, "D")
                corpus = corpus[dates <= text_period_end]
                dates = dates[dates <= text_period_end]
            assert len(corpus) > 0, f"No text data found for {text_period_start=} and {text_period_end=}"

            metadata = dict(corpus_filename=text_data_path, period_start=text_period_start, period_end=text_period_end)
            embs_kwargs = dict(metadata=metadata, history_length=self.hparams.text_history)
            if text_type == "bert":
                from src.utilities.text import get_or_create_embeddings

                model_name = "bert-base-uncased"
                text_features = get_or_create_embeddings(corpus, model_name, None, **embs_kwargs)
                self.text_emb_dim = len(text_features[0])  # First text feature

            elif "llama" in text_type.lower():
                from src.utilities.text import get_or_create_embeddings

                model_name = text_type.replace("llama", "Meta-Llama-3.1-8B")
                model_name = f"meta-llama/{model_name}"
                # try meta-llama/Llama-3.1-8B-Instruct
                cache_dir = os.path.join(os.environ.get("PSCRATCH", os.environ.get("HOME")), ".cache", "huggingface")
                text_features = get_or_create_embeddings(corpus, model_name, cache_dir, **embs_kwargs)
                self.text_emb_dim = len(text_features[0])  # First text feature

            elif text_type == "bow":
                from sklearn.feature_extraction.text import CountVectorizer

                log.info("Bag of words representation is used for text data.")
                vectorizer = CountVectorizer(stop_words="english")
                X = vectorizer.fit_transform(corpus)
                text_features = X.toarray().astype(np.float32)  # text_features=bow_array
                self.text_emb_dim = len(text_features[0])

            elif text_type == "tf-idf":
                from sklearn.feature_extraction.text import TfidfVectorizer

                log.info("Tf-idf representation is used for text data.")
                vectorizer = TfidfVectorizer(stop_words="english")
                text_features = vectorizer.fit_transform(corpus)
                text_features = text_features.toarray().astype(np.float32)
                self.text_emb_dim = text_features.shape[1]
                assert len(corpus) == len(text_features), f"{len(corpus)=} vs {len(text_features)=}"

            else:
                raise ValueError(f"Invalid text_type: {text_type}")

            self.text_data = {}
            self.raw_text_dataset = {}  # to analyse the text data, if needed
            for text, feature, date in zip(corpus, text_features, dates):
                # log.info(f"Text: {text[:10]}... Feature: {feature[:15]}...")
                if feature is None:
                    log.warning(f"Text data for {date=} is None. Skipping...")
                    continue
                date_feature = extract_date(date, shift_text_date)
                self.text_data[date_feature] = torch.from_numpy(feature).squeeze()
                self.raw_text_dataset[date_feature] = text

            # Compute how many days from training+val period are missing in the text data
            missing_dates = set()
            for split in ["train", "val"]:
                slice_ = getattr(self, f"{split}_slice")
                slice_datetimes = pd.date_range(slice_.start, slice_.stop, freq="D")
                slice_datetimes = [np.datetime64(x, "D") for x in slice_datetimes]
                missing_dates.update(set(slice_datetimes) - set(self.text_data.keys()))
                if split == "train":
                    # Go over first 3 dates in training period
                    for h in range(3):
                        date_h = slice_datetimes[h]
                        text_example = self.raw_text_dataset.get(date_h) or "No text"
                        text_example = text_example[:80].replace("\n", "\t")
                        feature_shape = self.text_data[date_h].shape if date_h in self.text_data else "None"
                        log.info(f"{h=}, {date_h=}, {text_example=}... {feature_shape=}")

                # print(list(slice_datetimes)[:10], list(self.text_data.keys())[:10])
            if missing_dates:
                missing_dates = sorted(set([str(x)[:7] for x in missing_dates]))  # Get unique years and months only
                log.info(
                    f"Missing {len(missing_dates)} dates (year-month only) in training+val period: {missing_dates}"
                )
                #  Missing >=98 dates (year-month only) in training+val period: [All the way to 1987-12, and ...,
                #  '1988-01', '1988-02', '1988-04', '1988-05', '1988-08', '1988-09', '1988-10', '1988-12', '1989-01',
                #  '1989-03', '1989-07', '1989-08', '1989-12', '1990-10', '1991-01', '1991-02', '1991-04', '1991-05',
                #  '1991-06', '1991-11', '1992-12', '1993-01', '1993-04', '1994-04', '1996-04', '1997-05', '1997-07',
                #  '1998-10', '1999-04', '1999-07', '1999-12', '2000-03', '2001-01', '2001-04', '2001-07', '2001-08',
                #  '2002-01', '2004-05', '2005-03', '2005-05', '2006-01', '2006-08', '2006-09', '2007-03', '2007-06',
                #  '2008-01', '2008-12', '2009-03', '2010-02', '2010-03', '2010-04', '2010-05', '2010-06', '2010-07',
                #  '2010-08', '2010-09', '2010-10', '2010-11', '2010-12', '2011-01', '2011-02', '2011-03', '2011-04',
                #  '2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11', '2011-12', '2012-02',
                #  '2012-03', '2012-10']

            log.info(f"Text embedding dimension: {self.text_emb_dim}")
            log.info(f"Text data loaded. Number of entries: {len(self.text_data)} n_texts: {len(corpus)}")

    @property
    def dataset_identifier(self) -> str:
        iden = f"ERA5_horizon{self.hparams.horizon}"
        return iden

    @property
    def sigma_data(self) -> float:
        return 1.0

    def __del__(self):
        # Close the Dask client when the dataset is no longer needed
        if hasattr(self, "client"):
            self.client.close()

    def get_horizon(self, split: str, dataloader_idx: int = 0) -> int:
        if split in ["val", "validate"] and dataloader_idx == 1:
            return self.hparams.prediction_horizon_long
        assert dataloader_idx in [0, None], f"Invalid dataloader_idx: {dataloader_idx}"
        if split in ["predict", "test"]:
            return self.hparams.prediction_horizon_long or self.hparams.horizon
        elif split in ["val", "validate"]:
            return self.hparams.prediction_horizon or self.hparams.horizon
        else:
            assert split in ["train", "fit"], f"Invalid split: {split}"
            return self.hparams.horizon

    def _check_args(self):
        super()._check_args()
        h = self.hparams.horizon
        assert isinstance(h, list) or h > 0, f"horizon must be > 0 or a list, but is {h}"

        # Warn if num_workers is too high when using dask
        if self.hparams.use_dask and self.hparams.num_dask_workers:
            cpus_available = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 32))
            dask_workers = int(self.hparams.num_dask_workers * self.dask_config["worker_saturation"])
            dask_threads = dask_workers * self.dask_config["threads_per_worker"]
            dataloader_workers = self.hparams.num_workers if self.hparams.num_workers != -1 else cpus_available

            total_workers = dask_threads + dataloader_workers
            if total_workers > cpus_available * 1.2:  # Allow some oversubscription
                log.warning(
                    f"⚠️  Potential CPU oversubscription detected!\n"
                    f"   Dask threads: {dask_threads} ({dask_workers} workers × {self.dask_config['threads_per_worker']} threads)\n"
                    f"   DataLoader workers: {dataloader_workers}\n"
                    f"   Total: {total_workers} > Available CPUs: {cpus_available}\n"
                    f"   Consider reducing datamodule.num_workers (e.g., to {max(2, cpus_available // 4)}) or num_dask_workers"
                )
        # if os.path.isdir(self.zarr_path):  # Check local directory
        #     assert os.path.isfile(join(self.zarr_path, ".zmetadata")), f"Could not find .zmetadata in data_dir: {self.zarr_path}"

    def _setup_dask_cluster(self):
        """Setup dask cluster for single-node or multi-node training with PyTorch DDP."""
        if not self.hparams.use_dask or self.hparams.num_dask_workers is None:
            return

        if hasattr(self, "client") and hasattr(self.client, "status") and self.client.status == "running":
            return  # Already setup

        # Get DDP info (if available)
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                rank = dist.get_rank()
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                # world_size = dist.get_world_size()
            else:
                rank = local_rank = 0
                # world_size = 1
        except (ImportError, RuntimeError):
            rank = local_rank = 0
            # world_size = 1

        cfg = self.dask_config

        # Calculate worker resources
        cpus_per_node = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 32))
        n_workers = int(self.hparams.num_dask_workers * cfg["worker_saturation"])
        threads_per_worker = cfg["threads_per_worker"]
        memory_limit = cfg["memory_limit"]

        if cfg["multinode"]:
            # Multi-node setup: use distributed scheduler
            # Use shared filesystem (SCRATCH on HPC systems) for scheduler file
            shared_dir = os.environ.get("SCRATCH", os.environ.get("PSCRATCH", "/tmp"))
            scheduler_file = os.environ.get(
                "DASK_SCHEDULER_FILE", f"{shared_dir}/dask-scheduler-{os.environ.get('SLURM_JOB_ID', 'local')}.json"
            )

            if rank == 0:
                # Global rank 0: create the scheduler on network interface
                import socket

                hostname = socket.gethostname()
                # Get the interface address (not localhost)
                interface_address = socket.gethostbyname(hostname)
                log.info(
                    f"[Rank {rank}] Creating Dask scheduler on {hostname} ({interface_address}) at port {cfg['scheduler_port']}"
                )

                # Create scheduler-only cluster first to write address quickly
                cluster = LocalCluster(
                    n_workers=0,  # Start with no workers
                    processes=True,
                    host=interface_address,  # Bind to network interface, not localhost
                    scheduler_port=cfg["scheduler_port"],
                    dashboard_address=f"{interface_address}:{cfg['dashboard_port']}",
                )

                # Get the scheduler address and save immediately
                scheduler_address = cluster.scheduler_address
                if "0.0.0.0" in scheduler_address or "127.0.0.1" in scheduler_address:
                    scheduler_address = f"tcp://{hostname}:{cfg['scheduler_port']}"

                # Save scheduler address to shared file BEFORE creating workers
                with open(scheduler_file, "w") as f:
                    f.write(scheduler_address)
                log.info(f"[Rank {rank}] Scheduler address {scheduler_address} saved to {scheduler_file}")

                # Now scale up workers
                log.info(f"[Rank {rank}] Adding {n_workers} workers with {threads_per_worker} threads each")
                cluster.scale(n_workers)

                # Connect as client
                self.client = Client(cluster)
                self.cluster = cluster

            else:
                # Other ranks: wait for scheduler and connect as clients
                import time

                wait_count = 0
                while not os.path.exists(scheduler_file):
                    time.sleep(0.5)
                    wait_count += 1
                    if wait_count > 240:  # 120 seconds timeout
                        raise TimeoutError(f"Scheduler file not found after 120s: {scheduler_file}")

                # Give scheduler extra time to be fully ready
                time.sleep(2)
                with open(scheduler_file) as f:
                    scheduler_address = f.read().strip()

                log.info(f"[Rank {rank}, Local {local_rank}] Connecting to scheduler at {scheduler_address}")
                self.client = Client(scheduler_address, timeout="60s")
        else:
            # Single-node setup: use LocalCluster
            n_workers = min(n_workers, cpus_per_node // threads_per_worker)
            log.info(f"Creating single-node Dask cluster with {n_workers} workers, {threads_per_worker} threads each")

            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
                processes=True,
                dashboard_address=f":{cfg['dashboard_port']}",
            )
            self.client = Client(cluster)

        # Configure dask for better performance
        dask.config.set(
            {
                "distributed.scheduler.work-stealing": False,  # Disable for large chunks
                "distributed.worker.memory.target": 0.85,
                "distributed.worker.memory.spill": 0.9,
                "distributed.worker.memory.pause": 0.95,
            }
        )
        if "scheduler" in cfg:
            print(f"Setting Dask scheduler to: {cfg['scheduler']}")
            dask.config.set(scheduler=cfg["scheduler"])

        if cfg["optimize_lustre"]:
            # Lustre-specific optimizations
            dask.config.set(
                {
                    "distributed.worker.memory.rebalance.measure": "optimistic",
                    "distributed.comm.timeouts.connect": "60s",
                    "distributed.comm.timeouts.tcp": "60s",
                }
            )

        log.info(f"Dask cluster ready. Dashboard at: {self.client.dashboard_link}")

    def _open_dataset_start(self, zarr_path, time_slice: slice) -> (xr.Dataset, int):
        # Open the dataset with xarray to get the time coordinate and compute the time offset for TensorStore
        try:
            ds = open_zarr_dataset(zarr_path, decode_times=True, chunks=None, mask_and_scale=False, consolidated=True)
        except Exception as e:
            raise RuntimeError(f"Could not open zarr dataset: {zarr_path}") from e

        # Compute integer time offset for TensorStore
        # Find the index of time_slice.start in the original time coordinate
        time_start_idx = int((ds["time"] == time_slice.start).argmax().values)
        time_subsample_here = extract_time_subsample(zarr_path, self.hparams.hourly_resolution)
        time_slice_here = slice(time_slice.start, time_slice.stop, time_subsample_here)
        ds = ds.sel(time=time_slice_here)

        # Crop spatially if needed
        if self.spatial_crop_inputs is not None:
            log.info(f"Applying spatial crop to inputs: {self.spatial_crop_inputs}")
            ds = ds.sel(**self.spatial_crop_inputs)
            log.info(f"New shape after spatial crop: {ds.dims}")

        # Assert that latitude is increasing order
        if not (np.diff(ds.latitude.data) > 0).all():
            ds = ds.reindex(latitude=list(reversed(ds.latitude.data)))
            assert (np.diff(ds.latitude.data) > 0).all(), "Latitude is not in increasing order after reindexing."
            log.info(f"Reindexed latitude to be increasing. Now: {ds.latitude.data[:5]} ... {ds.latitude.data[-5:]}")
        return ds, time_start_idx

    @property
    def lat_lon_dims(self):
        if self.hparams.lat_lon_format == "lat_lon":
            return "latitude", "longitude"
        else:
            return "longitude", "latitude"

    def get_split_dataset(self, split: str, time_slice: slice, zarr_path=None, **kwargs) -> ERA5DatasetBase:
        assert split in ["fit", "train", "validate", "val", "test", "predict"], f"Invalid split: {split}"
        zarr_path = zarr_path or self.zarr_path

        # Setup dask cluster if needed
        if self.hparams.use_dask:
            self._setup_dask_cluster()

        ds, time_start_idx = self._open_dataset_start(zarr_path, time_slice)
        ds_target = None
        if self.hparams.target_dataset is not None and self.hparams.target_dataset != self.hparams.dataset:
            target_path = self.hparams.target_dataset
            if not os.path.exists(target_path):
                target_path = find_path_from_dir_opts(self.hparams.data_dir, target_path)
            ds_target, _ = self._open_dataset_start(target_path, time_slice)
            # Check that order of spatial dims (lat, lon) vs (lon, lat) is the same in ds and ds_target
            ds_spatial_dims = [dim for dim in ds.dims if dim in ["latitude", "longitude", "lat", "lon"]]
            ds_target_spatial_dims = [dim for dim in ds_target.dims if dim in ["latitude", "longitude", "lat", "lon"]]
            if ds_spatial_dims != ds_target_spatial_dims:
                ds_target = ds_target.transpose(..., *ds_spatial_dims)

        kwargs["zarr_path"] = zarr_path
        kwargs["target_dataset"] = ds_target
        kwargs["time_offset"] = time_start_idx  # Integer offset for TensorStore to match xarray's pre-sliced dataset
        kwargs["window"] = self.hparams.window
        kwargs["static_fields"] = self.hparams.static_fields
        kwargs["forcing_fields"] = self.hparams.forcing_fields
        kwargs["use_dask"] = self.hparams.use_dask
        kwargs["load_type"] = self.hparams.load_type
        kwargs["hourly_resolution"] = self.hparams.hourly_resolution
        kwargs["spatial_crop_outputs"] = self.spatial_crop_outputs
        kwargs["output_mask_area"] = self.hparams.output_mask_area
        kwargs["text_skip_missing_dates"] = self.hparams.text_skip_missing_dates
        if split in ["fit", "train"]:
            kwargs["possible_initial_times"] = self.hparams.possible_initial_times
            # Set loss weights
            kwargs["loss_latitude_weighting"] = self.hparams.loss_latitude_weighting
            kwargs["loss_pressure_weighting"] = self.hparams.loss_pressure_weighting
            kwargs["loss_pressure_weighting_levels"] = self.hparams.loss_pressure_weighting_levels
            kwargs["loss_pressure_weighting_divide_by"] = self.hparams.loss_pressure_weighting_divide_by
            kwargs["loss_surface_vars_weighting"] = self.hparams.loss_surface_vars_weighting
            kwargs["loss_multipliers"] = self.hparams.loss_multipliers
            kwargs["spatial_crop_during_training"] = self.hparams.spatial_crop_during_training
        else:
            kwargs["possible_initial_times"] = self.hparams.possible_initial_times_eval

        kwargs["lat_lon_format"] = self.lat_lon_dims

        dset = self._get_split_dataset(ds, split, **kwargs)

        if self._latitude is None:
            self._latitude = ds.latitude  # dset.dataset.latitude
            self._longitude = ds.longitude
        split_id = split
        if "dataloader_idx" in kwargs and kwargs["dataloader_idx"] > 0:
            split_id += f"_{kwargs['dataloader_idx']}"
        self._split_to_time[split_id] = ds.time
        return dset

    @abstractmethod
    def _get_split_dataset(
        self, ds, split: str, time_slice: slice, dataloader_idx: int = 0, **kwargs
    ) -> ERA5DatasetBase:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        # Set the correct tensor datasets for the train, val, and test sets
        ds_splits = dict()
        if stage in ["fit", "validate", None]:
            if self.hparams.training_datasets is not None:
                train_sets = []
                for train_set in self.hparams.training_datasets:
                    for train_slice in self.train_slice:
                        train_set = find_path_from_dir_opts(self.hparams.data_dir, train_set)
                        train_set_ds = self.get_split_dataset("fit", train_slice, zarr_path=train_set)
                        log.info(f"Training dataset {train_set=}, {train_slice=} with {len(train_set_ds)=} samples.")
                        train_sets += [train_set_ds]

                ds_splits["train"] = torch.utils.data.ConcatDataset(train_sets)
                log.info(
                    f"Total training samples from {len(self.hparams.training_datasets)} datasets: {len(ds_splits['train'])}"
                )
            else:
                train_sets = []
                for train_slice in self.train_slice:
                    train_set_ds = self.get_split_dataset("fit", train_slice)
                    log.info(f"Training dataset slice {train_slice} with {len(train_set_ds)} samples loaded.")
                    train_sets += [train_set_ds]
                if len(train_sets) == 1:
                    ds_splits["train"] = train_sets[0]
                else:
                    ds_splits["train"] = torch.utils.data.ConcatDataset(train_sets)
                    log.info(f"Total training samples from {len(self.train_slice)} slices: {len(ds_splits['train'])}")

            val_kwargs = dict(split="validate", time_slice=self.val_slice, subsample=self.hparams.subsample_valid)
            ds_splits["val"] = [self.get_split_dataset(**val_kwargs, max_num_samples=self.hparams.max_val_samples)]
            if self.get_horizon("val", dataloader_idx=1) is not None:
                log.info(f"Using long inference horizon={self.get_horizon('val', dataloader_idx=1)} for validation")
                ds_splits["val"] += [self.get_split_dataset(**val_kwargs, max_num_samples=16, dataloader_idx=1)]

        if stage in ["test", None]:
            ds_splits["test"] = self.get_split_dataset("test", self.test_slice)
        if stage == "predict":
            ds_splits["predict"] = self.get_split_dataset("predict", self.predict_slice)

        for split, split_ds in ds_splits.items():
            if split_ds is None:
                continue
            # Save the tensor dataset to self._data_{split}
            setattr(self, f"_data_{split}", split_ds)
            assert getattr(self, f"_data_{split}") is not None, f"Could not create {split} dataset"

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)

    @property
    def validation_set_names(self) -> List[str]:
        return ["val", "inference"] if len(self._data_val) > 1 else ["val"]

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
    ) -> Dict[str, OneStepAggregator]:
        assert dataloader_idx in [0, 1], f"Invalid dataloader_idx: {dataloader_idx}"
        split_ds = getattr(self, f"_data_{split}")
        if split == "val" and isinstance(split_ds, list):
            split_ds = split_ds[0]  # just need it for the area weights

        hourly_res = self.hparams.hourly_resolution
        if "interpolation" in experiment_type.lower():
            split_horizon = self.hparams.horizon
            horizon_range = range(1, split_horizon)
        else:
            split_horizon = self.get_horizon(split, dataloader_idx)
            horizon_range = range(1, split_horizon + 1)

        aggregators_all = defaultdict(list)
        area_weights = to_torch_and_device(split_ds.area_weights_tensor, device)
        aggr_kwargs = dict(area_weights=area_weights, is_ensemble=is_ensemble)
        coords = {"longitude": self._longitude, "latitude": self._latitude}
        record_normed = self.hparams.log_normed and split_horizon <= 80  # save logging space for huge horizons
        record_abs_values = self.hparams.log_abs_values and split_horizon <= 80
        record_rmse = True
        if split_ds.mask is not None:
            masks = [None, split_ds.mask]
            assert self.crop_name != "", f"Please give your crop a name. Current crop: {self.spatial_crop_outputs}"
            mask_names = ["", f"{self.crop_name}/"]
        else:
            masks = [None]
            mask_names = [""]
        if record_rmse and verbose:
            log.info(f"Recording normed metrics for {split=}, {dataloader_idx=}, {split_horizon=}")
        if self.hparams.eval_resolution is not None:
            lat_res, lon_res = self.hparams.eval_resolution
            if f"{lat_res}x{lon_res}" in self.zarr_path or f"{lon_res}x{lat_res}" in self.zarr_path:
                log.info(
                    f"Evaluation resolution {lat_res}x{lon_res} matches dataset resolution. No regridding needed."
                )
            else:
                latitude_spacing = "with_poles" if len(self._latitude) % 2 == 1 else "without_poles"
                lon_scheme = (
                    "start_at_0"
                    if self._longitude[0] == 0
                    else "centered_at_0" if self._longitude[0] == -180 else None
                )
                dest_lat = regrid.latitude_values(lat_res, latitude_spacing=latitude_spacing)
                dest_lon = regrid.longitude_values(lon_res, longitude_scheme=lon_scheme)
                src_coords = coords
                coords = {"latitude": dest_lat, "longitude": dest_lon}
                to_regrid = ["preds", "gen_"]
                if self.hparams.target_dataset is None:
                    to_regrid += ["target_"]
                aggr_kwargs["preprocess_fn"] = regrid.ConservativeRegridder(
                    coords_source=src_coords,
                    coords_dest=coords,
                    which_substrings=to_regrid,
                )
                aggr_kwargs["area_weights"] = aggr_kwargs["preprocess_fn"](area_weights)

        for mask, mask_name in zip(masks, mask_names):
            aggr_kwargs["mask"] = mask

            snapshot_horizons_hours = [1 * hourly_res, 24, 5 * 24, 10 * 24, split_horizon * hourly_res]
            if not self.hparams.log_images or dataloader_idx != 1:
                use_snapshot_aggregator = False
            elif split == "val" and dataloader_idx == 1:
                assert len(self._data_val) > 1, "Full rollout is only supported for inference"
                # Save some example snapshots from the full rollout
                use_snapshot_aggregator = True if mask is None else False
            else:
                use_snapshot_aggregator = mask is None
            spectra_horizons_hours = snapshot_horizons_hours + [12, 3 * 24, 7 * 24, 14 * 24]

            # name=f"t{h * hourly_res}" is used for logging the appropriate lead time regardless of the hourly_res or
            # temporal resolution/subsampling of the dataset
            snapshot_var_names = ["temperature_850", "2m_temperature"] # ["temperature_850", "2m_temperature", "10m_u_component_of_wind"]
            # snapshot_var_names += ["10m_v_component_of_wind", "mean_sea_level_pressure"]
            snapshot_var_names = [f"{vr}_normed" for vr in snapshot_var_names] + ["geopotential_500"]
            snapshot_kwargs = {
                "var_names": snapshot_var_names,
                "preprocess_fn": lambda x: np.moveaxis(np.flip(x, axis=-1), -1, -2),
                "every_nth_epoch": self.hparams.every_nth_epoch_snapshot,
            }
            spectra_names = ["temperature_850", "2m_temperature"] #["2m_temperature", "10m_u_component_of_wind", "mean_sea_level_pressure"]
            spectra_levels = [50, 100, 500, 700, 850, 1000]
            spectra_names += [f"temperature_{lev}" for lev in spectra_levels]
            spectra_names += [f"geopotential_{lev}" for lev in spectra_levels]
            spectra_names += [f"u_component_of_wind_{lev}" for lev in spectra_levels]
            spectra_names += [f"v_component_of_wind_{lev}" for lev in spectra_levels]
            spectra_names += [f"specific_humidity_{lev}" for lev in spectra_levels]

            # NEW TODO: REMOVE

            # spectra_names = ["2m_temperature", "temperature_850", "geopotential_500", "total_column_water_vapour", "geopotential_250", "geopotential_300", "geopotential_700", "geopotential_1000"]

            
            spectra_kwargs = {
                "var_names": spectra_names,
                "spectra_type": "zonal_60_90",
                "spatial_dims": self.lat_lon_dims,
            }

            for h in horizon_range:
                h_to_hours = h * hourly_res
                record_spectra = (
                    self.hparams.log_spectra if h_to_hours in spectra_horizons_hours and mask is None else False
                )
                record_rmse_quantiles = (
                    (0.90, 0.95, 0.99)
                    if split == "val" and dataloader_idx == 0 and h == 1
                    else ()
                )
                aggregators_all[f"t{h}"].append(
                    OneStepAggregator(
                        use_snapshot_aggregator=use_snapshot_aggregator and h_to_hours in snapshot_horizons_hours,
                        name=f"{mask_name}t{h_to_hours}",
                        verbose=verbose and (h == 1),
                        record_metrics=self.hparams.log_metrics,
                        record_normed=record_normed,
                        record_rmse=record_rmse,
                        record_rmse_quantiles=record_rmse_quantiles,
                        record_abs_values=record_abs_values,
                        snapshot_kwargs=snapshot_kwargs,
                        # Preprocess the snapshots to flip latitudes and bring lats before lons for proper plotting
                        record_spectra=record_spectra,
                        spectra_kwargs=spectra_kwargs,
                        coords=coords,
                        **aggr_kwargs,
                    )
                )

        # Make it into list aggregators, so that it is easy to call the record_batch method
        for k, v in aggregators_all.items():
            name = f"t{int(k[1:]) * hourly_res}" if k.startswith("t") else None
            aggregators_all[k] = ListAggregator(v, verbose=False, name=name)

        if save_to_path is not None:
            # Specify ++module.save_predictions_filename="xarray" to save the predictions in xarray format
            aggregators_all["save_to_disk"] = MemEfficientSaveToDiskAggregator(
                final_dims_of_data=self.lat_lon_dims,
                is_ensemble=is_ensemble,
                preprocess_fn=aggr_kwargs.get("preprocess_fn", None),
                coords=coords,
                concat_dim_name="lead_time",
                batch_dim_name="datetime",
                save_to_path=save_to_path,
                max_ensemble_members=10,
                **kwargs,
            )

        return aggregators_all


class NaNCleaner(torch.nn.Module):
    def __init__(self, normalizer, mask, **vars_to_fill_values):
        super().__init__()
        self.normalizer = normalizer
        self.mask = to_torch_and_device(mask, None)  # Boolean mask where True indicates valid data
        assert len(vars_to_fill_values) > 0, "Please provide at least one variable to fill NaNs for."
        self.vars_to_fill_values = vars_to_fill_values

    def _apply(self, fn, recurse=True):
        self.normalizer._apply(fn, recurse=recurse)
        self.mask = fn(self.mask)

    def normalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for var, fill_value in self.vars_to_fill_values.items():
            tensor = tensors[var]
            tensors[var] = torch.where(self.mask, fill_value, tensor)
            # print(f"Num NaNs before: {torch.isnan(tensor).sum().item()} (and {torch.isnan(tensor[0,0,...]).sum().item()=}, after filling: {torch.isnan(tensors[var]).sum().item()}. {tensor.shape=}. Mask sum: {self.mask.sum().item()}, {self.mask.shape=}, {fill_value=}")
        return self.normalizer.normalize(tensors)

    def denormalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tensors = self.normalizer.denormalize(tensors)
        for var, fill_value in self.vars_to_fill_values.items():
            tensor = tensors[var]
            tensors[var] = torch.where(self.mask, fill_value, tensor)
        return tensors

    def normalized_to_residual_normalized(self, *args, **kwargs):
        return self.normalizer.normalized_to_residual_normalized(*args, **kwargs)

    def normalized_residual_to_normalized(self, *args, **kwargs):
        return self.normalizer.normalized_residual_to_normalized(*args, **kwargs)


class ERA5DataModule2D(ERA5DataModuleBase):
    def __init__(
        self,
        *args,
        input_vars: Sequence[str] = (
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "geopotential_500",
            "temperature_850",
        ),
        output_vars: Sequence[str] = (
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "geopotential_500",
            "temperature_850",
        ),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.all_vars = list(set(input_vars) | set(output_vars))
        # todo hard code skip res for output only vars
        self.normalizer = get_normalizer(
            self._normalizer_files["mean"],
            self._normalizer_files["std"],
            names=self.all_vars,
            is_2d_flattened=True,
            global_stds_res_path=self._normalizer_files["std_residual"],
            input_names=input_vars,
        )
        if "sea_surface_temperature" in self.all_vars:
            # Fill NaNs in sea_surface_temperature with 15 degC (normalized value will be computed inside NaNCleaner)
            fill_value_sst = xr.open_dataset(self._normalizer_files["min"])["sea_surface_temperature"].values.item()

            any_sst = open_zarr_dataset(self.zarr_path).isel(time=0)["sea_surface_temperature"]
            if not (np.diff(any_sst.latitude.data) > 0).all():  # Assert that latitude is increasing order
                any_sst = any_sst.reindex(latitude=list(reversed(any_sst.latitude.data)))
            if self.hparams.lat_lon_format == "lat_lon" and any_sst.dims[0] not in ["latitude", "lat"]:
                any_sst = any_sst.transpose("latitude", "longitude")
            elif self.hparams.lat_lon_format == "lon_lat" and any_sst.dims[0] not in ["longitude", "lon"]:
                any_sst = any_sst.transpose("longitude", "latitude")
            mask = np.isnan(any_sst.values)
            frac_masked = np.sum(mask) / mask.size
            log.info(
                f"Filling NaNs in sea_surface_temperature with {fill_value_sst} degC. Fraction of NaNs: {frac_masked:.4f}"
            )
            self.normalizer = NaNCleaner(
                self.normalizer,
                mask=mask,  # mask will be set inside the dataset
                sea_surface_temperature=fill_value_sst,
            )
        channel_axis = -3
        # ====== Don't do the following! Using set will change the order of the variables!!!!! ======
        # input_only_vars = set(input_vars) - set(output_vars)
        # in_vars_without_input_only = set(input_vars) - input_only_vars
        # ======================================================================================
        input_only_vars = [vari for vari in input_vars if vari not in output_vars]
        in_vars_without_input_only = [vari for vari in input_vars if vari not in input_only_vars]
        if len(input_only_vars) > 0:
            log.info(f"Input-only variables: {input_only_vars}")  # will be inputted with "dynamical_condition: key
        self.in_packer = Packer(in_vars_without_input_only, axis=channel_axis)
        self.in_only_packer = Packer(input_only_vars, axis=channel_axis) if len(input_only_vars) > 0 else None
        if self.spatial_crop_outputs is not None:
            channel_axis_unpack_outputs = channel_axis  # + 1  # lat and lon get flattened
        else:
            channel_axis_unpack_outputs = channel_axis
        self.out_packer = Packer(output_vars, axis_pack=channel_axis, axis_unpack=channel_axis_unpack_outputs)

    def _get_split_dataset(self, ds, split: str, dataloader_idx: int = 0, **kwargs) -> ERA5Dataset2D:
        dset = ERA5Dataset2D(
            dataset=ds,
            text_dataset=self.text_data,
            split=split,
            horizon=self.get_horizon(split, dataloader_idx),
            input_vars=self.hparams.input_vars,
            output_vars=self.hparams.output_vars,
            normalizer=self.normalizer,
            in_only_packer=self.in_only_packer,
            return_future_date_for_training=self.hparams.return_future_date_for_training,
            **kwargs,
        )
        return dset


class ERA5DatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: xr.Dataset,
        text_dataset: dict,
        split: str,
        horizon: int,
        static_fields: Sequence[str],
        zarr_path: str = None,
        target_dataset: Optional[xr.Dataset] = None,
        time_offset: int = 0,  # Time offset from train/val/test split for TensorStore
        forcing_fields: Sequence[str] = None,
        window: int = 1,
        hourly_resolution: int = 1,
        possible_initial_times: Optional[Sequence[str]] = None,
        spatial_crop_outputs: Optional[Dict[str, slice]] = None,
        output_mask_area: Optional[str] = None,
        spatial_crop_during_training: bool = False,
        loss_latitude_weighting: bool = False,
        max_num_samples: Optional[int] = None,
        subsample: int = 1,
        lat_lon_format: Tuple[str, str] = ("longitude", "latitude"),
        use_dask: bool = False,
        load_type: bool = "xarray",
        text_skip_missing_dates: bool = False,
        return_future_date_for_training: bool = False,
    ):
        self.dataset = dataset
        self.zarr_path = zarr_path
        self.target_dataset = target_dataset
        self.time_offset = time_offset  # Offset for TensorStore to match xarray's pre-sliced dataset
        self.subsample = subsample
        self.dataset_id = split
        self.horizon = horizon
        self.window = window
        self.last_ic_idx = window - 1
        self.max_num_samples = max_num_samples
        self.possible_initial_times = (
            [int(h) for h in possible_initial_times] if possible_initial_times is not None else None
        )
        all_times = self.dataset.time.values[
            self.last_ic_idx : -horizon
        ]  # keep only times for which we can predict horizon hours ahead
        ds_idxs = np.arange(len(all_times), dtype=int)
        if self.possible_initial_times is not None:
            all_hours = all_times.astype("datetime64[h]").astype(int) % 24
            # Create a mask for hours we want to keep as possible initial times
            valid_hours = np.isin(all_hours, self.possible_initial_times)
            ds_idxs = ds_idxs[valid_hours]

        if max_num_samples is not None:
            ds_idxs = subsample_preselected_indices(ds_idxs, max_num_samples)
        elif subsample > 1:
            raise ValueError("Please provide max_num_samples when subsampling is desired.")

        self.ds_idxs = ds_idxs
        self.length = len(ds_idxs)
        self.loss_latitude_weighting = loss_latitude_weighting
        self.use_dask = use_dask
        self.load_type = load_type
        self.lat_lon_format = lat_lon_format
        self.forcing_fields = forcing_fields or []
        self.text_skip_missing_dates = text_skip_missing_dates
        self.return_future_date_for_training = return_future_date_for_training
        # Get whether the dataset is formatted as lat/lon or lon/lat

        if self.length < 0:
            raise ValueError(
                f"Invalid length: {self.length} for split: {split}; len(self.dataset.time)={len(self.dataset.time)}, horizon: {horizon}, max_num_samples: {max_num_samples}"
            )

        dim_order = list(dataset.sizes.keys())
        self.requires_transpose = False
        if lat_lon_format == ("latitude", "longitude") and dim_order.index("latitude") > dim_order.index("longitude"):
            self.requires_transpose = True
        elif lat_lon_format == ("longitude", "latitude") and dim_order.index("latitude") < dim_order.index(
            "longitude"
        ):
            self.requires_transpose = True
        if self.requires_transpose:
            self.preproc_func = to_torch_tensor_and_transpose
        else:
            self.preproc_func = to_torch_tensor
        # Create static conditions
        if static_fields is not None and len(static_fields) > 0:
            static_conditions = []
            for i, static_field_name in enumerate(static_fields):
                if static_field_name in self.dataset.keys():
                    try:
                        static_field = self.dataset[static_field_name].compute().values
                    except AttributeError as e:
                        raise AttributeError(
                            f"Error when loading {static_field_name=}, {self.dataset[static_field_name]=}"
                        ) from e

                    assert np.all(np.isfinite(static_field)), f"Found NaNs in static_field: {static_field_name}"
                    static_conditions.append(static_field)
                elif static_field_name == "lat_lon_embeddings":
                    # Create lat/lon embeddings for each grid point
                    # Create lat x lon meshgrid
                    lats, lons = np.meshgrid(self.dataset.latitude, self.dataset.longitude)
                    # xx, yy = 10, 88
                    # a1 = lats[xx, yy], lons[xx, yy]
                    # a2 = self.dataset.isel(latitude=yy, longitude=xx)
                    # a2 = a2.latitude.values, a2.longitude.values
                    # assert np.allclose(a1, a2), f"lats[yy, xx], lons[yy, xx]: {a1}, a2: {a2}"
                    x = np.cos(lats) * np.cos(lons)
                    y = np.cos(lats) * np.sin(lons)
                    z = np.sin(lats)
                    # Check that no NaNs are present
                    assert np.all(np.isfinite(x)), f"Found NaNs in x: {x}"
                    assert np.all(np.isfinite(y)), f"Found NaNs in y: {y}"
                    assert np.all(np.isfinite(z)), f"Found NaNs in z: {z}"
                    if lat_lon_format == ("latitude", "longitude"):
                        x, y, z = x.T, y.T, z.T
                    # log.info(f"lat_lon_format: {lat_lon_format}, x.shape: {x.shape}, y.shape: {y.shape}, z.shape: {z.shape},"
                    #       f" last_static_field: {static_conditions[-1].shape}, dims: {self.dataset.dims}")
                    static_conditions += [x, y, z]
                else:
                    raise ValueError(f"Invalid static_field: {static_field_name}")

            # Stack static conditions along channel dimension
            num_static_fields = len(static_conditions)
            static_conditions = np.stack(static_conditions, axis=0)
            # Standardize static conditions across field dimension (i.e. use different mean/std for each static field)
            mean_sc = static_conditions.mean(axis=(-2, -1), keepdims=True)
            std_sc = static_conditions.std(axis=(-2, -1), keepdims=True)
            assert len(mean_sc) == len(std_sc) == num_static_fields
            static_conditions = (static_conditions - mean_sc) / std_sc
            self.static_conditions = torch.from_numpy(static_conditions).float()
        else:
            self.static_conditions = None
        self._area_weights = get_lat_weights(self.dataset)
        nlon = self.dataset.longitude.size
        repeat_shape = (nlon, 1)
        self._area_weights_tensor = torch.as_tensor(self.area_weights.values, dtype=torch.float32).repeat(repeat_shape)
        if lat_lon_format != ("longitude", "latitude"):
            self._area_weights_tensor = self._area_weights_tensor.T

        if spatial_crop_outputs is not None:
            # Compute output mask for spatial cropping of tensors
            for k, v in spatial_crop_outputs.items():
                assert k in ["latitude", "longitude"], f"Invalid spatial_crop_outputs key: {k}"
            # Get points within the spatial crop slices (e.g. latitude=slice(10, 20), longitude=slice(20, 30))
            lat_slice = spatial_crop_outputs.get("latitude", slice(None))
            lon_slice = spatial_crop_outputs.get("longitude", slice(None))
            mask = (
                (lat_slice.start <= self.dataset.latitude)
                & (self.dataset.latitude <= lat_slice.stop)
                & (lon_slice.start <= self.dataset.longitude)
                & (self.dataset.longitude <= lon_slice.stop)
            )

            if output_mask_area == "land":
                mask = mask & (self.dataset.land_sea_mask > 0.5)
            elif output_mask_area == "sea":
                mask = mask & (self.dataset.land_sea_mask < 0.5)
            elif output_mask_area is not None:
                raise ValueError(f"Invalid output_mask_area: {output_mask_area}")
            mask = mask.transpose(*lat_lon_format)
            self.mask = torch.from_numpy(mask.values)  # .nonzero(as_tuple=True)
            if split in ["train", "fit"] and spatial_crop_during_training is True:
                # Adjust area weights to fit the mask
                self.return_mask = self.mask
                self._area_weights_tensor = self._area_weights_tensor[self.mask]
            elif split in ["train", "fit"] and spatial_crop_during_training not in [False, True, None]:
                # Upweight the area weights for the masked area, but don't use the mask for cropping
                log.info(f"Upweighting area weights for the masked area by {spatial_crop_during_training}")
                self._area_weights_tensor[self.mask] *= spatial_crop_during_training
                self.return_mask = None
            else:
                # For othes (eval) splits, mask is only used inside aggregators (with special prefix(es))
                self.return_mask = None
                # log.info(f"Output mask won't be returned for split: ``{split}``")
            # self._area_weights_tensor_masked = self._area_weights_tensor[self.mask]
        else:
            assert output_mask_area is None, "output_mask_area is only supported when spatial_crop_outputs is not None"
            self.mask = self.return_mask = self._area_weights_tensor_masked = None
        self.text_dataset = text_dataset
        if self.text_dataset is not None:
            self._text_dim = len(next(iter(self.text_dataset.values())))

    @property
    def area_weights(self):
        return self._area_weights

    @property
    def area_weights_tensor(self):
        return self._area_weights_tensor

    @property
    def loss_weights_tensor(self) -> Optional[torch.Tensor]:
        weights = None
        if self.loss_latitude_weighting:
            weights = self.area_weights_tensor
        return weights

    def __len__(self):
        return self.length


class ERA5Dataset2D(ERA5DatasetBase):
    def __init__(
        self,
        *args,
        input_vars: Sequence[str],
        output_vars: Sequence[str],
        normalizer,
        in_only_packer: Packer = None,
        loss_pressure_weighting: bool = False,
        loss_pressure_weighting_levels: Union[str, List[int]] = "era5",  # can be "era5", "wb", or a list of levels
        loss_pressure_weighting_divide_by: str = "mean",  # can be "mean" or "sum"
        loss_surface_vars_weighting: Optional[str] = None,
        loss_multipliers: Optional[Dict[str, float]] = None,
        preselect_vars: bool = False,  # True,
        preprocess_to_tensor: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.normalizer = copy.copy(normalizer)
        self.normalizer.to("cpu")
        if loss_pressure_weighting is True:
            loss_pressure_weighting = "graphcast"
        self.loss_pressure_weighting = loss_pressure_weighting
        self.loss_pressure_weighting_levels = loss_pressure_weighting_levels
        self.loss_pressure_weighting_divide_by = loss_pressure_weighting_divide_by
        self.loss_surface_vars_weighting = loss_surface_vars_weighting
        self.loss_multipliers = loss_multipliers
        self.preprocess_to_tensor = preprocess_to_tensor
        # Need to flatten the 3D variables to 2D (by stacking the pressure levels)
        self.all_vars = set(input_vars) | set(output_vars)
        self.input_only_vars = set(input_vars) - set(output_vars)
        self.in_only_packer = in_only_packer
        self.skip_idxs = set()
        self.possible_2d_vars = [
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "sea_surface_temperature",
            "mean_top_downward_short_wave_radiation_flux",
            "total_column_water_vapour",
        ]
        possible_3d_vars = [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            # "z", "q", "t", "u", "v", "w
        ]
        self.vars2d = list(sorted([v for v in self.all_vars if v in self.possible_2d_vars]))
        vars_not_2d = [v for v in self.all_vars if v not in self.possible_2d_vars]
        self.var3d_to_levels = defaultdict(list)
        for v in vars_not_2d:
            var_name = "_".join(v.split("_")[:-1])
            if var_name in possible_3d_vars:
                p_level = int(v.split("_")[-1])
                self.var3d_to_levels[var_name].append(p_level)
            else:
                raise ValueError(f"Invalid variable: {v}")
        # Find all unique levels and filter them out
        # and sort the levels, such that higher levels come first (using the integer value)
        self.all_levels = set()
        for v in self.var3d_to_levels:
            self.var3d_to_levels[v] = sorted(self.var3d_to_levels[v], reverse=True)
            self.all_levels.update(self.var3d_to_levels[v])
        self.all_levels = sorted(self.all_levels)

        self.all_vars_stem = sorted(set(self.var3d_to_levels.keys()) | set(self.vars2d))
        self.dataset = self.dataset[self.all_vars_stem]
        self.dataset = self.dataset.sel(level=self.all_levels)
        ds = self.dataset
        # ds = ds.sel(level=self.all_levels).load() # makes faster but can get killed
        input_dims = dict(  # input_dims for the ML model, or the __getitem__ method below
            time=self.window + self.horizon,
            latitude=len(ds.latitude),
            longitude=len(ds.longitude),
        )
        if "level" in ds.dims and len(ds.level) >= 1:
            input_dims["level"] = len(ds.coords["level"])
        self.input_dims = input_dims
        self.bgen = None
        self.level_to_idx = {int(lvl): i for i, lvl in enumerate(ds.level.values)} if "level" in ds.sizes else None
        self._3d_ops = []
        for vr, levels in self.var3d_to_levels.items():
            ops = []
            for level in levels:
                idx = self.level_to_idx[int(level)]
                key = f"{vr}_{level}"
                ops.append((idx, key))
            self._3d_ops.append((vr, ops))

        if self.load_type == "xarray":
            pass
        elif self.load_type == "xbatcher":
            self.bgen = xbatcher.BatchGenerator(
                ds,
                input_dims=input_dims,
                preload_batch=False,
                input_overlap={"time": self.window + self.horizon - 1},
                # batch_dims={"time": 1},
            )
        elif self.load_type == "tstore":
            self.coords_time = self.dataset["time"][:].astype("datetime64[ns]").values
            self.coords_lon = self.dataset["longitude"][:].values
            self.coords_lat = self.dataset["latitude"][:].values  # We need the actual values now, not just size
            self.lat_size = self.coords_lat.size
            # 1. Open Stores Async (One store per parent variable)
            self.ts_specs = {}
            for parent in self.all_vars_stem:
                self.ts_specs[parent] = {
                    "driver": "zarr",
                    "kvstore": {"driver": "file", "path": os.path.join(self.zarr_path, parent)},
                    # OPTIMIZATION: Cache metadata in RAM to avoid re-reading .zarray files
                    # 'context': {
                    #     'cache_pool': {'total_bytes_limit': 100_000_000},  # 100MB cache
                    #     'data_copy_concurrency': {'limit': 8},  # Limit threads per store, 4 better than 8
                    #     'file_io_concurrency': {'limit': 8},
                    # }
                }

            # 2. Get Level Indices Map (so we slice integers, not search floats)
            # IMPORTANT: Read from original zarr file, not filtered self.dataset
            zarr_group = zarr.open(self.zarr_path, mode="r")
            z_levels_original = zarr_group["level"][:]
            self.level_to_idx = {int(lvl): i for i, lvl in enumerate(z_levels_original)}
            # 3. Pre-calculate "What to read" for 3D vars
            # Maps 'temperature_850' -> ('temperature', 3)
            self.ts_mapping_3d = {}
            for var, levels in self.var3d_to_levels.items():
                for level in levels:
                    self.ts_mapping_3d[f"{var}_{level}"] = (var, self.level_to_idx[int(level)])
        else:
            raise ValueError(f"Invalid load_type: {self.load_type}")

        if "val" in self.dataset_id:
            # log.info(f"Subsampling the dataset by a factor of {self.subsample}")
            # Print the 1,2,-2,-1 date indices to check if the subsampling is correct
            dates = [self.__get_date__(i) for i in [0, 1, 2, 3, -3, -2, -1] if i < self.__len__()]
            # dates = [self.__get_date__(i) for i in range(self.__len__())]
            # Dates are np.datetime64 objects, let's print only year, month, day, hour
            dates = [str(d) for d in dates]
            if self.max_num_samples is not None:
                log.info(f"Using {self.max_num_samples=} for split: `{self.dataset_id}`.\nDates examples: {dates}")
            else:
                log.info(f"Using {self.subsample=} for split: `{self.dataset_id}`.\nDates examples: {dates}")

    def _lazy_init_stores(self):
        """
        Called once per worker process to open connections.
        """
        if hasattr(self, "ts_stores"):
            return
        # OPTIMIZATION: Create a shared context for this worker
        # This prevents thread explosion when opening 82 variables
        # ctx = ts.Context({
        #     'file_io_concurrency': {'limit': 8},  # Global limit for this worker
        #     'data_copy_concurrency': {'limit': 8},
        # })
        self.ts_stores = {}
        # Open the actual connections now that we are safely inside the worker
        for parent, spec in self.ts_specs.items():
            self.ts_stores[parent] = ts.open(spec).result()

    @property
    def loss_weights_tensor(self) -> Optional[torch.Tensor]:
        weights = super().loss_weights_tensor  # may be None or cos(lat) weights
        if self.loss_pressure_weighting is not None or self.loss_surface_vars_weighting is not None:
            if self.loss_pressure_weighting in [False, None] or self.loss_pressure_weighting_levels is None:
                # Throw error if one is set but not the other
                log.warning(
                    f"Unexpected loss_pressure_weighting: {self.loss_pressure_weighting}, "
                    f"loss_pressure_weighting_levels: {self.loss_pressure_weighting_levels}"
                    f"\nIt is expected to set both or neither. Are you sure you want to set one but not the other?"
                )

            var_to_weight = torch.ones(len(self.output_vars))
            if self.loss_pressure_weighting is not None:
                if self.loss_pressure_weighting == "graphcast":
                    log.info("Applying GraphCast-like pressure weighting to the loss")
                    # all_levels = self.all_levels   # Actually used levels
                    if self.loss_pressure_weighting_levels == "era5":
                        # Not sure if results are sensitive at all to which levels are used here for computing the
                        # weighting normalization. Probably not, but it's worth checking.
                        all_levels = (
                            1,
                            2,
                            3,
                            5,
                            7,
                            10,
                            20,
                            30,
                            50,
                            70,
                            100,
                            125,
                            150,
                            175,
                            200,
                            225,
                            250,
                            300,
                            350,
                            400,
                            450,
                            500,
                            550,
                            600,
                            650,
                            700,
                            750,
                            775,
                            800,
                            825,
                            850,
                            875,
                            900,
                            925,
                            950,
                            975,
                            1000,
                        )  # ERA5 levels
                    elif self.loss_pressure_weighting_levels == "wb":
                        all_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
                    else:
                        assert isinstance(self.loss_pressure_weighting_levels, list)
                        all_levels = self.loss_pressure_weighting_levels

                    if self.loss_pressure_weighting_divide_by == "mean":
                        level_div = np.mean(all_levels)  # Graphcast code uses this
                    elif self.loss_pressure_weighting_divide_by == "sum":
                        level_div = np.sum(all_levels)  # Graphcast paper says this
                    else:
                        raise ValueError(
                            f"Invalid loss_pressure_weighting_divide_by: {self.loss_pressure_weighting_divide_by}"
                        )
                elif self.loss_pressure_weighting == "makani":
                    log.info("Applying Makani-like pressure weighting to the loss")
                    pass
                else:
                    raise ValueError(f"Invalid loss_pressure_weighting: {self.loss_pressure_weighting}")
                for i, ov in enumerate(self.output_vars):
                    if ov in self.possible_2d_vars:
                        # Keep the weight as 1 for 2D variables
                        continue
                    p_level = int(ov.split("_")[-1])
                    # Weight the pressure levels such that higher levels are weighted more
                    if self.loss_pressure_weighting == "graphcast":
                        var_to_weight[i] = p_level / level_div
                    elif self.loss_pressure_weighting == "makani":
                        var_to_weight[i] = 0.001 * p_level
                    else:
                        assert False, f"Invalid loss_pressure_weighting: {self.loss_pressure_weighting}"
                    # Github copilot suggested the ones below
                    # var_to_weight[i] = 1 / (1 + np.abs(p_level - level_mean))
                    # var_to_weight[i] = 1 - np.abs(p_level - level_mean) / level_mean

            if self.loss_surface_vars_weighting is not None:
                log.info(f"Applying surface variable weighting ``{self.loss_surface_vars_weighting}`` to the loss")
                # Weight the surface variables differently
                if self.loss_surface_vars_weighting == "graphcast":
                    fixed_var_weights = {
                        # Any variables not specified here are weighted as 1.0 (or with pressure weighting)
                        # A single-level variable, but an important headline variable
                        # and also one which we have struggled to get good performance
                        # on at short lead times, so leaving it weighted at 1.0, equal
                        # to the multi-level variables:
                        "2m_temperature": 1.0,
                        # New single-level variables, which we don't weight too highly
                        # to avoid hurting performance on other variables.
                        "10m_u_component_of_wind": 0.1,
                        "10m_v_component_of_wind": 0.1,
                        "mean_sea_level_pressure": 0.1,
                        "total_precipitation_6hr": 0.1,
                        "total_precipitation_12hr": 0.1,
                        "sea_surface_temperature": 0.1,
                    }
                elif self.loss_surface_vars_weighting == "exp":
                    fixed_var_weights = {
                        "2m_temperature": 1.0,
                        "10m_u_component_of_wind": 0.1,
                        "10m_v_component_of_wind": 0.1,
                        "mean_sea_level_pressure": 1.0,
                        "total_precipitation_6hr": 0.1,
                        "total_precipitation_12hr": 0.1,
                        "sea_surface_temperature": 0.5,
                    }
                else:
                    raise ValueError(f"Invalid loss_surface_vars_weighting: {self.loss_surface_vars_weighting}")

                for i, ov in enumerate(self.output_vars):
                    if ov in fixed_var_weights.keys():
                        assert var_to_weight[i] == 1.0, f"var_to_weight[{i}]: {var_to_weight[i]}"
                        var_to_weight[i] = fixed_var_weights[ov]
                # print loss weights
                # print({v: float(var_to_weight[i]) for i, v in enumerate(self.output_vars)})

            if self.loss_multipliers is not None:
                for i, ov in enumerate(self.output_vars):
                    if ov in self.possible_2d_vars:
                        var_name_stem = ov
                    else:
                        var_name_stem = "_".join(ov.split("_")[:-1])

                    if var_name_stem in self.loss_multipliers:
                        multiplier = self.loss_multipliers[var_name_stem]
                        assert multiplier > 0, f"Invalid multiplier: {multiplier} for variable: {ov}"
                        if multiplier == 1:
                            continue  # no need to log
                        log.info(f"Applying loss multiplier of {multiplier} to variable {ov}")
                        var_to_weight[i] = var_to_weight[i] * multiplier

            n_spatial_dims = 2 if weights is None else len(weights.shape)
            # Create singleton dimensions for the spatial dimensions (after the variable dimension)
            var_to_weight = var_to_weight.view(len(self.output_vars), *([1] * n_spatial_dims))
            if self.loss_pressure_weighting == "makani":
                # Renormalize to 1
                var_to_weight = var_to_weight / var_to_weight.sum()

            if weights is None:
                weights = var_to_weight
            else:
                assert len(weights.shape) <= 2, f"weights.shape: {weights.shape}"
                # Weights shape is either (H, W) or (H*W). We need to create a (C, H, W) tensor using the var_to_weight
                # tensor, where C is the number of output variables
                weights = var_to_weight * weights.unsqueeze(0)

        return weights

    def __get_date__(self, idx):
        try:
            idx_actual = int(self.ds_idxs[idx])
        except IndexError:
            return None
        if self.bgen is None:
            # Direct xarray indexing
            batch_start_time = self.dataset.coords["time"].values[idx_actual]
        else:
            batch = self.bgen[idx_actual].load()
            batch_start_time = batch.coords["time"].values[self.last_ic_idx]
        return batch_start_time  # .astype("datetime64[D]")

    @log_timing(name="dataloader_getitem", log_every_n=50)
    def __getitem__(self, idx):
        # print(f"Getting item {idx}/{self.__len__()} for split: `{self.dataset_id}`")
        if idx in self.skip_idxs:
            return self.__getitem__(idx + 1)
        idx_actual = int(self.ds_idxs[idx])
        time_slice = slice(idx_actual, idx_actual + self.window + self.horizon)

        # static conditions are time-independent variables such as land_sea_mask, altitude, etc.
        arrays = dict(static_condition=self.static_conditions) if self.static_conditions is not None else dict()
        # Output-only mask for training and evaluating on spatially cropped outputs
        if self.return_mask is not None:
            arrays["predictions_mask"] = self.return_mask

        if self.load_type == "tstore":
            assert (
                self.target_dataset is None
            ), "Target dataset loading is not yet implemented for TensorStore load_type"
            dynamics = {}
            self._lazy_init_stores()
            # Get time values from already-sliced dataset (no offset needed)
            time_vals = self.coords_time[time_slice]
            batch_start_time = time_vals[self.last_ic_idx]
            # 2. TensorStore Loading (The async part)
            # Apply offset when reading from original zarr to match xarray's pre-sliced dataset
            time_slice_offset = slice(time_slice.start + self.time_offset, time_slice.stop + self.time_offset)
            futures = {}
            for parent_var, store in self.ts_stores.items():
                futures[parent_var] = store[time_slice_offset].read()
            # 3. Compute Forcings (While waiting for Disk I/O)
            # We can do this CPU math while the GPU/Disk is busy fetching data
            if self.forcing_fields:
                forcings_data = compute_forcings_numpy(
                    time_vals=time_vals,
                    lon_vals=self.coords_lon,
                    lat_vals=self.coords_lat,
                    forcing_fields=self.forcing_fields,
                )
                arrays["dynamical_condition"] = torch.from_numpy(np.stack(list(forcings_data.values()), axis=1))

            # 4. Resolve Futures
            loaded_data = {k: self.preproc_func(f.result()) for k, f in futures.items()}

            # A. Add Disk Variables
            for vr in self.vars2d:
                dynamics[vr] = loaded_data[vr]

            for target_name, (parent_name, lvl_idx) in self.ts_mapping_3d.items():
                dynamics[target_name] = loaded_data[parent_name][:, lvl_idx, :, :]
        else:
            try:
                if self.load_type == "xarray":
                    # Direct xarray indexing without xbatcher
                    batch = self.dataset.isel(time=time_slice)
                    batch = batch.load()
                elif self.load_type == "xbatcher":
                    # Use xbatcher
                    batch = self.bgen[idx_actual]  # .load()
                    if self.use_dask:
                        batch = dask.compute(batch)[0].load()
                    else:
                        batch = batch.load()
            except OSError as e:
                new_idx = idx + 1
                log.warning(f"OSError: {e}. Trying to load a different batch {idx}->{new_idx}.")
                return self[new_idx]

            # You can access the time of the batch with batch.coords['time'], which is a DataArray of datetime64
            # To select the start time of the batch, you can use batch.coords['time'].values[0]
            batch_start_time = batch.coords["time"].values[self.last_ic_idx]  # e.g. 2020-01-01T00:00:00
            # if self.possible_initial_times is not None:
            #     batch_hour = batch_start_time.astype("datetime64[h]").astype("int") % 24
            #     if batch_hour not in self.possible_initial_times:
            #         raise ValueError(f"Invalid {batch_hour=} for {self.possible_initial_times=}")

            # log.info(f"idx: {idx}, batch.dims: {batch.dims}") #batch_time: {batch_time}")
            # arrays["dynamics"] = self.get_variables_ds(batch, preprocess_to_tensor=True, use_tqdm=False)
            dynamics = self._create_var_to_tensor_dict(batch)

            target_dynamics = None
            if self.target_dataset is not None:
                target_batch = self.target_dataset.isel(time=time_slice).load()
                assert np.all(
                    target_batch.time.values == batch.time.values
                ), f"{target_batch.time.values=}, {batch.time.values=}"
                target_dynamics = self._create_var_to_tensor_dict(target_batch)

            if len(self.forcing_fields) > 0:
                assert len(self.input_only_vars) == 0, "forcing_fields and input_only_vars cannot be used together"
                assert len(set(self.forcing_fields) & set(self.all_vars)) == 0
                add_derived_vars(batch)
                if set(self.forcing_fields) & {TISR}:
                    add_tisr_var(batch)

                dyn_arrays = []
                any_tensor = dynamics[next(iter(dynamics))]
                t, h, w = any_tensor.shape[-3], any_tensor.shape[-2], any_tensor.shape[-1]
                for vr in self.forcing_fields:
                    dyn_arr = torch.from_numpy(batch[vr].values)
                    if "day_progress" in vr:
                        # day_progress is (Time, lon) only
                        unsqueeze_dim = -1 if self.lat_lon_format == ("longitude", "latitude") else -2
                        dyn_arr = dyn_arr.unsqueeze(unsqueeze_dim)  # add lat dimension
                    elif "year_progress" in vr:
                        # year_progress is (Time, ) only
                        dyn_arr = dyn_arr.view(t, 1, 1)

                    dyn_arr = dyn_arr.expand(t, h, w)
                    dyn_arrays.append(dyn_arr)
                arrays["dynamical_condition"] = torch.stack(dyn_arrays, dim=1)

        if len(self.input_only_vars) > 0:
            # Return as separate key, "dynamical_condition"
            dynamical_condition = {vr: dynamics.pop(vr) for vr in self.input_only_vars}
            arrays["dynamical_condition"] = self.in_only_packer.pack(self.normalizer.normalize(dynamical_condition))

        arrays["dynamics"] = dynamics
        if target_dynamics is not None:
            arrays["target_dynamics"] = target_dynamics
        if self.dataset_id not in ["train", "fit"]:
            # np_datetime = batch_start_time.astype("datetime64[s]")  # needs custom collate_fn
            np_datetime = batch_start_time.astype("datetime64[s]").astype(np.int64)
            # print(f"{idx=}, {batch_start_time=}, {np_datetime=}")
            arrays["metadata"] = dict(datetime=np_datetime)
            # idx: 0, batch_start_time: 2021-01-01T00:00:00.000000000, metadata: {'datetime': 1609459200}
            # idx: 1, batch_start_time: 2021-12-27T00:00:00.000000000, metadata: {'datetime': 1640563200}
        if self.text_dataset is not None:
            date = batch_start_time.astype("datetime64[D]")
            if date not in self.text_dataset:  # e.g.: KeyError: numpy.datetime64('2018-06-18')
                if self.text_skip_missing_dates:  # Skip missing dates
                    log.warning(
                        f"[{idx=}] Date {date} ({type(date)=}) not found in text dataset. {len(self.skip_idxs)=} Skipping."
                    )
                    self.skip_idxs.add(idx)
                    return self.__getitem__(idx + 1)
                arrays["condition_non_spatial"] = torch.zeros(self._text_dim)
                # may be replaced with null embeddings inside model
                # arrays["condition_non_spatial"] = None  # this leads to a collate fn error
                # raise KeyError(f"Date {date} not found in text dataset")

            elif self.return_future_date_for_training and self.dataset_id in ["train", "fit"]:
                # Return multiple text dates for training only (using them during eval would be cheating)
                arrays["condition_non_spatial"] = torch.stack(
                    [
                        self.text_dataset.get(date + np.timedelta64(h, "D"), torch.zeros(self._text_dim))
                        for h in range(self.horizon + 1)
                    ],
                    dim=0,
                )
            else:
                # log.info(f"Date {date} found in text dataset.")
                arrays["condition_non_spatial"] = self.text_dataset[date]
            # print(f"{self.dataset_id} idx: {idx}, batch_start_time: {batch_start_time}, text: {arrays['text']}")
        #
        # # Count number of NaNs in the arrays
        # for key, value in arrays.items():
        #     if isinstance(value, dict):
        #         for subkey, subvalue in value.items():
        #             if torch.isnan(subvalue).any():
        #                 num_nans = torch.isnan(subvalue).sum().item()
        #                 log.warning(f"Found {num_nans} NaNs in arrays['{key}']['{subkey}']")
        #     else:
        #         if torch.isnan(value).any():
        #             num_nans = torch.isnan(value).sum().item()
        #             log.warning(f"Found {num_nans} NaNs in arrays['{key}']")
        return arrays

    def _create_var_to_tensor_dict(self, batch):
        dynamics = dict()
        # Tensorfy all 2D variables
        for vr in self.vars2d:
            dynamics[vr] = self.preproc_func(batch[vr].values)  # (T, H, W)

        # Tensorfy all 3D variables
        for vr, ops in self._3d_ops:
            # This approach with 3d_ops saves 10-30ms per batch versus the commented out code below it
            full_data = self.preproc_func(batch[vr].values)
            for idx, key in ops:
                dynamics[key] = full_data[:, idx]

        # for vr, levels in self.var3d_to_levels.items():
        #     full_var_data = batch[vr].values # (T, L, H, W)
        #     for level in levels:
        #         idx = self.level_to_idx[level]
        #         dynamics[f"{vr}_{level}"] = self.preproc_func(full_var_data[:, idx, ...])
        return dynamics

    def get_item_and_speed_test(self, idx, add_pid: bool = False, verbose: bool = False):
        t0 = time.time()
        if verbose:
            print_json(
                {
                    "event": "get-batch start",
                    "time": t0,
                    "idx": idx,
                    "pid": multiprocessing.current_process().pid if add_pid else None,
                }
            )
        arrays = self[idx]
        t1 = time.time()
        if verbose:
            print_json(
                {
                    "event": "get-batch end",
                    "time": t1,
                    "idx": idx,
                    "pid": multiprocessing.current_process().pid if add_pid else None,
                    "duration": t1 - t0,
                }
            )
        # t1 - t0 is the time taken to get the batch (in seconds)
        return arrays, t1 - t0


def print_json(obj):
    print(json.dumps(obj))


def to_torch_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x)


def to_torch_tensor_and_transpose(x: np.ndarray) -> torch.Tensor:
    return transpose_hw(torch.from_numpy(x))


@torch.jit.script
def transpose_hw(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-1, -2)


if __name__ == "__main__":
    dm = ERA5DataModule2D(
        data_dir="gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr",
        # Put the statistic files and text data at the root of the repository in the /data directory
        data_dir_stats="../../data/stats/",  # change to your local path
        text_data_path="../../data/text/meteorological_all_with_date.csv",  # change to your local path
        predict_slice=slice("2020-12-01", "2020-12-31"),
        static_fields=(
            "land_sea_mask",
            "soil_type",
            "geopotential_at_surface",
            "lat_lon_embeddings",
        ),
        horizon=3,
        text_type="bert",
    )
    dm.setup(stage="fit")
    x = dm._data_train[0]
    for k, v in x.items():
        print(f"{k}: {v[list(v.keys())[0]].shape if isinstance(v, dict) else v.shape}")
