from __future__ import annotations

import glob
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.utilities.utils import find_config_differences_return_as_joined_str, get_logger


# Override this in your project
# -----------------------------------------------------------------------
_ENTITY = "kvats-university-of-california-san-diego"
PROJECT = "test"
_PROJECT_TRAIN = None  # Set this when using a different project for logging than for reloading checkpoints
_TRAINING_RUN_PATH = None  # Set this when using a different run for training than for reloading checkpoints
# -----------------------------------------------------------------------

log = get_logger(__name__)

CACHE = dict()


def get_entity(entity: str = None) -> str:
    if entity is None:
        return _ENTITY or wandb.api.default_entity
    return entity


def get_project_train(project: str = None) -> str:
    if project is None:
        return _PROJECT_TRAIN or PROJECT
    return project


def get_training_run_path():
    return _TRAINING_RUN_PATH


def get_api(wandb_api: wandb.Api = None, timeout=100) -> wandb.Api:
    if wandb_api is None:
        try:
            wandb_api = wandb.Api(timeout=timeout)
        except wandb.errors.UsageError:
            wandb.login()
            wandb_api = wandb.Api(timeout=timeout)
    return wandb_api


def get_api_and_set_entity(entity: str = None, wandb_api: wandb.Api = None) -> wandb.Api:
    entity = get_entity(entity)
    api = get_api(wandb_api)
    api._default_entity = entity
    return api


def get_run_api(
    run_id: str = None,
    entity: str = None,
    project: str = None,
    run_path: str = None,
    wandb_api: wandb.Api = None,
) -> wandb.apis.public.Run:
    entity, project = get_entity(entity), project or PROJECT
    assert run_path is None or run_id is None, "Either run_path or run_id must be None"
    assert run_id is None or isinstance(run_id, str), f"run_id must be a string, but is {type(run_id)}: {run_id}"
    run_path = run_path or f"{entity}/{project}/{run_id}"
    api = get_api_and_set_entity(entity, wandb_api)
    return api.run(run_path)


def get_project_runs(
    entity: str = None, project: str = None, wandb_api: wandb.Api = None, **kwargs
) -> List[wandb.apis.public.Run]:
    """Filter with kwarg: filters: Optional[Dict[str, Any]] = None,"""
    entity, project = get_entity(entity), project or PROJECT
    return get_api_and_set_entity(entity, wandb_api).runs(f"{entity}/{project}", **kwargs)


def get_project_groups(
    entity: str = None, project: str = None, wandb_api: wandb.Api = None
) -> List[wandb.apis.public.Run]:
    runs = get_project_runs(entity, project, wandb_api)
    return list(set([run.group for run in runs]))


def get_runs_for_filter(
    entity: str = None,
    project: str = None,
    group: str = None,
    wandb_api: wandb.Api = None,
    filter_dict: Dict[str, Any] = None,
    filter_functions: Sequence[Callable] = None,
    only_ids: bool = False,
    verbose: bool = True,
    **kwargs,
) -> Union[List[wandb.apis.public.Run], List[str]]:
    """Get all runs for a given group"""
    extra_filters = {"group": group} if group is not None else None
    filter_wandb_api = get_filter_for_wandb(filter_dict, extra_filters=extra_filters)
    group_runs = get_project_runs(entity, project, wandb_api, filters=filter_wandb_api, **kwargs)  # {"group": group})
    try:
        n_runs = len(group_runs)
    except ValueError as e:
        log.warning(f"Error when loading groups with {entity=} {project=} err={e}")
        n_runs = 0
    except requests.exceptions.HTTPError as e:
        log.warning(f"Error when loading groups with {entity=} {project=} err={e}")
        n_runs = -1
    if n_runs == 0:
        if verbose:
            pass
            # print(f"----> No runs for {group=}! Did you mistype the group name? Entity/project: {entity}/{project}")
    elif filter_functions is not None:
        n_groups_before = len(group_runs)
        filter_functions = [filter_functions] if callable(filter_functions) else list(filter_functions)
        group_runs = [run for run in group_runs if all([f(run) for f in filter_functions])]
        if len(group_runs) == 0 and len(filter_functions) > 0 and verbose:
            print(f"Filter functions filtered out all {n_groups_before} runs for group {group}")
        # elif n_groups_before == 0:
        #     print(f"----> No runs for group {group}!! Did you mistype the group name?")

    if only_ids:
        group_runs = [run.id for run in group_runs]
    return group_runs


def get_runs_for_project(**kwargs):
    return get_runs_for_filter(group=None, **kwargs)


def get_run_apis(
    run_id: str = None,
    group: str = None,
    **kwargs,
) -> List[wandb.apis.public.Run]:
    assert run_id is None or group is None, "Either run_id or group must be None"
    assert run_id is not None or group is not None, "Either run_id or group must be not None"
    assert run_id is None or isinstance(run_id, str), f"run_id must be a string, but is {type(run_id)}: {run_id}"
    assert group is None or isinstance(group, str), f"group must be a string, but is {type(group)}: {group}"
    if run_id is not None:
        return [get_run_api(run_id=run_id, **kwargs)]
    else:
        return get_runs_for_filter(group=group, **kwargs)


def get_wandb_id_for_run(wandb_config: Dict[str, Any]) -> str:
    """Get a unique id for the current run. If on a Slurm cluster, use the job ID, otherwise generate a random id."""
    # Check for PARALLEL_RUN_ID first (for parallel runs), then fall back to SLURM_JOB_ID
    if "PARALLEL_RUN_ID" in os.environ:
        return str(os.environ["PARALLEL_RUN_ID"])
    elif "SLURM_JOB_ID" in os.environ:
        # we are on a Slurm cluster... using the job ID helps when requeuing jobs to resume the same run
        maybe_id = maybe_id_base = str(os.environ["SLURM_JOB_ID"])
        # Check if the wandb ID already exists on wandb
        try:
            for trial in range(1000):
                maybe_id = f"{maybe_id_base}v{trial}" if trial > 0 else maybe_id_base
                _ = get_run_api(run_id=maybe_id, entity=wandb_config["entity"], project=wandb_config["project"])
            return wandb.sdk.lib.runid.generate_id()  # we've tried 1000 times, so just generate a random id
        except Exception:
            return maybe_id  # the run does not exist yet
    else:
        # we are not on a Slurm cluster, so just generate a random id
        return wandb.sdk.lib.runid.generate_id()


def get_runs_for_group_with_any_metric(
    wandb_group: str,
    options: List[str] | str,
    option_to_key: Callable[[str], str] | None = None,
    wandb_api=None,
    metric: str = "crps",
    **wandb_kwargs,
) -> (Optional[List[wandb.apis.public.Run]], str):
    """Get all runs for a given group that have any of the given metrics."""
    options = [options] if isinstance(options, str) else options
    option_to_key = option_to_key or (lambda x: x)
    wandb_kwargs2 = wandb_kwargs.copy()
    group_runs, any_metric_key = None, None
    tried_options = []
    for s_i, sum_metric in enumerate(options):
        any_metric_key = f"{option_to_key(sum_metric)}/{metric}".replace("//", "/")
        tried_options.append(any_metric_key)
        filter_func = has_summary_metric(any_metric_key)
        if "filter_functions" not in wandb_kwargs:
            wandb_kwargs2["filter_functions"] = filter_func
        elif "filter_functions" in wandb_kwargs and len(options) > 1:
            wandb_kwargs2["filter_functions"] = wandb_kwargs["filter_functions"] + [filter_func]
        else:
            wandb_kwargs2["filter_functions"] = wandb_kwargs["filter_functions"]
        group_runs = get_runs_for_filter(group=wandb_group, wandb_api=wandb_api, verbose=False, **wandb_kwargs2)
        if len(group_runs) > 0:
            break
    if len(group_runs) == 0:
        logging.warning(
            f"No runs found for group {wandb_group}. "
            f"Possible splits: {options}.\nFull keys that were tried: {tried_options}"
        )
        return None, None
    return group_runs, any_metric_key.replace(f"/{metric}", "")


def parse_metric_pattern(metric: str) -> tuple:
    """
    Parse a metric pattern to extract prefix, step (if any), and suffix.

    Handles both:
    - 'prefix/t$/suffix' pattern (with placeholder)
    - 'prefix/t123/suffix' pattern (with specific step)

    Returns:
        tuple: (prefix, step, suffix, metric_base_name)
              step will be None for placeholder patterns or an int for specific step patterns
    """
    # First try matching the placeholder pattern
    placeholder_pattern = r"^(.*)/t\$/(.*)$"
    match = re.fullmatch(placeholder_pattern, metric)

    if match:
        prefix, suffix = match.group(1), match.group(2)
        step = None  # Placeholder pattern doesn't have a specific step
    else:
        # Try matching the specific step pattern
        step_pattern = r"^(.*)/t(\d+)/(.*)$"
        match = re.fullmatch(step_pattern, metric)

        if match:
            prefix, step_str, suffix = match.group(1), match.group(2), match.group(3)
            try:
                step = int(step_str)
            except ValueError:
                return None, None, None, None
        else:
            return None, None, None, None

    # Construct the base metric name (without the timestep part)
    metric_base_name = "/".join([prefix, suffix]).strip("/")

    return prefix, step, suffix, metric_base_name


def get_history_data(run, metric_base_name: str, x_axis_name: str = None, verbose: bool = True) -> tuple:
    """
    Get history data for a metric from a run.

    Args:
        run: The run object
        metric_base_name: The base name of the metric (without timestep part)
        x_axis_name: The name of the x-axis (lead time or time)
        verbose: Whether to print warnings

    Returns:
        tuple: (history_dict, lead_time_name) where history_dict maps steps to values
    """
    # Determine the lead time column name
    x_axis_name_brev = x_axis_name.lower().replace("_", "").replace(" ", "") if x_axis_name else None
    if x_axis_name is None or "leadtime" in x_axis_name_brev or x_axis_name_brev in ["timestep", "timestamp"]:
        x_axis_name = "lead_time" if "lead_time" in run.summary.keys() else "time"
    elif x_axis_name_brev in ["wavenumber", "zonalwavenumber"]:
        x_axis_name = "wavenumber"
    elif x_axis_name_brev in ["wavelength", "zonalwavelength"]:
        x_axis_name = "wavelength"
    elif verbose:
        print(f"Warning: Using x_axis_name '{x_axis_name}'.")

    # Get the history data
    run_history = run.history(keys=[metric_base_name], pandas=True, x_axis=x_axis_name)

    # Create a lookup dictionary for fast access
    history_dict = {}

    if run_history.empty:
        if verbose:
            print(f"Warning: No history found for {run.id=} {run.name=}, {metric_base_name=}, {x_axis_name=}")
    else:
        # Convert to numeric to handle potential non-numeric values
        numeric_steps = pd.to_numeric(run_history[x_axis_name], errors="coerce")

        # Create a dictionary mapping step values to metric values
        for i, step in enumerate(numeric_steps):
            if pd.notna(step):  # Skip NaN steps
                history_dict[int(step)] = run_history[metric_base_name].iloc[i]

    return history_dict, x_axis_name


def get_run_metric_for_timesteps(
    run, metric_template: str, timesteps: Sequence[int], x_label_name=None, return_time=False
) -> np.ndarray:
    """
    Efficiently retrieve metrics for multiple timesteps from a run.

    Args:
        run: The run object containing metrics
        metric_template: The metric pattern with 't$' as placeholder for timestep
        timesteps: Sequence of timesteps to retrieve

    Returns:
        numpy array of metric values for each timestep
    """
    assert "t$" in metric_template, f"metric must contain 't$' pattern, but is {metric_template}"

    # Parse the metric pattern
    prefix, _, suffix, metric_base_name = parse_metric_pattern(metric_template)

    if metric_base_name is None:
        # Fall back to the old implementation if pattern doesn't match
        return np.array(
            [
                get_run_metric_from_summary_or_line_plot(
                    run, metric_template.replace("$", str(t)), x_label_name, verbose=(i == 0)
                )
                for i, t in enumerate(timesteps)
            ]
        )

    # Get all history data at once
    history_dict, x_label_name = get_history_data(run, metric_base_name, x_label_name)
    if timesteps is None:
        # If no specific timesteps are provided, use all available ones
        timesteps = np.array(list(history_dict.keys()))  # sorted(history_dict.keys())
    # Now efficiently retrieve values for all requested timesteps
    data = []
    for i, t in enumerate(timesteps):
        # First check if the timestep exists in our history dictionary
        if t in history_dict:
            data.append(history_dict[t])
        else:
            # If not in history, try to get it from summary
            metric_t = metric_template.replace("$", str(t))
            if metric_t in run.summary:
                data.append(run.summary[metric_t])
            else:
                # Use None if not found in either place
                if i == 0:  # Only print warning for first timestep to avoid spam
                    log.warning(f"{run.id=} No data found for {metric_t}")
                data.append(None)
    if return_time:
        return np.array(data), timesteps
    return np.array(data)


def get_run_metric_from_summary_or_line_plot(run, metric: str, x_label_name=None, verbose=True) -> Any:
    """
    Try returning run.summary[metric] or find step T in run history if metric contains 't<T>'.

    Args:
        run: The run object
        metric: The metric name (possibly with timestep)
        verbose: Whether to print warnings

    Returns:
        The metric value if found, None otherwise
    """
    # First check if metric is directly in summary
    if metric in run.summary:
        return run.summary[metric]

    # Parse the metric pattern
    _, step, _, metric_base_name = parse_metric_pattern(metric)

    # If we couldn't parse the pattern or no specific step
    if metric_base_name is None or step is None:
        return None

    # Get history data
    history_dict, lead_time_name = get_history_data(run, metric_base_name, x_label_name, verbose)

    # Look for the specific step
    if step in history_dict:
        return history_dict[step]

    # Step not found in history
    if verbose:
        print(f"Warning: {run.id=} No matching rows found for step {step} in metric '{metric_base_name}'")

    return None


def get_wandb_ckpt_name(run_path: str, epoch: Union[str, int] = "best") -> str:
    """
    Get the wandb ckpt name for a given run_path and epoch.
    Args:
        run_path: ENTITY/PROJECT/RUN_ID
        epoch: If an int, the ckpt name will be the one for that epoch.
            If 'last' ('best') the latest ('best') epoch ckpt will be returned.

    Returns:
        The wandb ckpt file-name, that can be used as follows to restore the checkpoint locally:
           >>> run_path = "<ENTITY/PROJECT/RUN_ID>"
           >>> ckpt_name = get_wandb_ckpt_name(run_path, epoch)
           >>> wandb.restore(ckpt_name, run_path=run_path, replace=True, root=os.getcwd())
    """
    assert epoch in ["best", "last"] or isinstance(
        epoch, int
    ), f"epoch must be 'best', 'last' or an int, but is {epoch}"
    run_api = get_run_api(run_path=run_path)
    ckpt_files = [f.name for f in run_api.files() if f.name.endswith(".ckpt")]
    if epoch == "best":
        if "best.ckpt" in ckpt_files:
            ckpt_filename = "best.ckpt"
        else:
            raise ValueError(f"Could not find best.ckpt in {ckpt_files}")
    elif "last.ckpt" in ckpt_files and epoch == "last":
        ckpt_filename = "last.ckpt"
    else:
        if len(ckpt_files) == 0:
            raise ValueError(f"Wandb run {run_path} has no checkpoint files (.ckpt) saved in the cloud!")
        elif len(ckpt_files) >= 2:
            ckpt_epochs = [int(name.replace("epoch", "")[:3]) for name in ckpt_files]
            if epoch == "last":
                # Use checkpoint of latest epoch if epoch is not specified
                max_epoch = max(ckpt_epochs)
                ckpt_filename = [name for name in ckpt_files if str(max_epoch) in name][0]
                log.info(f"Multiple ckpt files exist: {ckpt_files}. Using latest epoch: {ckpt_filename}")
            else:
                # Use checkpoint with specified epoch
                ckpt_filename = [name for name in ckpt_files if str(epoch) in name]
                if len(ckpt_filename) == 0:
                    raise ValueError(f"There is no ckpt file for epoch={epoch}. Try one of the ones in {ckpt_epochs}!")
                ckpt_filename = ckpt_filename[0]
        else:
            ckpt_filename = ckpt_files[0]
            log.warning(f"Only one ckpt file exists: {ckpt_filename}. Using it...")
    return ckpt_filename


def restore_model_from_wandb_cloud(
    run_path: str,
    ckpt_filename: str = "best",  # was None
    **kwargs,
) -> str:
    """
    Restore the model from the wandb cloud to local file-system.
    Args:
        run_path: PROJECT/ENTITY/RUN_ID
        local_checkpoint_path: If not None, the model will be restored from this path.
        ckpt_filename: If not None, the model will be restored from this filename (in the cloud).

    Returns:
        The ckpt filename that can be used to reload the model locally.
    """

    entity, project, wandb_id = run_path.split("/")
    if ckpt_filename is None:
        ckpt_filename = get_wandb_ckpt_name(run_path, **kwargs)
        ckpt_filename = ckpt_filename.split("/")[-1]  # in case the file contains local dir structure

    expected_ckpt_path = os.path.join(os.getcwd(), ckpt_filename)
    if wandb_id in expected_ckpt_path:
        ckpt_path = expected_ckpt_path
    else:
        # rename best_model_fname to add a unique prefix to avoid conflicts with other runs
        # (e.g. if the same model is reloaded twice). Replace only filename part of the path, not the dir structure
        ckpt_path = os.path.join(os.getcwd(), f"{wandb_id}-{ckpt_filename}")

    ckpt_path_tmp = ckpt_path
    if not os.path.exists(ckpt_path):
        try:
            from src.utilities.s3utils import download_s3_object

            s3_file_path = f"{project}/checkpoints/{wandb_id}/{ckpt_filename}"
            download_s3_object(s3_file_path, ckpt_path, throw_error=True)
        except Exception:
            # IMPORTANT ARGS replace=True: see https://github.com/wandb/client/issues/3247
            ckpt_path_tmp = wandb.restore(ckpt_filename, run_path=run_path, replace=True, root=os.getcwd()).name
            assert os.path.abspath(ckpt_path_tmp) == expected_ckpt_path

    # if os.path.exists(ckpt_path):
    #     # if DDP and multiple processes are restoring the same model, this may happen. check if is_rank_zero
    #     if ckpt_path != ckpt_path_tmp:
    #         os.remove(ckpt_path)  # remove if one exists from before
    if os.path.exists(ckpt_path_tmp):
        os.rename(ckpt_path_tmp, ckpt_path)
    return ckpt_path


def load_hydra_config_from_wandb(
    run_path: str | wandb.apis.public.Run,
    local_dir: str = None,
    override_config: Optional[DictConfig] = None,
    override_key_value: List[str] = None,
    update_config_in_cloud: bool = False,
) -> DictConfig:
    """
    Args:
        run_path (str): the wandb ENTITY/PROJECT/ID (e.g. ID=2r0l33yc) corresponding to the config to-be-reloaded
        local_dir (str): if not None, the hydra_config file will be searched in this directory
        override_config (DictConfig): each of its keys will override the corresponding entry loaded from wandb
        override_key_value: each element is expected to have a "=" in it, like datamodule.num_workers=8
        update_config_in_cloud: if True, the config in the cloud will be updated with the new overrides
    """
    if override_config is not None and override_key_value is not None:
        log.warning("Both override_config and override_key_value are not None! ")
    if isinstance(run_path, wandb.apis.public.Run):
        run = run_path
        run_path = "/".join(run.path)
    else:
        assert isinstance(
            run_path, str
        ), f"run_path must be a string or wandb.apis.public.Run, but is {type(run_path)}"
        run = get_run_api(run_path=run_path)

    override_key_value = override_key_value or []
    if not isinstance(override_key_value, list):
        raise ValueError(f"override_key_value must be a list of strings, but has type {type(override_key_value)}")
    # copy overrides to new list
    overrides = list(override_key_value.copy())

    # Prefer global rank when available; LOCAL_RANK repeats across nodes and causes coordination bugs.
    rank_env = os.environ.get("RANK", os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0")))
    try:
        rank = int(rank_env)
    except (TypeError, ValueError):
        rank = 0

    # Check if hydra_config file exists locally
    work_dir = run.config.get("dirs/work_dir", run.config.get("dirs", {}).get("work_dir", None))
    possible_cfg_dirs = []
    if work_dir is not None:
        if run.id not in work_dir:
            work_dir = os.path.join(os.path.dirname(work_dir), run.id)
        possible_cfg_dirs = [os.path.join(work_dir, "wandb")]
    is_local = False
    hydra_config_files = []
    if isinstance(local_dir, str):
        if run.id not in local_dir and os.path.exists(os.path.join(local_dir, run.id)):
            local_dir = os.path.join(local_dir, run.id)
        if os.path.isfile(local_dir):
            local_dir = os.path.dirname(local_dir)
        possible_cfg_dirs.append(local_dir)
        possible_cfg_dirs.append(local_dir.replace("checkpoints", "configs"))
        possible_cfg_dirs.append(local_dir.replace("checkpoints", "wandb"))
    for possible_cfg_dir in possible_cfg_dirs:
        if possible_cfg_dir is not None and os.path.exists(possible_cfg_dir):
            if os.path.exists(os.path.join(possible_cfg_dir, "latest-run")):
                possible_cfg_dir = os.path.join(possible_cfg_dir, "latest-run")
            # Find hydra_config file in wandb directory or subdirectories using glob
            hydra_config_files = glob.glob(f"{possible_cfg_dir}/**/hydra_config*.yaml", recursive=True)
            if len(hydra_config_files) > 0:
                log.info(
                    f"[rank={rank}] Found {len(hydra_config_files)} files in {possible_cfg_dir}: {hydra_config_files}"
                )
                is_local = True
                break
            # torch.distributed.barrier()
    if not is_local or len(hydra_config_files) == 0:
        # Find latest hydra_config-v{VERSION}.yaml file in wandb cloud (Skip versions labeled as 'old')
        hydra_config_files = [f.name for f in run.files() if "hydra_config" in f.name and "old" not in f.name]
        is_local = False

    if len(hydra_config_files) == 0:
        raise ValueError(f"Could not find any hydra_config file in wandb run {run_path}")
    elif len(hydra_config_files) == 1:
        if not is_local:
            assert hydra_config_files[0].endswith(
                "hydra_config.yaml"
            ), f"Only one hydra_config file found: {hydra_config_files}"
    else:
        hydra_config_files = [f for f in hydra_config_files if "hydra_config-v" in f]
        assert len(hydra_config_files) > 0, f"Could not find any hydra_config-v file in wandb run {run_path}"
        # Sort by version number (largest is last, earliest are hydra_config.yaml and hydra_config-v1.yaml),
        hydra_config_files = sorted(hydra_config_files, key=lambda x: int(x.split("-v")[-1].split(".")[0]))

    hydra_config_file = hydra_config_files[-1]
    # if not hydra_config_file.endswith("hydra_config.yaml"):
    #     log.info(f" Reloading from hydra config file: {hydra_config_file}")

    if is_local:
        log.info(f"Loading local hydra config file: {hydra_config_file}")
    else:
        # Always restore to a rank-specific directory so each process has an independent file path.
        local_file_path = None

        restore_root = os.path.join(os.getcwd(), ".wandb_restore", run.id, f"rank-{rank}-pid-{os.getpid()}")
        os.makedirs(restore_root, exist_ok=True)
        log.info(f"[rank: {rank}] Downloading hydra config file: {hydra_config_file} to {restore_root}")
        wandb_restore_kwargs = dict(run_path=run_path, replace=True, root=restore_root)
        try:
            local_file = wandb.restore(hydra_config_file, **wandb_restore_kwargs)
            local_file_path = local_file.name
        except Exception as e:
            log.warning(f"[rank: {rank}] Error when restoring hydra config file with explicit root: {e}")
            try:
                local_file = wandb.restore(hydra_config_file, run_path=run_path, replace=True)
                local_file_path = local_file.name
            except Exception as e2:
                log.error(f"[rank: {rank}] Second error when restoring hydra config file: {e2}")

        if local_file_path is not None:
            hydra_config_file = local_file_path

    assert os.path.exists(hydra_config_file), f"[rank: {rank}] Could not find {hydra_config_file=}"

    try:
        config = OmegaConf.load(hydra_config_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"[rank: {rank}] Could not find {hydra_config_file=}") from e

    # remove overrides of the form k=v, where k has no dot in it. We don't support this.
    overrides = [o for o in overrides if "=" in o]  # and "." in o.split("=")[0]]
    if len(overrides) != len(override_key_value):
        diff = set(overrides) - set(override_key_value)
        print(f"Removed {len(diff)} overrides that are not in the form k=v: {diff}", overrides, override_key_value)
        log.warning(f"The following overrides were removed because they are not in the form k=v: {diff}")

    overrides += [
        f"logger.wandb.id={run.id}",
        f"logger.wandb.entity={run.entity}",
        f"logger.wandb.project={run.project}",
        f"logger.wandb.tags={run.tags}",
        f"logger.wandb.group={run.group}",
    ]
    overrides = OmegaConf.from_dotlist(overrides)
    config = OmegaConf.unsafe_merge(config, overrides)

    if override_config is not None:
        for k, v in override_config.items():
            if k in ["model", "trainer"] and isinstance(v, str):
                override_config.pop(k)  # remove key from override_config
                log.warning(f"Key {k} is a string, but it should be a DictConfig. Ignoring it.")
        # override config with override_config (which needs to be the second argument of OmegaConf.merge)
        config = OmegaConf.unsafe_merge(config, override_config)  # unsafe_merge since override_config is not needed

    if not is_local and os.path.exists(hydra_config_file):
        # Cleanup only rank-private restore artifacts; never delete potentially shared cwd files.
        abs_hydra_path = os.path.abspath(hydra_config_file)
        if os.path.join(os.getcwd(), ".wandb_restore") in abs_hydra_path:
            try:
                os.remove(hydra_config_file)
            except (FileNotFoundError, PermissionError) as e:
                log.debug(f"[rank: {rank}] Could not remove restored config file {hydra_config_file}: {e}")

    if run.id != config.logger.wandb.id and run.id in config.logger.wandb.name:
        config.logger.wandb.id = run.id
    assert str(config.logger.wandb.id) == str(
        run.id
    ), f"{config.logger.wandb.id=} != {run.id=}. {is_local=} \nFull Hydra config: {config}"
    if update_config_in_cloud:
        with open("hydra_config.yaml", "w") as fp:
            OmegaConf.save(config, f=fp.name, resolve=True)
        run.upload_file("hydra_config.yaml", root=".")
        os.remove("hydra_config.yaml")
    return config


def does_any_ckpt_file_exist(wandb_run: wandb.apis.public.Run, only_best_and_last: bool = True, local_dir: str = None):
    """
    Check if any checkpoint file exists in the wandb run.
    Args:
        wandb_run: the wandb run to check
        only_best_and_last: if True, only checks for 'best.ckpt' and 'last.ckpt' files, otherwise checks for all ckpt files
                Setting to true may speed up the check, since it will stop as soon as it finds one of the two files.
    """
    if local_dir is not None:
        local_dirs = [local_dir, os.path.dirname(local_dir.rstrip("/"))]
        for local_dir in local_dirs:
            if wandb_run.id not in local_dir:
                local_dir = os.path.join(local_dir, wandb_run.id)
            # Find with glob
            ckpt_files = glob.glob(f"{local_dir}/**/*.ckpt", recursive=True)
            if len(ckpt_files) > 0:
                log.info(f"Found {ckpt_files} ckpt files in local dir {local_dir} for run {wandb_run.id}")
                return True

    names = ["last.ckpt", "best.ckpt"] if only_best_and_last else None
    if isinstance(wandb_run.summary, dict) and "checkpoint/in_s3" in wandb_run.summary.keys():
        return True  # Using S3 storage

    try:
        return len([1 for f in wandb_run.files(names=names) if f.name.endswith(".ckpt")]) > 0
    except Exception as e:
        log.warning(f"Error when checking for checkpoint files in wandb run {wandb_run.id}: {e}")
        return False


def get_existing_wandb_group_runs(
    config: DictConfig, ckpt_must_exist: bool = False, **kwargs
) -> List[wandb.apis.public.Run]:
    if config.get("logger", None) is None or config.logger.get("wandb", None) is None:
        log.warning("No logger.wandb config found in config. Without it, it's not possible to find existing runs..")
        return []
    wandb_cfg = config.logger.wandb
    runs_in_group = get_runs_for_filter(entity=wandb_cfg.entity, project=wandb_cfg.project, group=wandb_cfg.group)
    try:
        n_runs = len(runs_in_group)
    except (ValueError, TypeError) as e:  # happens if project does not exist or empty
        log.warning(f"Error when loading runs for {wandb_cfg=}: {e}")
        return []
    if ckpt_must_exist:
        local_dir = config.ckpt_dir
        runs_in_group = [run for run in runs_in_group if does_any_ckpt_file_exist(run, **kwargs, local_dir=local_dir)]
        if len(runs_in_group) == 0 and n_runs > 0:
            runs_with_epoch_1plus = [run for run in runs_in_group if run.summary.get("epoch", 0) > 1]
            log.warning(
                f"No runs found for group {wandb_cfg.group} with checkpoints. "
                f"However, {n_runs} runs in group exist ({len(runs_with_epoch_1plus)} with epoch > 1). "
                f"{config.ckpt_dir=}."
            )
    return runs_in_group
    # other_seeds = [run.config.get('seed') for run in other_runs]
    # if config.seed in other_seeds:
    #    state = runs_in_group[other_seeds.index(config.seed)].state
    #    log.info(f"Found a run (state={state}) with the same seed (={this_seed}) in group {group}.")
    #    return True
    # return False


def reupload_run_history(run):
    """
    This function can be called when for weird reasons your logged metrics do not appear in run.summary.
    All metrics for each epoch (assumes that a key epoch=i for each epoch i was logged jointly with the metrics),
    will be reuploaded to the wandb run summary.
    """
    summary = {}
    for row in run.scan_history():
        if "epoch" not in row.keys() or any(["gradients/" in k for k in row.keys()]):
            continue
        summary.update(row)
    run.summary.update(summary)


#####################################################################
#
# Pre-filtering of wandb runs
#
def has_finished(run):
    return run.state == "finished"


def not_running(run):
    return run.state != "running"


def has_final_metric(run) -> bool:
    return "test/mse" in run.summary.keys() and "test/mse" in run.summary.keys()


def has_run_id(run_ids: str | List[str]) -> Callable:
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    return lambda run: any([run.id == rid for rid in run_ids])


def contains_in_run_name(name: str) -> Callable:
    return lambda run: name in run.name


def has_summary_metric(metric_name: str, check_non_nan: bool = False) -> Callable:
    metric_name = metric_name.replace("//", "/")

    def has_metric(run):
        return metric_name in run.summary.keys()  # or metric_name in run.summary_metrics.keys()

    def has_metric_non_nan(run):
        value = run.summary.get(metric_name)
        try:
            return value is not None and value not in {"NaN", "Infinity"} and not np.isnan(value)
        except Exception as e:
            raise ValueError(
                f"Error when checking metric {metric_name} in run {run.id}. Summary value: {value}, type: {type(value)}"
            ) from e

    return has_metric_non_nan if check_non_nan else has_metric


def has_summary_metric_any(metric_names: List[str], check_non_nan: bool = False) -> Callable:
    metric_names = [m.replace("//", "/") for m in metric_names]

    def has_metric(run):
        return any([m in run.summary.keys() for m in metric_names])

    def has_metric_non_nan(run):
        return any([m in run.summary.keys() and not np.isnan(run.summary[m]) for m in metric_names])

    return has_metric_non_nan if check_non_nan else has_metric


def has_summary_metric_lower_than(metric_name: str, lower_than: float) -> Callable:
    metric_name = metric_name.replace("//", "/")
    return lambda run: metric_name in run.summary.keys() and run.summary[metric_name] < lower_than


def has_summary_metric_greater_than(metric_name: str, greater_than: float) -> Callable:
    metric_name = metric_name.replace("//", "/")
    return lambda run: metric_name in run.summary.keys() and run.summary[metric_name] > greater_than


def has_minimum_runtime(min_minutes: float = 10.0) -> Callable:
    return lambda run: run.summary.get("_runtime", 0) > min_minutes * 60


def has_minimum_epoch(min_epoch: int = 10) -> Callable:
    def has_min_epoch(run):
        hist = run.history(keys=["epoch"])
        return len(hist) > 0 and max(hist["epoch"]) > min_epoch

    return has_min_epoch


def has_minimum_epoch_simple(min_epoch: int = 10) -> Callable:
    def has_min_epoch(run):
        return run.summary.get("epoch", 0) > min_epoch

    return has_min_epoch


def has_maximum_epoch_simple(max_epoch: int = 10) -> Callable:
    def has_min_epoch(run):
        return run.summary.get("epoch", np.inf) < max_epoch

    return has_min_epoch


def has_keys(keys: Union[str, List[str]]) -> Callable:
    keys = [keys] if isinstance(keys, str) else keys
    return lambda run: all([(k in run.summary.keys() or k in run.config.keys()) for k in keys])


def hasnt_keys(keys: Union[str, List[str]]) -> Callable:
    keys = [keys] if isinstance(keys, str) else keys
    return lambda run: all([(k not in run.summary.keys() and k not in run.config.keys()) for k in keys])


def has_max_metric_value(metric: str = "test/MERRA2/mse_epoch", max_metric_value: float = 1.0) -> Callable:
    return lambda run: run.summary[metric] <= max_metric_value


def has_tags(tags: Union[str, List[str]]) -> Callable:
    if isinstance(tags, str):
        tags = [tags]
    return lambda run: all([tag in run.tags for tag in tags])


def hasnt_tags(tags: Union[str, List[str]]) -> Callable:
    if isinstance(tags, str):
        tags = [tags]
    return lambda run: all([tag not in run.tags for tag in tags])


def hyperparams_list_api(replace_dot_and_slashes: bool = False, **hyperparams) -> Dict[str, Any]:
    filter_dict_for_api = {}
    for hyperparam, value in hyperparams.items():
        if replace_dot_and_slashes:
            if "/" in hyperparam:
                hyperparam = hyperparam.replace("/", ".")
            else:
                hyperparam = hyperparam.replace(".", "/")
        if (
            "config." not in hyperparam
            and "summary." not in hyperparam
            and "summary_metrics." not in hyperparam
            and hyperparam != "tags"
        ):
            # Automatically add config. prefix if not present
            hyperparam = f"config.{hyperparam}"
        filter_dict_for_api[hyperparam] = value
    return filter_dict_for_api


def has_config_values(**hyperparams) -> Callable:
    return lambda run: all(
        hyperparam in run.config and run.config[hyperparam] == value for hyperparam, value in hyperparams.items()
    )


def larger_than(**kwargs) -> Callable:
    return lambda run: all(
        hasattr(run.config, hyperparam) and value > run.config[hyperparam] for hyperparam, value in kwargs.items()
    )


def lower_than(**kwargs) -> Callable:
    return lambda run: all(
        hasattr(run.config, hyperparam) and value < run.config[hyperparam] for hyperparam, value in kwargs.items()
    )


str_to_run_pre_filter = {
    "has_finished": has_finished,
    "has_final_metric": has_final_metric,
}


def get_filter_functions_helper(epoch: int = None, finished: bool = True, config_values=None):
    filter_functions = []
    if finished:
        filter_functions.append(has_finished)
    if epoch is not None:
        filter_functions += [contains_in_run_name(f"{epoch}epoch")]
    if config_values is not None:
        filter_functions.append(has_config_values(**config_values))
    return filter_functions


#####################################################################
#
# Post-filtering of wandb runs (usually when you need to compare runs)
#


def non_unique_cols_dropper(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(cols_to_drop, axis=1)
    return df


def groupby(
    df: pd.DataFrame,
    group_by: Union[str, List[str]] = "seed",
    metrics: List[str] = "val/mse_epoch",
    keep_columns: List[str] = "model/name",
) -> pd.DataFrame:
    """
    Args:
        df: pandas DataFrame to be grouped
        group_by: str or list of str defining the columns to group by
        metrics: list of metrics to compute the group mean and std over
        keep_columns: list of columns to keep in the resulting grouped DataFrame
    Returns:
        A dataframe grouped by `group_by` with columns
        `metric`/mean and `metric`/std for each metric passed in `metrics` and all columns in `keep_columns` remain intact.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(keep_columns, str):
        keep_columns = [keep_columns]
    if isinstance(group_by, str):
        group_by = [group_by]

    grouped_df = df.groupby(group_by, as_index=False, dropna=False)
    agg_metrics = {m: ["mean", "std"] for m in metrics}
    agg_remain_intact = {c: "first" for c in keep_columns}
    # cols = [group_by] + keep_columns + metrics + ['id']
    stats = grouped_df.agg({**agg_metrics, **agg_remain_intact})
    stats.columns = [(f"{c[0]}/{c[1]}" if c[1] in ["mean", "std"] else c[0]) for c in stats.columns]
    for m in metrics:
        stats[f"{m}/std"].fillna(value=0, inplace=True)

    return stats


str_to_run_post_filter = {
    "unique_columns": non_unique_cols_dropper,
}


def get_wandb_filters_dict_list_from_list(filters_list) -> dict:
    if filters_list is None:
        filters_list = []
    elif not isinstance(filters_list, list):
        filters_list: List[Union[Callable, str]] = [filters_list]
    filters_wandb = []  # dict()
    for f in filters_list:
        if isinstance(f, str):
            f = str_to_run_pre_filter[f.lower()]
        filters_wandb.append(f)
        # filters_wandb = {**filters_wandb, **f}
    return filters_wandb


def get_topk_groups_per_hparam(
    hyperparam_filter: Dict[str, Any],
    monitor: str = "val/avg/crps",
    mode: str = None,
    group_aggregation_func: str = "avg",
    top_k: int = 3,
    min_num_of_runs_per_group: int = 1,
    filter_functions: List[Callable] = None,
    entity: str = None,
    project: str = None,
    **kwargs,
) -> Dict[str, List[wandb.apis.public.Run]]:
    assert min_num_of_runs_per_group > 0, f"min_num_of_runs_per_group must be > 0, got {min_num_of_runs_per_group}"
    if mode is None:
        mode = "max" if "max" in monitor else "min"
    filter_functions = filter_functions or []
    filter_functions += [has_summary_metric(monitor)]
    g_to_run = wandb_project_run_filtered(
        hyperparam_filter=hyperparam_filter,
        filter_functions=filter_functions,
        aggregate_into_groups=True,
        entity=entity,
        project=project,
        **kwargs,
    )

    def get_val_from_summary(summary):
        if isinstance(summary, dict) or hasattr(summary, "keys"):
            return summary[mode]
        return summary

    # Compute statistic for monitor for each group
    aggregate_func = {
        "avg": np.mean,
        "min": np.min,
        "max": np.max,
        "median": np.median,
    }[group_aggregation_func]
    g_to_summary = {
        g: aggregate_func([get_val_from_summary(r.summary[monitor]) for r in runs])
        for g, runs in g_to_run.items()
        if len(runs) >= min_num_of_runs_per_group
    }
    # Sort groups by statistic
    g_to_summary = {k: v for k, v in sorted(g_to_summary.items(), key=lambda item: item[1], reverse=mode == "max")}
    # Get top k groups
    topk_groups = list(g_to_summary.keys())[:top_k]
    # Get runs for top k groups
    topk_groups = {g: {"runs": g_to_run[g], "summary": g_to_summary[g]} for g in topk_groups}
    if top_k == 1 and len(topk_groups) == 1:
        group_name = list(topk_groups.keys())[0]
        topk_groups = topk_groups[group_name]
        topk_groups["group"] = group_name
    return topk_groups


def get_run_ids_for_hyperparams(hyperparams: dict, **kwargs) -> List[str]:
    runs = wandb_project_run_filtered(hyperparams, **kwargs)
    run_ids = [run.id for run in runs]
    return run_ids


def get_filter_for_wandb(
    filter_dict: Dict[str, Any], extra_filters: Dict[str, Any] = None, robust: bool = True
) -> Dict[str, Any]:
    if filter_dict is None:
        return {} if extra_filters is None else extra_filters
    if "$and" not in filter_dict and "$or" not in filter_dict:
        filter_wandb_api = hyperparams_list_api(**filter_dict)
        if isinstance(extra_filters, dict):
            filter_wandb_api = {**filter_wandb_api, **extra_filters}

        if robust:
            filter_wandb_api_v2 = hyperparams_list_api(replace_dot_and_slashes=True, **filter_dict)
            if isinstance(extra_filters, dict):
                filter_wandb_api_v2 = {**filter_wandb_api_v2, **extra_filters}
            filter_wandb_api = {"$or": [filter_wandb_api, filter_wandb_api_v2]}
        else:
            filter_wandb_api = {"$and": [filter_wandb_api]}  # MongoDB query lang
    else:
        filter_wandb_api = filter_dict
    return filter_wandb_api


def wandb_project_run_filtered(
    hyperparam_filter: Dict[str, Any] = None,
    extra_filters: Dict[str, Any] = None,
    filter_functions: Sequence[Callable] = None,
    order="-created_at",
    aggregate_into_groups: bool = False,
    entity: str = None,
    project: str = None,
    wandb_api=None,
    verbose: bool = True,
    robust: bool = True,
) -> Union[List[wandb.apis.public.Run], Dict[str, List[wandb.apis.public.Run]]]:
    """
    Args:
        hyperparam_filter: a dict str -> value, e.g. {'model/name': 'mlp', 'datamodule/exp_type': 'pristine'}
        filter_functions: A set of callable functions that take a wandb run and return a boolean (True/False) so that
                            any run with one or more return values being False is discarded/filtered out
        robust: If True, the hyperparam_filter will be applied in two ways: as is and with dots replaced by slashes
            (within an OR query). This is useful when wandb keys are stored in different formats.

    Note:
        For more complex/logical filters, see https://www.mongodb.com/docs/manual/reference/operator/query/
    """
    entity = get_entity(entity)
    project = project or PROJECT
    extra_filters = extra_filters or dict()
    filter_functions = filter_functions or []
    if not isinstance(filter_functions, list):
        filter_functions = [filter_functions]
    filter_functions = [(f if callable(f) else str_to_run_pre_filter[f.lower()]) for f in filter_functions]

    hyperparam_filter = hyperparam_filter or dict()
    api = get_api(wandb_api)

    if "group" in hyperparam_filter.keys() and "group" not in extra_filters.keys():
        hyperparam_filter = hyperparam_filter.copy()
        extra_filters = {**extra_filters, "group": hyperparam_filter.pop("group")}
    filter_wandb_api = get_filter_for_wandb(hyperparam_filter, extra_filters=extra_filters, robust=robust)

    runs_start = api.runs(f"{entity}/{project}", filters=filter_wandb_api, per_page=100, order=order)

    if len(filter_functions) > 0:
        runs = []
        for i, run in enumerate(runs_start):
            if all(f(run) for f in filter_functions):
                runs.append(run)
    else:
        runs = runs_start

    if verbose:
        log.info(f"#Filtered runs = {len(runs)}, (wandb API filtered {len(runs_start)})")
        if len(runs) == 0:
            log.warning(
                f" No runs found for given filters: {filter_wandb_api} in {entity}/{project}"
                f"\n #Runs before post-filtering with {filter_functions}: {len(runs_start)}"
            )
        else:
            log.info(f"Found {len(runs)} runs!")

    if aggregate_into_groups:
        groups = defaultdict(list)
        for run in runs:
            groups[run.group].append(run)
        return groups
    return runs


def get_ordered_runs_with_config_diff(
    order: str = None,
    metric: str = None,
    lower_is_better: bool = True,
    top_k: int = 5,
    every_k: int = 1,
    return_metric_value: bool = True,
    exclude_sub_dicts=None,
    replace_epoch_by_million_imgs: bool = True,
    verbose=True,
    **kwargs,
) -> Dict[str, wandb.apis.public.Run]:
    """
    Get the top k runs with the largest configuration differences.
    """
    if order is None:
        assert metric is not None, "One of order or metric must be specified"
        order = f"summary_metrics.{metric}"
        order = f"+{order}" if lower_is_better else f"-{order}"
        # For some reason, the below does not filter out runs with None values properly, so we do it manually below in post-processing
        if "extra_filters" not in kwargs:
            kwargs["extra_filters"] = dict()
        kwargs["extra_filters"][f"summary_metrics.{metric}"] = {"$exists": True}
        # kwargs["hyperparam_filter"] = {**kwargs["hyperparam_filter"], f"summary_metrics.{metric}": {"$ne": None}, f"summary_metrics.{metric}": {"$exists": True}}
        # kwargs["hyperparam_filter"] = {**kwargs["hyperparam_filter"], f"summary_metrics.{metric}": {"$ne": None}}
        # # # kwargs["hyperparam_filter"] = {**kwargs["hyperparam_filter"], f"summary_metrics.{metric}": {"$ne": None}, f"summary_metrics.{metric}": {"$exists": True}}
    else:
        assert metric is None, "Only one of order or metric must be specified"

    kwargs.pop("extra_filters", None)
    log.info(f"order={order}", kwargs)
    runs = wandb_project_run_filtered(order=order, verbose=verbose, **kwargs)
    if metric is not None:
        runs = [run for run in runs if run.summary_metrics.get(metric) is not None]
    if len(runs) == 0:
        return {}
    # if len(runs) < 2:
    # return {"runs": runs}
    best_runs = runs[: top_k * every_k : every_k]

    exclude_sub_dicts = exclude_sub_dicts or (
        "logger",
        "wandb",
        "callbacks",
        "model_checkpoint",
        "model_checkpoint_t8",
        "module",
        "early_stopping",
        "scheduler",
        "diffusion_config",
        "model_config",
        "datamodule_config",
        "++datamodule",
        "++diffusion",
        "++logger",
        "scheduler@module",
        "dirs",
        "slurm_job_id",
        "start_time",
        "ckpt_path",
        "n_gpus",
        "world_size",
        "pin_memory",
        "num_workers",
        "regression_overrides",
        "regression_run_id",
        "eval_batch_size",
        "batch_size",
        "inference_val_every_n_epochs",
        "save_prediction_batches",
        "ema_decay",
        "downsampling_method",
        "enable_inference_dropout",
        "regression_inference_dropout",
    )
    exclude_nested = {
        "datamodule": ["eval_batch_size", "batch_size", "lookback_window", "pin_memory", "num_workers"],
        "exp": ["inference_val_every_n_epochs"],
        "trainer": ["num_sanity_val_steps", "gpus"],
        "optim": ["effective_batch_size"],
    }
    # Define the keys for which we tolerate a certain percentage difference instead of an exact match
    keys_to_tolerated_percent_diff = {
        "model/params_not_trainable": 0.07,  # 7% difference
        "model/params_trainable": 0.07,  # 7% difference
        "model/params_total": 0.07,  # 7% difference
    }
    if replace_epoch_by_million_imgs:
        exclude_sub_dicts = list(exclude_sub_dicts) + ["epoch", "global_step"]

    exclude_sub_dicts = set(exclude_sub_dicts)
    # Get all unique config differences
    best_configs = []
    for run in best_runs:
        config = run.config
        config = {k: v for k, v in config.items() if k not in exclude_sub_dicts}
        if exclude_sub_dicts:
            try:
                config["Mimg."] = run.config.get("global_step", 0) * run.config.get("effective_batch_size", 0) / 1e6
            except TypeError:
                log.warning(f"Could not compute Mimg for run {run.id} (name={run.name})")
        for key, sub_keys in exclude_nested.items():
            if key in config:
                for sub_key in sub_keys:
                    if sub_key in config[key]:
                        del config[key][sub_key]

        best_configs.append(config)

    diffs = find_config_differences_return_as_joined_str(
        best_configs, sort_by_name=True, keys_to_tolerated_percent_diff=keys_to_tolerated_percent_diff
    )
    diff_to_run = dict()
    for run, diff in zip(best_runs, diffs):
        v1 = run.summary.get(metric)
        if v1 is None:
            if verbose:
                log.warning(f"Skipping run_id={run.id} because metric {metric} is not available")
            continue
        if diff in diff_to_run:
            other_run = diff_to_run[diff][0] if return_metric_value else diff_to_run[diff]
            if metric is not None:
                v2 = other_run.summary.get(metric)
                # close if first 3 digits are the same
                max_diff = 1 if "rel" in metric else 1e-3  # if relative in %, then allow up to 1%
                assert (
                    abs(v1 - v2) < max_diff
                ), f"Metric values are not the same even though the diff is the same: r1.id={run.id}, r2.id={other_run.id}; v1={v1:.5f}, v2={v2:.5f}"
            if verbose:
                log.warning(f"Duplicate diff found: {diff}, skipping run_id={run.id}. Keeping run_id={other_run.id}")
            continue
        if return_metric_value:
            diff_to_run[diff] = (run, v1)
        else:
            diff_to_run[diff] = run
    return diff_to_run


def runs_to_df(runs, metrics, skip_hps=None, baseline_run=None, aggregate_by="crps"):
    skip_hps = skip_hps or {}
    # List to store our data
    data = []

    for run in tqdm(runs, desc="Processing runs"):
        diffusion_cfg = run.config.get("diffusion")
        heun = run.config.get("diffusion.heun") or diffusion_cfg.get("heun")
        step = run.config.get("diffusion.step") or diffusion_cfg.get("step")
        s_churn = run.config.get("diffusion.S_churn") or diffusion_cfg.get("S_churn")
        if heun is None or step is None or s_churn is None:
            log.info(
                f"Skipping run {run.id} due to missing hyperparameters. heun={heun}, step={step}, s_churn={s_churn}. config={run.config}"
            )
            continue

        hps = {"heun": heun, "step": step, "churn": s_churn}
        hps.update(diffusion_cfg)
        hps.update(run.config.get("exp", {}))
        hps["with_time_emb"] = run.config.get("model", {}).get("with_time_emb")
        hps["horizon"] = run.config.get("datamodule", {}).get("horizon")
        hps["lookback_window"] = run.config.get("datamodule", {}).get("lookback_window")
        hps["ema_decay"] = run.config.get("exp", {}).get("ema_decay")
        hps["max_epochs"] = run.config.get("trainer", {}).get("max_epochs")
        hps["run_id"] = run.id
        run_metrics = {k: run.summary[k] for k in metrics}
        data.append({**hps, **run_metrics})

    # Create DataFrame
    df = pd.DataFrame(data)
    for key, values in skip_hps.items():
        # drop rows with specific values
        df = df[~df[key].isin(values)]
        for metric in metrics:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
            df[metric] = df[metric].replace([np.inf, -np.inf], np.nan)
    if baseline_run is not None:
        # Generate aggregated relative metrics
        aggregate_rel_metric = f"aggregated_relative_{aggregate_by}"
        df[aggregate_rel_metric] = 0.0
        for metric in metrics:
            if aggregate_by in metric:
                base_value = baseline_run.summary.get(metric)
                assert base_value is not None and isinstance(
                    base_value, float
                ), f"Could not find baseline value for metric {metric}"
                # to avoid TypeError: unsupported operand type(s) for -: 'str' and 'float'
                # we need to convert the metric to float (coerce will convert non-numeric values to NaN)
                df[metric] = pd.to_numeric(df[metric], errors="coerce")
                df[aggregate_rel_metric] += 100 * (df[metric] - base_value) / base_value
        df[aggregate_rel_metric] /= len(metrics)
    return df


def get_runs_df(
    get_metrics: bool = True,
    hyperparam_filter: dict = None,
    run_pre_filters: Union[str, List[Union[Callable, str]]] = "has_finished",
    run_post_filters: Union[str, List[str]] = None,
    verbose: int = 1,
    make_hashable_df: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """

    get_metrics:
    run_pre_filters:
    run_post_filters:
    verbose: 0, 1, or 2, where 0 = no output at all, 1 is a bit verbose
    """
    if run_post_filters is None:
        run_post_filters = []
    elif not isinstance(run_post_filters, list):
        run_post_filters: List[Union[Callable, str]] = [run_post_filters]
    run_post_filters = [(f if callable(f) else str_to_run_post_filter[f.lower()]) for f in run_post_filters]

    # Project is specified by <entity/project-name>
    runs = wandb_project_run_filtered(hyperparam_filter, run_pre_filters, **kwargs)
    summary_list = []
    config_list = []
    group_list = []
    name_list = []
    tag_list = []
    id_list = []
    for i, run in enumerate(runs):
        if i % 50 == 0:
            log.info(f"Going after run {i}")
        # if i > 100: break
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        if "model/_target_" not in run.config.keys():
            if verbose >= 1:
                print(f"Run {run.name} filtered out, as model/_target_ not in run.config.")
            continue

        id_list.append(str(run.id))
        tag_list.append(str(run.tags))
        if get_metrics:
            summary_list.append(run.summary._json_dict)
            # run.config is the hyperparameters
            config_list.append({k: v for k, v in run.config.items() if k not in run.summary.keys()})
        else:
            config_list.append(run.config)

        # run.name is the name of the run.
        name_list.append(run.name)
        group_list.append(run.group)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({"name": name_list, "id": id_list, "tags": tag_list})
    group_df = pd.DataFrame({"group": group_list})
    all_df = pd.concat([name_df, config_df, summary_df, group_df], axis=1)

    cols = [c for c in all_df.columns if not c.startswith("gradients/") and c != "graph_0"]
    all_df = all_df[cols]
    if all_df.empty:
        raise ValueError("Empty DF!")
    for post_filter in run_post_filters:
        all_df = post_filter(all_df)
    all_df = clean_hparams(all_df)
    if make_hashable_df:
        all_df = all_df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)

    return all_df


def fill_nan_if_present(df: pd.DataFrame, column_key: str, fill_value: Any, inplace=True) -> pd.DataFrame:
    if column_key in df.columns:
        df[column_key] = df[column_key].fillna(fill_value)  # , inplace=inplace)
        # df = df[column_key].apply(lambda x: fill_value if x != x else x)
    return df


def clean_hparams(df: pd.DataFrame):
    # Replace string representation of nan with real nan
    df.replace("NaN", np.nan, inplace=True)
    # df = df.where(pd.notnull(df), None).fillna(value=np.nan)

    # Combine/unify columns of optim/scheduler which might be present in stored params more than once
    combine_cols = [col for col in df.columns if col.startswith("model/optim") or col.startswith("model/scheduler")]
    for col in combine_cols:
        new_col = col.replace("model/", "").replace("optimizer", "optim")
        if not hasattr(df, new_col):
            continue
        getattr(df, new_col).fillna(getattr(df, col), inplace=True)
        del df[col]

    return df


def get_datetime_of_run(run: wandb.apis.public.Run, to_local_timezone: bool = True) -> datetime:
    """Get datetime of a run"""
    dt_str = run.createdAt  # a str like '2023-03-09T08:20:25'
    dt_utc = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    if to_local_timezone:
        return dt_utc.astimezone(tz=None)
    else:
        return dt_utc
    return datetime.fromtimestamp(run.summary["_timestamp"])


def get_unique_groups_for_run_ids(run_ids: Sequence[str], wandb_api: wandb.Api = None, **kwargs) -> List[str]:
    """Get unique groups for a list of run ids"""
    api = get_api(wandb_api)
    groups = []
    for run_id in run_ids:
        run = get_run_api(run_id, wandb_api=api, **kwargs)
        groups.append(run.group)
    return list(set(groups))


def get_unique_groups_for_hyperparam_filter(
    hyperparam_filter: dict,
    filter_functions: str | List[Union[Callable, str]] = None,
    **kwargs,  # 'has_finished'
) -> List[str]:
    """Get unique groups for a hyperparam filter

    Args:
        hyperparam_filter: dict of hyperparam filters.
        filter_functions: list of filter functions to apply to runs before getting groups.

    Examples:
         Use hyperparam_filter={'datamodule/horizon': 1, 'model/dim': 128} to get all runs with horizon=1 and dim=128
         or {'datamodule/horizon': 1, 'diffusion/timesteps': {'$gte': 10}} for horizon=1 and timesteps >= 10
    """
    runs = wandb_project_run_filtered(hyperparam_filter, filter_functions=filter_functions, **kwargs)
    groups = [run.group for run in runs]
    return list(set(groups))


def add_summary_metrics(
    run_id: str,
    metric_keys: Union[str, List[str]],
    metric_values: Union[float, List[float]],
    wandb_api: wandb.apis.public.Api = None,
    override: bool = False,
    **kwargs,
):
    """
    Add a metric to the summary of a run.
    """
    wandb_api = get_api(wandb_api)
    run = get_run_api(run_id, wandb_api=wandb_api, **kwargs)
    metric_keys = [metric_keys] if isinstance(metric_keys, str) else metric_keys
    metric_values = [metric_values] if isinstance(metric_values, float) else metric_values
    assert len(metric_keys) == len(
        metric_values
    ), f"metric_keys and metric_values must have same length, but got {len(metric_keys)} and {len(metric_values)}"

    for key, value in zip(metric_keys, metric_values):
        if key in run.summary.keys() and not override:
            print(f"Metric {key} already present in run {run_id}, skipping.")
            return
        run.summary[key] = value
    run.summary.update()


def metrics_of_runs_to_arrays(
    runs: Sequence[wandb.apis.public.Run],
    metrics: Sequence[str],
    columns: Sequence[Any],
    column_to_wandb_key: Callable[[Any], str] | Callable[[Any], List[str]],
    dropna_rows: bool = True,
) -> Dict[str, np.ndarray]:
    """Convert metrics of runs to arrays

    Args:
        runs (list): list of wandb runs (will be the rows of the arrays)
        metrics (list): list of metrics (one array will be created for each metric)
        columns (list): list of columns (will be the columns of the arrays)
        column_to_wandb_key (Callable): function to convert a given column to a wandb key (without metric suffix)
         If it returns a list of keys, the first one will be used to get the metric (if present).
    """

    def column_to_wandb_key_with_metric(wandb_key_stem, metric: str):
        if metric not in wandb_key_stem:
            wandb_key_stem = f"{wandb_key_stem}/{metric}"
        return wandb_key_stem.replace("//", "/")

    def get_summary_metric(run: wandb.apis.public.Run, metric: str, column: Any):
        wandb_keys = column_to_wandb_key(column)
        wandb_keys = [wandb_keys] if isinstance(wandb_keys, str) else wandb_keys
        for wandb_key_stem in wandb_keys:
            wandb_key = column_to_wandb_key_with_metric(wandb_key_stem, metric)
            if wandb_key in run.summary.keys():
                return run.summary[wandb_key]
        return np.nan

    nrows, ncols = len(runs), len(columns)
    arrays = {m: np.zeros((nrows, ncols)) for m in metrics}
    for r_i, run in enumerate(runs):
        if (
            run.project != "DYffusion"
            and np.isnan(get_summary_metric(run, metrics[0], columns[0]))
            and "None" not in column_to_wandb_key(None)
        ):
            full_metric_names = [column_to_wandb_key_with_metric(column_to_wandb_key(None), m) for m in metrics]
            run_metrics = get_summary_metrics_from_history(run, full_metric_names, robust=False)
            for m, fm in zip(metrics, full_metric_names):
                assert len(run_metrics[fm]) >= ncols, f"Expected {ncols} columns, got {len(run_metrics[fm])}"
                if len(run_metrics[fm]) > ncols:
                    run_metrics[fm] = run_metrics[fm][ncols]
                else:
                    arrays[m][r_i, :] = run_metrics[fm]
        else:
            for m in metrics:
                arrays[m][r_i, :] = [get_summary_metric(run, m, c) for c in columns]
    if dropna_rows:
        for m in metrics:
            arrays[m] = arrays[m][~np.isnan(arrays[m]).any(axis=1)]
    return arrays


def get_summary_metrics_from_history(run, metrics: Sequence[str], robust: bool = False):
    """Get summary metrics from history"""
    history = run.history(keys=metrics, pandas=True) if not robust else run.scan_history(keys=metrics)
    # history has one column per metric, one row per step, we want to return one numpy array per metric
    if robust:
        return {m: history[m].to_numpy() for m in metrics}
    else:
        return {m: history[m].to_numpy() for m in metrics}


def add_time_average_new(
    run, relative_run=None, times=range(1, 61), target_metric_prefix="test-wx/25ens_mems", metric="crps", cache=True
):
    """Add time average of a metric to a run."""
    times = list(times)

    time_avg_metric_name = f"time_avg_{times[0]}_{times[-1]}"
    if relative_run is not None:
        time_avg_metric_name = f"{time_avg_metric_name}_rel_{relative_run.id}"

    time_avg_metric_name = f"{target_metric_prefix}/{time_avg_metric_name}/{metric}"
    if time_avg_metric_name in run.summary and run.summary[time_avg_metric_name] is not None:
        return False
    if relative_run is not None and f"{time_avg_metric_name}_rel_{relative_run.id}" in run.summary:
        # Wrongly keyed, fix it
        run.summary[time_avg_metric_name] = run.summary[f"{time_avg_metric_name}_rel_{relative_run.id}"]
        del run.summary[f"{time_avg_metric_name}_rel_{relative_run.id}"]
        return True

    metrics = np.array([run.summary.get(f"{target_metric_prefix}/t{t}/{metric}") for t in times], dtype=np.float32)
    assert all([m is not None for m in metrics]), metrics
    if any(np.isnan(metrics)):
        print(f"NaN found in {run.name} (id={run.id})")
        run.summary[time_avg_metric_name] = np.nan
        return True

    if relative_run is not None:
        cached_key = f"{relative_run.id}_{times[0]}_{times[-1]}"
        if cache and cached_key not in CACHE:
            base_metrics = np.array([relative_run.summary.get(f"{target_metric_prefix}/t{t}/{metric}") for t in times])
            assert all([m is not None for m in base_metrics]), base_metrics
            CACHE[cached_key] = base_metrics

        base_metrics = CACHE[cached_key]
        # Compute relative metric (in %)
        metrics = (metrics - base_metrics) / base_metrics * 100

    try:
        time_avg = np.mean(metrics)
    except Exception as e:
        print(f"Error: {e}. Run: {run.name}. ID: {run.id}")
        print(f"Metrics: {metrics}")
        return False
    run.summary[time_avg_metric_name] = time_avg
    return time_avg


def wandb_run_summary_update(wandb_run: wandb.apis.public.Run):
    try:
        wandb_run.summary.update()
    except wandb.errors.CommError:
        logging.warning("Could not update wandb summary")
    # except requests.exceptions.HTTPError or requests.exceptions.ConnectionError:
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        # try again
        wandb_run.summary.update()
    except TypeError:
        pass  # wandb_run.update()
