import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
import numpy as np
import pytorch_lightning
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, ListConfig

from src.utilities import wandb_api
from src.utilities.utils import get_logger


log = get_logger(__name__)

try:
    torch.serialization.add_safe_globals([ListConfig])
except AttributeError:
    log.warning("torch.serialization.add_safe_globals([ListConfig]) not supported in this version of PyTorch")


def get_lightning_module(config: DictConfig, **kwargs):
    r"""Get the ML model, a subclass of :class:`~src.experiment_types._base_experiment.BaseExperiment`, as defined by the key value pairs in ``config.model``.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)
        **kwargs: Any additional keyword arguments for the model class (overrides any key in config, if present)

    Returns:
        BaseExperiment:
            The lightning module that you can directly use to train with pytorch-lightning

    Examples:

    .. code-block:: python

        from src.utilities.config_utils import get_config_from_hydra_compose_overrides

        config_mlp = get_config_from_hydra_compose_overrides(overrides=['model=mlp'])
        mlp_model = get_model(config_mlp)

        # Get a prediction for a (B, S, C) shaped input
        random_mlp_input = torch.randn(1, 100, 5)
        random_prediction = mlp_model.predict(random_mlp_input)
    """
    model = hydra.utils.instantiate(
        config.module,
        model_config=config.model,
        datamodule_config=config.datamodule,
        diffusion_config=config.get("diffusion", default_value=None),
        _recursive_=False,
        **kwargs,
    )

    return model


def get_datamodule(config: DictConfig):
    r"""Get the datamodule, as defined by the key value pairs in ``config.datamodule``. A datamodule defines the data-loading logic as well as data related (hyper-)parameters like the batch size, number of workers, etc.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)

    Returns:
        Base_DataModule:
            A datamodule that you can directly use to train pytorch-lightning models

    Examples:

    .. code-block:: python

        from src.utilities.config_utils import get_config_from_hydra_compose_overrides

        cfg = get_config_from_hydra_compose_overrides(overrides=['datamodule=icosahedron', 'datamodule.order=5'])
        ico_dm = get_datamodule(cfg)
    """
    data_module = hydra.utils.instantiate(
        config.datamodule,
        _recursive_=False,
        model_config=config.model,
    )
    return data_module


def get_model_and_data(config: DictConfig):
    r"""Get the model and datamodule. This is a convenience function that wraps around :meth:`get_model` and :meth:`get_datamodule`.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)

    Returns:
        (BaseExperiment, Base_DataModule): A tuple of (module, datamodule), that you can directly use to train with pytorch-lightning

    Examples:

    .. code-block:: python

        from src.utilities.config_utils import get_config_from_hydra_compose_overrides

        cfg = get_config_from_hydra_compose_overrides(overrides=['datamodule=icosahedron', 'model=mlp'])
        mlp_model, icosahedron_data = get_model_and_data(cfg)

        # Use the data from datamodule (its ``train_dataloader()``), to train the model for 10 epochs
        trainer = pl.Trainer(max_epochs=10, devices=1)
        trainer.fit(model=model, datamodule=icosahedron_data)

    """
    data_module = get_datamodule(config)
    model = get_lightning_module(config)
    if config.module.get("torch_compile") == "module":
        log.info("Compiling LightningModule with torch.compile()...")
        model = torch.compile(model)
    return model, data_module


def reload_model_from_config_and_ckpt(
    config: DictConfig,
    ckpt_path: str,
    device: Optional[torch.device] = None,
    also_datamodule: bool = False,
    also_ckpt: bool = False,
    model: pytorch_lightning.LightningModule = None,
    use_ema_weights_only: bool = False,
    reload_strict: bool = False,
    exclude_state_dict_keys: List[str] = None,
    print_name: str = "",
) -> Dict[str, Any]:
    r"""Load a model as defined by ``config.model`` and reload its weights from ``model_path``.

    Args:
        config (DictConfig): The config to use to reload the model
        ckpt_path (str): The path to the model checkpoint (its weights)
        device (torch.device): The device to load the model on. Defaults to 'cuda' if available, else 'cpu'.
        also_datamodule (bool): If True, also reload the datamodule from the config. Defaults to True.
        also_ckpt (bool): If True, also returns the checkpoint from ``model_path``. Defaults to False.
        model (LightningModule): If provided, the model to reload the weights into. If None, a new model is instantiated.
        use_ema_weights_only (bool): If True, only the EMA weights are loaded. Defaults to False.
        reload_strict (bool): If True, the model weights are loaded strictly (i.e. all keys must match). Defaults to False.
        exclude_state_dict_keys (List[str]): A list of keys to exclude from the state_dict when loading the model. Defaults to None.
        print_name (str): A string to print when reloading the model. Defaults to "".

    Returns:
        BaseModel: The reloaded model if load_datamodule is ``False``, otherwise a tuple of (reloaded-model, datamodule)

    Examples:

    .. code-block:: python

        # If you used wandb to save the model, you can use the following to reload it
        from src.utilities.wandb_api import load_hydra_config_from_wandb

        run_path = ENTITY/PROJECT/RUN_ID   # wandb run id (you can find it on the wandb URL after runs/, e.g. 1f5ehvll)
        config = load_hydra_config_from_wandb(run_path, override_kwargs=['datamodule.num_workers=4', 'trainer.gpus=-1'])

        model, datamodule = reload_model_from_config_and_ckpt(config, model_path, load_datamodule=True)

        # Test the reloaded model
        trainer = hydra.utils.instantiate(config.trainer, _recursive_=False)
        trainer.test(model=model, datamodule=datamodule)

    """
    if model is None:
        model_provided = False
        model, data_module = get_model_and_data(config) if also_datamodule else (get_lightning_module(config), None)
    else:
        model_provided = True
        assert not also_datamodule, "If model is provided, also_datamodule must be False"
        data_module = None
    # Reload model
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # rename weights (sometimes needed for backwards compatibility)
    state_dict, has_been_renamed = rename_state_dict_keys_and_save(model_state, ckpt_path, model, use_ema_weights_only)
    reload_strict = reload_strict or has_been_renamed  # force strict if we renamed the state dict keys
    # Reload weights
    state_dict = load_state_dict_and_analyze_weight_changes(
        model, state_dict, strict=reload_strict, exclude_keys=exclude_state_dict_keys
    )
    # Using the returned state_dict, ensures that
    # we don't pass on any excluded keys (e.g. for freezing all but those layers)

    to_return = {
        "model": model,
        "datamodule": data_module,
        "state_dict": state_dict,
        "epoch": model_state.get("epoch"),
        "global_step": model_state.get("global_step"),
        "wandb": model_state.get("wandb", None),
    }
    file_size = os.path.getsize(ckpt_path)
    print_name = f"{print_name}: " if print_name else ""
    str_to_print = f"Reloaded {print_name}{ckpt_path}" + (" into provided model" if model_provided else "") + "."
    if model_state.get("epoch") is not None:
        str_to_print += f" Epoch={model_state['epoch']}"
    if model_state.get("global_step") is not None:
        str_to_print += f" Global_step={model_state['global_step']}"
    str_to_print += f" File size [in MB]: {file_size / 1e6:.2f}."
    if model_state.get("wandb") is not None:
        str_to_print += f"\nRun ID: {model_state['wandb']['id']}\t Name: {model_state['wandb']['name']}"
    log.info(str_to_print)
    if also_ckpt:
        to_return["ckpt"] = model_state
    return to_return


def get_checkpoint_from_path_or_wandb(
    model_checkpoint: Optional[torch.nn.Module] = None,
    model_checkpoint_path: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
    model_name: Optional[str] = "model",
    reload_kwargs: Optional[Dict[str, Any]] = None,
    model_overrides: Optional[List[str]] = None,
) -> torch.nn.Module:
    if model_checkpoint is not None:
        assert model_checkpoint_path is None, "must provide either model_checkpoint or model_checkpoint_path"
        assert wandb_run_id is None, "must provide either model_checkpoint or wandb_run_id"
        model = model_checkpoint
    # elif model_checkpoint_path is not None:
    #     raise NotImplementedError('Todo: implement loading from checkpoint path')
    #     assert wandb_run_path is None, 'must provide either model_checkpoint or wandb_run_path'
    #
    elif wandb_run_id is not None:
        # assert model_checkpoint_path is None, 'must provide either wandb_run_path or model_checkpoint_path'
        override_key_value = model_overrides or []
        override_key_value += ["module.verbose=False"]
        reload_kwargs = reload_kwargs or {}
        model = reload_checkpoint(
            run_id=wandb_run_id,
            override_key_value=override_key_value,
            local_checkpoint_path=model_checkpoint_path,
            **reload_kwargs,
        )["model"]
    else:
        raise ValueError("Provide either model_checkpoint, model_checkpoint_path or wandb_run_id")
    return model


def reload_checkpoint(
    run_id: Optional[str] = None,
    config: DictConfig = None,
    ckpt_filename: Optional[str] = None,
    epoch: Union[str, int] = "best",
    override_key_value: List[str] = None,
    local_checkpoint_path: str = None,
    use_ema_weights_only: bool = False,
    entity: str = None,
    project: str = None,
    ckpt_path_only: bool = False,
    **reload_kwargs,
) -> dict:
    """
    Reload model checkpoint based on only the Wandb run ID or a local checkpoint path.

    Args:
        run_id (Optional[str]): the wandb run ID (e.g. 2r0l33yc) corresponding to the model to-be-reloaded. If None, config must be provided.
        config (DictConfig): If provided, the config to use to reload the model. If None, the config is loaded from Wandb using run_id.
        ckpt_filename (str): the filename of the checkpoint to be reloaded (e.g. 'last.ckpt')
        override_key_value: each element is expected to have a "=" in it, like datamodule.num_workers=8
        local_checkpoint_path (str): If not None, the path to the local checkpoint to be reloaded.
        entity (str): the wandb entity corresponding to the model to-be-reloaded. Only used if config is None.
        project (str): the project entity corresponding to the model to-be-reloaded. Only used if config is None.
        ckpt_path_only (bool): If True, only return the path to the checkpoint, not the model itself. Defaults to False.
    """
    if run_id is not None:
        run_id = str(run_id).strip()

    if use_ema_weights_only:
        override_key_value = override_key_value or []
        override_key_value += ["module.use_ema=False"]  # EMA weights are reloaded into normal model, rest is discarded

    # Load configuration object
    run_path = None
    if config is None:
        if run_id is None:
            raise ValueError("Either run_id or config must be provided")
        entity, project = wandb_api.get_entity(entity), project or wandb_api.get_project_train()
        run_path = f"{entity}/{project}/{run_id}"
        config = wandb_api.load_hydra_config_from_wandb(
            run_path, local_dir=local_checkpoint_path, override_key_value=override_key_value
        )
    elif isinstance(config, str):
        config = get_local_path_and_download_if_needed(config)
        if os.path.isfile(config):
            # If config is a path to a file, load it
            config = compose_hydra_config(config_path=config, overrides=override_key_value)
        else:
            raise ValueError(f"{config=} is not a valid path to a config file ({os.getcwd()=}).")

    # Infer path to checkpoint
    if local_checkpoint_path is True:
        # Find local checkpoint
        assert ckpt_filename is not None, "If local_checkpoint_path is True, please specify ckpt_filename"
        if run_id is None:
            raise ValueError("Cannot use local_checkpoint_path=True without run_id")
        local_checkpoint_path = get_local_ckpt_path(
            config,
            wandb_run_id=run_id,
            ckpt_filename=ckpt_filename,
            throw_error_if_local_not_found=False,
        )
        print(local_checkpoint_path)
    if isinstance(local_checkpoint_path, (str,)):
        ckpt_path = get_local_path_and_download_if_needed(local_checkpoint_path, ckpt_filename, abs_path=True)
        if os.path.isdir(ckpt_path):
            if run_id and os.path.exists(os.path.join(ckpt_path, run_id, ckpt_filename)):
                ckpt_path = os.path.join(ckpt_path, run_id, ckpt_filename)
            else:
                ckpt_path = os.path.join(ckpt_path, ckpt_filename)
        assert os.path.isfile(ckpt_path), f"Could not find local ckpt {ckpt_path=} in {os.getcwd()}"
        log.info(f"Restoring model from local absolute path: {ckpt_path}")
    else:
        # Restore model from wandb cloud or S3 storage
        run_path = f"{entity}/{project}/{run_id}"
        ckpt_path = wandb_api.restore_model_from_wandb_cloud(run_path, epoch=epoch, ckpt_filename=ckpt_filename)

    assert isinstance(local_checkpoint_path, str), f"{type(local_checkpoint_path)=} not a str"
    train_id = config.logger.wandb.get("training_id", None)
    possible_ids = [str(config.logger.wandb.id), str(train_id) if train_id is not None else None]
    assert str(run_id) in possible_ids, f"{run_id=} not in {possible_ids}."
    if ckpt_path_only:
        return ckpt_path
    reloaded_model_data = reload_model_from_config_and_ckpt(
        config, ckpt_path, use_ema_weights_only=use_ema_weights_only, **reload_kwargs
    )
    if reloaded_model_data.get("wandb") is not None:
        reloaded_id = reloaded_model_data["wandb"].get("id")
        if reloaded_id != run_id and str(reloaded_id) not in str(run_id):
            reloaded_model_data["wandb"]["id"] = run_id #override
            #raise ValueError(f"run_id={run_id} != state_dict['wandb']['id']={reloaded_model_data['wandb']['id']}")
    # config.trainer.resume_from_checkpoint = ckpt_path
    # os.remove(ckpt_path) if os.path.exists(ckpt_path) else None  # delete the downloaded ckpt
    # Reload weights
    return {**reloaded_model_data, "config": config, "ckpt_path": ckpt_path}


def compose_hydra_config(config_path: str = "../configs", overrides: List[str] = None) -> DictConfig:
    """Compose a Hydra configuration from a given config path and overrides."""
    GlobalHydra.instance().clear()
    if config_path.endswith(".yaml"):
        config_file = os.path.basename(config_path)
        config_path = os.path.dirname(config_path)
    else:
        config_file = "main_config.yaml"
    if "src/" in config_path:
        # Hydra requires relative path (relative to this __file__)
        config_path = config_path.replace("src/", "../")
    with hydra.initialize(version_base=None, config_path=config_path):
        config = hydra.compose(config_name=config_file, overrides=overrides)
    return config


def find_wandb_run_dir(local_dir: str, run_id: str) -> str:
    """
    Find any subdirectory of local_dir that contains wandb_run.id as one of its subdirectories.

    Args:
        local_dir (str): Root directory to start the search
        run_id (str): WandB run ID to look for

    Returns:
        str: Full path of the directory containing the run_id, or None if not found

    Example:
        find_wandb_run_dir("/home", "4628219")
        # might return "/home/checkpoints/4628219" if it exists
    """
    local_path = Path(local_dir)

    # Walk through all subdirectories
    for root, dirs, _ in os.walk(local_path):
        # Check if run_id matches any directory name
        if run_id in dirs:
            print(f"Found {run_id=} ckpt in {root=}, {dirs=}")
            return str(Path(root) / run_id)

    return None


def get_local_ckpt_path(
    config: DictConfig,
    wandb_run_id: str,
    ckpt_filename: str = "last.ckpt",
    throw_error_if_local_not_found: bool = False,
) -> Optional[str]:
    potential_dirs = []
    work_dirs = [config.work_dir.replace("-test", ""), os.environ.get("WORK_DIR", None)]
    for work_dir in work_dirs:
        if work_dir is None or not os.path.exists(work_dir):
            continue
        potential_dirs.extend(
            [
                # config.ckpt_dir,
                os.path.join(work_dir, "checkpoints"),
                os.path.join(work_dir, wandb_run_id, "checkpoints"),
                os.path.join(os.path.dirname(work_dir), "checkpoints", wandb_run_id),
                os.path.join(os.getcwd(), "results", "checkpoints"),
            ]
        )
    if os.environ.get("PSCRATCH", None) is not None:
        for script_dir in ["ns", "sm", "", "era5"]:
            potential_dirs.append(
                os.path.join(os.environ["PSCRATCH"], "results", script_dir, "checkpoints", wandb_run_id)
            )
            potential_dirs.append(
                os.path.join(os.environ["PSCRATCH"], "results", script_dir, wandb_run_id, "checkpoints")
            )

    for callback_k in config.get("callbacks", {}).keys():
        if "checkpoint" in callback_k and config.callbacks[callback_k] is not None:
            if config.callbacks[callback_k].get("dirpath", None) is not None:
                potential_dirs.append(config.callbacks[callback_k].dirpath)

    for i, local_dir in enumerate(potential_dirs):
        # log.info(f"Checking {local_dir}. {os.path.exists(local_dir)=}, {wandb_run_id=}")
        if not os.path.exists(local_dir):
            continue
        if wandb_run_id not in local_dir:
            # Find any subdir of local_dir with wandb_run.id as one subdir
            if os.path.exists(os.path.join(local_dir, wandb_run_id)):
                local_dir = os.path.join(local_dir, wandb_run_id)
            else:
                local_dir = find_wandb_run_dir(local_dir, wandb_run_id)
                if local_dir is None:
                    continue
            if os.path.exists(os.path.join(local_dir, "checkpoints")):
                local_dir = os.path.join(local_dir, "checkpoints")

        ckpt_files = [f for f in os.listdir(local_dir) if f.endswith(".ckpt")]
        if ckpt_filename == "last.ckpt":
            ckpt_files = [f for f in ckpt_files if "last" in f]
            if len(ckpt_files) == 0:
                continue
            elif len(ckpt_files) == 1:
                latest_ckpt_file = ckpt_files[0]
            else:
                # Get their epoch numbers from inside the file
                # epochs = [torch.load(os.path.join(local_dir, f), weights_only=True)["epoch"] for f in ckpt_files]
                epochs = []
                for f in ckpt_files:
                    try:
                        ep = torch.load(os.path.join(local_dir, f), weights_only=False, map_location="cpu")["epoch"]
                        epochs.append(ep)
                    except RuntimeError:
                        log.warning(f"Could not load {f=} to get epoch number. Skipping this file.")
                        continue
                # Find the ckpt file with the latest epoch
                latest_ckpt_file = ckpt_files[np.argmax(epochs)]
                log.info(
                    f"Found multiple last-v<V>.ckpt files. Using the one with the highest epoch: {latest_ckpt_file}. ckpt_to_epoch: {dict(zip(ckpt_files, epochs))}"
                )
            return os.path.join(local_dir, latest_ckpt_file)

        elif ckpt_filename in ["earliest_epoch", "latest_epoch", "earliest_epoch_any", "latest_epoch_any"]:
            # Find the earliest epoch ckpt file
            # Ckpt dir has files like:
            # - Kolmogorov-H32-ERDM-1.0t-edm-0.002-80.0sigma_128x128-Vl_epoch103_seed11.ckpt
            # - Kolmogorov-H32-ERDM-1.0t-edm-0.002-80.0sigma_128x128-Vl_epoch033_seed11.ckpt
            if ckpt_filename in ["earliest_epoch_any", "latest_epoch_any"]:
                ckpt_files = [f for f in ckpt_files if "epoch" in f]
            else:
                ckpt_files = [f for f in ckpt_files if "epoch" in f and "epochepoch=" not in f and "_step" not in f]
            if len(ckpt_files) == 0:
                continue

            # Function to extract the epoch number from the filename
            def get_epoch_number(filename):
                if "_any" in ckpt_filename:
                    filename = filename.replace("epochepoch=", "epoch")  # Fix for a bug in the filename
                match = re.search(r"_epoch(\d+)_", filename)
                if match is None:
                    log.warning(f"Could not find epoch number in {filename=}. Skipping this file.")
                    return -1
                return int(match.group(1))

            # Find the ckpt file with the earliest epoch
            min_or_max = min if ckpt_filename == "earliest_epoch" else max
            earliest_ckpt_file = min_or_max(ckpt_files, key=lambda f: get_epoch_number(f))
            log.info(f"For ckpt_filename={ckpt_filename}, found ckpt file: {earliest_ckpt_file} in {local_dir}")
            return os.path.join(local_dir, earliest_ckpt_file)

        ckpt_path = os.path.join(local_dir, ckpt_filename)
        print(ckpt_path)
        if os.path.exists(ckpt_path):
            return ckpt_path
        else:
            log.warning(f"{local_dir} exists but could not find {ckpt_filename=}. Files in dir: {ckpt_files}.")
    if ckpt_filename in ["earliest_epoch", "latest_epoch", "earliest_epoch_any", "latest_epoch_any"]:
        raise NotImplementedError(f"Could not find {ckpt_filename=} in any of the potential dirs: {potential_dirs}")
    if throw_error_if_local_not_found:
        raise FileNotFoundError(
            f"Could not find ckpt file {ckpt_filename} in any of the potential dirs: {potential_dirs}"
        )
    return None


def load_state_dict_and_analyze_weight_changes(
    model, state_dict, strict=False, exclude_keys: List[str] = None, num_examples=8
):
    """
    Analyzes weight changes after loading partial state dict.
    Args:
        model: PyTorch model
        state_dict: State dict to load
        strict: Whether to strictly enforce that the keys in state_dict match the keys returned by model.state_dict()
        exclude_keys: List of keys to exclude from the state_dict (won't be loaded!)
        num_examples: Number of first/last layers to show
    """
    exclude_keys = exclude_keys or []
    if not isinstance(exclude_keys, list):
        exclude_keys = list(exclude_keys)
    # add _orig_mod. prefix for when using torch compile
    exclude_keys += [ek.replace("model.", "model._orig_mod.") for ek in exclude_keys]
    exclude_keys = set(exclude_keys)
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}
    unloaded_keys = [k for k in state_dict.keys() if k not in orig_state or k in exclude_keys]
    if exclude_keys:
        assert not strict, "Cannot exclude keys when strict=True"
        state_dict = {k: v for k, v in state_dict.items() if k not in exclude_keys}

    model.load_state_dict(state_dict, strict=strict)

    changed = []
    unchanged = []
    total_params = changed_params = 0

    for key, new_val in model.state_dict().items():
        if key in orig_state:
            params = new_val.numel()
            total_params += params
            if not torch.equal(orig_state[key], new_val):
                num_changed = (orig_state[key] != new_val).sum().item()
                changed_params += num_changed
                changed.append(f"{key}: {num_changed}/{params} params ({num_changed / params * 100:.1f}%)")
            else:
                unchanged.append(key)

    # If (almost) fully changed, no unloaded keys then no need to print the analysis
    if round(changed_params / total_params * 100) == 100 and len(unloaded_keys) == 0:
        return

    output_str = [
        f"Total reloaded parameters: {changed_params:,}/{total_params:,} ({changed_params / total_params * 100:.1f}%)"
    ]

    if changed:
        output_str.append(f"Changed (i.e. reloaded) layers ({len(changed)} total; showing params changed/total):")
        output_str.extend(changed[:num_examples])
        remaining = len(changed) - 2 * num_examples
        if remaining > 0:
            output_str.append(f"... and {remaining} layers in between ...")
        if len(changed) > num_examples:
            output_str.extend(changed[-num_examples:])
    else:
        output_str.append("No layers were changed")

    if unchanged:
        output_str.append(f"\nUnchanged layers (layers without any change; {len(unchanged)} total):")
        output_str.extend(unchanged[:num_examples])
        remaining = len(unchanged) - 2 * num_examples
        if remaining > 0:
            output_str.append(f"... and {remaining} layers in between ...")
        if len(unchanged) > num_examples:
            output_str.extend(unchanged[-num_examples:])
    else:
        output_str.append("No layers were unchanged")

    if unloaded_keys:
        output_str.append(
            f"\nUnloaded keys (layers in the state dict but not in the model; {len(unloaded_keys)} total):"
        )
        output_str.extend(unloaded_keys[:num_examples])
        remaining = len(unloaded_keys) - 2 * num_examples
        if remaining > 0:
            output_str.append(f"... and {remaining} keys in between ...")
        if len(unloaded_keys) > num_examples:
            output_str.extend(unloaded_keys[-num_examples:])
    else:
        output_str.append("No unloaded keys")

    log.info("\n".join(output_str))
    return state_dict


def analyze_weight_changes_concise(model, state_dict, n_examples: int = 4):
    """Analyzes weight changes after loading partial state dict."""
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state_dict, strict=False)

    changed = []
    unchanged = []
    total_params = changed_params = 0

    for key, new_val in model.state_dict().items():
        if key in orig_state:
            params = new_val.numel()
            total_params += params
            if not torch.equal(orig_state[key], new_val):
                num_changed = (orig_state[key] != new_val).sum().item()
                changed_params += num_changed
                changed.append(f"{key}: {num_changed}/{params} params ({num_changed / params * 100:.1f}%)")
            else:
                unchanged.append(key)

    log.info(
        f"Total changed parameters: {changed_params:,}/{total_params:,} ({changed_params / total_params * 100:.1f}%)"
    )
    log.info(
        "Changed layers (showing params changed/total): "
        + ";  ".join(changed[:n_examples])
        + (f"\n... and {len(changed) - n_examples} more layers" if len(changed) > n_examples else "")
    )
    log.info(
        f"Unchanged layers (Layers without any change; {len(unchanged)} total): "
        + ";  ".join(unchanged[:n_examples])
        + (f"\n... and {len(unchanged) - n_examples} more layers" if len(unchanged) > n_examples else "")
    )


def rename_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> (Dict[str, torch.Tensor], bool):
    #  Missing key(s) in state_dict: "model.downs.0.2.fn.fn.to_qkv.1.weight", "model.downs.1.2.fn.fn.to_qkv.1.weight",
    #  Unexpected key(s) in state_dict: "model.downs.0.2.fn.fn.to_qkv.weight", "model.downs.1.2.fn.fn.to_qkv.weight",
    # rename weights
    renamed = False
    for k in list(state_dict.keys()):
        if "fn.to_qkv.weight" in k and "mid_attn" not in k:
            state_dict[k.replace("fn.to_qkv.weight", "fn.to_qkv.1.weight")] = state_dict.pop(k)
            renamed = True

    return state_dict, renamed


def rename_state_dict_keys_and_save(
    torch_model_state, ckpt_path: str, model: nn.Module, use_ema_weights_only: bool = False
) -> (Dict[str, torch.Tensor], bool):
    """Renames the state dict keys and saves the renamed state dict back to the checkpoint."""
    state_dict, has_been_renamed = rename_state_dict_keys(torch_model_state["state_dict"])
    if has_been_renamed:
        # Save the renamed model state
        torch_model_state["state_dict"] = state_dict
        torch.save(torch_model_state, ckpt_path)
    # Check if model XOR state_dict are (not) torch.compiled.
    # If one of them is but the other is not, we not to remove (or add) _orig_mod. to the state_dict keys.
    #  However, we don't want to save the edited state_dict back to the checkpoint, so we do this here.
    model_is_compiled = hasattr(model, "_orig_mod") or (hasattr(model, "model") and hasattr(model.model, "_orig_mod"))
    state_dict_is_compiled = any(["_orig_mod" in k for k in state_dict.keys()])

    if model_is_compiled and not state_dict_is_compiled:
        # Add _orig_mod to the state_dict keys
        log.info("Adding _orig_mod to the state_dict keys since the model is compiled but ckpt is not.")
        state_dict = {
            k.replace("model.", "model._orig_mod.").replace("model_ema.", "model_ema._orig_mod."): v
            for k, v in state_dict.items()
        }
        has_been_renamed = True
    elif not model_is_compiled and state_dict_is_compiled:
        # Remove _orig_mod from the state_dict keys
        log.info("Removing _orig_mod from the state_dict keys since the model is not compiled but the ckpt was.")
        state_dict = {k.replace("._orig_mod", ".").replace("..", "."): v for k, v in state_dict.items()}
        has_been_renamed = True

    if use_ema_weights_only:
        log.info("Using only EMA weights from the state_dict.")
        # Remove all keys that do not start with "model_ema"
        state_dict_new = {}
        for k in state_dict.keys():
            if "model." in k and "model_ema." not in k:
                # Use EMA values for the model weights with key k
                #   e.g.: model.model.out_norm.weight -> model_ema.modelmap_layer0weight
                k_ema = k.replace(".", "").replace("model", "model_ema.", 1)
                if k_ema in state_dict:
                    state_dict_new[k] = state_dict[k_ema]
                else:
                    log.info(f"Key {k_ema} not found in state_dict. Using non-EMA value for {k}.")
                    state_dict_new[k] = state_dict[k]

        state_dict = state_dict_new
        # state_dict = {k: v for k, v in state_dict.items() if "model_ema." in k}
        # Remove "model_ema." from the keys
        # state_dict = {k.replace("model_ema.", ""): v for k, v in state_dict.items()}
        has_been_renamed = True

    return state_dict, has_been_renamed


def download_model_from_hf(
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    hf_path: Optional[str] = None,
    cache_dir: str = "auto",
):
    """
    Downloads a model file from Hugging Face Hub

    Args:
        repo_id (str): Hugging Face repository ID, e.g. "username/repo_name"
        filename (str): Name of the file to download, e.g. "model.pt"
        hf_path (str): Path to the model on Hugging Face Hub. "<repo_id>/<filename>"
        cache_dir (str): Local directory to save the model

    Returns:
        str: Path to the downloaded file
    """
    if hf_path is not None:
        assert repo_id is None and filename is None, "hf_path should be used alone"
        # After last / is the filename
        repo_id, filename = hf_path.rsplit("/", 1)
    else:
        assert repo_id is not None and filename is not None, "repo_id and filename should be used together"

    if filename.endswith(".ckpt") or filename.endswith(".pt"):
        dtype = "model"
        cache_dir = ".cache/models/" if cache_dir == "auto" else cache_dir
    elif filename.endswith(".yaml"):
        dtype = "config"
        cache_dir = ".cache/configs/" if cache_dir == "auto" else cache_dir
    else:
        dtype = "data"
        cache_dir = ".cache/data/" if cache_dir == "auto" else cache_dir

    os.makedirs(cache_dir, exist_ok=True)

    log.info(f"Downloading {dtype} from HuggingFace Hub: {repo_id}/{filename}. Saving to {cache_dir=}")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)

    return model_path


def get_local_path_and_download_if_needed(path: str, filename: str = None, abs_path: bool = False) -> Optional[str]:
    """
    Convert a local path to an absolute path and download the file if it is a Hugging Face Hub path (starts with "hf:")
    """
    if path is None:
        return None

    if os.path.isfile(path):
        pass
    elif filename is not None and not path.endswith(filename):
        # If a filename is provided, append it to the path
        path = os.path.join(path, filename)

    if path.startswith("hf:"):
        # Reload from Hugging Face if path starts with "hf://" or "hf:"
        hf_path = path.replace("hf://", "").replace("hf:", "")
        path = download_model_from_hf(hf_path=hf_path)

    if abs_path:
        path = os.path.abspath(path)
    return path
