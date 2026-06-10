from __future__ import annotations

from typing import Any, Dict

import hydra
import pytorch_lightning
import torch
from omegaconf import DictConfig

from src.datamodules.abstract_datamodule import BaseDataModule
from src.experiment_types._base_experiment import BaseExperiment
from src.utilities.utils import (
    get_logger,
)


"""
In this file you can find helper functions to avoid model/data loading and reloading boilerplate code
"""

log = get_logger(__name__)


def get_lightning_module(config: DictConfig, **kwargs) -> BaseExperiment:
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


def get_datamodule(config: DictConfig) -> BaseDataModule:
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


def get_model_and_data(config: DictConfig) -> (BaseExperiment, BaseDataModule):
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


class NoTorchModuleWrapper:
    """A wrapper to avoid registering the model as a torch module"""

    def __init__(self, module: torch.nn.Module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def __setattr__(self, name, value):
        if name == "module":
            super().__setattr__(name, value)  # avoid recursion
        else:
            setattr(self.module, name, value)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def get_simple_trainer(**kwargs) -> pytorch_lightning.Trainer:
    devices = kwargs.get("devices", 1 if torch.cuda.is_available() else None)
    accelerator = kwargs.get("accelerator", "gpu" if torch.cuda.is_available() else None)
    return pytorch_lightning.Trainer(
        devices=devices,
        accelerator=accelerator,
        **kwargs,
    )


def run_inference(
    module: pytorch_lightning.LightningModule,
    datamodule: pytorch_lightning.LightningDataModule,
    trainer: pytorch_lightning.Trainer = None,
    trainer_kwargs: Dict[str, Any] = None,
):
    trainer = trainer or get_simple_trainer(**(trainer_kwargs or {}))
    results = trainer.predict(module, datamodule=datamodule)
    results = module._evaluation_get_preds(results, split="predict")
    if hasattr(datamodule, "numpy_results_to_xr_dataset"):
        results = datamodule.numpy_results_to_xr_dataset(results, split="predict")
    return results
