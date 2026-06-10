import os

import hydra
from omegaconf import DictConfig

from src.train import run_model
from src.utilities.utils import get_logger

log = get_logger(__name__)

if "CONFIG_PATH" in os.environ:
    # Split config path and config name from config path (split by last '/')
    config_path, config_name = os.environ["CONFIG_PATH"].rsplit("/", 1)
    log.info(f"Using config path from environment variable: {os.environ['CONFIG_PATH']}")
else:
    config_path = "src/configs/"
    config_name = "main_config.yaml"


@hydra.main(config_path=config_path, config_name=config_name, version_base=None)
def main(config: DictConfig) -> float:
    """Run/train model based on the config file configs/main_config.yaml (and any command-line overrides)."""
    # import torch
    # torch.multiprocessing.set_sharing_strategy('file_system')  # ulimit -n 32768
    os.environ["WORK_DIR"] = config.work_dir
    return run_model(config)


if __name__ == "__main__":
    if "WANDB_API_KEY" in os.environ:
        import wandb

        wandb.login(key=os.environ["WANDB_API_KEY"])

    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
