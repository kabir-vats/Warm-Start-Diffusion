# Warm-Start Diffusion ERA5-MINI

This repository contains the pieces needed for the assignment training setup:

- ADM/Dhariwal U-Net backbone for ERA5-MINI.
- ERA5-MINI datamodule and WeatherBench-style evaluation helpers.
- EDM diffusion, warm-start EDM, weighted MAE, and weighted CRPS objectives.
- The four ERA5-MINI training launchers under `scripts/era5-mini/`.

The data paths in the configs still point at the server locations used by the
original experiments. Override `datamodule.data_dir`, `datamodule.dataset`, and
`datamodule.data_dir_stats` on the command line if your server paths differ.

## Setup

```bash
conda env create -f environment/env_train_gpu.yaml
conda activate warm-start-era5-mini
pip install -e .
```

## Training Scripts

```bash
bash scripts/era5-mini/15o-12h1-wmse-edm256d-33lr33mlr-128ebs-uniform.sh
bash scripts/era5-mini/finetune_1.61.6_1.6_1.6_WSTRAIN.sh
bash scripts/era5-mini/1o-12h1-wmae-adm256d-13lr13mlr-w2-005a0mwd-32ebs-s20k.sh
bash scripts/era5-mini/1o-12h1-wcrps-adm256d-13lr13mlr-w2-005a0mwd-32ebs-s20k.sh
```

All launchers pass extra trailing arguments through to Hydra, for example:

```bash
bash scripts/era5-mini/15o-12h1-wmse-edm256d-33lr33mlr-128ebs-uniform.sh trainer.devices=1 logger.wandb.mode=offline
```
