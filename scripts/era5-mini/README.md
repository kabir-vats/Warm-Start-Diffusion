# ERA5-MINI Training Launchers

Only the assignment-relevant ERA5-MINI launchers were copied:

- `15o-12h1-wmse-edm256d-33lr33mlr-128ebs-uniform.sh`: baseline EDM diffusion without warm start.
- `finetune_1.61.6_1.6_1.6_WSTRAIN.sh`: warm-start EDM fine-tuning with warm-start training enabled.
- `1o-12h1-wmae-adm256d-13lr13mlr-w2-005a0mwd-32ebs-s20k.sh`: weighted MAE deterministic/autoregressive ADM objective.
- `1o-12h1-wcrps-adm256d-13lr13mlr-w2-005a0mwd-32ebs-s20k.sh`: weighted CRPS ensemble/autoregressive ADM objective.

Each script changes to the repository root and invokes `python run.py` with the
ERA5-MINI Hydra overrides from the donor repo.
