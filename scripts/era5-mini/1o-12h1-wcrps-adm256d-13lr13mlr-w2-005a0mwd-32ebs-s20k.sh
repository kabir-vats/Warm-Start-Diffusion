#!/bin/bash

cd "$(dirname "$0")"/../..
# export name of file as environment variable (to know which script created the run) if not already set
if [ -z "$SCRIPT_NAME" ]; then
  export SCRIPT_NAME=$(basename "$0")
fi
# Run the training script
# "$@" makes sure to use any extra command line arguments supplied here with bash <script>.sh <args>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run.py \
  experiment=era5_adm_mini datamodule=era5_mini logger.wandb.project=ERA5_MINI \
  ++datamodule.possible_initial_times_eval=[0,12] \
  ++datamodule.train_slice=[["1979-01-01T00:00:00","2019-12-31T23:00:00"],["1979-01-01T06:00:00","2019-12-31T23:00:00"]] \
  datamodule.hourly_resolution=12 datamodule.prediction_horizon=15 \
  datamodule.prediction_horizon_long=null ++module.inference_val_every_n_epochs=null datamodule.window=2 \
  trainer.num_sanity_val_steps=0 datamodule.eval_batch_size=4 datamodule.batch_size=32 \
  module.monitor="val/avg/rmse_normed" callbacks.model_checkpoint_t2m=null \
  model.upsample_dims=null trainer.deterministic=False trainer.benchmark=False \
  module.residual_pred=True model.model_channels=256 \
  model.loss_function="wcrps" model.num_training_ensemble_members=2 module.num_predictions=5 \
  model.dropout=0.1 module.enable_inference_dropout=True module.scheduler.warmup_steps=500 \
  trainer.max_epochs=10 \
  module.optimizer.lr=7e-5 module.optimizer.muon.lr=7e-3 module.optimizer.weight_decay=0.1 module.optimizer.muon.wd=0 \
  module.torch_compile=null ++module.dyn_cond_from_inputs=False ++model.train_ensemble_type="batched" \
  ++datamodule.loss_latitude_weighting=True module.learned_channel_variance_loss=False \
  'model.channel_mult=[1,2,4]' ++datamodule.max_val_samples=512 \
  ++module.empty_cache_at_autoregressive_step=True datamodule.num_workers=8 \
  ++datamodule.loss_pressure_weighting=null ++datamodule.loss_surface_vars_weighting=null \
  ++datamodule.loss_latitude_weighting=True \
  name_suffix="1.5o-12h-W2r-MC-3layers_73MLR" suffix="CRPS-2" \
  ++module.from_pretrained_checkpoint_run_id="50730599" ++module.from_pretrained_checkpoint_filename="ERA5-12h-1AR_1.5o-12h-W2r-MC-3layers_333lr_epoch010_seed11.ckpt" \
  "$@"