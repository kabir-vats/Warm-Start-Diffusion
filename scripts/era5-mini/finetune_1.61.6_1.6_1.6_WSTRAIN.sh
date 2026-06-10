#!/bin/bash

cd "$(dirname "$0")"/../..
# export name of file as environment variable (to know which script created the run) if not already set
if [ -z "$SCRIPT_NAME" ]; then
  export SCRIPT_NAME=$(basename "$0")
fi
# Run the training script
# "$@" makes sure to use any extra command line arguments supplied here with bash <script>.sh <args>
# ++diffusion.channel_noise_mult='[1,2,1.905,1.6,1.176,0.556,0.6667,0.364]'\
# ++diffusion.channel_noise_mult='[1,0.5,0.525,0.625,0.85,1.8,1.5,2.75]'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run.py \
  experiment=era5_edm_unet datamodule=era5_mini logger.wandb.project=ERA5_MINI \
  ++datamodule.possible_initial_times_eval=[0,12] \
  ++datamodule.train_slice=[["1979-01-01T00:00:00","2019-12-31T23:00:00"],["1979-01-01T06:00:00","2019-12-31T23:00:00"]] \
  datamodule.hourly_resolution=12 datamodule.prediction_horizon=15 \
  datamodule.prediction_horizon_long=null ++module.inference_val_every_n_epochs=null datamodule.window=2 \
  trainer.num_sanity_val_steps=0 datamodule.eval_batch_size=4 datamodule.batch_size=128 \
  module.monitor="val/avg/crps_normed" callbacks.model_checkpoint_t2m=null \
  model.upsample_dims=null trainer.deterministic=False trainer.benchmark=False \
  module.residual_pred=True model.model_channels=256 \
  model.num_training_ensemble_members=null module.num_predictions=5 \
  model.dropout=0.1 module.enable_inference_dropout=False module.scheduler.warmup_steps=500 \
  trainer.max_epochs=null ++trainer.max_steps=4000 \
  diffusion.loss_function=wmse diffusion.P_mean=-1.6 diffusion.P_std=1.6 diffusion.sigma_max_inf=400 \
  diffusion.sigma_min=0.002 diffusion.num_steps=20 ++module.learned_channel_variance_loss=False ++diffusion.use_noise_logvar=True \
  module.optimizer.lr=3e-4 module.optimizer.muon.lr=3e-3 module.optimizer.weight_decay=0.1 module.optimizer.muon.wd=0 \
  module.torch_compile=null ++module.dyn_cond_from_inputs=False ++model.train_ensemble_type="batched" \
  ++datamodule.loss_latitude_weighting=True ++diffusion.compute_loss_per_sigma=True \
  'model.channel_mult=[1,2,4]' ++datamodule.max_val_samples=512  \
  ++module.allow_validation_size_indivisible_on_ddp=True \
  ++module.empty_cache_at_autoregressive_step=True datamodule.num_workers=8 \
  ++datamodule.loss_pressure_weighting=null ++datamodule.loss_surface_vars_weighting=null \
  ++datamodule.loss_latitude_weighting=True \
  name_suffix="EDM-12h-W2r-3l_WSFT-1" suffix="128ebs_pm-1.6_pstd1.6" \
  ++module.from_pretrained_checkpoint_run_id="50740391" ++module.from_pretrained_checkpoint_filename="last.ckpt" \
  ++diffusion.warm_start=True ++diffusion.warm_start_steps=6 ++diffusion.warm_start_dropout=True \
  ++diffusion.guidance_ckpt_filename="last.ckpt" ++diffusion.guidance_run_id="51297606" \
  ++diffusion.warm_start_train=True ++diffusion.warm_start_max=0.9 ++diffusion.warm_start_min=0.002 \
  "$@"
