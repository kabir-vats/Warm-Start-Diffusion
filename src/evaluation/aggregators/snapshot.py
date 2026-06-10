from typing import Any, List, Mapping, Optional

import matplotlib.pyplot as plt
import torch


try:
    import seaborn as sns

    # Apply Seaborn styles for enhanced aesthetics
    sns.set(
        context="talk", style="white", palette="colorblind", font="serif", font_scale=1, rc={"lines.linewidth": 2.5}
    )
except ImportError:
    pass


class SnapshotAggregator:
    """
    An aggregator that records the first sample of the last batch of data.
    > The way it works is that it gets called once per batch, but in the end (when using get_logs)
    it only returns information based on the last batch.
    """

    _captions = {
        "full-field": "{name} one step full field for last samples; (left) generated and (right) target.",  # noqa: E501
        "residual": "{name} one step residual for last samples; (left) generated and (right) target.",  # noqa: E501
        "error": "{name} one step error (generated - target) for last sample.",
    }

    def __init__(
        self,
        is_ensemble: bool,
        target_time: Optional[int] = None,
        var_names: Optional[List[str]] = None,
        every_nth_epoch: int = 1,
        preprocess_fn=None,
    ):
        self.is_ensemble = is_ensemble
        assert target_time is None or target_time > 0
        self.target_time = target_time  # account for 0-indexing not needed because initial condition is included
        self.target_time_in_batch = None
        self.var_names = var_names
        self.every_nth_epoch = every_nth_epoch
        self.preprocess_fn = preprocess_fn if preprocess_fn is not None else lambda x: x

    @torch.inference_mode()
    def update(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any] = None,
        i_time_start: int = 0,
    ):
        if self.target_time is not None:
            data_steps = target_data_norm[list(target_data_norm.keys())[0]].shape[1]
            diff = self.target_time - i_time_start
            # target time needs to be in the batch (between i_time_start and i_time_start + data_steps)
            if diff < 0 or diff >= data_steps:
                return  # skip this batch, since it doesn't contain the target time
            else:
                self.target_time_in_batch = diff

        def to_cpu(x):
            return {k: v.cpu() for k, v in x.items()} if isinstance(x, dict) else x.cpu()

        self._target_data = to_cpu(target_data)
        self._gen_data = to_cpu(gen_data)
        self._target_data_norm = to_cpu(target_data_norm)
        self._gen_data_norm = to_cpu(gen_data_norm)
        self._metadata = metadata if metadata is not None else {}
        if self.target_time is not None:
            assert (
                self.target_time_in_batch <= data_steps
            ), f"target_time={self.target_time}, time_in_batch={self.target_time_in_batch} is larger than the number of timesteps in the data={data_steps}!"

    @torch.inference_mode()
    def compute(self, prefix: str = "", epoch: int = None):
        """
        Returns logs as can be reported to WandB.

        Args:
            prefix: Label to prepend to all log keys.
            epoch: Current epoch number.
        """
        if self.every_nth_epoch > 1 and epoch >= 3 and epoch % self.every_nth_epoch != 0:
            return None
        if self.target_time_in_batch is None and self.target_time is not None:
            return None  # skip this batch, since it doesn't contain the target time

        image_logs = {}
        max_snapshots = 2  # 3
        names = self.var_names
        if names is None:
            names = list(self._gen_data_norm.keys()) if hasattr(self._gen_data_norm, "keys") else [None]
        for name in names:
            name_label = f"/{name}" if name is not None else ""
            if name is not None and "normed" in name:
                gen_data = self._gen_data_norm
                target_data = self._target_data_norm
                name = name.replace("_normed", "")
            else:
                gen_data = self._gen_data
                target_data = self._target_data

            # Take the first sample in batch
            snapshots_pred = gen_data[name] if name is not None else gen_data
            snapshots_pred = snapshots_pred[:max_snapshots, 0] if self.is_ensemble else snapshots_pred[0].unsqueeze(0)
            target_for_image = target_data[name][0] if name is not None else target_data[0]
            if name is None:
                assert snapshots_pred.shape[1] == 1, f"{snapshots_pred.shape=} but expected 1 variable only ({name=})."
                snapshots_pred = snapshots_pred.squeeze(1)
                target_for_image = target_for_image.squeeze(0)
            target_datetime = self._metadata["datetime"][0] if "datetime" in self._metadata else None
            input_for_image = None
            # Select target time
            if self.target_time is not None:
                snapshots_pred = snapshots_pred[:, self.target_time_in_batch]
                target_for_image = target_for_image[self.target_time_in_batch]
                if input_for_image is not None:
                    input_for_image = input_for_image[self.target_time_in_batch]

            n_ens_members = snapshots_pred.shape[0]
            figsize1 = ((n_ens_members + 1) * 5, 5)
            figsize2 = (n_ens_members * 5, 5)
            fig_full_field, ax_full_field = plt.subplots(
                1, n_ens_members + 1, figsize=figsize1, sharex=True, sharey=True
            )
            fig_error, ax_error = plt.subplots(1, n_ens_members, figsize=figsize2, sharex=True, sharey=True)
            ax_error = [ax_error] if n_ens_members == 1 else ax_error
            # Compute vmin and vmax
            vmin = min(snapshots_pred.min(), target_for_image.min())
            vmax = max(snapshots_pred.max(), target_for_image.max())
            # Plot full field and compute errors. Plot with colorbar using same vmin and vmax (different for error vs full field)
            errors = [snapshots_pred[i] - target_for_image for i in range(n_ens_members)]
            vmin_error = min([error.min() for error in errors])
            vmax_error = max([error.max() for error in errors])
            if abs(vmin_error) > abs(vmax_error):
                vmax_error = -vmin_error
            else:
                vmin_error = -vmax_error  # make sure 0 is in the middle of the colorbar
            # Preprocess (e.g. flip) the images so that they are plotted correctly
            snapshots_pred = self.preprocess_fn(snapshots_pred.cpu().numpy())
            target_for_image = self.preprocess_fn(target_for_image.cpu().numpy())
            errors = [self.preprocess_fn(error.cpu().numpy()) for error in errors]

            for i in range(n_ens_members):
                # Plot full field with colorbar
                pcm_ff = ax_full_field[i].imshow(snapshots_pred[i], vmin=vmin, vmax=vmax)
                ax_ff_title = f"Generated {i}" if n_ens_members > 1 else "Generated"
                ax_full_field[i].set_title(ax_ff_title)
                # Plot error with red blue colorbar
                pcm_err = ax_error[i].imshow(errors[i], vmin=vmin_error, vmax=vmax_error, cmap="seismic")
                ax_error_title = rf"$\hat{{y}}_{i} - y$" if n_ens_members > 1 else r"$\hat{y} - y$"
                ax_error[i].set_title(ax_error_title)

            pcm_ff = ax_full_field[-1].imshow(target_for_image, vmin=vmin, vmax=vmax)
            ax_full_field[-1].set_title("Target")
            # Create colorbar's beneath the images horizontally. To make it less thick: shrink=0.8, pad=0.03, location="bottom", aspect=40, fraction=0.05
            cbar_kwargs = dict(location="bottom", shrink=0.8, pad=0.03, fraction=0.08)
            # Defaults are: shrink=1, pad=0.05, fraction=0.15
            fig_full_field.colorbar(pcm_ff, ax=ax_full_field, **cbar_kwargs)
            fig_error.colorbar(pcm_err, ax=ax_error, **cbar_kwargs)
            # Add a main title to the figure which is the target time
            y = 0.8  # 0.9, 0.95 are way too high
            if target_datetime is not None:
                fig_full_field.suptitle(f"Target time: {target_datetime}", y=y)
                fig_error.suptitle(f"Target time: {target_datetime}", y=y)
            # Disable ticks
            for ax in ax_full_field:
                ax.axis("off")
            for ax in ax_error:
                ax.axis("off")

            # fig_full_field.tight_layout()
            # fig_error.tight_layout()
            image_logs[f"image-full-field{name_label}"] = fig_full_field
            image_logs[f"image-error{name_label}"] = fig_error

            # small_gap = torch.zeros((target_for_image.shape[-2], 2)).to(snapshots_pred.device, dtype=torch.float)
            # gap = torch.zeros((target_for_image.shape[-2], 4)).to(
            #     snapshots_pred.device, dtype=torch.float
            # )  # gap between images in wandb (so we can see them separately)
            # # Create image tensors
            # image_error, image_full_field, image_residual = [], [], []
            # for i in range(snapshots_pred.shape[0]):
            #     image_full_field += [snapshots_pred[i]]
            #     image_error += [snapshots_pred[i] - target_for_image]
            #     if input_for_image is not None:
            #         image_residual += [snapshots_pred[i] - input_for_image]
            #     if i == snapshots_pred.shape[0] - 1:
            #         image_full_field += [gap, target_for_image]
            #         if input_for_image is not None:
            #             image_residual += [gap, target_for_image - input_for_image]
            #     else:
            #         image_full_field += [small_gap]
            #         image_residual += [small_gap]
            #         image_error += [small_gap]

            # images = {}
            # images["error"] = torch.cat(image_error, dim=1)
            # images["full-field"] = torch.cat(image_full_field, dim=1)
            # if input_for_image is not None:
            #     images["residual"] = torch.cat(image_residual, dim=1)

            # for key, data in images.items():
            #     caption = self._captions[key].format(name=name)
            #     caption += f" vmin={data.min():.4g}, vmax={data.max():.4g}."
            #     data = np.flip(data.cpu().numpy(), axis=-2)
            #     wandb_image = wandb.Image(data, caption=caption)
            #     image_logs[f"image-{key}/{name}"] = wandb_image

        prefix = prefix + "/" if prefix else ""
        image_logs = {f"{prefix}{key}": image_logs[key] for key in image_logs}  # todo: use datetime as key?
        return {}, image_logs, {}
