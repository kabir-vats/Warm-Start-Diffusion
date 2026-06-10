from omegaconf import DictConfig


def get_dims_of_dataset(datamodule_config: DictConfig):
    """Returns the number of features for the given dataset."""
    target = datamodule_config.get("_target_", datamodule_config.get("name"))
    conditional_dim = conditional_non_spatial_dim = 0
    spatial_dims_out = None
    if "era5" not in target:
        raise ValueError(f"Only ERA5-style datamodules are included in this transplant, got: {target}")

    if hasattr(datamodule_config, "variables_3d"):
        input_dim = {"3d": len(datamodule_config.variables_3d), "2d": len(datamodule_config.variables_2d)}
        output_dim = {"3d": len(datamodule_config.variables_3d), "2d": len(datamodule_config.variables_2d)}
    else:
        input_dim = len(datamodule_config.input_vars)
        output_dim = len(datamodule_config.output_vars)
    spatial_crop_inputs = datamodule_config.get("spatial_crop_inputs", None)
    if spatial_crop_inputs is not None:
        crop_lat, crop_lon = tuple(spatial_crop_inputs["latitude"]), tuple(spatial_crop_inputs["longitude"])
        if crop_lat == (10, 70) and crop_lon == (190, 310):
            spatial_dims = (80, 40)
        else:
            raise ValueError(f"Unknown spatial crop for inputs: {crop_lat}, {crop_lon}")
    else:
        dataset_str = datamodule_config.dataset if hasattr(datamodule_config, "dataset") else datamodule_config.data_dir
        if "64x32" in dataset_str:
            spatial_dims = (64, 32)
        elif "240x121" in dataset_str:
            spatial_dims = (240, 121)
        elif "360x181" in dataset_str:
            spatial_dims = (360, 181)
        else:
            raise ValueError(f"Unknown dataset spatial dimensions in: {dataset_str}")

    static_fields = datamodule_config.get("static_fields", []) or []
    forcings = datamodule_config.get("forcing_fields", []) or []
    conditional_dim = len(static_fields) + len(forcings)
    if "lat_lon_embeddings" in static_fields:
        conditional_dim += 2

    if datamodule_config.text_data_path is not None:
        text_emb_type = datamodule_config.text_type
        if text_emb_type in ["tf-idf", None]:
            conditional_non_spatial_dim = 17783  # 7187
        elif text_emb_type == "bert":
            conditional_non_spatial_dim = 768
        elif text_emb_type == "bow":
            conditional_non_spatial_dim = 7187
        elif "llama" in text_emb_type:
            conditional_non_spatial_dim = 4096
        else:
            raise ValueError(f"Unknown text embedding type: {text_emb_type}")

    return {
        "input": input_dim,
        "output": output_dim,
        "spatial_in": spatial_dims,
        "spatial_out": spatial_dims_out if spatial_dims_out is not None else spatial_dims,
        "conditional": conditional_dim,
        "conditional_non_spatial_dim": conditional_non_spatial_dim,
    }
