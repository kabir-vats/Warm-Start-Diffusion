import os
import re
import time
from typing import Dict, List, Optional, Sequence

from omegaconf import DictConfig

from src.utilities.utils import replace_substrings


def _shared_prefix(config: DictConfig, init_prefix: str = "") -> str:
    """This is a prefix for naming the runs for a more agreeable logging."""
    s = init_prefix if isinstance(init_prefix, str) else ""
    if not config.get("model"):
        return s
    # Find mixer type if it is a transformer model (e.g. self-attention or FNO mixing)
    kwargs = dict(mixer=config.model.mixer._target_) if config.model.get("mixer") else dict()
    s += clean_name(config.model._target_, **kwargs)
    return s.lstrip("_")


def get_name_for_hydra_config_class(config: DictConfig) -> Optional[str]:
    """Will return a string that can describe the class of the (sub-)config."""
    if "name" in config and config.get("name") is not None:
        return config.get("name")
    elif "_target_" in config:
        return config._target_.split(".")[-1]
    return None


def get_clean_float_name(lr: float) -> str:
    """Stringify floats <1 into very short format (use for learning rates, weight-decay etc.)"""
    # basically, map Ae-B to AB (if lr<1e-5, else map 0.0001 to 1e-4)
    # convert first to scientific notation:
    if lr >= 0.1:
        return str(lr)
    lr_e = f"{lr:.1e}"  # 1e-2 -> 1.0e-02, 0.03 -> 3.0e-02
    # now, split at the e into the mantissa and the exponent
    lr_a, lr_b = lr_e.split("e-")
    # if the decimal point is 0 (e.g 1.0, 3.0, ...), we return a simple string
    if lr_a[-1] == "0":
        return f"{lr_a[0]}{int(lr_b)}"
    else:
        return str(lr).replace("e-", "")


def remove_float_prefix(string, prefix_name: str = "lr", separator="_") -> str:
    # Remove the lr and/or wd substrings like:
    # 0.0003lr_0.01wd -> ''
    # 0.0003lr -> ''
    # 0.0003lr_0.5lrecs_0.01wd -> '0.5lrecs'
    # 0.0003lr_0.5lrecs -> '0.5lrecs'
    # 0.0003lr_0.5lrecs_0.01wd_0.5lrecs -> '0.5lrecs_0.5lrecs'
    if prefix_name not in string:
        return string
    part1, part2 = string.split(prefix_name)
    # split at '_' and keep all but the last part
    part1keep = "_".join(part1.split(separator)[:-1])
    return part1keep + part2


def get_loss_name(loss):
    if isinstance(loss, str):
        loss_name = loss.lower()
    elif loss.get("_target_", "").endswith("LpLoss"):
        p, is_relative = loss.get("p", 2), loss.get("relative")
        loss_name = f"l{p}r" if is_relative else f"l{p}a"
    else:
        assert loss.get("_target_") is not None, f"Unknown loss ``{loss}``"
        loss_name = loss.get("_target_").split(".")[-1].lower().replace("loss_function", "").replace("loss", "")
    return loss_name


def get_detailed_name(config, add_unique_suffix: bool = True) -> str:
    """This is a detailed name for naming the runs for logging."""
    s = config.get("name") + "_" if config.get("name") is not None else ""
    hor = config.datamodule.get("horizon", 1)
    if (
        hor > 1
        and f"H{hor}" not in s
        and f"horizon{hor}" not in s.lower()
        and f"h{hor}" not in s.lower()
        and f"{hor}h" not in s.lower()
        and f"{hor}l" not in s.lower()
    ):
        # print(f"WARNING: horizon {hor} not in name, but should be ({s=})")
        s = s[:-1] + f"-MH{hor}_"

    s += str(config.get("name_suffix")) + "_" if config.get("name_suffix") is not None else ""
    s += _shared_prefix(config) + "_"

    w = config.datamodule.get("window", 1)
    if w > 1:
        s += f"{w}w_"

    if config.datamodule.get("train_start_date") is not None:
        s += f"{config.datamodule.train_start_date}tst_"

    if config.get("model") is None:
        return s.rstrip("_-").lstrip("_-")  # for "naive" baselines, e.g. climatology

    use_ema, ema_decay = config.module.get("use_ema", False), config.module.get("ema_decay", 0.9999)
    if use_ema:
        s += "EMA_"
        if ema_decay != 0.9999:
            s = s.replace("EMA", f"EMA{config.module.ema_decay}")

    is_diffusion = config.get("diffusion") is not None
    if is_diffusion:
        diff_class = config.diffusion.get("_target_")
        assert diff_class is not None, f"Diffusion class not found in {config.diffusion=}"
        if config.diffusion.get("interpolator_run_id"):
            replace = {
                "glkxjzy5": "i400ep",
                #
                "d5h1wl5c": "v2",
                "d4n7nz9x": "25Dr",  # NavierStokes-IpolNext16h_0k-_SimpleUnet_EMA0.995_64d_34lr_25Dr_14wd__11seed
                "keuriykh": "25DrNoEMA",  # NavierStokes-IpolNext16h_0k-_SimpleUnet_64d_34lr_25Dr_14wd__11seed
                "v588kkqe": "1k",  # NavierStokes-IpolNext16h_1k-_SimpleUnet_64d_34lr_15Dr_14wd__11seed
                "a5zyp6br": "5DrEMA",  # NavierStokes-IpolNext16h_0k-_SimpleUnet_EMA0.995_64d_34lr_5Dr_14wd__11seed
                "4wli59pv": "10DrEMA",  # NavierStokes-IpolNext16h_0k-_SimpleUnet_EMA0.995_64d_34lr_10Dr_14wd__11seed
                "v80hx8mk": "20DrEMA",  # NavierStokes-IpolNext16h_0k-_SimpleUnet_EMA0.995_64d_34lr_20Dr_14wd__11seed
                "8crimln7": "v1",  # NavierStokes-IpolNext8h_0k-_SimpleUnet_64d_34lr_15Dr_14wd__11seed
                "cmtop9ii": "EMA",  # NavierStokes-IpolNext8h_0k-_SimpleUnet_EMA0.995_64d_34lr_15Dr_14wd__11seed
                "tmdh3b7k": "5DrEMA",  # NavierStokes-IpolNext8h_0k-_SimpleUnet_EMA0.995_64d_34lr_5Dr_14wd__11seed
                "9yjnrwbk": "5Dr",  # NavierStokes-IpolNext8h_0k-_SimpleUnet_64d_34lr_5Dr_14wd__11seed
                "vfijvt6z": "25Dr",  # NavierStokes-IpolNext8h_0k-_SimpleUnet_64d_34lr_25Dr_14wd__11seed
                "tntl355t": "35Dr",  # NavierStokes-IpolNext8h_0k-_SimpleUnet_64d_34lr_35Dr_14wd__11seed
                "nctrnwbj": "v2",  # NavierStokes-IpolNext32h_0k-_SimpleUnet_EMA0.995_64d_34lr_15Dr_14wd__11seed
                "pvsn1ihu": "v1",  # NavierStokes-Interpolation8h_SimpleUnet_64d_34lr_15Dr_14wd__11seed
                "usysw7ei": "EMA",  # NavierStokes-Interpolation8h_SimpleUnet_EMA0.995_64d_34lr_15Dr_14wd__11seed
                "s31adn8h": "EMA",  # NavierStokes-Interpolation16h_SimpleUnet_EMA0.995_64d_34lr_15Dr_14wd__11seed
                "2ikmohr7": "25DrNoEMA",  # NavierStokes-Interpolation16h_SimpleUnet_64d_34lr_25Dr_14wd__11seed
                "j5gb4z38": "v1",
                "05pohfes": "EMA",  # NavierStokes-IpolNext4h_0k-_SimpleUnet_EMA0.995_64d_34lr_15Dr_14wd__11seed
                "iq62idey": "20DrEMA",  # NavierStokes-IpolNext4h_0k-_SimpleUnet_EMA0.995_64d_34lr_20Dr_14wd__11seed
                "f77arfd2": "10Dr",
                "x2cjsyw3": "20Dr",
                "j9tka412": "25Dr",
                "hzvh7rcb": "v1",
                "ncna54nl": "10Dr",
                "9fd75zce": "20Dr",
                "e8jutiu5": "25Dr",
                "y8fxsyfv": "30Dr",  # NavierStokes-IpolNext6h_0k-_SimpleUnet_64d_34lr_30Dr_14wd__11seed
                "3btgvj8x": "35Dr",  # NavierStokes-IpolNext6h_0k-_SimpleUnet_64d_34lr_35Dr_14wd__11seed
                "corexj3c": "30DrEMA",  # NavierStokes-IpolNext6h_0k-_SimpleUnet_EMA0.995_64d_34lr_30Dr_14wd__11seed
                "g1hh5fnh": "35DrEMA",  # NavierStokes-IpolNext6h_0k-_SimpleUnet_EMA0.995_64d_34lr_35Dr_14wd__11seed
                "7qejbrky": "10DrEMA",  # NavierStokes-IpolNext6h_0k-_SimpleUnet_EMA0.995_64d_34lr_10Dr_14wd__11seed
                "xzez57uh": "EMA",  # NavierStokes-IpolNext6h_0k-_SimpleUnet_EMA0.995_64d_34lr_15Dr_14wd__11seed
                "ct72dks4": "20DrEMA",  # NavierStokes-IpolNext6h_0k-_SimpleUnet_EMA0.995_64d_34lr_20Dr_14wd__11seed
                "ghbwo5sc": "25DrEMA",  # NavierStokes-IpolNext6h_0k-_SimpleUnet_EMA0.995_64d_34lr_25Dr_14wd__11seed_
                "9m2l2d0e": "5Dr",  # NavierStokes-Interpolation6h_SimpleUnet_64d_34lr_5Dr_14wd__11seed
                "vlkz2rqf": "10Dr",  # NavierStokes-Interpolation6h_SimpleUnet_64d_34lr_10Dr_14wd__11seed
                "ytuvmdyc": "15Dr",  # NavierStokes-Interpolation6h_SimpleUnet_64d_34lr_15Dr_14wd__11seed
                "z2ifhs9g": "20Dr",  # NavierStokes-Interpolation6h_SimpleUnet_64d_34lr_20Dr_14wd__11seed
                "12bd09ka": "25Dr",  # NavierStokes-Interpolation6h_SimpleUnet_64d_34lr_25Dr_14wd__11seed
                "7bylug9j": "30Dr",  # NavierStokes-Interpolation6h_SimpleUnet_64d_34lr_30Dr_14wd__11seed
                "nc0hdj5x": "35Dr",  # NavierStokes-Interpolation6h_SimpleUnet_64d_34lr_35Dr_14wd__11seed
                "jyjth0jk": "5DrEMA",  # NavierStokes-Interpolation6h_SimpleUnet_EMA0.995_64d_34lr_5Dr_14wd__11seed
                "44ksd99h": "10DrEMA",  # NavierStokes-Interpolation6h_SimpleUnet_EMA0.995_64d_34lr_10Dr_14wd__11seed
                "jkiupieb": "15DrEMA",  # NavierStokes-Interpolation6h_SimpleUnet_EMA0.995_64d_34lr_15Dr_14wd__11seed
                "oyi05k1x": "20DrEMA",  # NavierStokes-Interpolation6h_SimpleUnet_EMA0.995_64d_34lr_20Dr_14wd__11seed
                "iflevf9f": "5Dr",  # NavierStokes-IpolNext12h_0k-_SimpleUnet_64d_34lr_5Dr_14wd__11seed
                "qbc0o6tv": "10Dr",  # NavierStokes-IpolNext12h_0k-_SimpleUnet_64d_34lr_10Dr_14wd__11seed
                "coc7r6xv": "15Dr",  # NavierStokes-IpolNext12h_0k-_SimpleUnet_64d_34lr_15Dr_14wd__11seed
                "yiiyog8a": "20Dr",  # NavierStokes-IpolNext12h_0k-_SimpleUnet_64d_34lr_20Dr_14wd__11seed
                "0b39ismd": "25Dr",  # NavierStokes-IpolNext12h_0k-_SimpleUnet_64d_34lr_25Dr_14wd__11seed
                "9m9jr4fj": "5DrEMA",  # NavierStokes-IpolNext12h_0k-_SimpleUnet_EMA0.995_64d_34lr_5Dr_14wd__11seed
                "05nrdwkt": "10DrEMA",  # NavierStokes-IpolNext12h_0k-_SimpleUnet_EMA0.995_64d_34lr_10Dr_14wd__11seed
                # Spring Mesh
                "2sqdeqlu": "5DrEMA",  # SpringMesh-IpolNext6h_SimpleCNN_EMA0.995_64d_44lr_5Dr_14wd__11seed
                "czocieyv": "10DrEMA",  # SpringMesh-IpolNext6h_SimpleCNN_EMA0.995_64d_44lr_10Dr_14wd__11seed
                "wcdgeb88": "15DrEMA",  # SpringMesh-IpolNext6h_SimpleCNN_EMA0.995_64d_44lr_15Dr_14wd__11seed
                "qaudgrcb": "20DrEMA",  # SpringMesh-IpolNext6h_SimpleCNN_EMA0.995_64d_44lr_20Dr_14wd__11seed
                "v36fbo7n": "25DrEMA",  # SpringMesh-IpolNext6h_SimpleCNN_EMA0.995_64d_44lr_25Dr_14wd__11seed
                "ngnfc3nj": "30DrEMA",  # SpringMesh-IpolNext6h_SimpleCNN_EMA0.995_64d_44lr_30Dr_14wd__11seed
                "nhggkeuk": "35DrEMA",  # SpringMesh-IpolNext6h_SimpleCNN_EMA0.995_64d_44lr_35Dr_14wd__11seed
                "q5lt6sjg": "40DrEMA",  # SpringMesh-IpolNext6h_SimpleCNN_EMA0.995_64d_44lr_40Dr_14wd__11seed
                "wm43eedx": "10Dr",  # SpringMesh-IpolNext6h_SimpleCNN_64d_44lr_10Dr_14wd__11seed
                "5jos7poq": "3x80Ur",  # SpringMesh-IpolNext6h_UNetR_EMA0.995_64x1-2d_44lr_80at80b80b1Dr_14wd__11seed
                #      H16
                "tjflueeh": "20DrEMA",  # SpringMesh-IpolNext16h_SimpleCNN_EMA0.995_64d_44lr_20Dr_14wd__11seed
                "fnai4sza": "15DrEMA",  # SpringMesh-IpolNext16h_SimpleCNN_EMA0.995_64d_44lr_15Dr_14wd__11seed
                "xb7wmaty": "10DrEMA",  # SpringMesh-IpolNext16h_SimpleCNN_EMA0.995_64d_44lr_10Dr_14wd__11seed
                "sgnfcb6r": "5DrEMA",  # SpringMesh-IpolNext16h_SimpleCNN_EMA0.995_64d_44lr_5Dr_14wd__11seed
                #     H67
                "fa3oi3qb": "5DrEMA",  # SpringMesh-IpolNext67h_SimpleCNN_EMA0.995_64d_44lr_5Dr_14wd__11seed
                "nxo6kaqi": "10DrEMA",  # SpringMesh-IpolNext67h_SimpleCNN_EMA0.995_64d_44lr_10Dr_14wd__11seed
                "zl9uzhs5": "15DrEMA",  # SpringMesh-IpolNext67h_SimpleCNN_EMA0.995_64d_44lr_15Dr_14wd__11seed
                "5vhmhims": "20DrEMA",  # SpringMesh-IpolNext67h_SimpleCNN_EMA0.995_64d_44lr_20Dr_14wd__11seed
                #     H134
                "g7qyvdu4": "5DrEMA",  # SpringMesh-IpolNext134h_SimpleCNN_EMA0.995_64d_44lr_5Dr_14wd__11seed
                "zwe06h9f": "10DrEMA",  # SpringMesh-IpolNext134h_SimpleCNN_EMA0.995_64d_44lr_10Dr_14wd__11seed
                "rcnan96n": "15DrEMA",  # SpringMesh-IpolNext134h_SimpleCNN_EMA0.995_64d_44lr_15Dr_14wd__11seed
            }
            i_id = replace.get(
                config.diffusion.interpolator_run_id,
                config.diffusion.interpolator_run_id,
            )
            if config.diffusion.get("interpolator_artificial_steps_run_id"):
                s += f"{i_id}+{config.diffusion.interpolator_artificial_steps_run_id}-ipolID_"
            else:
                s += f"{i_id}-ipolID_"

        default = "linear" if "mcvd" in diff_class.lower() else "cosine"
        if config.diffusion.get("beta_schedule", default) != default:
            s += f"{config.diffusion.beta_schedule}_"

        if config.diffusion.get("prediction_mode", "raw") == "residual":
            s += "Res_"

        extra1 = config.diffusion.get("additional_interpolation_steps", 0)
        extra2 = config.diffusion.get("additional_interpolation_steps_factor", 0)
        if config.diffusion.get("schedule") == "linear":
            if extra2 > 0:
                # additional_steps = config.diffusion.additional_interpolation_steps_factor * (config.datamodule.horizon - 2)
                if config.diffusion.get("interpolate_before_t1", False):
                    s += f"{extra2}k-Xa_"
                else:
                    s += f"{extra2}k-Xb_"
        elif config.diffusion.get("schedule") == "before_t1_only":
            if extra1 > 0:
                s += f"{extra1}k-preT1_"

        elif config.diffusion.get("schedule") == "before_t1_then_linear":
            assert extra1 > 0 and extra2 > 0, "Both extra1 and extra2 must be > 0"
            s += f"{extra1}preX{extra2}postT1_"

        elif config.diffusion.get("schedule") == "exp_a_b":
            s += f"{config.diffusion.get('a')}a{config.diffusion.get('b')}b_"
        elif config.diffusion.get("schedule") == "edm_paper_reversed":
            s = s.replace("edm_paper_reversed", "edmPR")

        fcond = config.diffusion.get("forward_conditioning")
        if fcond is not None and fcond != "none":
            s += f"{fcond}-fcond_" if "noise" not in fcond else f"{fcond}_"

        if config.diffusion.get("time_encoding", "dynamics") != "dynamics":
            tenc = config.diffusion.get("time_encoding")
            if tenc == "continuous":
                s += "ContTime_"
            elif tenc == "dynamics":
                pass  # s += "DynT_"
            else:
                s += f"{config.diffusion.time_encoding}-timeEnc_"

        if config.diffusion.get("condition_on_x_last") in [False]:
            s += "NoXlast"

        if config.diffusion.get("use_separate_heads", False):
            if config.diffusion.use_separate_heads in [True, "v1"]:
                s += "Heads_"
            elif config.diffusion.use_separate_heads == "v2":
                dout = config.diffusion.get("interpolator_head_dropout")
                if dout is not None:
                    s += f"HeadsV2-{int(dout * 100)}_"
                else:
                    s += "HeadsV2_"
    elif config.module.get("artificial_steps_factor", 1) > 1:
        s += f"{config.module.artificial_steps_factor}Tfactor_"

    if config.module.get("multi_step_loss"):
        lam1, lam2 = config.module.lam_loss, config.module.lam_multi_step_loss
        s += f"{lam1}x{lam2}{config.module.multi_step_loss}loss_"

    hdims = config.model.get("hidden_dims")
    if hdims is None:
        num_L = config.model.get("num_layers") or config.model.get("depth")
        if num_L is None:
            dim_mults = config.model.get("dim_mults") or config.model.get("channel_mult")
            if dim_mults is None:
                pass
            elif tuple(dim_mults) == (1, 2, 4):
                num_L = "3"
            else:
                num_L = "-".join([str(d) for d in dim_mults])

        possible_dim_names = ["dim", "hidden_dim", "embed_dim", "hidden_size", "model_channels"]
        hdim = None
        for name in possible_dim_names:
            hdim = config.model.get(name)
            if hdim is not None:
                break

        if hdim is not None:
            hdims = f"{hdim}x{num_L}" if num_L is not None else f"{hdim}"
    elif all([h == hdims[0] for h in hdims]):
        hdims = f"{hdims[0]}x{len(hdims)}"
    else:
        hdims = str(hdims)

    s += f"{hdims}d_" if hdims is not None else ""
    if config.model.get("mlp_ratio", 2.0) != 2.0:
        s += f"{config.model.mlp_ratio}dxMLP_"

    if is_diffusion and config.diffusion.get("loss_function_interpolation") is not None:
        loss_ipol = get_loss_name(config.diffusion.loss_function_interpolation)
        loss_fcast = get_loss_name(config.diffusion.loss_function_forecast)
        s += f"{loss_ipol.upper()}-{loss_fcast.upper()}_"
    elif is_diffusion and config.diffusion.get("loss_function") is not None:
        loss = config.diffusion.get("loss_function")
        loss = get_loss_name(loss)
        if loss not in ["mse", "l2"]:
            s += f"{loss.upper()}_"
    else:
        loss = config.model.get("loss_function")
        loss = get_loss_name(loss)
        if loss in ["mse", "l2"]:
            pass
        elif loss in ["l2_rel", "l1_rel"]:
            s += f"{loss.upper().replace('_REL', 'rel')}_"
        else:
            s += f"{loss.upper()}_"

    if config.model.get("patch_size") is not None:
        p = config.model.patch_size
        p1, p2 = p if isinstance(p, (list, tuple)) else (p, p)
        s += f"{p1}x{p2}patch_" if p1 != p2 else f"{p1}ps_"
    time_emb = config.model.get("with_time_emb", False)
    if time_emb not in [False, True, "scale_shift"]:
        s += f"{time_emb}_"
    if (isinstance(time_emb, str) and "scale_shift" in time_emb) and not config.model.get(
        "time_scale_shift_before_filter"
    ):
        s += "tSSA_"  # time scale shift after filter

    optim = config.module.get("optimizer")
    if optim is not None:
        if "adamw" not in optim.name.lower():
            s += f"{optim.name.replace('Fused', '').replace('fused', '')}_"
        if "fused" in optim.name.lower() or optim.get("fused", False):
            s = s[:-1] + "F_"
    scheduler_cfg = config.module.get("scheduler")
    if scheduler_cfg is not None and ("lr_max" in scheduler_cfg or "lr_start" in scheduler_cfg):
        lr_start = get_clean_float_name(scheduler_cfg.get("lr_start", 0))
        lr_max = get_clean_float_name(scheduler_cfg.get("lr_max", 0))
        lr_min = get_clean_float_name(scheduler_cfg.get("lr_min", 0))
        wup_steps = scheduler_cfg.get("warm_up_steps", 0)
        s += f"{lr_start}-{lr_max}-{lr_min}lr_" if wup_steps > 0 else f"{lr_max}-{lr_min}lr_"
        if wup_steps != 500:
            s = s[:-1]
            s += f"{scheduler_cfg.warm_up_steps / 100}Kw_"
        if scheduler_cfg.get("max_decay_steps", 1000) != 1000:
            s = s[:-1]
            s += f"{scheduler_cfg.max_decay_steps / 100}Kd_"
    else:
        lr = config.get("base_lr") or optim.get("lr")
        s += f"{get_clean_float_name(lr)}lr_"
        if optim.get("muon", {}).get("lr") is not None:
            s = f"{s[:-1]}{get_clean_float_name(optim.muon.lr)}mlr_"
            if optim.muon.momentum is not None and optim.muon.momentum != 0.95:
                s += f"{get_clean_float_name(optim.muon.momentum)}mMom_"
            if optim.muon.wd is not None and optim.muon.wd > 0:
                s += f"{get_clean_float_name(optim.muon.wd)}mwd_"
        if scheduler_cfg is not None:
            name_or_tgt = scheduler_cfg.get("name") or scheduler_cfg.get("_target_")
            if "warmup_epochs" in scheduler_cfg or "warmup_steps" in scheduler_cfg:
                wup = scheduler_cfg.warmup_epochs if "warmup_epochs" in scheduler_cfg else scheduler_cfg.warmup_steps
                tmax = scheduler_cfg.get("max_epochs") or scheduler_cfg.get("max_steps")
                if scheduler_cfg.get("decay_step_start") is not None:
                    s += f"WuP{wup}:Ds{scheduler_cfg.decay_step_start}_"
                elif name_or_tgt == "cosine_hard_restarts":
                    s += f"LCR{wup}:{tmax}:{scheduler_cfg.num_cycles}_"
                else:
                    s += f"LC{wup}:{tmax}_"

    if is_diffusion:
        lam1 = config.diffusion.get("lambda_reconstruction")
        lam2 = config.diffusion.get("lambda_reconstruction2")
        lam_cy = config.diffusion.get("lambda_consistency")
        lam_ipol = config.diffusion.get("lambda_interpolation")
        lam_ipol2 = config.diffusion.get("lambda_interpolation2")
        all_lams = [lam1, lam2, lam_cy, lam_ipol, lam_ipol2]
        nonzero_lams = len([1 for lam in all_lams if lam is not None and lam > 0])
        uniform_lams = [
            1 / nonzero_lams if nonzero_lams > 0 else 0,
            0.33 if nonzero_lams == 3 else 0,
        ]
        if config.diffusion.get("detach_interpolated_data", False):
            s += "detXi_"
        if config.diffusion.get("lambda_reconstruction2", 0) > 0:
            if lam1 == lam2 and lam2 == lam_ipol:
                s += f"{lam1}lams_"
            elif lam1 == lam2:
                s += f"{lam1}lRecs_"
            else:
                s += f"{lam1}-{lam2}lRecs_"

            if config.diffusion.get("reconstruction2_detach_x_last", False):
                s += "detX0_"
        elif lam1 is not None and lam1 not in uniform_lams:
            s += f"{lam1}lRec_"

        if lam_cy is not None and lam_cy > 0:
            strat = config.diffusion.get("consistency_strategy").replace("detach", "det").replace("net", "NN")
            if not use_ema and "ema" in strat and ema_decay != 0.999:
                strat = strat.replace("ema", f"ema{ema_decay}")
            if lam_cy not in uniform_lams:
                s += f"{lam_cy}lCy-{strat}_"
            else:
                s += f"{strat}_"

        if lam_ipol is not None and lam_ipol > 0 and lam_ipol2 is not None and lam_ipol2 > 0:
            lams_ipols = lam_ipol if lam_ipol == lam_ipol2 else f"{lam_ipol}-{lam_ipol2}"
            s += f"{lams_ipols}lIpols_"
        elif lam_ipol is not None and lam_ipol != 0.5 and (lam_ipol != lam1 or lam_ipol != lam2):
            s += f"{lam_ipol}lIpol_"

        if config.diffusion.get("lambda_reverse_diffusion", 0) > 0:
            s += f"{config.diffusion.get('lambda_reverse_diffusion')}lRD_"

    dropout = {
        "": config.model.get("dropout", 0),
        "in": config.model.get("input_dropout", 0),
        "pos": config.model.get("pos_emb_dropout", 0),
        "at": config.model.get("attn_dropout", 0),
        "b": config.model.get("block_dropout", 0),
        "b1": config.model.get("block_dropout1", 0),
        "ft": config.model.get("dropout_filter", 0),
        "mlp": config.model.get("dropout_mlp", 0),
    }
    any_nonzero = any([d > 0 for d in dropout.values() if d is not None])
    for k, d in dropout.items():
        if d is not None and d > 0:
            s += f"{int(d * 100)}{k}"
    if any_nonzero:  # remove redundant 'Dr_'   (should be done for all dropout later on #todo)
        s += "Dr_"

    if any_nonzero and is_diffusion and config.diffusion.get("enable_interpolator_dropout", False):
        s += "iDr_"  # interpolator dropout

    if config.model.get("drop_path_rate", 0) > 0:
        s += f"{int(config.model.drop_path_rate * 100)}dpr_"

    if config.module.optimizer.get("weight_decay") and config.module.optimizer.get("weight_decay") > 0:
        s += f"{get_clean_float_name(config.module.optimizer.get('weight_decay'))}wd_"

    if config.get("suffix", "") != "":
        s += f"{config.get('suffix')}_"

    wandb_cfg = config.get("logger", {}).get("wandb", {})
    if wandb_cfg.get("resume_run_id") and wandb_cfg.get("id", "$") != wandb_cfg.get("resume_run_id", "$"):
        s += f"{wandb_cfg.get('resume_run_id')}rID_"

    if add_unique_suffix:
        s += f"{config.get('seed')}seed"
        s += "_" + time.strftime("%Hh%Mm%b%d")
        wandb_id = wandb_cfg.get("id")
        if wandb_id is not None:
            s += f"_{wandb_id}"

    return s.replace("None", "").rstrip("_-").lstrip("_-")


def clean_name(class_name, mixer=None, dm_type=None) -> str:
    """This names the model class paths with a more concise name."""
    if "AFNONet" in class_name or "Transformer" in class_name:
        if mixer is None or "AFNO" in mixer:
            s = "AFNO"
        elif "SelfAttention" in mixer:
            s = "self-attention"
        else:
            raise ValueError(class_name)
    elif "SphericalFourierNeuralOperatorNet" in class_name:
        return "SFNO"
    elif "UnetConvNext" in class_name:
        s = "UnetConvNext"
    elif "unet_simple" in class_name:
        s = "SimpleUnet"
    elif "AutoencoderKL" in class_name:
        s = "LDM"
    elif "SimpleChannelOnlyMLP" in class_name:
        s = "SiMLP"
    elif "MLP" in class_name:
        s = "MLP"
    elif "Unet" in class_name:
        s = "UNetR"
    elif "SimpleConvNet" in class_name:
        s = "SimpleCNN"
    elif "graph_network" in class_name:
        s = "GraphNet"
    elif "CNN_Net" in class_name:
        s = "CNN"
    elif "NCSN" in class_name:
        s = "NCSN"
    elif class_name == "src.models.gan.GAN":
        s = "SimpleGAN"
    elif "edm.DhariwalUNet" in class_name:
        s = "ADM"
    elif "DhariwalUNetOld" in class_name:
        s = "ADM-old"
    elif "DhariwalUNet3D" in class_name:
        s = "ADM3d"
    elif "edm2.UNet" in class_name:
        s = "ADM2"
    elif "HealpixUNet" in class_name:
        s = "ADM-HPX"
    elif "stormer.vit_adaln.ViTAdaLN" in class_name:
        s = "StormerViT"
    else:
        raise ValueError(f"Unknown class name: {class_name}, did you forget to add it to the clean_name function?")

    return s


def get_group_name(config) -> str:
    """
    This is a group name for wandb logging.
    On Wandb, the runs of the same group are averaged out when selecting grouping by `group`
    """
    # s = get_name_for_hydra_config_class(config.model)
    # s = s or _shared_prefix(config, init_prefix=s)
    return get_detailed_name(config, add_unique_suffix=False)


def var_names_to_clean_name() -> Dict[str, str]:
    """This is a clean name for the variables (e.g. for plotting)"""
    var_dict = {
        "tas": "Air Temperature",
        "psl": "Sea-level Pressure",
        "ps": "Surface Pressure",
        "pr": "Precipitation",
        "sst": "Sea Surface Temperature",
    }
    return var_dict


variable_name_to_metadata_dict = {
    "DLWRFsfc": {"units": "W/m**2", "long_name": "surface downward longwave flux"},
    "DSWRFsfc": {
        "units": "W/m**2",
        "long_name": "averaged surface downward shortwave flux",
    },
    "DSWRFtoa": {
        "units": "W/m**2",
        "long_name": "top of atmos downward shortwave flux",
    },
    "GRAUPELsfc": {
        "units": "kg/m**2/s",
        "long_name": "bucket surface graupel precipitation rate",
    },
    "HGTsfc": {"units": "m", "long_name": "surface height"},
    "ICEsfc": {
        "units": "kg/m**2/s",
        "long_name": "bucket surface ice precipitation rate",
    },
    "LHTFLsfc": {"units": "w/m**2", "long_name": "surface latent heat flux"},
    "PRATEsfc": {
        "units": "kg/m**2/s",
        "long_name": "bucket surface precipitation rate",
    },
    "PRESsfc": {"units": "Pa", "long_name": "surface pressure"},
    "SHTFLsfc": {"units": "w/m**2", "long_name": "surface sensible heat flux"},
    "SNOWsfc": {
        "units": "kg/m**2/s",
        "long_name": "bucket surface snow precipitation rate",
    },
    "ULWRFsfc": {"units": "W/m**2", "long_name": "surface upward longwave flux"},
    "ULWRFtoa": {"units": "W/m**2", "long_name": "top of atmos upward longwave flux"},
    "USWRFsfc": {
        "units": "W/m**2",
        "long_name": "averaged surface upward shortwave flux",
    },
    "USWRFtoa": {"units": "W/m**2", "long_name": "top of atmos upward shortwave flux"},
    "air_temperature_0": {"units": "K", "long_name": "temperature level-0"},
    "air_temperature_1": {"units": "K", "long_name": "temperature level-1"},
    "air_temperature_2": {"units": "K", "long_name": "temperature level-2"},
    "air_temperature_3": {"units": "K", "long_name": "temperature level-3"},
    "air_temperature_4": {"units": "K", "long_name": "temperature level-4"},
    "air_temperature_5": {"units": "K", "long_name": "temperature level-5"},
    "air_temperature_6": {"units": "K", "long_name": "temperature level-6"},
    "air_temperature_7": {"units": "K", "long_name": "temperature level-7"},
    "ak_0": {"units": "Pa", "long_name": "ak"},
    "ak_1": {"units": "Pa", "long_name": "ak"},
    "ak_2": {"units": "Pa", "long_name": "ak"},
    "ak_3": {"units": "Pa", "long_name": "ak"},
    "ak_4": {"units": "Pa", "long_name": "ak"},
    "ak_5": {"units": "Pa", "long_name": "ak"},
    "ak_6": {"units": "Pa", "long_name": "ak"},
    "ak_7": {"units": "Pa", "long_name": "ak"},
    "ak_8": {"units": "Pa", "long_name": "ak"},
    "bk_0": {"units": "", "long_name": "bk"},
    "bk_1": {"units": "", "long_name": "bk"},
    "bk_2": {"units": "", "long_name": "bk"},
    "bk_3": {"units": "", "long_name": "bk"},
    "bk_4": {"units": "", "long_name": "bk"},
    "bk_5": {"units": "", "long_name": "bk"},
    "bk_6": {"units": "", "long_name": "bk"},
    "bk_7": {"units": "", "long_name": "bk"},
    "bk_8": {"units": "", "long_name": "bk"},
    "eastward_wind_0": {"units": "m/sec", "long_name": "zonal wind level-0"},
    "eastward_wind_1": {"units": "m/sec", "long_name": "zonal wind level-1"},
    "eastward_wind_2": {"units": "m/sec", "long_name": "zonal wind level-2"},
    "eastward_wind_3": {"units": "m/sec", "long_name": "zonal wind level-3"},
    "eastward_wind_4": {"units": "m/sec", "long_name": "zonal wind level-4"},
    "eastward_wind_5": {"units": "m/sec", "long_name": "zonal wind level-5"},
    "eastward_wind_6": {"units": "m/sec", "long_name": "zonal wind level-6"},
    "eastward_wind_7": {"units": "m/sec", "long_name": "zonal wind level-7"},
    "land_fraction": {
        "units": "dimensionless",
        "long_name": "fraction of grid cell area occupied by land",
    },
    "northward_wind_0": {"units": "m/sec", "long_name": "meridional wind level-0"},
    "northward_wind_1": {"units": "m/sec", "long_name": "meridional wind level-1"},
    "northward_wind_2": {"units": "m/sec", "long_name": "meridional wind level-2"},
    "northward_wind_3": {"units": "m/sec", "long_name": "meridional wind level-3"},
    "northward_wind_4": {"units": "m/sec", "long_name": "meridional wind level-4"},
    "northward_wind_5": {"units": "m/sec", "long_name": "meridional wind level-5"},
    "northward_wind_6": {"units": "m/sec", "long_name": "meridional wind level-6"},
    "northward_wind_7": {"units": "m/sec", "long_name": "meridional wind level-7"},
    "ocean_fraction": {
        "units": "dimensionless",
        "long_name": "fraction of grid cell area occupied by ocean",
    },
    "sea_ice_fraction": {
        "units": "dimensionless",
        "long_name": "fraction of grid cell area occupied by sea ice",
    },
    "soil_moisture": {
        "units": "kg/m**2",
        "long_name": "total column soil moisture content",
    },
    "specific_total_water_0": {
        "units": "kg/kg",
        "long_name": "specific total water level-0",
    },
    "specific_total_water_1": {
        "units": "kg/kg",
        "long_name": "specific total water level-1",
    },
    "specific_total_water_2": {
        "units": "kg/kg",
        "long_name": "specific total water level-2",
    },
    "specific_total_water_3": {
        "units": "kg/kg",
        "long_name": "specific total water level-3",
    },
    "specific_total_water_4": {
        "units": "kg/kg",
        "long_name": "specific total water level-4",
    },
    "specific_total_water_5": {
        "units": "kg/kg",
        "long_name": "specific total water level-5",
    },
    "specific_total_water_6": {
        "units": "kg/kg",
        "long_name": "specific total water level-6",
    },
    "specific_total_water_7": {
        "units": "kg/kg",
        "long_name": "specific total water level-7",
    },
    "surface_temperature": {"units": "K", "long_name": "surface temperature"},
    "tendency_of_total_water_path": {
        "units": "kg/m^2/s",
        "long_name": "time derivative of total water path",
    },
    "tendency_of_total_water_path_due_to_advection": {
        "units": "kg/m^2/s",
        "long_name": "tendency of total water path due to advection",
    },
    "total_water_path": {"units": "kg/m^2", "long_name": "total water path"},
    "10m_u_component_of_wind": {"units": "m/s", "short_name": "10u", "long_name": "10m u component of wind"},
    "10m_v_component_of_wind": {"units": "m/s", "short_name": "10v", "long_name": "10m v component of wind"},
    "2m_temperature": {"units": "K", "short_name": "2t", "long_name": "2m temperature"},
    "sea_surface_temperature": {"units": "K", "short_name": "sst", "long_name": "Sea surface temperature"},
    "mean_sea_level_pressure": {"units": "Pa", "short_name": "mslp", "long_name": "mean sea level pressure"},
    "geopotential": {"units": "m^2/s^2", "short_name": "z", "long_name": "geopotential"},
    "temperature": {"units": "K", "short_name": "t", "long_name": "temperature"},
    "specific_humidity": {"units": "kg/kg", "short_name": "q", "long_name": "specific humidity"},
    "u_component_of_wind": {"units": "m/s", "short_name": "u", "long_name": "u component of wind"},
    "v_component_of_wind": {"units": "m/s", "short_name": "v", "long_name": "v component of wind"},
    "vertical_velocity": {"units": None, "short_name": "w", "long_name": "vertical velocity"},
}


def variable_name_to_metadata(variable: str) -> Dict[str, str]:
    """This is a clean name for the variables (e.g. for plotting)"""
    # Check if variable ends with _NUMBER, if so split it
    match = re.match(r"^(.+?)(?:_(\d+))?$", variable)
    if match:
        variable = match.group(1)
        number = match.group(2)
    else:
        number = None

    if variable not in variable_name_to_metadata_dict:
        return {"units": "", "long_name": variable, "short_name": variable}
    metadata_dict_var = variable_name_to_metadata_dict[variable].copy()
    add_number = ["long_name", "short_name"]
    if number is not None:  # Add number to long_name and short_name (e.g. geopotential_500)
        for key in add_number:
            if key in metadata_dict_var:
                metadata_dict_var[key] += f"{number}"

    return metadata_dict_var


def full_variable_name_with_units(variable: str, formatted: bool = True, capitalize: bool = True) -> str:
    """This is a full name for the variable (e.g. for plotting)"""
    data = variable_name_to_metadata(variable)
    long_name = data.get("long_name", variable)
    if capitalize:
        long_name = long_name.capitalize()
    # Make long name bold in latex, and units italic
    if formatted is True:
        name = long_name.replace("_", " ").replace(" ", "\\ ")
        if data["units"] == "":
            return f"$\\bf{{{name}}}$"
        else:
            return f'$\\bf{{{name}}}$ [$\\it{{{data["units"]}}}$]'
    elif formatted == "units":
        if data["units"] == "":
            return f"{long_name}"
        else:
            return f'{long_name} [$\\it{{{data["units"]}}}$]'
    else:
        if data["units"] == "":
            return f"{long_name}"
        else:
            return f'{long_name} [{data["units"]}]'


def formatted_units(variable: str) -> str:
    """This is a full name for the variable (e.g. for plotting)"""
    data = variable_name_to_metadata(variable)
    return f"[$\\it{{{data['units']}}}$]"


def formatted_name(variable: str, capitalize: bool = False, use_short: bool = False) -> str:
    """This is a full name for the variable (e.g. for plotting)"""
    data = variable_name_to_metadata(variable)
    name = data.get("long_name", variable) if not use_short else data.get("short_name", variable)
    if capitalize:
        name = name.capitalize()
    name = name.replace("_", " ").replace(" ", "\\ ")
    return f"$\\bf{{{name}}}$"


def get_formatted_name_and_units(variable: str, **kwargs) -> (str, str):
    var_name = formatted_name(variable, **kwargs)
    units = formatted_units(variable)
    return var_name, units


def clean_metric_name(metric: str) -> str:
    """This is a clean name for the metrics (e.g. for plotting)"""
    # Match a string that starts with 't' followed by digits
    match = re.match(r"^t(\d+)$", metric)
    if match:
        # Extract the number and format it
        number = match.group(1)
        return f"$t = {number}$"

    # Otherwise, check if the metric is in the dictionary
    metric_dict = {
        "mae": "MAE",
        "mse": "MSE",
        "crps": "CRPS",
        "rmse": "RMSE",
        "bias": "Bias",
        "mape": "MAPE",
        "ssr": "SSR",
        # "ssr": "Spread / RMSE",
        "ssr_abs_dist": "abs(1 - Spread / RMSE)",
        "ssr_squared_dist": "(1 - Spread / RMSE)^2",
        "nll": "NLL",
        "r2": "R2",
        "corr": "Correlation",
        "corrcoef": "Correlation",
        "corr_mem_avg": "Corr. Mem. Avg.",
        "corr_spearman": "Spearman Correlation",
        "corr_kendall": "Kendall Correlation",
        "corr_pearson": "Pearson Correlation",
        "grad_mag_percent_diff": "Gradient Mag. % Diff",
    }
    for k in ["crps", "ssr", "rmse", "grad_mag_percent_diff", "bias"]:
        metric_dict[f"weighted_{k}"] = metric_dict[k]

    return metric_dict.get(metric.lower(), metric)


def normalize_run_name(
    name: str,
    remove_lr_from_label: bool = False,
    remove_weight_decay_from_label: bool = False,
) -> str:
    """This is a clean name for the runs (e.g. for plotting)"""
    #                 group_name = group_name.replace("PacificSubset", "Pac").replace('MultiHorizon', 'MH')
    #             if len(group_name) >= 128:
    #                 group_name = group_name.replace('MixUpPretrainedInterpolator', 'DY2S')
    #                 group_name = group_name.replace('-ipolID_', '-iID_')
    #                 group_name = group_name.replace('-fcond', '')
    replacements = {
        "L1_UNetResNet": "UNetResNet-L1",
        "PacificSubset": "Pac",
        "MultiHorizon": "MH",
        "MixUpPretrainedInterpolator": "DY2S",
        "-ipolID_": "-iID_",
        "ipolNfactor": "xN",
        "data+noise-v1": "D+N",
        "data+noise-v2": "D+Nv2",
        "-fcond": "",
        "UNetResNet": "",
        "64x3h": "",
    }
    name = replace_substrings(name, replacements, ignore_case=True).replace("__", "_")
    replacements2 = {
        "_qer51x9m-iID": "-v2",
        "_3jqv8m4g-iID": "-v1",
    }
    name = replace_substrings(name, replacements2, ignore_case=True).replace("__", "_")

    if remove_weight_decay_from_label:
        # remove any '_0.001wd', '_1e-05wd', '_0.01wd' from the label
        name = re.sub(r"_\d(e-)?.\d+wd", "", name)  # \d+ means one or more digits
    if remove_lr_from_label:
        # remove any '_6e-05lr', '_1e-05lr', '_0.0001lr' from the label
        name = re.sub(r"_\d(e-)?.\d+lr", "", name)  # \d+ means one or more digits
    return name


def get_label_names_for_wandb_group_names(
    wandb_groups: Sequence[str],
    labels: Optional[List[str]] = None,
    remove_lr_from_label: bool = False,
    remove_weight_decay_from_label: bool = False,
) -> (List[str], List[str]):
    if labels is None:
        labels = wandb_groups
        labels = [
            normalize_run_name(
                label,
                remove_lr_from_label=remove_lr_from_label,
                remove_weight_decay_from_label=remove_weight_decay_from_label,
            )
            for label in labels
        ]
        # get the longest common prefix of wandb_groups
        common_prefix = os.path.commonprefix(labels)
        # lstrip and rstrip anything that is shared across all labels
        labels = [label.replace(common_prefix, "").lstrip("-_") for label in labels]
        # remove common suffix
        common_suffix = os.path.commonprefix([label[::-1] for label in labels])
        if len(common_suffix) > 1:
            labels = [label[: -len(common_suffix)].rstrip("-_") for label in labels]
        # while len(set([l[-1] for l in labels])) == 1:
        #    labels = [l[:-1] for l in labels]
    # wandb_groups = sorted(wandb_groups)  # sorted(get_unique_groups_for_run_ids(wandb_run_ids, **wandb_kwargs))
    # sort labels based on sorted wandb_groups
    labels = [label for _, label in sorted(zip(wandb_groups, labels))]
    wandb_groups = sorted(wandb_groups)

    assert len(wandb_groups) == len(labels), "wandb_groups and labels must have same length"
    return wandb_groups, labels
