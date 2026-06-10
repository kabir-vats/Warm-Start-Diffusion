# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import SiLU
from torch.nn.functional import silu

from src.models._base_model import BaseModel
from src.models.modules.mlp import Mlp
from src.utilities.timing import log_timing
from src.utilities.utils import get_logger


log = get_logger(__name__)

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode="kaiming_normal", init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=None,
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
        dimension: int = 2,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode, fan_in=in_channels * kernel**dimension, fan_out=out_channels * kernel**dimension
        )
        if dimension == 1:
            self.conv_func = torch.nn.functional.conv1d
            self.tconv_func = torch.nn.functional.conv_transpose1d
            weight_shape = [out_channels, in_channels, kernel]
        elif dimension == 2:
            self.conv_func = torch.nn.functional.conv2d
            self.tconv_func = torch.nn.functional.conv_transpose2d
            weight_shape = [out_channels, in_channels, kernel, kernel]
        elif dimension == 3:
            self.conv_func = torch.nn.functional.conv3d
            self.tconv_func = torch.nn.functional.conv_transpose3d
            weight_shape = [out_channels, in_channels, kernel, kernel, kernel]
        else:
            raise ValueError(f"Invalid dimension {dimension}")
        self._singleton_dims = [1] * dimension
        self.weight = torch.nn.Parameter(weight_init(weight_shape, **init_kwargs) * init_weight) if kernel else None
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        )
        if resample_filter is None:
            resample_filter = [1] * dimension
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:  # SongUnet only
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = self.conv_func(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = self.conv_func(x, w, padding=w_pad + f_pad)
            x = self.conv_func(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                # print(f"{x.shape=}, {f.shape=}, {f.mul(4).tile([self.in_channels, 1, 1, 1]).shape=}, {f_pad=}.")
                x = self.tconv_func(
                    x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad
                )
                # print(f"output shape after conv_transpose2d: {x.shape}") # ([14, 384, 240, 128])
            if self.down:
                x = self.conv_func(
                    x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad
                )
            if w is not None:
                x = self.conv_func(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, *self._singleton_dims))
        return x


# ----------------------------------------------------------------------------
# Group normalization.


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps
        )
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum("ncq,nck->nqk", q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32))
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


class AttentionOpCausal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        # Create causal mask: lower triangular matrix of ones
        seq_len = q.shape[2]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(q.device)

        # Compute attention scores
        attn_scores = torch.einsum("ncq,nck->nqk", q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32))

        # Apply mask by setting masked positions to -inf before softmax
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # Apply softmax and convert back to original dtype
        w = attn_scores.softmax(dim=2).to(q.dtype)

        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        cross_attention=False,
        context_channels: int = False,
        create_context_length: Optional[int] = None,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=None,
        resample_proj=False,
        adaptive_scale=True,
        emb_init_identity=False,  # Initialize the timestep embedding so that it has no effect.
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
        separate_attn_qkv=False,
        dimension: int = 2,
    ):
        assert adaptive_scale
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.attention = attention
        self.num_heads = (
            0 if attention is False else num_heads if num_heads is not None else out_channels // channels_per_head
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            dimension=dimension,
            **init,
        )
        if emb_channels is not None:
            affine_init = init_zero if emb_init_identity else init
            self.affine_scale = Linear(in_features=emb_channels, out_features=out_channels, **affine_init)
            self.affine_shift = Linear(in_features=emb_channels, out_features=out_channels, **affine_init)
        else:
            self.affine_scale = self.affine_shift = None

        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, dimension=dimension, **init_zero
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                dimension=dimension,
                **init,
            )

        init_attn = init_attn if init_attn is not None else init
        attn_kwargs = dict(kernel=1, dimension=dimension)
        self.separate_attn_qkv = separate_attn_qkv
        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            if separate_attn_qkv:
                #  this will make it a bit slower
                self.to_q = Conv2d(**attn_kwargs, in_channels=out_channels, out_channels=out_channels, **init_attn)
                self.to_k = Conv2d(**attn_kwargs, in_channels=out_channels, out_channels=out_channels, **init_attn)
                self.to_v = Conv2d(**attn_kwargs, in_channels=out_channels, out_channels=out_channels, **init_attn)
            else:
                self.qkv = Conv2d(**attn_kwargs, in_channels=out_channels, out_channels=out_channels * 3, **init_attn)
            self.proj = Conv2d(**attn_kwargs, in_channels=out_channels, out_channels=out_channels, **init_zero)

        if cross_attention:
            attn_kwargs_c = dict(kernel=1, dimension=1)
            self.norm3 = GroupNorm(num_channels=out_channels, eps=eps)
            self.to_q_xattn = Conv2d(**attn_kwargs, in_channels=out_channels, out_channels=out_channels, **init_attn)
            if create_context_length is not None:
                if isinstance(create_context_length, int):
                    self.preprocess_cond = Linear(context_channels, out_channels * create_context_length, **init)
                else:
                    create_context_length = int(create_context_length.replace("non_linear", ""))
                    self.preprocess_cond = torch.nn.Sequential(
                        Linear(context_channels, out_channels * create_context_length, **init),
                        SiLU(),
                        Linear(out_channels * create_context_length, out_channels * create_context_length, **init),
                    )
                context_channels = out_channels
                self.reshape_cond = lambda c: rearrange(c, "b (c l) -> b c l", c=out_channels, l=create_context_length)
                self.cond_pos_embed = torch.nn.Parameter(torch.randn(1, out_channels, create_context_length) * 0.02)
            else:
                self.preprocess_cond = torch.nn.Identity()
                self.reshape_cond = lambda c: c.unsqueeze(-1)
                self.cond_pos_embed = None

            self.to_k_xattn = Conv2d(
                **attn_kwargs_c, in_channels=context_channels, out_channels=out_channels, **init_attn
            )
            self.to_v_xattn = Conv2d(
                **attn_kwargs_c, in_channels=context_channels, out_channels=out_channels, **init_attn
            )
            self.proj_cattn = Conv2d(**attn_kwargs, in_channels=out_channels, out_channels=out_channels, **init_zero)

    def forward(self, x, emb, shift=None, scale=None, context=None):
        orig = x
        # print(f"UNetBlock: {x.shape=}, {self.in_channels=}, {self.out_channels=}, {self.emb_channels=}")
        x = self.conv0(silu(self.norm0(x)))

        if self.affine_scale is None:
            assert emb is None
            x = silu(self.norm1(x))
        else:  # elif emb is not None:
            if shift is None and scale is None:
                scale = self.affine_scale(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
                shift = self.affine_shift(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
                # params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
                # scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))  # shift + (1 + scale) * norm(x)

        x = self.conv1(self.dropout(x))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            b, c, h, w = x.shape
            nh = self.num_heads
            # x.shape = (B, C, H, W) and let B' = B * num_heads and C' = C // num_heads
            xnorm = self.norm2(x)
            if self.separate_attn_qkv:
                q = rearrange(self.to_q(xnorm), "b (nh c) h w -> (b nh) c (h w)", nh=nh)
                k = rearrange(self.to_k(xnorm), "b (nh c) h w -> (b nh) c (h w)", nh=nh)
                v = rearrange(self.to_v(xnorm), "b (nh c) h w -> (b nh) c (h w)", nh=nh)
            else:
                B2, C2 = x.shape[0] * self.num_heads, x.shape[1] // self.num_heads  # B2 = B * num_heads,  H * W
                q, k, v = self.qkv(xnorm).reshape(B2, C2, 3, -1).unbind(2)
            # q = k = v.shape = (B', C', H * W)
            # print(f"{q.shape=}, {k.shape=}, {v.shape=}, {x.shape=}")
            if self.attention == "causal":
                attn_w = AttentionOpCausal.apply(q, k)
            else:
                attn_w = AttentionOp.apply(q, k)  # w.shape = (B', H * W, H * W)
            a = torch.einsum("nqk,nck->ncq", attn_w, v)  # a.shape = (B', C', H * W)
            a = rearrange(a, "(b nh) c (h w) -> b (nh c) h w", b=b, nh=nh, h=h, w=w)
            x = self.proj(a).add_(x) * self.skip_scale

        if hasattr(self, "norm3"):
            B2, C2 = x.shape[0] * self.num_heads, x.shape[1] // self.num_heads  # B2 = B * num_heads,  H * W
            q = self.to_q_xattn(self.norm3(x)).reshape(
                B2, C2, -1
            )  # to_k(x).shape = x = (B, C, H, W), q = (B2, C2, H * W)
            context = self.reshape_cond(self.preprocess_cond(context))
            if self.cond_pos_embed is not None:
                context = context + self.cond_pos_embed
            # print(f"{x.shape=}, {self.to_q(self.norm3(x)).shape=}, {q.shape=}, {self.to_k(context).shape=}")
            k, v = self.to_k_xattn(context).reshape(B2, C2, -1), self.to_v_xattn(context).reshape(B2, C2, -1)

            w = AttentionOp.apply(q, k)  # w.shape = (B', H * W, seq_len_context)
            a = torch.einsum("nqk,nck->ncq", w, v)  # a.shape = (B', C3, H * W)
            x = self.proj_cattn(a.reshape(*x.shape)).add_(x) * self.skip_scale
            # print(f"c attn {x.shape=}, {q.shape=}, {context.shape=}, {k.shape=}, {v.shape=}, {w.shape=}, {a.shape=}")

        return x


class _TemporalAttentionBlock(torch.nn.Module):
    """Similar as attention in UNetBlock but with temporal attention"""

    def __init__(
        self,
        dim,
        num_heads=None,
        channels_per_head=None,
        eps=1e-5,
        init_zero=dict(init_weight=0),
        init_attn=None,
        causal: bool = False,
        non_linear_proj: bool = False,
        cond_dim: int = None,
        cond_type: str = "linear",
    ):
        super().__init__()
        # Assume that input is of shape (B * H * W, C, T), reshaped from (B, C, T, H, W)
        assert (
            num_heads is not None or channels_per_head is not None
        ), "num_heads or channels_per_head must be provided"
        in_channels_qkv = dim
        if cond_type == "concat":
            in_channels_qkv += 1

        self.num_heads = num_heads if num_heads is not None else dim // channels_per_head
        self.norm = GroupNorm(num_channels=in_channels_qkv, eps=eps)
        self.qkv = Conv2d(
            in_channels=in_channels_qkv, out_channels=dim * 3, kernel=1, **(init_attn or {}), dimension=1
        )
        # 1x1 conv on C
        self.proj = Conv2d(in_channels=dim, out_channels=dim, kernel=1, **init_zero, dimension=1)  # 1x1 conv on C
        self.cond_type = cond_type
        self.emb_map = None
        if cond_dim is None:
            self.affine = None
        elif cond_type == "linear":
            self.affine = Linear(cond_dim, 2 * dim, **init_zero)
        elif cond_type == "conv":
            self.affine = torch.nn.Conv1d(cond_dim, 2 * dim, kernel_size=3, padding=1)
        elif cond_type == "non-linear":
            self.emb_map = torch.nn.Linear(cond_dim, cond_dim)
            self.affine = torch.nn.Linear(cond_dim, 2 * dim, bias=True)
        elif cond_type == "non-linear-conv":
            self.emb_map = torch.nn.Conv1d(cond_dim, cond_dim, kernel_size=3, padding=1)
            self.affine = torch.nn.Conv1d(cond_dim, 2 * dim, kernel_size=3, padding=1)
        else:
            raise ValueError(f"Unknown cond_type: {cond_type}")

        self.causal = causal
        if causal:
            log.info("Using causal temporal attention")

    def forward(self, x, emb, h: int = None, w: int = None):
        skip = x
        B2, C2 = x.shape[0] * self.num_heads, x.shape[1] // self.num_heads  # B2 = B * num_heads * H * W
        # log.info(f"t-attn in: {x.shape=}, {B2=}, {C2=}")
        q, k, v = self.qkv(self.norm(x)).reshape(B2, C2, 3, -1).unbind(2)  # q = k = v.shape = (B', C', T)
        weight = AttentionOp.apply(q, k) if not self.causal else AttentionOpCausal.apply(q, k)
        attn = torch.einsum("nqk,nck->ncq", weight, v)
        gate = None
        if self.cond_type is not None:
            if "conv" in self.cond_type:
                emb = rearrange(emb, "(b t) c -> b c t", t=self.T)
                if self.emb_map is not None:
                    emb = silu(self.emb_map(emb))
                shift, scale = self.affine(emb).chunk(2, dim=1)
            else:
                emb = rearrange(emb, "(b t) c -> b t c", t=self.T)
                if self.emb_map is not None:
                    emb = silu(self.emb_map(emb))

                # shift, scale, gate = self.affine(emb).chunk(3, dim=-1)
                # shift, scale, gate = shift.transpose(1, 2), scale.transpose(1, 2), gate.transpose(1, 2)
                shift, scale = self.affine(emb).chunk(2, dim=-1)
                shift, scale = shift.transpose(1, 2), scale.transpose(1, 2)

            if shift.shape[0] > 1:
                shift = repeat(shift, "b c t -> (b h w) c t", h=h, w=w)
                scale = repeat(scale, "b c t -> (b h w) c t", h=h, w=w)
                gate = repeat(gate, "b c t -> (b h w) c t", h=h, w=w) if gate is not None else None

        if self.affine is None:
            x = self.proj(attn.reshape(*x.shape))
        else:
            if gate is None:
                x = self.proj(torch.addcmul(shift, attn.reshape(*x.shape), scale + 1))  # shift + (1 + scale) * norm(x)
            else:
                x = gate * self.proj(modulate(attn.reshape(*x.shape), shift, scale))
        # if False:
        #     x = self.conv5(silu(torch.addcmul(shift, self.norm5(x), scale + 1)))
        if self.alpha is not None:
            # if 1 (at init), pass through (identity). Can be useful when fine-tuning a non-temporal model
            with torch.no_grad():
                self.alpha.clamp_(0, 1)
            return self.alpha * skip + (1 - self.alpha) * x
        else:
            x = skip + x
        return x


# TemporalAttentionBlock = get_einops_wrapped_module(
#     _TemporalAttentionBlock, "b c t h w", "(b h w) c t"
# )


class TemporalAttentionBlock(_TemporalAttentionBlock):
    def __init__(
        self,
        dim: int,
        n_temporal_channels=None,
        pos_encoding: str = None,
        mixing_factor: bool = False,
        cond_type: str = "linear",
        **kwargs,
    ):
        super().__init__(dim=dim, cond_type=cond_type, **kwargs)
        self.T = n_temporal_channels
        self.alpha = torch.nn.Parameter(torch.ones(1)) if mixing_factor else None
        if pos_encoding == "learned":
            log.info(f"Using learned temporal positional encoding with max_frames={n_temporal_channels}")
            self.temporal_pos_encoding = torch.nn.Parameter(torch.randn(1, dim, n_temporal_channels) * 0.02)
        # todo: implement sinusoidal positional encoding based on the timestep arange(0, max_frames)
        else:
            self.temporal_pos_encoding = None
            assert pos_encoding is None, f"Unknown pos_encoding: {pos_encoding}"

    def forward(self, x, emb):
        h, w = x.shape[-2:]
        x = rearrange(x, "(b t) c h w -> (b h w) c t", t=self.T)
        if self.temporal_pos_encoding is not None:
            x = x + self.temporal_pos_encoding
        # Run attn
        x = super().forward(x, emb, h=h, w=w)
        x = rearrange(x, "(b h w) c t -> (b t) c h w", t=self.T, h=h, w=w)
        return x


class TemporalAttentionBlock2(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = None,
        channels_per_head=None,
        n_temporal_channels=None,
        pos_encoding: str = None,
        mlp_ratio=4.0,
        cond_dim: int = None,
        causal: bool = False,
        **kwargs,
    ):
        super().__init__()
        raise ImportError(
            "TemporalAttentionBlock2 depends on the optional attention/xformers stack "
            "that is not included in this ERA5-MINI transplant."
        )

        assert causal is False, "Causal attention not implemented"
        self.T = n_temporal_channels
        assert (
            num_heads is not None or channels_per_head is not None
        ), "num_heads or channels_per_head must be provided"
        self.num_heads = num_heads if num_heads is not None else dim // channels_per_head

        self.norm1 = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(dim, num_heads=self.num_heads, qkv_bias=True, **kwargs)
        self.norm2 = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)

        def approx_gelu():
            return torch.nn.GELU(approximate="tanh")

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.affine = torch.nn.Linear(cond_dim, 6 * dim, bias=True) if cond_dim is not None else None
        if cond_dim is not None:
            log.info(f"Using conditional temporal attention with {cond_dim=}")

        if pos_encoding == "learned":
            log.info(f"Using learned temporal positional encoding with max_frames={n_temporal_channels}")
            self.temporal_pos_encoding = torch.nn.Parameter(torch.randn(1, dim, n_temporal_channels) * 0.02)
        else:
            self.temporal_pos_encoding = None
            assert pos_encoding is None, f"Unknown pos_encoding: {pos_encoding}"

    def forward(self, x, emb):
        # NOTE: If x has very large batch sizes, xformers can be problematic (see https://github.com/facebookresearch/xformers/issues/845)
        h, w = x.shape[-2:]
        x = rearrange(x, "(b t) c h w -> (b h w) t c", t=self.T)
        # inputs.shape=[2, 69, 8, 240, 121] - [2*8, 69, 240, 121] -> [2*240*121, 8, 69], time.shape=torch.Size([2, 8])
        # shift_msa.shape=torch.Size([16, 256]), scale_msa.shape=torch.Size([16, 256]),
        # emb.shape=torch.Size([16, 1024]), x.shape=torch.Size([61440, 8, 256])
        if self.temporal_pos_encoding is not None:
            x = x + self.temporal_pos_encoding
        # Run attn
        if self.affine is None:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            # repeat from (b*t, c) -> (b, t, c, h, w) -> (b*h*w, t, c)
            def reshape_emb(emb):
                return repeat(rearrange(emb, "(b t) c -> b t c", t=self.T), "b t c -> (b h w) t c", h=h, w=w)

            # emb = repeat(rearrange(emb, "(b t) c -> b t c", t=self.T), "b t c -> (b h w) t c", h=h, w=w)
            # todo: replace affine with a reshape + 1d conv along time dimension (+non-linearity?)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.affine(emb).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa = reshape_emb(shift_msa), reshape_emb(scale_msa), reshape_emb(gate_msa)
            shift_mlp, scale_mlp, gate_mlp = reshape_emb(shift_mlp), reshape_emb(scale_mlp), reshape_emb(gate_mlp)
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        x = rearrange(x, "(b h w) t c -> (b t) c h w", t=self.T, h=h, w=w)
        return x


class TemporalAttentionBlock3(UNetBlock):
    def __init__(
        self,
        dim: int,
        n_temporal_channels=None,
        pos_encoding: str = None,
        cond_dim: int = None,
        cond_type: str = "linear",
        init_attn=None,
        conv_dim: int = 1,
        **kwargs,
    ):
        super().__init__(
            in_channels=dim,
            out_channels=dim,
            emb_channels=cond_dim,
            adaptive_scale=cond_dim is not None,
            emb_init_identity=True,
            dimension=conv_dim,
            init=init_attn,
            init_attn=init_attn,
            **kwargs,
        )
        self.T = n_temporal_channels
        self.conv_dim = conv_dim
        if pos_encoding == "learned":
            log.info(f"Using learned temporal positional encoding with max_frames={n_temporal_channels}")
            self.temporal_pos_encoding = torch.nn.Parameter(torch.randn(1, dim, n_temporal_channels) * 0.02)
        else:
            self.temporal_pos_encoding = None
            assert pos_encoding is None, f"Unknown pos_encoding: {pos_encoding}"
        self.cond_type = cond_type
        if cond_type == "conv":
            self.affine = Conv2d(cond_dim, 2 * dim, kernel=3, **init_attn, dimension=1)

    def forward(self, x, emb):
        h, w = x.shape[-2:]
        from_pattern = "(b t) c h w"
        if self.conv_dim == 1:
            to_pattern = "(b h w) c t"
        elif self.conv_dim == 3:
            to_pattern = "b c t h w"
        else:
            raise ValueError(f"Invalid conv_dim: {self.conv_dim}")
        x = rearrange(x, f"{from_pattern} -> {to_pattern}", t=self.T)
        if self.temporal_pos_encoding is not None:
            x = x + self.temporal_pos_encoding

        shift, scale = None, None
        if self.affine is None:
            del emb
        else:
            if self.cond_type == "conv":
                emb = rearrange(emb, "(b t) c -> b c t", t=self.T) if emb.ndim == 2 else emb
                shift, scale = self.affine(emb).chunk(2, dim=1)
            else:
                emb = rearrange(emb, "(b t) c -> b t c", t=self.T)
                shift, scale = self.affine(emb).chunk(2, dim=-1)
                shift, scale = shift.transpose(1, 2), scale.transpose(1, 2)
            if self.conv_dim == 3:
                shift = shift.unsqueeze(-1).unsqueeze(-1)
                scale = scale.unsqueeze(-1).unsqueeze(-1)
            elif shift.shape[0] > 1 and self.conv_dim == 1:
                shift = repeat(shift, "b c t -> (b h w) c t", h=h, w=w)
                scale = repeat(scale, "b c t -> (b h w) c t", h=h, w=w)

        # Run attn
        x = super().forward(x, emb=None, shift=shift, scale=scale)
        x = rearrange(x, f"{to_pattern} -> {from_pattern}", t=self.T, h=h, w=w)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion
temporal_blocks = (TemporalAttentionBlock, TemporalAttentionBlock2, TemporalAttentionBlock3)


class DhariwalUNet(BaseModel):
    def __init__(
        self,
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 3, 4],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[32, 16, 8],  # List of resolutions with self-attention.
        attn_levels=None,  # List of levels with self-attention.
        channels_per_head=64,  # Number of channels per self-attention head.
        dropout=0.10,  # List of resolutions with self-attention.
        dropout_mode: str = None,
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        with_time_emb: bool = True,  # Include time embedding in the output.
        emb_init_identity: bool = False,  # Initialize the timestep adaptive LN so that it has no effect.
        map_emb_type: str = "mlp",  # Type of embedding mapper: mlp, fourier, positional
        outer_sample_mode: str = None,  # bilinear or nearest
        upsample_dims: tuple = None,  # (256, 256) or (128, 120) etc. This is used for the input image size.
        upsample_hidden_spatial_dims_auto: bool = False,  # Automatically upsample hidden spatial dims when uneven.
        upsample_outputs_by: int = 1,
        separate_attn_qkv=False,
        in_channel_cross_attn: bool = False,
        non_spatial_conditioning_mode: str = None,
        non_spatial_cond_hdim: int = None,
        create_context_length: int = None,
        null_embedding_for_non_spatial_cond: str = "zeros",  # zeros, noop, learn
        keep_spatial_dims: bool = False,  # Whether to keep spatial dims fixed
        **kwargs,
    ):
        super().__init__(**kwargs)
        # assert (
        # self.spatial_shape_out[0] == self.spatial_shape_out[1] and len(self.spatial_shape_out) == 2
        # ), "Only square images are supported."
        self.save_hyperparameters()
        if self.hparams.debug_mode:
            self.log_text.info("Running in debug mode with toy parameters.")
            self.hparams.channel_mult = channel_mult = (1, 2, 4)
            self.hparams.model_channels = model_channels = 32
            self.hparams.num_blocks = num_blocks = 2
            self.hparams.attn_resolutions = attn_resolutions = []
            self.hparams.attn_levels = attn_levels = [2]
            self.hparams.channels_per_head = channels_per_head = 16

        img_resolution = self.spatial_shape_out[0]  # Image resolution at input/output.
        in_channels = self.num_input_channels + self.num_conditional_channels
        out_channels = self.num_output_channels
        attn_levels = attn_levels or []

        self.label_dropout = label_dropout
        init = self.init = dict(init_mode="kaiming_uniform", init_weight=np.sqrt(1 / 3), init_bias=np.sqrt(1 / 3))
        init_zero = self.init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)

        self.initialize_non_spatial_conditioning(
            non_spatial_conditioning_mode, non_spatial_cond_hdim, null_embedding_for_non_spatial_cond
        )
        # Mapping.
        self.with_time_emb, self.augment_dim, self.label_dim = with_time_emb, augment_dim, label_dim
        self.map_emb_type = map_emb_type
        if with_time_emb or non_spatial_conditioning_mode == "adaLN" or self.hparams.noise_dim > 0:
            emb_channels = self.emb_channels = int(model_channels * channel_mult_emb)
            if with_time_emb:
                self.map_noise = PositionalEmbedding(num_channels=model_channels)
                map_layer_in_c = model_channels
            else:
                self.map_noise = torch.nn.Identity()
                if non_spatial_conditioning_mode == "adaLN":
                    map_layer_in_c = self.non_spatial_cond_hdim
                else:
                    map_layer_in_c = self.hparams.noise_dim

            self.map_layer0 = Linear(in_features=map_layer_in_c, out_features=emb_channels, **init)
            if map_emb_type == "mlp":
                self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        else:
            self.map_noise = None
            emb_channels = self.emb_channels = None

        use_cross_attn = self.non_spatial_conditioning_mode == "cross_attn"
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=channels_per_head,  # caution: if dim < channels_per_head, no attention will be applied
            dropout=dropout,
            init=init,
            init_zero=init_zero,
            emb_init_identity=emb_init_identity,
            context_channels=self.non_spatial_cond_hdim,
            create_context_length=create_context_length,
            separate_attn_qkv=separate_attn_qkv,
        )
        enc_block_kwargs, dec_block_kwargs = block_kwargs.copy(), block_kwargs.copy()
        if dropout_mode is not None:
            if "decoder_only" in dropout_mode:
                enc_block_kwargs["dropout"] = 0.0
            elif "encoder_only" in dropout_mode:
                dec_block_kwargs["dropout"] = 0.0

        if augment_dim:
            self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero)
        else:
            self.map_augment = None

        if label_dim:
            self.map_label = Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
        else:
            self.map_label = None

        if upsample_dims is not None:
            assert outer_sample_mode is not None, "Need to provide outer_sample_mode when upsample_dims is provided."
            # Upsample (45, 90) -> (48, 96) to be easier to divide by 2 multiple times
            self.upsampler = torch.nn.Upsample(size=tuple(upsample_dims), mode=outer_sample_mode)
        else:
            if upsample_hidden_spatial_dims_auto and outer_sample_mode is None:
                self.hparams.outer_sample_mode = "bilinear"
            self.upsampler = None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        skips = []
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            use_attn = level in attn_levels or res in attn_resolutions
            if use_attn:
                self.log_text.info(f"Using attention at enc. {res=}, {level=}, {mult=}, {len(channel_mult)=}")
            if level == 0:
                cin = cout
                cout = model_channels * mult
                if in_channel_cross_attn:
                    raise ImportError(
                        "in_channel_cross_attn depends on the optional Stormer/ClimaX modules "
                        "that are not included in this ERA5-MINI transplant."
                    )
                else:
                    self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                down = not keep_spatial_dims
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=down, **enc_block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=use_attn,
                    cross_attention=use_attn and use_cross_attn,
                    **enc_block_kwargs,
                )

        skips += [block.out_channels for block in self.enc.values()]
        self.condition_readouts = ("mean", "sum", "max")
        self.condition_head = None
        if self.predict_non_spatial_condition:
            condition_head_in_dim = len(self.condition_readouts) * cout
            if self.non_spatial_conditioning_mode == "adaLN":
                condition_head_in_dim += self.time_dim
            # Add readout layer for condition prediction
            self.condition_head = torch.nn.Sequential(
                torch.nn.Linear(condition_head_in_dim, condition_head_in_dim),
                torch.nn.GELU(),
                torch.nn.Linear(condition_head_in_dim, self.non_spatial_cond_hdim),
            )

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        num_blocks_decoder = (2 + num_blocks) * len(channel_mult) + 1
        if dropout_mode == "decoder_only_dec":
            dropouts_decoder = np.linspace(0.01, dropout, num_blocks_decoder).tolist()
        elif dropout_mode == "decoder_only_inc":
            dropouts_decoder = np.linspace(dropout, 0.01, num_blocks_decoder).tolist()
        else:
            dropouts_decoder = [dropout] * num_blocks_decoder

        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            use_attn = level in attn_levels or res in attn_resolutions  # Note that level is reversed here.
            # use_attn = (len(channel_mult) - 1 - level in attn_levels) or res in attn_resolutions
            if use_attn:
                self.log_text.info(
                    f"Using attention at dec. resolution {res}, {level=}, {mult=}, {len(channel_mult)=}"
                )
            if level == len(channel_mult) - 1:
                attn = not keep_spatial_dims
                dec_block_kwargs["dropout"] = dropouts_decoder.pop()
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    attention=attn,
                    cross_attention=use_cross_attn,
                    **dec_block_kwargs,
                )
                dec_block_kwargs["dropout"] = dropouts_decoder.pop()
                self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **dec_block_kwargs)
            else:
                up = not keep_spatial_dims
                dec_block_kwargs["dropout"] = dropouts_decoder.pop()
                self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=up, **dec_block_kwargs)

            for idx in range(num_blocks + 1):  # actual block count = num_blocks + 1
                cin = cout + skips.pop()
                cout = model_channels * mult
                super_res = upsample_outputs_by > 1 and idx == num_blocks and level == 0
                dec_block_kwargs["dropout"] = dropouts_decoder.pop()
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=use_attn,
                    cross_attention=use_attn and use_cross_attn,
                    up=super_res,
                    **dec_block_kwargs,
                )
        assert len(dropouts_decoder) == 0, f"{dropouts_decoder=}, {num_blocks_decoder=}"
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def encode(
        self,
        inputs,
        time=None,  # was called noise_labels
        time_emb_2=None,
        class_labels=None,
        augment_labels=None,
        condition=None,
        dynamical_condition=None,
        condition_non_spatial=None,
        static_condition=None,
        return_time_emb: bool = False,
        skip_temporal_blocks: bool = False,
        metadata=None,
        member: int = None,
    ):
        x = self.concat_condition_if_needed(inputs, condition, dynamical_condition, static_condition)
        if condition_non_spatial is not None:
            condition_non_spatial = self.preprocess_non_spatial_conditioning(condition_non_spatial)
        orig_x_shape = x.shape[-2:]
        x = self.upsampler(x) if self.upsampler is not None else x

        # Mapping.
        if self.non_spatial_conditioning_mode == "adaLN":
            assert time is None, "time is not None but non_spatial_conditioning_mode is adaLN"
            # Text emb is null if all features are zero. Check for each batch element and replace with null embedding.
            # Calculate L2 norm along embedding dimension
            assert len(condition_non_spatial.shape) == 2, f"{condition_non_spatial.shape=}"
            ada_ln_input = condition_non_spatial
        elif time is not None:
            ada_ln_input = time
        elif self.hparams.noise_dim > 0:
            ada_ln_input = torch.randn(x.shape[0], self.hparams.noise_dim, device=x.device)
        else:
            assert self.map_noise is None, f"time is None but {self.map_noise=}. {self.with_time_emb=}"

        if self.map_noise is not None:
            emb = self.map_noise(ada_ln_input)
            if self.map_augment is not None and augment_labels is not None:
                emb = emb + self.map_augment(augment_labels)
            emb = self.map_layer0(emb)
            if self.map_emb_type == "silu":
                emb = silu(emb)
            elif self.map_emb_type == "mlp":
                emb = silu(emb)
                emb = self.map_layer1(emb)
                if self.map_label is not None:
                    tmp = class_labels
                    if self.training and self.label_dropout:
                        tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
                    emb = emb + self.map_label(tmp)
                emb = silu(emb)  # emb = silu(linear(silu(linear(emb))))
            else:
                assert self.map_emb_type in ["linear"], f"{self.map_emb_type=}"
        else:
            emb = None
        if time_emb_2 is None:
            time_emb_2 = emb

        for_cross_attn = condition_non_spatial if self.non_spatial_conditioning_mode == "cross_attn" else None

        # blocks_with_emb = (UNetBlock, TemporalAttentionBlock, TemporalAttentionBlock2)
        # Encoder.
        skips = []
        for k, block in self.enc.items():
            # todo: first input layer shouldnt be used by Muon?!
            if isinstance(block, UNetBlock):
                x = block(x, emb, context=for_cross_attn)
                skips.append(x)
            elif isinstance(block, temporal_blocks):
                x = block(x, time_emb_2) if not skip_temporal_blocks else x
            else:
                x = block(x)
                skips.append(x)
            # 360x360_conv   torch.Size([4, 256, 360, 181])
            # 180x180_down   torch.Size([4, 256, 180, 90])
            # 180x180_block2 torch.Size([4, 512, 180, 90])
            # 90x90_down     torch.Size([4, 512, 90, 45])
            # 90x90_block0   torch.Size([4, 768, 90, 45])
            # 45x45_down     torch.Size([4, 768, 45, 22])
            # 45x45_block0   torch.Size([4, 1024, 45, 22])

        if self.condition_head is not None:
            # Add condition prediction
            # Compute statistics of the bottleneck layer
            if self.condition_readouts == ("mean", "sum", "max"):
                cond_head_input = torch.cat(
                    [x.mean(dim=(-2, -1)), x.sum(dim=(-2, -1)), torch.amax(x, dim=(-2, -1))], dim=1
                )
            else:
                raise ValueError(f"Unknown condition readouts: {self.condition_readouts}")
            # concat with time embedding
            if emb is not None:
                cond_head_input = torch.cat((cond_head_input, emb), dim=1)
            condition_pred = self.condition_head(cond_head_input)
        else:
            condition_pred = None

        return dict(
            x=x,
            skips=skips,
            condition_pred=condition_pred,
            emb=emb,
            time_emb_2=time_emb_2,
            orig_x_shape=orig_x_shape,
            for_cross_attn=for_cross_attn,
        )

    def decode(
        self,
        x,
        skips,
        emb,
        time_emb_2,
        orig_x_shape,
        for_cross_attn=None,
        skip_temporal_blocks: bool = False,
        condition_pred=None,
    ):
        # Decoder.
        for block_name, block in self.dec.items():
            # log.info(f"block: {block_name}, {x.shape=}, {skips[-1].shape=}, {x.shape[1] != block.in_channels}")
            # print(f"dec block: {block_name}, {x.shape=}, {skips[-1].shape=}, {emb is None}")
            if not isinstance(block, temporal_blocks) and x.shape[1] != block.in_channels:
                skip = skips.pop()
                if self.hparams.upsample_hidden_spatial_dims_auto and skip.shape[-2:] != x.shape[-2:]:
                    # self.log_text.info(f"Upsampling skip connection from {x.shape[-2:]} to {skip.shape[-2:]}")
                    x = F.interpolate(x, size=skip.shape[-2:], mode=self.hparams.outer_sample_mode)
                x = torch.cat([x, skip], dim=1)
            if isinstance(block, UNetBlock):
                x = block(x, emb, context=for_cross_attn)
            elif isinstance(block, temporal_blocks):
                x = block(x, time_emb_2) if not skip_temporal_blocks else x
            else:
                x = block(x)

        # log.info(f"final x shape: {x.shape}")
        if self.upsampler is not None:
            # x = F.interpolate(x, orig_x_shape, mode='bilinear', align_corners=False)
            x = F.interpolate(x, size=orig_x_shape, mode=self.hparams.outer_sample_mode)

        x = self.out_conv(silu(self.out_norm(x)))
        return x

    @log_timing(name="model_forward", log_every_n=50)
    def forward(
        self,
        inputs,
        time=None,  # was called noise_labels
        time_emb_2=None,
        class_labels=None,
        augment_labels=None,
        condition=None,
        dynamical_condition=None,
        condition_non_spatial=None,
        static_condition=None,
        return_time_emb: bool = False,
        skip_temporal_blocks: bool = False,
        metadata=None,
        member: int = None,
    ):
        enc_data = self.encode(
            inputs,
            time=time,
            time_emb_2=time_emb_2,
            class_labels=class_labels,
            augment_labels=augment_labels,
            condition=condition,
            dynamical_condition=dynamical_condition,
            condition_non_spatial=condition_non_spatial,
            static_condition=static_condition,
            return_time_emb=return_time_emb,
            skip_temporal_blocks=skip_temporal_blocks,
            metadata=metadata,
            member=member,
        )

        x = self.decode(
            **enc_data,
            skip_temporal_blocks=skip_temporal_blocks,
        )
        condition_pred = enc_data["condition_pred"]
        return_dict = dict(preds=x, condition_non_spatial=condition_pred) if condition_pred is not None else x

        if return_time_emb:
            return return_dict, enc_data["emb"]
        return return_dict

    def ensemble_forward(self, postprocess_func, *inputs, n_ens_mems: int = None, **kwargs):
        if self.hparams.dropout_mode is None or "encoder_only" in self.hparams.dropout_mode:
            return super()._ensemble_forward(postprocess_func, *inputs, n_ens_mems=n_ens_mems, **kwargs)
        elif "decoder_only" in self.hparams.dropout_mode:
            enc_data = self.encode(*inputs, **kwargs)
            enc_data = {
                k: torch.cat([v for _ in range(n_ens_mems)], dim=0) if torch.is_tensor(v) else v
                for k, v in enc_data.items()
            }
            enc_data["skips"] = [torch.cat([s for _ in range(n_ens_mems)], dim=0) for s in enc_data["skips"]]
            x = postprocess_func(self.decode(**enc_data))
            x = x.reshape(n_ens_mems, -1, *x.shape[1:])  # (ens, batch, 1)
            return x
        else:
            raise ValueError(f"Unknown dropout_mode: {self.hparams.dropout_mode}")


class DhariwalUNetTemporal(DhariwalUNet):
    def __init__(
        self,
        temporal_op: Optional[str] = None,  # Temporal operation: 'attn', 'causal_attn'.
        temporal_resolutions: Optional[List[int]] = None,  # List of resolutions to use temporal operation.
        init_temporal_op: bool = True,  # use temporal block after first conv or not
        temporal_op_before_down: Optional[bool] = None,
        # use temporal block before downsampling or not. If False, use after downsampling.
        final_temporal_op: bool = False,  # use temporal block after the last block or not
        temporal_pos_encoding: Optional[str] = None,  # frame position embedding.
        use_mixing_factor: bool = False,
        temporal_attn_type: str = "adm",  # or "vit"
        causal_attn: bool = False,
        use_time_emb_for_temporal_layers: str = None,
        extra_time_emb_mlp: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temporal_op = temporal_op
        assert temporal_op in ["attn", "causal_attn"], f"Unknown temporal operation: {temporal_op}"
        assert self.num_temporal_channels is not None, f"num_temporal_channels: {self.num_temporal_channels}"

        # Create new ModuleDict with both original and temporal layers in correct order
        last_block = self.hparams.num_blocks - 1
        max_res = self.spatial_shape_out[0]
        kv_out_channels = None
        t_attn_kwargs = dict(
            channels_per_head=self.hparams.channels_per_head,
            n_temporal_channels=self.num_temporal_channels,
            pos_encoding=temporal_pos_encoding,
            cond_dim=self.emb_channels if use_time_emb_for_temporal_layers else None,
        )
        self.cond_type = use_time_emb_for_temporal_layers
        if temporal_attn_type == "adm":
            temporal_attn_class = TemporalAttentionBlock
            t_attn_kwargs["init_zero"] = self.init_zero
            t_attn_kwargs["init_attn"] = self.init
            t_attn_kwargs["mixing_factor"] = use_mixing_factor
            t_attn_kwargs["cond_type"] = use_time_emb_for_temporal_layers
            t_attn_kwargs["causal"] = causal_attn
        elif "adm_block" in temporal_attn_type:
            temporal_attn_class = TemporalAttentionBlock3
            t_attn_kwargs["init_zero"] = self.init_zero
            t_attn_kwargs["init_attn"] = self.init
            t_attn_kwargs["cond_type"] = use_time_emb_for_temporal_layers
            # Check if ends with a number for conv dimensions (1 vs 3)
            conv_dim = 1
            if temporal_attn_type[-1].isdigit():
                t_attn_kwargs["conv_dim"] = conv_dim = int(temporal_attn_type[-1])
            if conv_dim == 1:
                t_attn_kwargs["attention"] = "causal" if causal_attn else True
            else:
                t_attn_kwargs["attention"] = False
        elif temporal_attn_type == "vit":
            t_attn_kwargs["causal"] = causal_attn
            temporal_attn_class = TemporalAttentionBlock2

        else:
            raise ValueError(f"Unknown attn_type: {temporal_attn_type}")

        self.extra_time_emb_mlp = extra_time_emb_mlp
        if extra_time_emb_mlp:
            log.info("Using extra time embedding MLP")
            model_channels = self.hparams.model_channels
            self.map_noise_T = PositionalEmbedding(num_channels=model_channels)
            if use_time_emb_for_temporal_layers == "linear":
                self.map_layer0_T = Linear(in_features=model_channels, out_features=self.emb_channels, **self.init)
                self.map_layer1_T = Linear(in_features=self.emb_channels, out_features=self.emb_channels, **self.init)
            elif use_time_emb_for_temporal_layers == "conv":
                log.info("Using conv for time embedding")
                self.map_layer0_T = Conv2d(model_channels, self.emb_channels, kernel=3, **self.init, dimension=1)
                self.map_layer1_T = Conv2d(self.emb_channels, self.emb_channels, kernel=3, **self.init, dimension=1)
            else:
                raise ValueError(f"Unknown time_emb_for_temporal_layers: {use_time_emb_for_temporal_layers}")

        new_enc = torch.nn.ModuleDict()
        for k, v in self.enc.items():
            res = int(k.split("x")[0])
            if temporal_resolutions is not None and res not in temporal_resolutions:
                new_enc[k] = v
                kv_out_channels = v.out_channels
                continue

            if "_down" in k and temporal_op_before_down is True:
                # Insert temporal block before downsampling
                new_enc[f"{res}x{res}_temporal_before_down"] = temporal_attn_class(kv_out_channels, **t_attn_kwargs)

            new_enc[k] = v  # Add original layer
            kv_out_channels = v.out_channels

            if "_conv" in k and init_temporal_op:
                # Insert temporal block after the first conv layer
                assert res == max_res, f"Temporal block should be added only to the first layer. {res=}, {max_res=}"
                new_enc[f"{res}x{res}_temporal_init"] = temporal_attn_class(kv_out_channels, **t_attn_kwargs)
            if "_down" in k and temporal_op_before_down is False:
                # Insert temporal block after downsampling
                new_enc[f"{res}x{res}_temporal_after_down"] = temporal_attn_class(kv_out_channels, **t_attn_kwargs)

            if k == f"{res}x{res}_block{last_block}":
                # Insert temporal block after the last block
                new_enc[f"{res}x{res}_temporal_last_block"] = temporal_attn_class(kv_out_channels, **t_attn_kwargs)

        # Replace the original enc with our new ordered version
        self.enc = new_enc

        new_dec = torch.nn.ModuleDict()
        for k, v in self.dec.items():
            res = int(k.split("x")[0])
            if temporal_resolutions is not None and res not in temporal_resolutions:
                new_dec[k] = v
                kv_out_channels = v.out_channels
                continue

            if "_up" in k and temporal_op_before_down is True:
                # Insert temporal block before upsampling
                new_dec[f"{res}x{res}_temporal_before_up"] = temporal_attn_class(kv_out_channels, **t_attn_kwargs)

            new_dec[k] = v  # Add original layer
            kv_out_channels = v.out_channels

            if "_in1" in k:
                # Insert temporal block after the first two decoder conv layers
                new_dec[f"{res}x{res}_temporal_in2"] = temporal_attn_class(kv_out_channels, **t_attn_kwargs)

            if "_up" in k and temporal_op_before_down is False:
                # Insert temporal block after upsampling
                new_dec[f"{res}x{res}_temporal_after_up"] = temporal_attn_class(kv_out_channels, **t_attn_kwargs)

            if res == max_res and not final_temporal_op:
                continue  # Skip the last block
            elif k == f"{res}x{res}_block{last_block+1}":
                # Insert temporal block after the last block  (note that one more block is added in the decoder)
                new_dec[f"{res}x{res}_temporal_last_block"] = temporal_attn_class(kv_out_channels, **t_attn_kwargs)

        # Replace the original dec with our new ordered version
        self.dec = new_dec

        #  Print the new enc and dec
        log.info(f"enc = {' '.join([k for k in self.enc.keys()])}")  # print(f"{self.enc=}")
        # enc = 256x256_conv 256x256_block0 256x256_block1 256x256_temporal_last_block 128x128_down 128x128_block0
        # 128x128_block1 128x128_temporal_last_block 64x64_down 64x64_block0 64x64_block1 64x64_temporal_last_block
        log.info(f"dec = {' '.join([k for k in self.dec.keys()])}")  # print(f"{self.dec=}")
        # dec = 64x64_in0 64x64_in1 64x64_temporal_in2 64x64_block0 64x64_block1 64x64_block2 64x64_temporal_last_block
        # 128x128_up 128x128_block0 128x128_block1 128x128_block2 128x128_temporal_last_block 256x256_up
        # 256x256_block0 256x256_block1 256x256_block2
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out adaLN modulation layers in blocks:
        for m in self.enc.values():
            if isinstance(m, (TemporalAttentionBlock, TemporalAttentionBlock2)):
                if m.affine is not None:
                    m.affine.weight.data.zero_()
                    m.affine.bias.data.zero_()
        for m in self.dec.values():
            if isinstance(m, (TemporalAttentionBlock, TemporalAttentionBlock2)):
                if m.affine is not None:
                    m.affine.weight.data.zero_()
                    m.affine.bias.data.zero_()

    def forward(self, inputs, time=None, **kwargs):
        # assert len(time.shape) == 2, f"{time.shape=}"
        b, c, t = inputs.shape[:3]
        # b, t_dim = time.shape
        # assert inputs.shape[0] == b, f"{inputs.shape[0]=}, b: {b}"
        # assert inputs.shape[2] == t_dim, f"{inputs.shape[2]=}, t_dim: {t_dim}"
        time = rearrange(time, "b t -> (b t)")
        x = rearrange(inputs, "b c t h w -> (b t) c h w")  # concat with cond: (b t) c h w -> (b t) (c+cond) h w
        if kwargs.get("static_condition") is not None:
            # Copy the static condition to all time steps b c' h w -> (b t) c' h w
            kwargs["static_condition"] = repeat(kwargs["static_condition"], "b c2 h w -> (b t) c2 h w", t=t)

        emb_t = None
        if self.extra_time_emb_mlp:
            emb_t = self.map_noise_T(time)
            if self.cond_type == "conv":
                emb_t = rearrange(emb_t, "(b t) c -> b c t", b=b, t=t)

            emb_t = silu(self.map_layer1_T(silu(self.map_layer0_T(emb_t))))

        # self.log_text.info(f"\n{inputs.shape=}, {x.shape=}, time.shape: {time.shape if time is not None else None}")
        x = super().forward(x, time=time, time_emb_2=emb_t, **kwargs)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b)
        return x


if __name__ == "__main__":
    n_non_spatial_c = 396
    unet = DhariwalUNet(
        model_channels=64,
        num_input_channels=3,
        num_output_channels=6,
        spatial_shape_in=(45, 90),
        spatial_shape_out=(45, 90),
        upsample_dims=(48, 96),
        outer_sample_mode="bilinear",
        non_spatial_conditioning_mode="cross_attn",
        num_conditional_channels_non_spatial=n_non_spatial_c,
        non_spatial_cond_hdim=None,
        with_time_emb=False,
        attn_levels=range(3),
        create_context_length=32,
    )
    B = 10
    x = torch.rand(B, 3, 45, 90)
    cond = torch.randn(B, n_non_spatial_c)
    y = unet(x, condition_non_spatial=cond)
    # print(unet.print_intermediate_shapes(x, n_non_spatial_c))
