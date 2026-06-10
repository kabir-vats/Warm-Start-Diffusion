from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.modules import convs
from src.utilities.utils import get_logger


log = get_logger(__name__)

PADDING_MODE_SET = None  # Global variable to store the padding mode so that it is set only once.


# Padding
# @torch.compile
def pad_before_conv2d_circular_width_only(input: Tensor, weight: Tensor, padding: int, torch_func, **kwargs) -> Tensor:
    """
    Args:
        torch_func: F.conv2d or F.conv_transpose2d
        input: (B, C_in, H, W)
        weight: (C_out, C_in, H_k, W_k)
        padding: will be equally applied to left, right, top, bottom. Mode is circular for width, zero for height.
    """
    if padding == 0:
        return torch_func(input, weight, **kwargs)
    if padding < 0:
        raise ValueError("padding should be a non-negative integer")
    # Pad circularly around width with F.pad
    # print(f"before pad: {input.shape}")
    input = F.pad(input, (padding, padding, 0, 0), mode="circular")
    # print(f"after pad: {input.shape} with padding={padding}. {torch_func=}")
    # Conv2d, using zero-padding on height only
    return torch_func(input, weight, **kwargs, padding=(padding, 0))


# @torch.compile
def pad_before_conv2d_circular_height_only(
    input: Tensor, weight: Tensor, padding: int, torch_func, **kwargs
) -> Tensor:
    """
    Args:
        torch_func: F.conv2d or F.conv_transpose2d
        input: (B, C_in, H, W)
        weight: (C_out, C_in, H_k, W_k)
        padding: will be equally applied to left, right, top, bottom. Mode is circular for height, zero for width.
    """
    if padding == 0:
        return torch_func(input, weight, **kwargs)
    if padding < 0:
        raise ValueError("padding should be a non-negative integer")
    # Pad circularly around height
    input = F.pad(input, (0, 0, padding, padding), mode="circular")
    # Conv2d, using zero-padding on width only
    return torch_func(input, weight, **kwargs, padding=(0, padding))


class ConvWithCircularPaddingWidthOnly(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        conv_class: torch.nn.Module,
        **kwargs,
    ):
        super(ConvWithCircularPaddingWidthOnly, self).__init__()
        # Width padding is done before calling conv2d.
        kwargs["padding_mode"] = "zeros"  # Inside Conv, only pad with zeros for height
        kwargs["padding"] = (padding, 0)
        self.pad_width = padding
        self.conv = conv_class(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        return self.conv(x)


class ConvWithCircularPaddingHeightOnly(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        conv_class: torch.nn.Module,
        **kwargs,
    ):
        super(ConvWithCircularPaddingHeightOnly, self).__init__()
        # Height padding is done before calling conv2d.
        kwargs["padding_mode"] = "zeros"  # Inside Conv, only pad with zeros for width
        kwargs["padding"] = (0, padding)
        self.pad_height = padding
        self.conv = conv_class(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad_height, self.pad_height), mode="circular")
        return self.conv(x)


def set_global_padding_mode(padding_mode: str, conv1d_too: bool = False):
    # Check if already set by other modules
    global PADDING_MODE_SET
    if PADDING_MODE_SET is not None:
        #if PADDING_MODE_SET != padding_mode:
            #raise ValueError(f"Padding mode is already set to ``{PADDING_MODE_SET}``.")
        return

    PADDING_MODE_SET = padding_mode  # Set the global padding mode

    log.info(f"Setting padding mode of Conv's to ``{padding_mode}`` globally.")
    if padding_mode in ["circular_width_only", "circular_height_only"]:
        # Pad circularly around width (height) only, zero-pad around height (width).
        if conv1d_too:
            torch.nn.Conv1d = lambda x: NotImplementedError(
                f"{padding_mode=} padding is not implemented for 1D convolutions."
            )
        torch.nn.Conv3d = lambda x: NotImplementedError(
            f"{padding_mode=} padding is not implemented for 3D convolutions."
        )
        if padding_mode == "circular_width_only":
            pad_class = ConvWithCircularPaddingWidthOnly
            pad_class_f = pad_before_conv2d_circular_width_only
        else:
            pad_class = ConvWithCircularPaddingHeightOnly
            pad_class_f = pad_before_conv2d_circular_height_only

        convs.WeightStandardizedConv2d = partial(pad_class, conv_class=convs.WeightStandardizedConv2d)
        torch.nn.Conv2d = partial(pad_class, conv_class=torch.nn.Conv2d)
        torch.nn.ConvTranspose2d = partial(pad_class, conv_class=torch.nn.ConvTranspose2d)
        torch.nn.functional.conv2d = partial(pad_class_f, torch_func=torch.nn.functional.conv2d)
        torch.nn.functional.conv_transpose2d = partial(pad_class_f, torch_func=torch.nn.functional.conv_transpose2d)
    else:
        convs.WeightStandardizedConv2d = partial(convs.WeightStandardizedConv2d, padding_mode=padding_mode)
        torch.nn.Conv2d = partial(torch.nn.Conv2d, padding_mode=padding_mode)
        torch.nn.Conv3d = partial(torch.nn.Conv3d, padding_mode=padding_mode)
        torch.nn.ConvTranspose2d = partial(torch.nn.ConvTranspose2d, padding_mode=padding_mode)
        torch.nn.ConvTranspose3d = partial(torch.nn.ConvTranspose3d, padding_mode=padding_mode)
        if conv1d_too:
            torch.nn.Conv1d = partial(torch.nn.Conv1d, padding_mode=padding_mode)
            torch.nn.ConvTranspose1d = partial(torch.nn.ConvTranspose1d, padding_mode=padding_mode)


if __name__ == "__main__":
    # Check F.pad
    input = torch.arange(9).reshape(1, 1, 3, 3).float()
    print(f"{input}")
    input_pad_circ1 = F.pad(input, (1, 1, 1, 1), mode="circular")
    print(f"{input_pad_circ1}")
    input_pad_circ2 = F.pad(input, (1, 1, 0, 0), mode="circular")  # pad around width only
    print(f"input_pad_circ2 ({input_pad_circ2.shape})=\n{input_pad_circ2}")
    # Zero pad in height
    input_pad_circ3 = F.pad(input_pad_circ2, (0, 0, 1, 1), mode="constant")
    print(f"input_pad_circ3 ({input_pad_circ3.shape})=\n{input_pad_circ3}")
    # Check if conv on input_pad_circ3 is the same as conv on input_pad_circ2 with padding on height only
    weight = torch.randn(1, 1, 3, 3)  # kernel size 3x3 with padding 1 --> Keep the same size
    out1 = F.conv2d(input_pad_circ3, weight, padding=0)
    out2 = F.conv2d(input_pad_circ2, weight, padding=(1, 0))
    print(
        f"Shape of input: {input.shape}, input_pad_circ3: {input_pad_circ3.shape}, input_pad_circ2: {input_pad_circ2.shape}"
    )
    print(f"Shape of out1: {out1.shape}, out2: {out2.shape}")
    # print(f"out1: {out1}")
    # print(f"out2: {out2}")
    print(f"out1 == out2: {torch.allclose(out1, out2)}")
    out3 = pad_before_conv2d_circular_width_only(input, weight, padding=1, torch_func=F.conv2d)
    # print(f"out3: {out3}")
    print(f"out1 == out3: {torch.allclose(out1, out3)}")
    # Check if Conv2D with adjusted padding is the same
    conv = ConvWithCircularPaddingWidthOnly(1, 1, 3, 1, conv_class=torch.nn.Conv2d, bias=False)
    conv.conv.weight.data = weight
    out4 = conv(input)
    print(f"out1 == out4: {torch.allclose(out1, out4)}")
