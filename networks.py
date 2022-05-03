"""Implements a custom U-Net.

Reference:
  Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for
  Biomedical Image Segmentation, MICCAI 2015.
  https://doi.org/10.1007/978-3-319-24574-4_28
"""

import torch
import torch.nn as nn
import warnings
from typing import Union, Tuple, Optional
from torch.nn.modules.batchnorm import _NormBase

_ACT_SHORTCUT = dict(
    LeakyReLU=dict(_type="LeakyReLU", negative_slope=0.2, inplace=True),
    ReLU=dict(_type="ReLU", inplace=True),
)


_NORM_SHORTCUT = dict(
    IN=dict(_type="InstanceNorm2d", affine=True),
    BN=dict(_type="BatchNorm2d", affine=True),
)


def _repeat_to_tuple(x: Union[Tuple, int]) -> Tuple:
    if isinstance(x, int):
        return x, x
    return x


def activation(conf: Union[dict, str, None]):
    if conf is None:
        return None
    if isinstance(conf, str):
        conf = _ACT_SHORTCUT.get(conf) or dict(_type=conf)
    conf = conf.copy()
    act_class = getattr(nn, conf.pop("_type"))
    return act_class(**conf)


def normalization(num_features, conf: Union[dict, str, None]):
    if conf is None:
        return None
    if isinstance(conf, str):
        conf = _NORM_SHORTCUT.get(conf) or dict(_type=conf)
    conf = conf.copy()
    assert "_type" in conf, f"must specify _type in normalization"
    norm_layer = getattr(nn, conf.pop("_type"))
    return norm_layer(num_features, **conf)


def convolution_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Tuple, int],
    stride: Union[Tuple, int] = 1,
    padding: Union[Tuple, int, str] = 0,
    dilation: Union[Tuple, int] = 1,
    groups=1,
    bias: Union[bool, str] = "auto",
    padding_mode="zeros",
    conv_conf: Optional[Union[dict, str]] = None,
    norm_conf: Optional[Union[dict, str]] = None,
    act_conf: Optional[Union[dict, str]] = "ReLU",
    order: str = "CNA",
):
    assert isinstance(order, str) and set(order) == set("CNA")
    # Convert ambiguity parameters to tuple.
    #
    # The parameters kernel_size, stride, padding, dilation can either be:
    # a single int – in which case the same value
    # is used for the height and width dimension
    # a tuple of two ints – in which case,
    # the first int is used for the height dimension,
    # and the second int for the width dimension
    conv_kwargs = dict(
        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    for k, v in conv_kwargs.items():
        conv_kwargs[k] = _repeat_to_tuple(v)

    # If specified as `auto`, it will be decided by the normalization.
    # bias will be set as True if `normalization` is None, otherwise False.
    assert bias in [True, False, "auto"]
    # if the conv layer is before a norm layer, bias is unnecessary.
    bias = bias if isinstance(bias, bool) else (norm_conf is None)

    conv_kwargs.update(
        dict(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
    )

    conv_conf = conv_conf or dict()
    conv_conf.update(conv_kwargs)

    conv = nn.Conv2d(**conv_conf)

    channels = out_channels if order.index("C") < order.index("N") else in_channels
    norm = normalization(channels, norm_conf)
    # `_NormBase` is Common base of _InstanceNorm and _BatchNorm
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html
    if isinstance(norm, _NormBase) and bias:
        warnings.warn("Unnecessary conv bias before batch/instance norm")

    act = activation(act_conf)

    layers = []
    for t in order:
        if t == "C":
            layers.append(conv)
        if t == "N" and norm is not None:
            layers.append(norm)
        if t == "A" and act is not None:
            layers.append(act)
    return layers


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        act_conf="ReLU",
        norm_conf=None,
    ):
        super(DownBlock, self).__init__()
        num_convs = 2

        convs = []
        prev_channels = in_channels
        channels = out_channels
        for _ in range(num_convs):
            convs.extend(
                convolution_layer(
                    prev_channels,
                    channels,
                    kernel_size=3,
                    padding="same",
                    act_conf=act_conf,
                    norm_conf=norm_conf,
                )
            )
            prev_channels = channels
        self.convs = nn.Sequential(*convs)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.convs(x)
        return skip, self.down(skip)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        act_conf="ReLU",
        norm_conf=None,
    ):
        super(UpBlock, self).__init__()
        padding = "same"
        num_convs = 2

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv = nn.Sequential(
            *convolution_layer(
                in_channels,
                out_channels,
                kernel_size=(2, 2),
                padding=padding,
                act_conf=act_conf,
                norm_conf=norm_conf,
            )
        )
        convs = []
        prev_channels = in_channels
        for _ in range(num_convs):
            channels = out_channels
            convs.extend(
                convolution_layer(
                    prev_channels,
                    channels,
                    kernel_size=(3, 3),
                    padding=padding,
                    act_conf=act_conf,
                    norm_conf=norm_conf,
                )
            )
            prev_channels = channels
        self.convs = nn.Sequential(*convs)

    def forward(self, x, skip):
        x = self.up_conv(self.up(x))
        return self.convs(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scales=4,
        bottleneck_channels=1024,
        bottleneck_layers=2,
        act_conf="ReLU",
        norm_conf=None,
    ):
        super().__init__()

        channels = [bottleneck_channels // 2 ** i for i in range(scales, 0, -1)]
        self.down_blocks = nn.ModuleList()
        prev_channel = in_channels
        for channel in channels:
            self.down_blocks.append(
                DownBlock(prev_channel, channel, act_conf, norm_conf)
            )
            prev_channel = channel

        bottlenecks = []
        for _ in range(bottleneck_layers):
            bottlenecks.extend(
                convolution_layer(
                    prev_channel,
                    bottleneck_channels,
                    kernel_size=3,
                    padding="same",
                    act_conf=act_conf,
                    norm_conf=norm_conf,
                )
            )
            prev_channel = bottleneck_channels
        self.bottlenecks = nn.Sequential(*bottlenecks)

        self.up_blocks = nn.ModuleList()
        for channel in reversed(channels):
            self.up_blocks.append(UpBlock(2 * channel, channel, act_conf, norm_conf))

        self.output_layer = nn.Sequential(
            *convolution_layer(
                channels[0],
                out_channels,
                kernel_size=1,
                norm_conf=None,
                act_conf="Sigmoid",
            ),
        )

    def forward(self, x):
        skips = []
        for block in self.down_blocks:
            skip, x = block(x)
            skips.append(skip)
        x = self.bottlenecks(x)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)
        return self.output_layer(x)


def _test():
    from torchinfo import summary

    unet = UNet(3, 3)
    print(unet)
    summary(unet, input_size=(1, 3, 512, 512))
    x = torch.randn(1, 3, 512, 512)
    x = (x - x.min()) / (x.max() - x.min())
    y = unet(x)
    print(x.min(), x.max(), x.mean())
    print(y.min(), y.max(), y.mean())


if __name__ == "__main__":
    _test()
