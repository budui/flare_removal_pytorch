"""Implements a custom U-Net.

Reference:
  Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for
  Biomedical Image Segmentation, MICCAI 2015.
  https://doi.org/10.1007/978-3-319-24574-4_28
"""

import torch
import torch.nn as nn

from networks.base import convolution_layer


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

        self.scales = scales
        self.bottleneck_channels = bottleneck_channels
        self.bottleneck_layers = bottleneck_layers

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

    def __str__(self):
        attrs = ["scales", "bottleneck_channels", "bottleneck_layers"]
        attr_str = "".join([f"\t{a}={getattr(self, a)}\n" for a in attrs])
        return f"{self.__class__.__name__}(\n{attr_str})"

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
    torch.set_grad_enabled(False)
    from torchinfo import summary

    unet = UNet(3, 3)
    print(unet)
    summary(unet, input_size=(1, 3, 512, 512))
    x = torch.randn(1, 3, 512, 512)
    x = (x - x.min()) / (x.max() - x.min())
    y = unet(x)
    print(x.size(), x.min(), x.max(), x.mean())
    print(y.size(), y.min(), y.max(), y.mean())


if __name__ == "__main__":
    _test()
