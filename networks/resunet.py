import torch
import torch.nn as nn

from networks.base import convolution_layer


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        padding,
        norm_conf="IN",
        act_conf="ReLU",
    ):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            *convolution_layer(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                stride=stride,
                norm_conf=norm_conf,
                act_conf=act_conf,
                order="NAC",
            ),
            *convolution_layer(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_conf=norm_conf,
                act_conf=act_conf,
                order="NAC",
            ),
        )
        self.skip = nn.Sequential(
            *convolution_layer(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                norm_conf=norm_conf,
                act_conf=None,
            )
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filters=(64, 128, 256, 512),
        norm_conf="IN",
        act_conf="ReLU",
    ):
        super(ResUNet, self).__init__()

        self.input = nn.Sequential(
            *convolution_layer(
                in_channels,
                filters[0],
                kernel_size=3,
                padding=1,
                norm_conf=norm_conf,
                act_conf=act_conf,
            ),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    filters[i],
                    filters[i + 1],
                    stride=2,
                    padding=1,
                    norm_conf=norm_conf,
                    act_conf=act_conf,
                )
                for i in range(3)
            ]
        )
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(filters[i], filters[i], kernel_size=2, stride=2)
                for i in [3, 2, 1]
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    filters[i] + filters[i - 1],
                    filters[i - 1],
                    stride=1,
                    padding=1,
                    norm_conf=norm_conf,
                    act_conf=act_conf,
                )
                for i in [3, 2, 1]
            ]
        )

        self.output = nn.Sequential(
            *convolution_layer(
                filters[0],
                out_channels,
                kernel_size=1,
                norm_conf=None,
                act_conf="Sigmoid",
            ),
        )

    def forward(self, x):
        x1 = self.input(x) + self.input_skip(x)
        x2 = self.down_blocks[0](x1)
        x3 = self.down_blocks[1](x2)
        out = self.up[0](self.down_blocks[2](x3))
        out = self.up_blocks[0](torch.cat([out, x3], dim=1))
        out = self.up[1](out)
        out = self.up_blocks[1](torch.cat([out, x2], dim=1))
        out = self.up[2](out)
        out = self.up_blocks[2](torch.cat([out, x1], dim=1))
        return self.output(out)


def _test():
    torch.set_grad_enabled(False)
    from torchinfo import summary

    unet = ResUNet(3, 3)
    print(unet)
    summary(unet, input_size=(1, 3, 512, 512))
    x = torch.randn(1, 3, 512, 512)
    x = (x - x.min()) / (x.max() - x.min())
    y = unet(x)
    print(x.size(), x.min(), x.max(), x.mean())
    print(y.size(), y.min(), y.max(), y.mean())


if __name__ == "__main__":
    _test()
