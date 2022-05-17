import torch

from torch import nn
import torch.nn.functional as F
from networks.base import convolution_layer


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_conf="ReLU", norm_conf=None):
        super(DenseBlock, self).__init__()
        self.main = nn.Sequential(
            *convolution_layer(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                norm_conf=norm_conf,
                act_conf=act_conf,
            )
        )

    def forward(self, x):
        return torch.cat([x, self.main(x)], dim=1)


class ResidualDenseBlock(nn.Module):
    def __init__(
        self, in_channels, base_channels, num_layers, act_conf="ReLU", norm_conf=None
    ):
        super(ResidualDenseBlock, self).__init__()

        self.main = nn.Sequential(
            *[
                DenseBlock(
                    in_channels + base_channels * i,
                    base_channels,
                    act_conf=act_conf,
                    norm_conf=norm_conf,
                )
                for i in range(num_layers)
            ]
        )

        # local feature fusion
        self.lff = nn.Conv2d(
            in_channels + base_channels * num_layers, base_channels, kernel_size=1
        )

    def forward(self, x):
        return self.lff(self.main(x)) + x


class ResidualDenseNetwork(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        rdb_channels=64,
        num_rdbs=3,
        num_dense_units=4,
        act_conf="ReLU",
        norm_conf=None,
    ):
        super(ResidualDenseNetwork, self).__init__()

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # residual dense blocks
        self.rdbs = nn.ModuleList()
        in_c = base_channels
        for _ in range(num_rdbs):
            self.rdbs.append(
                ResidualDenseBlock(
                    in_c,
                    rdb_channels,
                    num_dense_units,
                    act_conf=act_conf,
                    norm_conf=norm_conf,
                )
            )
            in_c = rdb_channels

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(rdb_channels * num_rdbs, base_channels, kernel_size=1),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        skip1 = self.sfe1(x)
        out = self.sfe2(skip1)

        local_features = []
        for rdb in self.rdbs:
            out = rdb(out)
            local_features.append(out)
        out = self.gff(torch.cat(local_features, dim=1)) + skip1
        out = self.out(out) + x
        return out


class RDNRestoration(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        scales=(1, 4, 8),
        base_channels=64,
        rdb_channels=64,
        num_rdbs=3,
        num_dense_units=2,
        act_conf="ReLU",
        norm_conf=None,
    ):

        super(RDNRestoration, self).__init__()

        self.scales = scales
        for s in self.scales:
            self.add_module(
                f"x{s}",
                ResidualDenseNetwork(
                    in_channels,
                    out_channels,
                    base_channels,
                    rdb_channels,
                    num_rdbs,
                    num_dense_units,
                    act_conf,
                    norm_conf,
                ),
            )

        self.out = nn.Conv2d(
            out_channels * len(scales), out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        outs = dict()
        for s in self.scales:
            if s == 1:
                x_in = x
            else:
                x_in = F.interpolate(
                    x, scale_factor=1 / s, mode="bilinear", align_corners=False
                )
            outs[s] = getattr(self, f"x{s}")(x_in)
        out = torch.cat(
            [F.interpolate(o, scale_factor=s) for s, o in outs.items()], dim=1
        )
        return self.out(out)


def _test():
    torch.set_grad_enabled(False)
    from torchinfo import summary

    unet = RDNRestoration(3, 3)
    print(unet)
    summary(unet, input_size=(1, 3, 512, 512))
    x = torch.randn(1, 3, 512, 512)
    x = (x - x.min()) / (x.max() - x.min())
    y = unet(x)
    print(x.size(), x.min(), x.max(), x.mean())
    print(y.size(), y.min(), y.max(), y.mean())


if __name__ == "__main__":
    _test()
