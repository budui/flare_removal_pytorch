from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import convolution_layer, activation


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, act_conf="ReLU"):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            activation(act_conf),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.main(x)


def upconv(
    in_channels,
    out_channels,
    up=2,
    norm_conf=None,
    act_conf="ReLU",
    up_conf="PixelShuffle",
):
    outc = out_channels if up_conf != "PixelShuffle" else out_channels * (up ** 2)
    layers = convolution_layer(
        in_channels,
        outc,
        kernel_size=3,
        stride=1,
        padding=3 // 2,
        norm_conf=norm_conf,
        act_conf=act_conf,
        order="CNA",
    )
    if up_conf == "PixelShuffle":
        layers.insert(1, nn.PixelShuffle(up))
        return nn.Sequential(*layers)
    else:
        return nn.Sequential(nn.Upsample(scale_factor=up, mode="bilinear"), *layers)


def kernel2d_conv(feat_in, kernel, ksize):
    """

    :param feat_in: <N,C,H,W>
    :param kernel: <N,C*ks*ks,H,W>
    :param ksize: ks
    :return:

    If you have some problems in installing the CUDA FAC layer,
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad_sz = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad_sz, pad_sz, pad_sz, pad_sz), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)  # <N,H,W,C,ks*ks>

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, dim=-1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out


class DISCNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=32,
        cond_in_channels=0,
        cond_multi_scale=True,
        dynamic_kernel_size=5,
        norm_conf=None,
        act_conf="ReLU",
    ):
        super(DISCNet, self).__init__()

        self.cond_multi_scale = cond_multi_scale
        self.cond_in_channels = cond_in_channels
        self.dynamic_kernel_size = dynamic_kernel_size

        inc = in_channels
        outc = base_channels
        for i in range(3):
            stride = 1 if i == 0 else 2
            self.add_module(
                f"enc{i}", self.block(inc, outc, norm_conf, act_conf, stride)
            )
            inc = outc
            outc = 2 * outc

        self.dec1_up = upconv(
            4 * base_channels,
            2 * base_channels,
            up=2,
            act_conf=act_conf,
            norm_conf=norm_conf,
        )
        self.dec1 = nn.Sequential(
            ResidualBlock(2 * base_channels, act_conf),
            ResidualBlock(2 * base_channels, act_conf),
        )
        self.dec2_up = upconv(
            2 * base_channels,
            1 * base_channels,
            up=2,
            act_conf=act_conf,
            norm_conf=norm_conf,
        )
        self.dec2 = nn.Sequential(
            ResidualBlock(base_channels, act_conf),
            ResidualBlock(base_channels, act_conf),
        )

        self.out = nn.Sequential(
            *self.conv3x3(base_channels, out_channels, norm_conf, act_conf)
        )

        self.dk_ids = ()
        if self.cond_in_channels:
            inc = self.cond_in_channels
            outc = base_channels

            if self.cond_multi_scale:
                self.dk_ids = (0, 1, 2)
            else:
                self.dk_ids = (2,)

            for i in range(3):
                stride = 1 if i == 0 else 2
                self.add_module(
                    f"cond_enc{i}", self.block(inc, outc, norm_conf, act_conf, stride)
                )

                if i in self.dk_ids:
                    self.add_module(
                        f"dynamic_kernel{i}",
                        nn.Sequential(
                            *self.block(
                                outc, outc, norm_conf, act_conf, is_sequential=False
                            ),
                            *self.conv3x3(
                                outc,
                                outc * dynamic_kernel_size ** 2,
                                norm_conf,
                                act_conf,
                                kernel_size=1,
                            ),
                        ),
                    )

                inc = outc
                outc = outc * 2

    @staticmethod
    def conv3x3(in_channel, out_channel, norm_conf, act_conf, stride=1, kernel_size=3):
        return convolution_layer(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            norm_conf=norm_conf,
            act_conf=act_conf,
        )

    def block(self, inc, outc, norm_conf, act_conf, stride=1, is_sequential=True):
        layers = [
            *self.conv3x3(inc, outc, norm_conf, act_conf, stride),
            ResidualBlock(outc, act_conf),
            ResidualBlock(outc, act_conf),
        ]
        if is_sequential:
            return nn.Sequential(*layers)
        return layers

    def forward(self, x, cond=None, cond_is_x=False):
        cond = x if cond_is_x else cond

        cond_kernels = deque()
        if self.cond_in_channels:
            f = cond
            for i in range(3):
                f = getattr(self, f"cond_enc{i}")(f)
                if i in self.dk_ids:
                    cond_kernels.append(getattr(self, f"dynamic_kernel{i}")(f))

        outs = []
        y = x
        for i in range(3):
            y = getattr(self, f"enc{i}")(y)
            if i in self.dk_ids:
                outs.append(
                    kernel2d_conv(y, cond_kernels.popleft(), self.dynamic_kernel_size)
                )
            else:
                outs.append(y)
        # outs: [out0, out1, out2]

        out = self.dec1(self.dec1_up(outs[-1]) + outs[-2])
        out = self.dec2(self.dec2_up(out) + outs[-3])
        return self.out(out) + x


def _test():
    torch.set_grad_enabled(False)
    from torchinfo import summary

    disc = DISCNet(cond_in_channels=3)
    print(disc)
    summary(disc, input_size=[(1, 3, 512, 512), (1, 3, 512, 512)])
    # x = torch.randn(1, 3, 512, 512)
    # x = (x - x.min()) / (x.max() - x.min())
    # y = disc(x)


if __name__ == "__main__":
    _test()
