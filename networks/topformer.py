import torch
from torch import nn
import torch.nn.functional as F
from networks.base import convolution_layer, normalization, activation
import math


def _make_divisible(v, divisor, min_value=None):
    min_value = divisor if min_value is None else min_value
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        act_conf="ReLU",
        norm_conf="BN",
    ):
        super(InvertedResidual, self).__init__()

        hidden_channels = int(round(in_channels * expand_ratio))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.with_residual = stride == 1 and out_channels == in_channels

        layers = []
        if expand_ratio != 1:
            layers.extend(
                convolution_layer(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_conf=norm_conf,
                    act_conf=act_conf,
                )
            )
        layers.extend(
            convolution_layer(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_channels,
                norm_conf=norm_conf,
                act_conf=act_conf,
            )
            + convolution_layer(
                hidden_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_conf=norm_conf,
                act_conf=None,
            )
        )
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        if self.with_residual:
            return self.main(x) + x
        return self.main(x)


class TokenPyramidModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_indices,
        body_configs,
        channel_multiplier=1,
        act_conf="ReLU",
        norm_conf="BN",
    ):
        super(TokenPyramidModule, self).__init__()
        self.out_indices = out_indices
        self.stem = nn.Sequential(
            *convolution_layer(
                in_channels,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_conf="BN",
                act_conf="ReLU",
            )
        )
        self.layers = nn.ModuleList()

        in_channels = 16
        for i, (kernel_size, expand_ratio, channel, stride) in enumerate(body_configs):
            out_channels = _make_divisible(channel * channel_multiplier, 8)
            self.layers.append(
                InvertedResidual(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm_conf=norm_conf,
                    act_conf=act_conf,
                )
            )
            in_channels = out_channels

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super(PyramidPoolAgg, self).__init__()
        self.stride = stride

    def forward(self, inputs):
        batch, channel, height, width = inputs[-1].shape
        height = (height - 1) // self.stride + 1
        width = (width - 1) // self.stride + 1
        return torch.cat(
            [F.adaptive_max_pool2d(x, (height, width)) for x in inputs], dim=1
        )


def conv1x1(in_channel, out_channel, norm_conf, act_conf=None, order="CNA"):
    return nn.Sequential(
        *convolution_layer(
            in_channel,
            out_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_conf=norm_conf,
            act_conf=act_conf,
            order=order,
        )
    )


class Attention(nn.Module):
    def __init__(
        self,
        channels,
        qk_channels,
        num_heads,
        attention_ratio=4,
        act_conf=None,
        norm_conf="BN",
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = qk_channels ** -0.5
        self.qk_channels = qk_channels
        self.total_qk_channels = qk_channels * num_heads

        self.v_channels = int(attention_ratio * qk_channels)
        self.total_v_channels = self.v_channels * num_heads

        self.to_q = conv1x1(channels, self.total_qk_channels, norm_conf)
        self.to_k = conv1x1(channels, self.total_qk_channels, norm_conf)
        self.to_v = conv1x1(channels, self.total_v_channels, norm_conf)
        self.proj = conv1x1(
            self.total_v_channels, channels, norm_conf, act_conf=act_conf, order="ACN"
        )

    def forward(self, x):
        b, c, h, w = x.shape

        q = (
            self.to_q(x)
            .reshape(b, self.num_heads, self.qk_channels, -1)
            .permute(0, 1, 3, 2)
        )  # B num_heads h*w key_channel
        k = self.to_k(x).reshape(b, self.num_heads, self.qk_channels, -1)
        v = (
            self.to_v(x)
            .reshape(b, self.num_heads, self.v_channels, -1)
            .permute(0, 1, 3, 2)
        )

        attn = torch.matmul(q, k).softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, self.total_v_channels, h, w)
        out = self.proj(out)
        return out


class DropPath(nn.Module):
    def __init__(self, p):
        super(DropPath, self).__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = 1 - self.p + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        out = x.div(1 - self.p) * random_tensor
        return out


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        drop=0.0,
        act_conf="ReLU",
        norm_conf="BN",
    ):
        super(MLP, self).__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = conv1x1(in_channels, hidden_channels, norm_conf=norm_conf)
        self.dwconv = nn.Sequential(
            *convolution_layer(
                hidden_channels,
                hidden_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=True,
                groups=hidden_channels,
                act_conf=act_conf,
            )
        )
        self.fc2 = conv1x1(hidden_channels, out_channels, norm_conf=norm_conf)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        channels,
        qk_channels,
        num_heads,
        attention_ratio=2,
        mlp_ratio=4,
        act_conf="ReLU",
        norm_conf="BN",
        drop_path=0.0,
        mlp_drop=0,
    ):
        super(Transformer, self).__init__()

        self.mha = Attention(
            channels, qk_channels, num_heads, attention_ratio, act_conf, norm_conf
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        mlp_channels = int(channels * mlp_ratio)
        self.mlp = MLP(
            channels,
            mlp_channels,
            act_conf=act_conf,
            drop=mlp_drop,
            norm_conf=norm_conf,
        )

    def forward(self, x):
        x = x + self.drop_path(self.mha(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum(nn.Module):
    def __init__(self, in_channels, out_channels, norm_conf="BN"):
        super(InjectionMultiSum, self).__init__()

        self.local_embedding = conv1x1(in_channels, out_channels, norm_conf=norm_conf)
        self.global_embedding = conv1x1(in_channels, out_channels, norm_conf=norm_conf)
        self.global_act = conv1x1(in_channels, out_channels, norm_conf=norm_conf)
        self.act = HSigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        b, c, h, w = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(
            self.act(global_act), size=(h, w), mode="bilinear", align_corners=False
        )

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(
            global_feat, size=(h, w), mode="bilinear", align_corners=False
        )

        out = local_feat * sig_act + global_feat
        return out


class TopFormer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_transformer=4,
        qk_channels=16,
        num_heads=8,
        attention_ratio=2,
        mlp_ratio=2,
        drop_path_rate=0.0,
        trans_act_conf="ReLU6",
        tpm_act_conf="ReLU",
        norm_conf="BN",
        out_act_conf=None,
    ):
        super(TopFormer, self).__init__()

        cfg = [
            # kernel_size, expand_ratio, channel, stride
            [3, 1, 16, 1],  # 1/2        0.464K  17.461M
            [3, 4, 32, 2],  # 1/4 1      3.44K   64.878M
            [3, 3, 32, 1],  # 4.44K   41.772M
            [5, 3, 64, 2],  # 1/8 3      6.776K  29.146M
            [5, 3, 64, 1],  # 13.16K  30.952M
            [3, 3, 128, 2],  # 1/16 5     16.12K  18.369M
            [3, 3, 128, 1],  # 41.68K  24.508M
            [5, 6, 160, 2],  # 1/32 7     0.129M  36.385M
            [5, 6, 160, 1],  # 0.335M  49.298M
            [3, 6, 160, 1],  # 0.335M  49.298M
        ]

        self.channels = [16, 32, 64, 128, 160]
        self.embedding_channels = sum(self.channels)
        drop_path = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_transformer)
        ]

        self.tpm = TokenPyramidModule(
            in_channels,
            [0, 2, 4, 6, 9],
            cfg,
            norm_conf=norm_conf,
            act_conf=tpm_act_conf,
        )
        self.ppa = PyramidPoolAgg(stride=2)
        self.sase = nn.Sequential(
            *[
                Transformer(
                    self.embedding_channels,
                    qk_channels=qk_channels,
                    num_heads=num_heads,
                    attention_ratio=attention_ratio,
                    mlp_ratio=mlp_ratio,
                    act_conf=trans_act_conf,
                    norm_conf=norm_conf,
                    drop_path=drop_path[i],
                    mlp_drop=0,
                )
                for i in range(num_transformer)
            ]
        )

        self.sim = nn.ModuleList()
        self.decode_out_indices = [1, 2, 3]
        self.sim_out_channels = [None, 256, 256, 256]
        for i in range(len(self.channels)):
            if i in self.decode_out_indices:
                self.sim.append(
                    InjectionMultiSum(
                        self.channels[i], self.sim_out_channels[i], norm_conf=norm_conf
                    )
                )
            else:
                self.sim.append(nn.Identity())

        # self.up_conv = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        #     *convolution_layer(
        #         self.sim_out_channels[0],
        #         64,
        #         kernel_size=3,
        #         padding=1,
        #         act_conf=tpm_act_conf,
        #         norm_conf=norm_conf,
        #     )
        # )
        self.output_layer = nn.Sequential(
            *convolution_layer(
                self.sim_out_channels[-1],
                out_channels,
                kernel_size=1,
                norm_conf=None,
                act_conf=out_act_conf,
            ),
        )
        self.init()

    def init(self):
        def apply_fn(m):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.apply(apply_fn)

    def forward(self, x):
        local_tokens = self.tpm(x)
        out = self.ppa(local_tokens)
        out = self.sase(out)
        global_tokens = out.split(self.channels, dim=1)
        outs = []
        for i in range(len(self.channels)):
            if i in self.decode_out_indices:
                outs.append(self.sim[i](local_tokens[i], global_tokens[i]))

        out = outs[0]
        for o in outs[1:]:
            out += F.interpolate(o, size=out.size()[2:], mode="bilinear", align_corners=False)
        out = self.output_layer(out)
        return out


def _test():
    torch.set_grad_enabled(False)

    tf = TopFormer(out_channels=1, out_act_conf=None)
    print(tf)

    x = torch.randn([1, 3, 256, 256])
    out = tf(x)
    print(out.size())

    from torchinfo import summary

    summary(tf, input_size=(1, 3, 256, 256))


if __name__ == "__main__":
    _test()
