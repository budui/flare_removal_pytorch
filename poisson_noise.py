import argparse
import functools
import warnings
from pathlib import Path
from typing import Sequence, Union, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.nn.modules.batchnorm import _NormBase
from torchvision import transforms


@functools.lru_cache(maxsize=300)
def load_data(filepath):
    return torch.from_numpy(np.load(filepath))


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


class FourierUnit(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            groups=1,
            norm_conf="BN",
            act_conf="ReLU",
            fft_norm="ortho",
    ):
        super(FourierUnit, self).__init__()

        self.fft_norm = fft_norm

        self.spectrum_conv = nn.Sequential(
            *convolution_layer(
                in_channels=2 * in_channels,
                out_channels=2 * out_channels,
                kernel_size=1,
                groups=groups,
                norm_conf=norm_conf,
                act_conf=act_conf,
            )
        )

    def forward(self, x):
        _, c, h, w = x.shape

        x = torch.fft.rfftn(x, dim=(-2, -1), norm=self.fft_norm)
        x = torch.cat([x.real, x.imag], dim=1)
        x = self.spectrum_conv(x)
        x = torch.complex(*x.split(c, dim=1))
        x = torch.fft.irfftn(x, s=(h, w), dim=(-2, -1), norm=self.fft_norm)
        return x


class SpectralTransform(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            groups=1,
            norm_conf="BN",
            act_conf="ReLU",
            enable_lfu=False,
            fft_norm="ortho",
    ):
        super(SpectralTransform, self).__init__()

        assert out_channels % 2 == 0
        assert stride in [1, 2]

        self.enable_lfu = enable_lfu
        self.stride = stride

        if self.stride == 2:
            self.down = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.conv1 = nn.Sequential(
            *convolution_layer(
                in_channels,
                out_channels // 2,
                kernel_size=1,
                groups=groups,
                norm_conf=norm_conf,
                act_conf=act_conf,
            )
        )
        fu_kwargs = dict(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            groups=groups,
            norm_conf=norm_conf,
            act_conf=act_conf,
            fft_norm=fft_norm,
        )
        self.fu = FourierUnit(**fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(**fu_kwargs)
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=(1, 1),
            groups=groups,
            bias=False,
        )

    def forward(self, x):
        if self.stride == 2:
            x = self.down(x)
        x = self.conv1(x)
        y = self.fu(x)
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            ly = torch.cat(
                torch.split(x[:, : c // 4], split_s, dim=-2), dim=1
            ).contiguous()
            ly = torch.cat(torch.split(ly, split_s, dim=-1), dim=1).contiguous()
            ly = self.lfu(ly)
            ly = ly.repeat(1, 1, split_no, split_no).contiguous()
            return self.conv2(x + y + ly)
        return self.conv2(x + y)


class Zero(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Zero, self).__init__()

    def forward(self, x: torch.Tensor):
        return x.new_zeros(1)


class FastFourierConvolution(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            alpha_in: float,
            alpha_out: float,
            kernel_size: Union[Tuple, int],
            stride: Union[Tuple, int] = 1,
            padding: Union[Tuple, int, str] = 0,
            dilation: Union[Tuple, int] = 1,
            groups=1,
            bias: Union[bool, str] = False,
            padding_mode="zeros",
            norm_conf: Optional[Union[dict, str]] = "BN",
            act_conf: Optional[Union[dict, str]] = "ReLU",
            enable_lfu: bool = False,
            fft_norm: str = "ortho",
    ):
        super(FastFourierConvolution, self).__init__()
        if isinstance(stride, Sequence):
            assert stride[0] == stride[1], "only support same stride tuple/list"
            stride = stride[0]
        assert stride in [1, 2], (
            f"{self.__class__.__name__} current only support stride=1 or 2 "
            f"but got {stride}"
        )
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1

        self.in_channels_g = int(in_channels * alpha_in)
        self.in_channels_l = in_channels - self.in_channels_g

        self.out_channels_g = int(out_channels * alpha_out)
        self.out_channels_l = out_channels - self.out_channels_g

        for name in ["l2l", "g2l", "l2g"]:
            t_in, t_out = name.split("2")
            c_in = getattr(self, f"in_channels_{t_in}")
            c_out = getattr(self, f"out_channels_{t_out}")
            if c_in > 0 and c_out > 0:
                self.add_module(
                    f"block_{name}",
                    nn.Conv2d(
                        getattr(self, f"in_channels_{t_in}"),
                        getattr(self, f"out_channels_{t_out}"),
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        padding_mode=padding_mode,
                        bias=bias,
                    ),
                )
            else:
                setattr(self, f"block_{name}", lambda _: 0.0)
        if self.in_channels_g > 0 and self.out_channels_g > 0:
            self.block_g2g = SpectralTransform(
                self.in_channels_g,
                self.out_channels_g,
                stride=stride,
                groups=groups,
                norm_conf=norm_conf,
                act_conf=act_conf,
                enable_lfu=enable_lfu,
                fft_norm=fft_norm,
            )
        else:
            self.block_g2g = lambda _: 0.0

    @staticmethod
    def zero(_: torch.Tensor):
        return 0

    def forward(self, x):
        x_l, x_g = x[:, : self.in_channels_l], x[:, self.in_channels_l:]
        out_l, out_g = None, None
        if self.out_channels_g > 0:
            out_g = self.block_g2g(x_g) + self.block_l2g(x_l)
        if self.out_channels_l > 0:
            out_l = self.block_g2l(x_g) + self.block_l2l(x_l)

        if out_g is None:
            out = out_l
        elif out_l is None:
            out = out_g
        else:
            out = torch.cat([out_l, out_g], dim=1)
        return out


class MLP(nn.Module):
    def __init__(self, in_channels, num_layers=4, base_channels=512):
        super(MLP, self).__init__()
        in_c = in_channels
        out_c = base_channels
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_c, out_c, bias=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_c = out_c

        layers.append(nn.Linear(base_channels, 1, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.main(x)


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, p=0.5, size=(1, 64, 64), length=10000):
        self.length = length
        self.size = size
        self.p = p

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.p < torch.rand(1):
            return (torch.rand(*self.size) - 0.5) * 3.4641016151377544, 1
        return torch.randn(*self.size), 0


class NPYDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path_a,
            path_b,
            path_real,
            transform,
            residual=True,
            train_ratio=0.7,
            is_train=True,
    ):
        super(NPYDataset, self).__init__()
        self.path_a = Path(path_a)
        self.path_b = Path(path_b)
        self.path_real = Path(path_real)
        self.residual = residual
        self.files = self.split_train(train_ratio, is_train)
        self.transform = transform

    def split_train(self, train_ratio, is_train):
        files = [f.name for f in self.path_a.glob("*.npy")]
        files = sorted(files, key=lambda x: x)
        train_items = files[: int(len(files) * train_ratio)]
        if is_train:
            return train_items
        train_items = set(train_items)
        return [f for f in files if f not in train_items]

    def load_image(self, name, is_a, base_image):
        root = self.path_a if is_a else self.path_b
        image = load_data((root / name).as_posix()) - base_image
        return self.transform(image)

    def __getitem__(self, idx):
        base_image = (
            load_data((self.path_real / self.files[idx // 2]).as_posix())
            if self.residual
            else 0
        )
        return self.load_image(self.files[idx // 2], idx % 2 == 0, base_image), idx % 2

    def __len__(self):
        return len(self.files) * 2

    def __repr__(self):
        return f"NPYDataset(residual={self.residual}, length={len(self)})"


def conv2d(**kwargs):
    t = "ffc_in1_out1"
    if t == "conv2d":
        return nn.Conv2d(**kwargs)
    elif t == "ffc_in1_out1":
        kwargs["alpha_in"] = 1.0
        kwargs["alpha_out"] = 1.0
        return FastFourierConvolution(**kwargs)
    elif t == "sn_conv2d":
        return torch.nn.utils.spectral_norm(nn.Conv2d(**kwargs))
    else:
        raise ValueError()


class NoNormDiscriminator(nn.Module):
    def __init__(
            self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gan_type="lsgan"
    ):
        super(NoNormDiscriminator, self).__init__()
        self.gan_type = gan_type
        kw = 3
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            conv2d(
                in_channels=input_nc,
                out_channels=ndf,
                kernel_size=kw,
                stride=1,
                padding=padw,
            ),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                conv2d(
                    in_channels=ndf * nf_mult_prev,
                    out_channels=ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=True,
                ),
                nn.LeakyReLU(0.2, True),
            ]

        sequence += [
            nn.Conv2d(
                in_channels=ndf * nf_mult,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=padw,
            )
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class PoissonNoise(nn.Module):
    def __init__(self, rate):
        super(PoissonNoise, self).__init__()
        self.rate = rate

    def forward(self, x):
        return torch.poisson(x / self.rate) * self.rate


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).view(data.size(0), -1).mean(dim=1, keepdim=True)
        loss = F.binary_cross_entropy_with_logits(
            output, target.float().unsqueeze(dim=1)
        )
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).view(data.size(0), -1).mean(dim=1, keepdim=True)
            test_loss += F.binary_cross_entropy_with_logits(
                output, target.float().unsqueeze(dim=1), reduction="sum"
            ).item()  # sum up batch loss

            pred = (output > 0).long()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 4, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [
            transforms.RandomCrop(size=(64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )
    # train_dataset = NPYDataset(
    #     "/data/Classifier_data/train_noisy_fake",
    #     "/data/Classifier_data/train_noisy_real",
    #     "/data/Classifier_data/train_clean",
    #     transform,
    #     is_train=True,
    # )
    # test_dataset = NPYDataset(
    #     "/data/Classifier_data/train_noisy_fake",
    #     "/data/Classifier_data/train_noisy_real",
    #     "/data/Classifier_data/train_clean",
    #     transform,
    #     is_train=False,
    # )
    train_dataset = NoiseDataset(length=10000, p=0.5)
    test_dataset = NoiseDataset(length=1000, p=0.4)
    print("train_dataset: ", train_dataset)
    print("test_dataset: ", test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = NoNormDiscriminator(4).to(device)
    # model = models.resnet18()
    # model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(512, 1, bias=True)
    # model = model.to(device)
    # model = MLP(1 * 64 * 64).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "result.pt")


if __name__ == "__main__":
    main()
