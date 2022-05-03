from collections import OrderedDict
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = "alex", version: str = "0.1"):
    # build url
    url = (
        "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
        + f"master/lpips/weights/v{version}/{net_type}.pth"
    )

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url,
        progress=True,
        map_location=torch.device("cpu"),
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace("lin", "")
        new_key = new_key.replace("model.", "")
        new_state_dict[new_key] = val

    return new_state_dict


def get_network(net_type: str):
    if net_type == "alex":
        return AlexNet()
    elif net_type == "squeeze":
        return SqueezeNet()
    elif net_type == "vgg":
        return VGG16()
    else:
        raise NotImplementedError("choose net_type from [alex, squeeze, vgg].")


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__(
            [
                nn.Sequential(
                    nn.Identity(), nn.Conv2d(nc, 1, (1, 1), (1, 1), 0, bias=False)
                )
                for nc in n_channels_list
            ]
        )

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            "mean", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor, value_range=None):
        # image x should be RGB, IMPORTANT: normalized to [-1,1]
        if value_range is not None and value_range != (-1, 1):
            low, high = value_range
            x = (x - (high + low) / 2) * (2 / (high - low))

        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(True).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)


class LPIPSLoss(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """

    def __init__(self, net_type: str = "alex", version: str = "0.1"):
        assert version in ["0.1"], "v0.1 is only supported now"

        super(LPIPSLoss, self).__init__()

        self.net_type = net_type
        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def __repr__(self):
        return f"{self.__class__.__name__}(net_type={self.net_type})"

    def forward(self, x: torch.Tensor, y: torch.Tensor, value_range=None):
        """

        :param x:  image x should be RGB, IMPORTANT: normalized to [-1,1]
        :param y:  image y should be RGB, IMPORTANT: normalized to [-1,1]
        :param value_range: (low, high), default is None, which means
            input images have been normalized before call this function
        :return:
        """
        feat_x, feat_y = self.net(x, value_range), self.net(y, value_range)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / x.shape[0]
