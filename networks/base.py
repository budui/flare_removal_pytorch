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
