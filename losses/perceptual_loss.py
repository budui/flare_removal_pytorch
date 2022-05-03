from typing import Mapping, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models.vgg as vgg


class PerceptualVGG(nn.Module):
    def __init__(
        self, layer_names: Tuple[str], vgg_type="vgg19", enable_input_normalize=True
    ):
        super().__init__()

        self.enable_input_normalize = enable_input_normalize
        self.layer_names = layer_names
        self.vgg_type = vgg_type

        indices_of_layer = self.indices_of_layer(self.vgg_type)
        self.layers = {indices_of_layer[n]: n for n in self.layer_names}

        self.features = self._features(tuple(self.layers.keys()), self.vgg_type)

        if self.enable_input_normalize:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            # the std is for image with range [0, 1]
            self.register_buffer(
                "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

        self.requires_grad_(False)

    @staticmethod
    def indices_of_layer(vgg_type) -> Mapping[str, int]:
        cfg_type = dict(
            [("vgg11", "A"), ("vgg13", "B"), ("vgg16", "D"), ("vgg19", "E")]
        )
        assert vgg_type in cfg_type, (
            f"{vgg_type} is not a valid vgg type that"
            f" predefined in {list(cfg_type.keys())}"
        )

        bid, lid, pid = 1, 1, 1
        names = []
        # Please refer function `make_layers` in following link
        # https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html
        for v in vgg.cfgs[cfg_type[vgg_type]]:
            if v == "M":
                names.append(f"pool{pid}")
                bid, lid, pid = bid + 1, 1, pid + 1
            else:
                names.extend([f"conv{bid}_{lid}", f"relu{bid}_{lid}"])
                lid += 1
        return {n: i for i, n in enumerate(names)}

    @staticmethod
    def _features(layer_indices: Sequence[int], vgg_type="vgg19"):
        """Borrow layers from pretrained VGG networks

        :param vgg_type: ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
            'vgg16_bn','vgg19_bn', 'vgg19']
        :return:
        """
        _vgg = getattr(vgg, vgg_type)(pretrained=True)
        num_layers = max(layer_indices) + 1
        assert len(_vgg.features) >= num_layers
        # only borrow layers that will be used from _vgg to avoid unused params
        return _vgg.features[:num_layers]

    def forward(self, x, value_range=None):
        """
        Importantly, the input image must in the range [0, 1].
        User can specify `value_range` (low, high) to insure normalization
        """
        if value_range is not None:
            low, high = value_range
            x = (x - low) / (high - low)

        if self.enable_input_normalize:
            x = (x - self.mean) / self.std

        features = {}

        for i, l in enumerate(self.features):
            x = l(x)
            if i in self.layers:
                features[self.layers[i]] = x.clone()

        return features


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        layers: Union[Mapping[str, float], Sequence[str]],
        criterion="L1",
        vgg_type="vgg19",
        enable_input_normalize=True,
    ):
        super().__init__()

        self.layers = (
            layers if isinstance(layers, Mapping) else {l: 1.0 for l in layers}
        )
        self.vgg_type = vgg_type

        self.feature = PerceptualVGG(
            tuple(self.layers.keys()), vgg_type, enable_input_normalize
        )

        criterion = criterion.upper()
        assert criterion in [
            "L1",
            "MSE",
        ], f"{criterion} criterion has not been supported"
        if criterion == "L1":
            self.criterion = nn.L1Loss()
        elif criterion == "MSE":
            self.criterion = nn.MSELoss()

    def __repr__(self):
        attrs = ["layers", "criterion", "vgg_type"]
        s = ",\n".join(map(lambda a: f"{a}={getattr(self, a)}", attrs))
        return f"{self.__class__.__name__}(\n{s}\n)"

    def forward(self, image, ground_truth, value_range=None):
        features1 = self.feature(image, value_range)
        features2 = self.feature(ground_truth.detach(), value_range)

        loss = 0.0
        for k in features1.keys():
            loss += self.layers[k] * self.criterion(features1[k], features2[k])

        return loss
