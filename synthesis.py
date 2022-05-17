import random
from typing import Union, Sequence

import kornia
import kornia.geometry.transform as KT
import numpy as np
import torch
import torch.nn.functional
import torchvision.transforms as T
import torchvision.transforms.functional
import torchvision.utils
import cv2
import skimage
import skimage.measure
import skimage.morphology

import utils

# Small number added to near-zero quantities to avoid numerical instability.
_EPS = 1e-7


def adjust_gamma(image: torch.Tensor, gamma: float, gain: float = 1.0) -> torch.Tensor:
    # Apply the gamma correction
    x_adjust: torch.Tensor = torch.pow(image, gamma).mul(gain)
    # Truncate between pixel values
    out: torch.Tensor = torch.clamp(x_adjust, 0.0, 1.0)
    return out


def remove_dc_component(image: torch.Tensor) -> torch.Tensor:
    """Removes the DC component in the background.

    :param image: Image tensor with shape [N, C, H, W], or [C, H, W].
    :return: Image(s) with DC background removed. The white level
     (maximum pixel value) stays the same.
    """

    image_min = image.amin([-1, -2], keepdim=True)
    image_max = image.amax([-1, -2], keepdim=True)
    return (image - image_min) * image_max / (image_max - image_min + _EPS)


def normalize_white_balance(image):
    """Normalizes the RGB channels so the image appears neutral.

    :param image: Image tensor with shape [C, H, W], or [B, C, H, W].
    :return: Image(s) with equal channel mean. (The channel mean may be
     different across images for batched input.)
    """

    channel_mean = image.mean([-1, -2], keepdim=True)
    max_of_mean = channel_mean.amax([-1, -2, -3], keepdim=True)

    normalized = max_of_mean * image / (channel_mean + _EPS)
    return normalized


def quantize_8(image):
    """
    Converts and quantizes an image to 2^8 discrete levels in [0, 1].
    :param image:
    :return:
    """
    return (image * 255).to(torch.uint8).float() * (1.0 / 255.0)


class RandomHorizontalFlip:
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        flipped = torch.rand(tensor.size(0)) < self.p
        tensor[flipped] = torch.flip(tensor[flipped], [3])
        return tensor


class RandomVerticalFlip:
    """Applies the :class:`~torchvision.transforms.RandomVerticalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        flipped = torch.rand(tensor.size(0)) < self.p
        tensor[flipped] = torch.flip(tensor[flipped], [2])
        return tensor


class RandomCrop:
    """Applies the :class:`~torchvision.transforms.RandomCrop` transform to a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image.
            Default is None, i.e no padding.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, padding=None, device="cpu"):
        self.size = size
        self.padding = padding
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros(
                (
                    tensor.size(0),
                    tensor.size(1),
                    tensor.size(2) + self.padding * 2,
                    tensor.size(3) + self.padding * 2,
                ),
                dtype=tensor.dtype,
                device=self.device,
            )
            padded[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ] = tensor
        else:
            padded = tensor

        h, w = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (tensor.size(0),), device=self.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),), device=self.device)

        rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[
            :,
            torch.arange(tensor.size(0))[:, None, None],
            rows[:, torch.arange(th)[:, None]],
            columns[:, None],
        ]
        return padded.permute(1, 0, 2, 3)


def random_resize(image, min_resolution, low, high):
    h, w = image.shape[-2:]
    factor = min(h / min_resolution, w / min_resolution)
    h, w = h / factor, w / factor
    size_scale = np.random.uniform(low, high, 1).item()
    image = torch.nn.functional.interpolate(
        image,
        (int(h * size_scale), int(w * size_scale)),
        mode="bilinear",
        align_corners=False,
    )
    if image.shape[-1] <= min_resolution or image.shape[-2] <= min_resolution:
        px = max(min_resolution - image.shape[-1] + 1, 0)
        py = max(min_resolution - image.shape[-2] + 1, 0)
        ptop = torch.randint(py, [1]).item() if py > 0 else 0
        pleft = torch.randint(px, [1]).item() if px > 0 else 0

        image = torch.nn.functional.pad(
            image,
            pad=[pleft, px - pleft, ptop, py - ptop],
            mode="replicate",
        )
    return image


def uniform_tensor(size, a, b, dtype=None, device=None, requires_grad=False):
    u = torch.rand(*size, dtype=dtype, device=device, requires_grad=requires_grad)
    return (b - a) * u + a


def add_flare_paper(
    scene: torch.Tensor,
    flare: torch.Tensor,
    apply_affine: bool = True,
    resolution: Union[Sequence, int] = 512,
    flare_max_gain: float = 10.0,
    noise_strength: float = 0.01,
):
    batch_size = scene.shape[0]
    device = scene.device
    resolution = (resolution, resolution) if isinstance(resolution, int) else resolution

    gamma = random.uniform(1.8, 2.2)
    flare_linear = adjust_gamma(flare, gamma)
    flare_linear = remove_dc_component(flare_linear)

    if apply_affine:
        rotation = uniform_tensor([batch_size], -180, 180, device=device)
        shift = torch.randn(batch_size, 2, device=device).mul_(10)
        shear = uniform_tensor([batch_size, 2], -np.pi / 9, np.pi / 9, device=device)
        scale = uniform_tensor([batch_size, 2], 0.9, 1.2, device=device)

        padding_mode = "zeros"
        flare_linear = KT.rotate(flare_linear, rotation, padding_mode=padding_mode)
        flare_linear = KT.shear(flare_linear, shear, padding_mode=padding_mode)
        flare_linear = KT.scale(flare_linear, scale, padding_mode=padding_mode)
        flare_linear = KT.translate(flare_linear, shift, padding_mode=padding_mode)

    flare_linear.clamp_(min=0.0, max=1.0)
    flare_linear = T.Compose(
        [
            T.CenterCrop(resolution),
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
        ]
    )(flare_linear)

    # First normalize the white balance. Then apply random white balance.
    flare_linear = normalize_white_balance(flare_linear)
    rgb_gains = uniform_tensor([flare_linear.size(1)], 0, flare_max_gain, device=device)
    rgb_gains = rgb_gains.view(1, flare_linear.size(1), 1, 1)
    flare_linear *= rgb_gains

    # Further augmentation on flare patterns: random blur and DC offset.
    flare_linear = T.functional.gaussian_blur(
        flare_linear, kernel_size=[21, 21], sigma=uniform_tensor([1], 0.1, 3).item()
    )
    flare_linear = flare_linear + uniform_tensor(
        [batch_size, 1, 1, 1], -0.02, 0.02, device=device
    )
    flare_linear = flare_linear.clamp_(min=0.0, max=1.0)

    flare_srgb = adjust_gamma(flare_linear, 1 / gamma)

    scene_linear = adjust_gamma(scene, gamma)
    scene_linear = T.Compose(
        [
            T.RandomCrop(resolution),
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
        ]
    )(scene_linear)

    # Additive Gaussian noise. The Gaussian's variance is drawn from a
    # Chi-squared distribution. This is equivalent to drawing the Gaussian's
    # standard deviation from a truncated normal distribution, as shown below.
    noise = torch.randn_like(scene_linear) * torch.abs(
        torch.randn(1, device=scene_linear.device) * noise_strength
    )
    scene_linear += noise

    # Random digital gain.
    # varying the intensity scale
    scene_linear = (scene_linear * random.uniform(0, 1.2)).clamp(0, 1)

    scene_srgb = adjust_gamma(scene_linear, 1 / gamma)

    # Combine the flare-free scene with a flare pattern to produce a synthetic
    # running example.F
    combined_linear = scene_linear + flare_linear
    combined_srgb = adjust_gamma(combined_linear, 1.0 / gamma)
    combined_srgb.clamp_(min=0.0, max=1.0)

    return (
        quantize_8(scene_srgb),
        quantize_8(flare_srgb),
        quantize_8(combined_srgb),
        gamma,
    )


def add_flare(
    scene: torch.Tensor,
    flare: torch.Tensor,
    resize_scale=(0.5, 1.5),
    apply_affine: bool = True,
    apply_random_white_balance: bool = False,
    resolution: Union[Sequence, int] = 512,
    flare_max_gain: float = 2.0,
    noise_strength: float = 0.01,
):
    """Adds flare to natural images.

    Here the natural images are in sRGB. They are first linearized before flare
    patterns are added. The result is then converted back to sRGB.

    :param scene:  Natural image batch in sRGB.
    :param flare: Lens flare image batch in sRGB.
    :param resize_scale: flare image size scale over resolution before crop.
    :param apply_affine: Whether to apply affine transformation.
    :param apply_random_white_balance: Whether to apply white balance.
    :param resolution: Resolution of training images.
    :param flare_max_gain: Maximum gain applied to the flare images in the
     linear domain. RGB gains are applied randomly and independently, not
     exceeding this maximum.
    :param noise_strength: Strength of the additive Gaussian noise. For
      each image, the Gaussian variance is drawn from a scaled Chi-squared
      distribution, where the scale is defined by `noise`.
    :return:
    """
    flare_resolution = resolution
    flare = random_resize(flare, flare_resolution, *resize_scale)

    # Since the gamma encoding is unknown, we use a random value so that
    # the models will hopefully generalize to a reasonable range of gammas.
    gamma = np.random.uniform(1.8, 2.2)
    flare_linear = adjust_gamma(flare, gamma)
    flare_linear = remove_dc_component(flare_linear)

    batch_size = flare_linear.shape[0]
    device = flare_linear.device

    if apply_affine:
        rotation = torch.empty(batch_size, device=device).uniform_(-180, 180)
        shear = torch.empty(batch_size, 2, device=device).uniform_(
            -np.pi / 9, np.pi / 9
        )
        scale = torch.empty(batch_size, 2, device=device).uniform_(0.9, 1.2)
        shift = torch.randn(batch_size, 2, device=device).mul_(10)
        padding_mode = "reflection"
        flare_linear = KT.rotate(flare_linear, rotation, padding_mode=padding_mode)
        flare_linear = KT.shear(flare_linear, shear, padding_mode=padding_mode)
        flare_linear = KT.scale(flare_linear, scale, padding_mode=padding_mode)
        flare_linear = KT.translate(flare_linear, shift, padding_mode=padding_mode)

    flare_linear.clamp_(min=0.0, max=1.0)
    basic_transforms = T.Compose(
        [
            # T.CenterCrop(flare_resolution),
            RandomCrop(flare_resolution),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
        ]
    )
    flare_linear = basic_transforms(flare_linear)

    # First normalize the white balance. Then apply random white balance.
    channel_size = flare_linear.shape[1]
    if apply_random_white_balance:
        flare_linear = normalize_white_balance(flare_linear)

        rgb_gains = flare_linear.new_empty(batch_size, channel_size).uniform_(
            0, flare_max_gain
        )
        flare_linear *= rgb_gains.view(batch_size, channel_size, 1, 1)  # NCHW
    else:
        rgb_gains = flare_linear.new_empty(batch_size, 1).uniform_(0, flare_max_gain)
        rgb_gains = rgb_gains + flare_linear.new_empty(
            batch_size, channel_size
        ).uniform_(0, flare_max_gain * 0.2)
        flare_linear *= rgb_gains.view(batch_size, -1, 1, 1)  # NCHW

    # Further augmentation on flare patterns: random blur and DC offset.
    flare_linear = T.functional.gaussian_blur(
        flare_linear,
        kernel_size=[21, 21],
        sigma=torch.empty(1, device=device).uniform_(0.1, 3).item(),
    )
    flare_linear = flare_linear + flare_linear.new_empty(
        [batch_size, 1, 1, 1]
    ).uniform_(-0.02, 0.02)
    flare_linear = flare_linear.clamp_(min=0.0, max=1.0)

    flare_srgb = adjust_gamma(flare_linear, 1 / gamma)

    scene_linear = adjust_gamma(scene, gamma)
    basic_transforms = T.Compose(
        [
            RandomCrop(resolution),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
        ]
    )
    scene_linear = basic_transforms(scene_linear)

    # Additive Gaussian noise. The Gaussian's variance is drawn from a
    # Chi-squared distribution. This is equivalent to drawing the Gaussian's
    # standard deviation from a truncated normal distribution, as shown below.
    noise = torch.randn_like(scene_linear, device=device) * torch.abs(
        torch.randn(1, device=device) * noise_strength
    )
    scene_linear += noise

    # Random digital gain.
    # varying the intensity scale
    scene_linear = (scene_linear * np.random.uniform(0, 1.2)).clamp(0, 1)

    scene_srgb = adjust_gamma(scene_linear, 1 / gamma)

    # Combine the flare-free scene with a flare pattern to produce a synthetic
    # running example.
    combined_linear = scene_linear + flare_linear
    combined_srgb = adjust_gamma(combined_linear, 1.0 / gamma)
    combined_srgb.clamp_(min=0.0, max=1.0)

    return (
        quantize_8(scene_srgb),
        quantize_8(flare_srgb),
        quantize_8(combined_srgb),
        gamma,
    )


def remove_flare(combined, flare, gamma=2.2):
    # Avoid zero. Otherwise, the gradient of pow() below will be undefined when
    # gamma < 1.
    combined = combined.clamp(_EPS, 1.0)
    flare = flare.clamp(_EPS, 1.0)

    combined_linear = torch.pow(combined, gamma)
    flare_linear = torch.pow(flare, gamma)

    scene_linear = combined_linear - flare_linear
    # Avoid zero. Otherwise, the gradient of pow() below will be undefined when
    # gamma < 1.
    scene_linear = scene_linear.clamp(_EPS, 1.0)
    scene = torch.pow(scene_linear, 1.0 / gamma)
    return scene


def get_highlight_mask(image, threshold=0.99):
    binary_mask = image.mean(dim=1, keepdim=True) > threshold
    binary_mask = binary_mask.to(image.dtype)
    return binary_mask


def _create_disk_kernel(kernel_size):
    x = np.arange(kernel_size) - (kernel_size - 1) / 2
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    kernel = np.float32(rr <= np.max(x)) + _EPS
    kernel = kernel / np.sum(kernel)
    return kernel


def blend_light_source(input_scene, pred_scene):
    binary_mask = (get_highlight_mask(input_scene) > 0.5).to("cpu", torch.bool)
    binary_mask = binary_mask.squeeze(dim=1)  # (b, h, w)
    binary_mask = binary_mask.numpy()

    labeled = skimage.measure.label(binary_mask)
    properties = skimage.measure.regionprops(labeled)
    max_diameter = 0
    for p in properties:
        # The diameter of a circle with the same area as the region.
        max_diameter = max(max_diameter, p["equivalent_diameter"])

    mask = np.float32(binary_mask)

    kernel_size = round(1.5 * max_diameter)
    if kernel_size > 0:
        kernel = _create_disk_kernel(kernel_size)
        mask = cv2.filter2D(mask, -1, kernel)
        mask = np.clip(mask * 3.0, 0.0, 1.0)
        mask_rgb = np.stack([mask] * 3, axis=1)

        mask_rgb = torch.from_numpy(mask_rgb).to(input_scene.device, torch.float32)
        blend = input_scene * mask_rgb + pred_scene * (1 - mask_rgb)
    else:
        blend = pred_scene
    return blend


def flare_to_mask(flare, threshold=0.25, dilation_kernel_size=5):
    mask = (flare.amax(dim=1, keepdim=True) > threshold).float()
    mask = kornia.morphology.dilation(
        mask,
        torch.ones(dilation_kernel_size, dilation_kernel_size, device=flare.device),
    )
    return mask


def _test():
    scene = utils.load_image(
        r"C:\Users\wangr\Desktop\大疆项目\数据集\Flickr\synthetic\transmission_layer\1136.jpg",
        # size=(300, 300),
    )
    flare = utils.load_image(
        # r"C:\Users\wangr\Downloads\DJI_0319 (中).JPG",
        # r"C:\Users\wangr\Desktop\大疆项目\数据集\lens-flare\simulated\aperture0000_blur00_crop02.png",
        # r"C:\Users\wangr\Desktop\大疆项目\数据集\lab_mavic3cine_flare\filtered_flare\DJI_0295.JPG",
        r"C:\Users\wangr\Desktop\大疆项目\数据集\lens-flare\captured\frame_1297.png",
        # size=(683, 512),
    )

    scene = scene.repeat(4, 1, 1, 1)
    flare = flare.repeat(4, 1, 1, 1)

    print(f"{flare.size()=}, {scene.size()=}")

    combined = []
    scenes = []
    flares = []
    flare_segs = []
    for _ in range(4):
        scene_srgb, flare_srgb, combined_srgb, gamma = add_flare(
            scene,
            flare,
        )
        combined.append(combined_srgb)
        scenes.append(scene_srgb)
        flares.append(flare_srgb)
        flare_segs.append(flare_to_mask(flare_srgb))

    torchvision.utils.save_image(
        torch.cat(combined, dim=0), r"C:\Users\wangr\Downloads\combined.jpg", nrow=4
    )
    torchvision.utils.save_image(
        torch.cat(scenes, dim=0), r"C:\Users\wangr\Downloads\scene.jpg", nrow=4
    )
    torchvision.utils.save_image(
        torch.cat(flares, dim=0), r"C:\Users\wangr\Downloads\flare.jpg", nrow=4
    )
    torchvision.utils.save_image(
        torch.cat(flare_segs, dim=0), r"C:\Users\wangr\Downloads\flare_segs.jpg", nrow=4
    )


if __name__ == "__main__":
    _test()
