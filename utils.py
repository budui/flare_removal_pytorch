from copy import deepcopy
from typing import Iterable, Union, Optional, MutableMapping

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


def instantiate(module, description):
    class_name, args = class_name_and_args(description)
    return getattr(module, class_name)(**args)


def class_name_and_args(description):
    if isinstance(description, str):
        return description, dict()
    if isinstance(description, MutableMapping):
        if "_type" in description:
            args = deepcopy(description)
            return args.pop("_type"), args
        elif len(description) == 1:
            class_name, arguments = tuple(description.items())[0]
            arguments = dict(arguments.items())
            return class_name, arguments
        else:
            raise ValueError(
                f"Invalid `description`, Mapping `description` must contain "
                f"the type information, but got {description}"
            )
    else:
        raise TypeError(
            f"`description` must be `MutableMapping` or a str,"
            f" but got {type(description)}"
        )


def grid_transpose(
    tensors: Union[torch.Tensor, Iterable], original_nrow: Optional[int] = None
) -> torch.Tensor:
    """
    batch tensors transpose.
    :param tensors: Tensor[(ROW*COL)*D1*...], or Iterable of same size tensors.
    :param original_nrow: original ROW
    :return: Tensor[(COL*ROW)*D1*...]
    """
    assert torch.is_tensor(tensors) or isinstance(tensors, Iterable)
    if not torch.is_tensor(tensors) and isinstance(tensors, Iterable):
        seen_size = None
        grid = []
        for tensor in tensors:
            if seen_size is None:
                seen_size = tensor.size()
                original_nrow = original_nrow or len(tensor)
            elif tensor.size() != seen_size:
                raise ValueError("expect all tensor in images have the same size.")
            grid.append(tensor)
        tensors = torch.cat(grid)

    assert original_nrow is not None

    cell_size = tensors.size()[1:]

    tensors = tensors.reshape(-1, original_nrow, *cell_size)
    tensors = torch.transpose(tensors, 0, 1)
    return torch.reshape(tensors, (-1, *cell_size))


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert("RGB")
    if size is not None:
        size = (size, size) if isinstance(size, int) else size
        img = img.resize(size, Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize(
            (int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS
        )
    return to_tensor(img).unsqueeze_(0)


def save_image(filename, data):
    img = data.mul(255.0).clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)
