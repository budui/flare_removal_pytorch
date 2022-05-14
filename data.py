import random
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Callable, MutableMapping, Mapping

import torchvision
from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file, default_loader


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


def pipeline(pipeline_description: Iterable) -> Callable:
    transforms_list = []
    for pd in pipeline_description:
        class_name, args = class_name_and_args(pd)
        transforms_list.append(getattr(torchvision.transforms, class_name)(**args))

    return torchvision.transforms.Compose(transforms_list)


class ImageDataset(Dataset):
    def __init__(self, folders, transform, recursive=False, return_image_path=False):
        if isinstance(folders, (str, Path)):
            folders = [
                folders,
            ]
        folders = [Path(f) for f in folders]
        for f in folders:
            assert f.exists(), f"{f} not exist, can not build ImageDataset"

        self.folders = folders
        self.recursive = recursive
        self.return_image_path = return_image_path
        self.files = self.list_image_files(self.folders, recursive=recursive)
        self.transform = transform if callable(transform) else pipeline(transform)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        attrs = ["folders", "return_image_path", "recursive"]
        attr_str = "".join([f"\t{a}={getattr(self, a)}\n" for a in attrs])
        return f"{self.__class__.__name__}(\n{attr_str})"

    @staticmethod
    def list_image_files(folders, recursive=False):
        pattern = "**/*" if recursive else "*"
        image_files = []
        for f in folders:
            if not f.exists():
                continue
            files = [file for file in f.glob(pattern) if is_image_file(file.name)]
            image_files.extend(files)
        return image_files

    def __getitem__(self, idx):
        file_path = self.files[idx]
        out = dict(image=self.transform(default_loader(file_path)))
        if self.return_image_path:
            out["path"] = file_path
        return out


class UnpairedDataset(Dataset):
    def __init__(
        self, folders_a, folders_b, transform, recursive=False, return_image_path=False
    ):
        if isinstance(transform, Mapping):
            transform_a = transform["A"]
            transform_b = transform["B"]
        else:
            transform_a = transform
            transform_b = transform

        self.dataset_a = ImageDataset(
            folders_a, transform_a, recursive, return_image_path
        )
        self.dataset_b = ImageDataset(
            folders_b, transform_b, recursive, return_image_path
        )

    def __len__(self):
        return max(len(self.dataset_b), len(self.dataset_a))

    def __getitem__(self, idx):
        j = random.randint(0, len(self.dataset_b) - 1)
        result_a = self.dataset_a[idx % len(self.dataset_a)]
        result_b = self.dataset_b[j]
        return dict(a=result_a, b=result_b)

    def __repr__(self):
        attrs = ["dataset_a", "dataset_b"]
        attr_str = "".join([f"\t{a}={getattr(self, a)}\n" for a in attrs])
        return f"{self.__class__.__name__}(\n{attr_str})"
