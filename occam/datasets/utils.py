import sys
import os
import torch
import wandb
import numpy as np
from PIL import Image
from typing import Any
from torchvision.datasets import ImageFolder
import torchvision
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd


from stuned.utility.utils import (
    get_project_root_path,
    str_is_number,
    load_from_pickle,
    optionally_make_dir,
)
from stuned.local_datasets.imagenet1k import (
    DEFAULT_MEAN,
    DEFAULT_STD,
)


# local modules
sys.path.insert(0, get_project_root_path())
import occam
sys.path.pop(0)


IMAGE_NORMALIZATION_CONST = 255
DATA_PATH = os.path.join(get_project_root_path(), "data")
JSON_PATH = os.path.join(get_project_root_path(), "jsons")
DATASETS_PATH = os.path.join(DATA_PATH, "datasets")
CSV_PATH = os.path.join(DATA_PATH, "csvs")


def open_pil_image(image_path):
    return (
        np.array(Image.open(image_path).convert("RGB"))
        / IMAGE_NORMALIZATION_CONST
    )


def make_wandb_image(tensor, caption=None):
    return wandb.Image(
        unnormalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy(),
        caption=caption,
    )


# function is taken from: https://github.com/bethgelab/model-vs-human/blob/master/modelvshuman/models/pytorch/simclr/utils/modules.py#L21
def unnormalize(tensor, mean=[0], std=[1], inplace=False):
    """Unnormalize a tensor image by first multiplying by std (channel-wise) and then adding the mean (channel-wise)

    Args:
        tensor (Tensor): Tensor image of size (N, C, H, W) to be de-standarized.
        mean (sequence): Sequence of original means for each channel.
        std (sequence): Sequence of original standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Unnormalized Tensor image.

    """

    if not torch.is_tensor(tensor):
        raise TypeError(
            "tensor should be a torch tensor. Got {}.".format(type(tensor))
        )

    if tensor.ndimension() != 4:
        raise ValueError(
            "Expected tensor to be a tensor image of size (N, C, H, W). Got tensor.size() = "
            "{}.".format(tensor.size())
        )
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)

    if (std == 0).any():
        raise ValueError(
            "std evaluated to zero after conversion to {}, leading to division by zero.".format(
                dtype
            )
        )

    if mean.ndim == 1:
        mean = mean[None, :, None, None]
    if std.ndim == 1:
        std = std[None, :, None, None]

    tensor.mul_(std).add_(mean)
    return tensor


def unnormalize_in1k(image):
    return unnormalize(image, DEFAULT_MEAN, DEFAULT_STD)


def torch_max_func(tensor, axis):
    return torch.max(tensor, axis=axis).values


# based on https://github.com/bethgelab/model-vs-human/blob/master/modelvshuman/datasets/decision_mappings.py
class ToClassesMapping:
    def __init__(self, indices_for_category, aggregation_function=torch.mean):
        self.aggregation_function = aggregation_function
        self.indices_for_category = indices_for_category
        self.categories = self.indices_for_category.categories

    def check_input(self, probabilities):
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()

    def __call__(self, probabilities):
        """
        probabilities: (batch_size, num_classes)
        returns: (batch_size, num_categories)
        """

        aggregated_class_probabilities = []

        for category in self.categories:
            indices = self.indices_for_category(category)
            values = probabilities[:, indices]
            aggregated_value = self.aggregation_function(values, axis=-1)
            aggregated_class_probabilities.append(aggregated_value.unsqueeze(1))

        aggregated_class_probabilities = torch.cat(
            aggregated_class_probabilities, dim=1
        )

        return aggregated_class_probabilities


def make_to_classes_mapping(
    indices_for_category, aggregation_function=torch.mean
):
    return ToClassesMapping(indices_for_category, aggregation_function)


class ModuleDelegatingWrapper(torch.nn.Module):
    def __init__(self, inner_module: torch.nn.Module):
        """
        A wrapper around torch.nn.Module to delegate calls to an inner module.

        Args:
            inner_module (torch.nn.Module): The module to wrap.
        """
        super().__init__()
        self.inner_module = inner_module

    def forward(self, *args, **kwargs) -> Any:
        """
        Delegates the forward pass to the inner module.
        """
        return self.inner_module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the inner module unless the attribute exists in this wrapper.
        """
        if name != "inner_module" and hasattr(self.inner_module, name):
            return getattr(self.inner_module, name)
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Delegate attribute setting to the inner module unless setting the 'inner_module' or wrapper attributes.
        """
        if name != "inner_module" and hasattr(self.inner_module, name):
            setattr(self.inner_module, name, value)
        else:
            super().__setattr__(name, value)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Delegate the call to the inner module.
        """
        return self.inner_module(*args, **kwargs)


def make_model_classes_wrapper(model, make_mapper):
    return ModelClassesWrapper(model, make_mapper)


class ModelClassesWrapper(ModuleDelegatingWrapper):
    def __init__(self, model, make_mapper):
        super().__init__(model)
        self.mapper = make_mapper()

    def __call__(self, x):

        logits = self.inner_module(x)

        return self.mapper(logits)


def get_collate_fn_in_d(drop_paths):
    def collate_fn_in_d(examples):
        images = []
        labels = []
        paths = []
        for example in examples:
            images.append(example["images"])
            # we take only first label as we are not interested in top-5 accuracy
            labels.append(torch.tensor(example["labels"][0], dtype=torch.long))
            paths.append(example["path"])
        if drop_paths:
            return torch.stack(images), torch.stack(labels)
        else:
            return torch.stack(images), torch.stack(labels), paths

    return collate_fn_in_d


def make_custom_folder_path2label(dataset_path):

    transform = None
    return_path = True
    masks = None
    mask_transform = None

    dataset = CustomImageFolder(
        dataset_path,
        transform=transform,
        return_path=return_path,
        masks=masks,
        mask_transform=mask_transform,
    )

    return dataset.samples


class CustomImageFolder(ImageFolder):
    def __init__(
        self, root, return_path=False, masks=None, mask_transform=None, **kwargs
    ):
        super().__init__(root, **kwargs)
        self.masks = masks
        self.return_path = return_path
        self.mask_transform = mask_transform
        if self.transform is not None and self.mask_transform is not None:
            resize_from_image = self.transform.transforms[0]
            resize_from_mask = self.mask_transform.transforms[1]
            assert isinstance(resize_from_image, torchvision.transforms.Resize)
            assert isinstance(resize_from_mask, torchvision.transforms.Resize)
            if isinstance(resize_from_image.size, tuple):
                assert resize_from_image.size[0] == resize_from_image.size[1]
                assert resize_from_mask.size[0] == resize_from_mask.size[1]
                resize_from_mask = resize_from_image
            else:
                resize_from_image_size = resize_from_image.size

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return_value = [sample, target]

        if self.masks is not None:
            image_name = os.path.basename(path)
            image_class = os.path.basename(os.path.dirname(path))
            image_type = os.path.basename(
                os.path.dirname(os.path.dirname(path))
            )
            mask_id = f"{image_type}_{image_class}_{image_name}"

            mask = self.masks[mask_id]["mask"]
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
                assert len(mask.shape) == 3
                mask = mask.repeat(3, 1, 1)
            return_value.append(mask)

        if self.return_path:
            return_value.append(path)

        return return_value

    def find_classes(self, dir):
        classes = os.listdir(dir)
        class_to_idx = {}
        for i, class_name in enumerate(sorted(classes)):
            if str_is_number(class_name):
                class_id = int(class_name)
            else:
                class_id = i
            class_to_idx[class_name] = class_id
        return classes, class_to_idx


def make_custom_folder_dataloader(
    dataset_path,
    transform,
    batch_size=128,
    num_workers=4,
    return_path=False,
    masks=None,
    mask_transform=None,
):
    if mask_transform is not None:
        assert masks is not None
    dataset = CustomImageFolder(
        dataset_path,
        transform=transform,
        return_path=return_path,
        masks=masks,
        mask_transform=mask_transform,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )


def make_mapping_dict_generic_from_folder(
    images_folder,
    masks_path,
    bboxes_path,
    separate_masks_folder,
):
    return make_mapping_dict_generic(
        images_folder,
        masks_path,
        bboxes_path=bboxes_path,
        separate_masks_folder=separate_masks_folder,
        path2label_func=make_custom_folder_path2label,
    )


def make_mapping_dict_generic(
    images_folder,  # can contain dataset_kwargs: (images_folder, dataset_kwargs)
    masks_path,  # can contain bboxes_path: (masks_path, bboxes_path)
    bboxes_path,
    separate_masks_folder,
    path2label_func,
):
    dataset_kwargs = {}
    if isinstance(images_folder, (list, tuple)):
        images_folder, dataset_kwargs = images_folder
    path2label = path2label_func(images_folder)

    if isinstance(masks_path, (list, tuple)):
        masks_path, bboxes_path = masks_path

    mapping_dict = make_mapping_dict_from_folder(
        path2label=path2label,
        masks_path=masks_path,
        separate_masks_folder=separate_masks_folder,
        bboxes_path=bboxes_path,
    )
    return mapping_dict


# mapping dict for CounterAnimal
def make_mapping_dict_from_folder(
    path2label,
    masks_path,
    separate_masks_folder,
    bboxes_path,
    assert_shape=False,
):
    """
    Generates a mapping dictionary linking image paths to corresponding mask paths, bounding box paths,
    and labels. Optionally validates the shape consistency between images and masks.

    Args:
        path2label (list): A list of tuples where each tuple contains the path to an image and its corresponding label.
        masks_path (str): Path to the pickle file containing preloaded masks.
        separate_masks_folder (str): Directory to save the extracted and processed mask files.
        bboxes_path (str or None): Path to bounding boxes information. If None, bounding boxes are not used.
        assert_shape (bool, optional): If True, asserts that the image and mask shapes match. Defaults to False.

    Returns:
        dict: A dictionary where each key is an image path, and the value is a tuple containing:
            - mask_path (str): Path to the saved mask file.
            - bbox_path (str or None): Path to the bounding box information (or None if not applicable).
            - label: Label associated with the image.

    Raises:
        AssertionError: If a mask corresponding to an image is not found in the preloaded masks or if `assert_shape`
                        is True and the image and mask shapes do not match.
        NotImplementedError: If `bboxes_path` is provided (functionality for handling bounding boxes is not implemented).
    """

    def get_bbox_path(path, bboxes_folder):
        if os.path.basename(bboxes_folder) == "val":
            bboxes_type = "val"
        else:
            assert (
                os.path.basename(bboxes_folder) == "Annotation"
            ), "train bboxes should be in Annotation folder"
            bboxes_type = "train"

        class_id = os.path.basename(os.path.dirname(path))
        if bboxes_type == "val":
            folder_path = bboxes_folder
        else:
            folder_path = os.path.join(bboxes_folder, class_id)
        return os.path.join(
            folder_path, os.path.basename(path).replace(".JPEG", ".xml")
        )

    res = {}
    masks = load_from_pickle(masks_path)
    renamed_masks = {}
    for key, value in masks.items():
        renamed_masks[subpath(key, 2)] = value

    masks = renamed_masks

    os.makedirs(separate_masks_folder, exist_ok=True)
    for path, label in tqdm(path2label):
        mask_id = subpath(path, 2)

        assert mask_id in masks

        mask = masks[mask_id]["mask"]

        if assert_shape:
            image = open_pil_image(path)

            assert image.shape[:-1] == mask.shape

        if bboxes_path is None:
            bbox_path = None
        else:
            bbox_path = get_bbox_path(path, bboxes_path)

        mask_path = os.path.join(
            separate_masks_folder,
            make_mask_name_from_path(path),
        )

        torch.save(mask, mask_path)
        res[path] = (mask_path, bbox_path, label)

    return res


def subpath(path, k, sep=""):
    return sep.join(path.split(os.sep)[-k:])


def load_xml(path_to_xml):
    root = ET.parse(path_to_xml).getroot()
    return root


def make_mask_name_from_path(path):
    return path.replace(os.sep, "@") + ".mask"


def make_source_df(mapping_dict):
    res = {
        "source_image_path": [],
        "classification_label": [],
        "image_to_label": [],
        "main_object_label": [],
        "mask_path": [],
        "mask_value": [],
        "bbox_path": [],
        "metadata": [],
    }

    for image_path, image_data in tqdm(mapping_dict.items()):
        mask_path = image_data[0]
        bbox_path = image_data[1]
        image_label = image_data[2]

        assert mask_path is not None

        main_object_label = 1
        masks = torch.load(mask_path, weights_only=False)

        all_mask_values = np.unique(masks).tolist()

        for mask_value in all_mask_values:

            res["source_image_path"].append(image_path)
            res["classification_label"].append(image_label)
            res["image_to_label"].append(None)
            res["main_object_label"].append(main_object_label)
            res["mask_path"].append(mask_path)
            res["mask_value"].append(mask_value)
            res["bbox_path"].append(bbox_path)
            res["metadata"].append(None)

    df = pd.DataFrame(res)
    return df


def to_parquet(df, path):
    optionally_make_dir(path)
    df.to_parquet(path)
