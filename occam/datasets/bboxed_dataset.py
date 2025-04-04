# import wget
import os
import sys

# import shutil
import torch
import pandas as pd

# from tqdm import tqdm
# import xml.etree.ElementTree as ET
import numpy as np

# import matplotlib.pyplot as plt
import torchvision
import PIL

# import json
import yaml
from sklearn.metrics import PrecisionRecallDisplay, roc_auc_score, roc_curve
from stuned.utility.utils import (
    get_project_root_path,
    load_from_pickle,
    get_with_assert,
    get_hash,
)
from torch.utils.data import DataLoader, Subset, Dataset
import copy
from sklearn.model_selection import train_test_split
from detectron2.data.detection_utils import read_image


from stuned.utility.utils import (
    show_images,
    load_from_pickle,
    append_dict,
    apply_random_seed,
)
from stuned.local_datasets.imagenet1k import get_imagenet_dataset
from stuned.local_datasets.transforms import (
    DEFAULT_RESIZE_IN,
    DEFAULT_SIZE_IN,
    make_transforms,
    make_default_test_transforms_imagenet,
)


# local modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("")), "src"))
# import densifier
from occam.datasets.utils import open_pil_image, subpath, load_xml
from occam.robust_classification.masking import (
    apply_visual_prompts,
    # _build_timm_model,
    is_background,
)

# from occam.utility.utils_for_notebooks import (
#     make_symlink_cmd,
#     tensor_for_matplotlib,
#     load_xml
# )
# from occam.detection.uncertainty_scores import (
#     div_continous_unique_per_sample,
#     average_energy_per_sample,
#     ens_entropy_per_sample,
#     entropy,
#     ens_conf_per_sample,
#     get_probs
# )
sys.path.pop(0)


MIN_NON_ZERO_PIXELS = 100
NUM_CORNER_PIXELS_FOR_BG = 5
EXTENDED_BBOXED_DATASET_ITEM_LEN = 10
EVAL_TRANSFORM_APPLIED_MASK_CONFIG = {
    "transforms_list": [
        "from_class-ToTensor",
        "from_class-Resize",
        "from_class-CenterCrop",
        "from_class-Normalize",
    ],
    "from_class-Normalize": {
        "class": "torchvision.transforms.Normalize",
        "kwargs": {"std": [0.229, 0.224, 0.225], "mean": [0.485, 0.456, 0.406]},
    },
    "from_class-ToTensor": {"class": "torchvision.transforms.ToTensor"},
    "from_class-CenterCrop": {
        "class": "torchvision.transforms.CenterCrop",
        "kwargs": {"size": 224},
    },
    "from_class-Resize": {
        "class": "torchvision.transforms.Resize",
        "kwargs": {
            "size": 224  # important to keep the size as mask is already resized to the full image size during applying
        },
    },
}


def make_bbox(bbox_path, n_channels=3):
    def get_from_root(root, prefix, name, cast=int):
        return cast(root.find(prefix + name).text)

    xml = load_xml(bbox_path)

    object_prefix = "object/bndbox/"
    x_min, y_min, x_max, y_max = (
        get_from_root(xml, object_prefix, "xmin"),
        get_from_root(xml, object_prefix, "ymin"),
        get_from_root(xml, object_prefix, "xmax"),
        get_from_root(xml, object_prefix, "ymax"),
    )
    size_prefix = "size/"
    width, height = (
        get_from_root(xml, size_prefix, "width"),
        get_from_root(xml, size_prefix, "height"),
    )

    bbox = torch.zeros((1, height, width))
    bbox[:, y_min:y_max, x_min:x_max] = 1
    return torch.cat([bbox] * n_channels, dim=0)


def compute_bbox_fit_score(mask, bbox, extended_output=False):
    # intersection = (mask * bbox).sum()
    # outside_bbox = (mask * (1 - bbox)).sum()
    # bbox_fit_score = intersection / max(1, outside_bbox)  # to filter out background
    mask_shape_len = len(mask.shape)
    assert mask.shape == bbox.shape
    if mask_shape_len == 4:
        mask = mask[0]
        bbox = bbox[0]
    assert mask_shape_len == 3

    # make sure that mask and bbox are binary even after transform with interpolation
    mask = mask == 1
    bbox = bbox == 1
    bbox_fit_score, intersection, union = iou(mask, bbox)

    bbox_fit_score = bbox_fit_score.mean().item()
    intersection = intersection.mean().item()
    union = union.mean().item()

    if extended_output:
        # return bbox_fit_score, intersection, outside_bbox
        return bbox_fit_score, intersection, union
    else:
        return bbox_fit_score


def iou(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    Computes the Intersection over Union (IoU) between predicted and target tensors.

    Args:
    - pred (torch.Tensor): Predicted binary mask or bounding box, shape (N, H, W) for masks or (N, 4) for boxes.
    - target (torch.Tensor): Ground truth binary mask or bounding box, shape (N, H, W) for masks or (N, 4) for boxes.
    - eps (float): Small value to avoid division by zero.

    Returns:
    - torch.Tensor: IoU scores for each instance.
    """
    # Check if the input is a mask or a bounding box

    if pred.dim() == 3 and target.dim() == 3:  # Case for masks
        intersection = (pred & target).float().sum((1, 2))
        union = (pred | target).float().sum((1, 2))

    else:
        raise ValueError("Unsupported input dimensions for pred and target")

    iou = (intersection + eps) / (union + eps)
    return iou, intersection, union


# TODO(Alex | 22.10.2024): Keep either this or previous version, avoid code duplication
class ImageNetBBoxAnnotationsV2(Dataset):
    """
    A PyTorch Dataset for ImageNet bounding box annotations with support for detection
    and classification tasks, including optional extended outputs and caching.

    Args:
        csv_path (str): Path to the CSV or Parquet file containing annotation data.
        image_transform (callable): Transformation applied to the images.
        mask_transform (callable): Transformation applied to the masks.
        dataset_task (str, optional): Specifies the dataset task. Options are "detection" or "classification".
            Defaults to "detection".
        extended_output (bool, optional): If True, additional metadata or extended outputs are included.
            Defaults to False.
        cache_path (str, optional): Path to a directory for caching processed data. Defaults to
            a "cache" folder in the project root path.
        allow_bbox_shape_mismatch (bool, optional): If True, allows bounding boxes with shape mismatches.
            Defaults to True.
        images_list (list, optional): List of specific images to include in the dataset.
            Defaults to None.
        foreground_keyword (str, optional): Keyword for identifying foreground object labels and metadata.
            Relevant for csv files that have scores according to some foreground detection method.
            If provided, 'is_object' label will be copied from the column '<foreground_keyword>_label'. Defaults to None.
        apply_mask (bool, optional): If True, applied_mask contains image with applied mask,
            otherwise applied_mask is a tuple (image, mask). Defaults to True.

    Attributes:
        csv (pd.DataFrame): DataFrame containing the parsed annotations.
        image_transform (callable): Transformation applied to the images.
        mask_transform (callable): Transformation applied to the masks.
        full_transform (callable): Combined transformation for masks and images.
        dataset_task (str): Dataset task ("detection" or "classification").
        extended_output (bool): Indicates whether extended output is enabled.
        current_cache_path (str or None): Path to the current cache directory based on the annotation data hash.
            If provided, applied masks and bounding boxes are cached after each getitem call.
        allow_bbox_shape_mismatch (bool): Indicates whether bounding box and image shape mismatches are allowed.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves a sample from the dataset at the specified index.

    Raises:
        AssertionError: If `csv_path` is not a valid CSV or Parquet file, or if the dataset task
                        is not "detection" or "classification".
        ValueError: If required columns are missing from the annotation data.
    """

    def __init__(
        self,
        csv_path,
        image_transform,
        mask_transform,
        dataset_task="detection",
        extended_output=False,
        cache_path=os.path.join(get_project_root_path(), "data", "cache"),
        allow_bbox_shape_mismatch=True,
        images_list=None,
        foreground_keyword=None,
        apply_mask=True,
        filter_keyword=None,
    ):
        super().__init__()
        # self.csv_path = csv_path
        if csv_path[-8:] == ".parquet":
            self.csv = pd.read_parquet(csv_path)
        else:
            assert csv_path[-4:] == ".csv"
            self.csv = pd.read_csv(csv_path, na_values=None)
        self.image_transform = image_transform
        self.mask_transform = mask_transform  # TODO(Alex | 08.11.2024): should be renamed to common_transform and specific_transform
        self.full_transform = torchvision.transforms.Compose(
            [self.mask_transform, self.image_transform]
        )
        self.dataset_task = dataset_task
        self.extended_output = extended_output
        if cache_path is not None:
            csv_hash = get_hash(self.csv)
            self.current_cache_path = os.path.join(cache_path, csv_hash)
            print(f"Current cache path: {self.current_cache_path}")
            os.makedirs(self.current_cache_path, exist_ok=True)
            # self.cache_path = cache_path
        else:
            self.current_cache_path = None
        self.allow_bbox_shape_mismatch = allow_bbox_shape_mismatch

        # filter by image names
        if images_list is not None:
            # self.active_indices = []
            pattern = "|".join(
                [
                    clean_regex(f"{image_basename.split('.')[0]}")
                    for image_basename in images_list
                ]
            )
            self.csv = self.csv[
                self.csv["source_image_path"].str.contains(pattern, regex=True)
            ]

        # filter by filtering heuristics
        if filter_keyword is not None:
            assert filter_keyword in self.csv.columns, (
                f"Filter keyword '{filter_keyword}' not found in the dataframe columns. "
                f"Columns:\n {self.csv.columns}."
            )
            self.csv = self.csv[self.csv[filter_keyword] == 1]

        # is_main_object is the one that has the highest foreground score
        if foreground_keyword is not None:
            label_key = f"{foreground_keyword}_label"
            assert (
                label_key in self.csv.columns
            ), f"Foreground keyword '{label_key}' not found in the CSV columns. Columns:\n {self.csv.columns}."

            self.csv["main_object_label"] = self.csv[label_key]
            self.csv["metadata"] = self.csv[f"{foreground_keyword}_metadata"]

        if self.dataset_task == "classification":
            self.csv = self.csv[self.csv["main_object_label"] == 1]
        else:
            assert self.dataset_task == "detection"

        self.apply_mask = apply_mask

        # # for classification keep only image with target object
        # for idx in self.mapping_dict.keys():
        #     item = self.mapping_dict[idx]
        #     if item[-1] == 1:
        #         self.active_indices.append(idx)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        csv_row = self.csv.iloc[idx]

        return_tuple = None

        if self.dataset_task == "classification" and not self.extended_output:
            if self.apply_mask:
                # check whether applied mask is already in cache
                applied_mask = try_to_get_from_cache(
                    current_cache_path=self.current_cache_path,
                    idx=idx,
                    make_func=None,
                    obj_type="applied_mask_without_transform",
                )
                if applied_mask is not None:
                    # TODO(Alex | 26.10.2024): keep dims order inside apply_mask
                    applied_mask = np.transpose(applied_mask, (1, 2, 0))
                    applied_mask = self.full_transform(applied_mask)
                    classification_label = csv_row.iloc[1]
                    return_tuple = (applied_mask, classification_label)

        if return_tuple is None:
            # (
            #     source_image_path,
            #     classification_label,
            #     image_to_label,
            #     main_object_label,
            #     mask_path,
            #     mask_value,
            #     bbox_path
            # ) = csv_row
            source_image_path = csv_row.iloc[0]
            classification_label = csv_row.iloc[1]
            image_to_label = csv_row.iloc[2]
            main_object_label = csv_row.iloc[3]
            mask_path = csv_row.iloc[4]
            mask_value = csv_row.iloc[5]
            bbox_path = csv_row.iloc[6]
            if len(csv_row) > 7:
                metadata = csv_row.iloc[7]
                if (
                    isinstance(metadata, str)
                    and metadata[0] == "{"
                    and metadata[-1] == "}"
                ):
                    # metadata = json.loads(metadata)
                    metadata = yaml.safe_load(metadata)

                    if isinstance(metadata, dict) and len(metadata) == 1:
                        metadata = None  # avoid collate issues when dicts have different set of keys

            else:
                metadata = None

            return_tuple = get_return_tuple(
                # self,
                # csv_row
                idx,
                source_image_path,
                classification_label,
                image_to_label,
                main_object_label,
                mask_path,
                mask_value,
                bbox_path,
                current_cache_path=self.current_cache_path,
                dataset_task=self.dataset_task,
                extended_output=self.extended_output,
                allow_bbox_shape_mismatch=self.allow_bbox_shape_mismatch,
                mask_transform=self.mask_transform,
                image_transform=self.image_transform,
                full_transform=self.full_transform,
                metadata=metadata,
                apply_mask=self.apply_mask,
            )

        # for el in return_tuple:
        #     print(type(el)) # tmp

        # to work with collate we can't leave Nones here
        cleaned_return_tuple = []
        for el in return_tuple:
            if el is None:
                cleaned_return_tuple.append("None")
            else:
                cleaned_return_tuple.append(el)
        return_tuple = cleaned_return_tuple

        return return_tuple


def clean_regex(regex):
    return (
        regex.replace(".", "\.")
        .replace("/", "\/")
        .replace("(", "\(")
        .replace(")", "\)")
    )


def prepare_bbox_maker(bbox_path):
    def bbox_maker():
        return make_bbox(bbox_path, n_channels=1)  # will concat 3 dims later

    return bbox_maker


def prepare_applied_mask_maker(image, mask):
    def make_applied_mask():
        return apply_visual_prompts(
            image,
            mask,
            visual_prompt_type=("naive_gray", "rectangle_crop_resize"),
            enforce_square_shape=False,
        ).squeeze(0)

    return make_applied_mask


def try_to_get_from_cache(current_cache_path, idx, make_func, obj_type):
    if make_func is None:
        return None

    if current_cache_path is not None:
        cache_path = os.path.join(current_cache_path, f"{obj_type}_{idx}.pt")
        if os.path.exists(cache_path):
            res = torch.load(cache_path)
        else:
            # bbox = make_bbox(bbox_path)
            # if make_func is None:
            #     res = None
            # else:
            res = make_func()
            torch.save(res, cache_path)
            # torch.save(bbox, cache_path)
    else:
        res = make_func()
    return res


def load_mask(mask_path, mask_value):
    all_masks = torch.load(mask_path, weights_only=False)
    assert all_masks.max() >= mask_value, (
        f"mask_value: {mask_value} is greater than the maximum mask "
        f"value: {all_masks.max()} for {mask_path}"
    )
    mask = all_masks == mask_value
    return mask, all_masks


def get_return_tuple(
    # self,
    # csv_row
    idx,
    source_image_path,
    classification_label,
    image_to_label,
    main_object_label,
    mask_path,
    mask_value,
    bbox_path,
    current_cache_path,
    dataset_task,
    extended_output,
    allow_bbox_shape_mismatch,
    mask_transform,
    image_transform,
    full_transform,
    metadata=None,
    apply_mask=True,
):
    def transform_image_mask_bbox(
        image, mask, bbox, image_transform, mask_transform, seed=None
    ):
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        apply_random_seed(seed)
        image = mask_transform(image)  # all but normalization
        image = image_transform(image)  # normalization
        apply_random_seed(seed)
        bbox = mask_transform(bbox)
        apply_random_seed(seed)
        mask = mask_transform(mask)
        bbox = (bbox > 0).to(image.dtype)  # to avoid interpolation artifacts
        mask = (mask > 0).to(image.dtype)  # to avoid interpolation artifacts

        bbox = torch.cat([bbox] * 3, dim=0)
        mask = torch.cat([mask] * 3, dim=0)
        assert image.shape == bbox.shape == mask.shape
        return image, mask, bbox

    def transform_before_mask_apply(
        idx,
        image,
        mask,
        bbox,
        mask_transform,
        image_transform,
        current_cache_path,
    ):
        assert mask is not None
        # assert bbox is not None
        # if bbox is None:
        #     bbox = np.ones_like(mask)

        # # stacked_image_mask_bbox = np.concatenate([image, mask, bbox], axis=2)

        # # stacked_image_mask_bbox = mask_transform(stacked_image_mask_bbox)
        # # # apply the same transform to all chunks
        # # image, mask, bbox = torch.split(stacked_image_mask_bbox, [3, 1, 1], dim=0)
        # bbox = (bbox > 0).to(bbox.dtype) # to avoid interpolation artifacts
        # mask = (mask > 0).to(mask.dtype) # to avoid interpolation artifacts

        # bbox = torch.cat([bbox] * 3, dim=0)
        # mask = torch.cat([mask] * 3, dim=0)

        # image = image_transform(image)

        # seed = torch.randint(0, 2**32, (1,)).item()

        # apply_random_seed(seed)
        # image = mask_transform(image) # all but normalization
        # image = image_transform(image) # normalization
        # apply_random_seed(seed)
        # bbox = mask_transform(bbox)
        # apply_random_seed(seed)
        # mask = mask_transform(mask)

        # bbox = (bbox > 0).to(image.dtype) # to avoid interpolation artifacts
        # mask = (mask > 0).to(image.dtype) # to avoid interpolation artifacts

        # bbox = torch.cat([bbox] * 3, dim=0)
        # mask = torch.cat([mask] * 3, dim=0)
        # assert image.shape == bbox.shape == mask.shape

        image, mask, bbox = transform_image_mask_bbox(
            image, mask, bbox, image_transform, mask_transform
        )

        # assert image.shape == bbox.shape

        # can have empty mask after aggresive transform, e.g. strong crop
        if mask.max() > 0:
            applied_mask = try_to_get_from_cache(
                current_cache_path=current_cache_path,
                idx=idx,
                make_func=prepare_applied_mask_maker(
                    image.unsqueeze(0), mask.unsqueeze(0)
                ),
                obj_type="applied_mask",
            )
        else:
            applied_mask = (
                torch.zeros_like(image, dtype=torch.float32) * 0.5
            )  # gray image
        # prepare_applied_mask_maker returns float, while zeros_like returns double, we want to always use float
        return image, mask, bbox, applied_mask

    def transform_after_mask_apply(image, mask, transform, current_cache_path):
        mask = np.concatenate([mask] * image.shape[-1], axis=-1)
        applied_mask = try_to_get_from_cache(
            current_cache_path=current_cache_path,
            idx=idx,
            make_func=prepare_applied_mask_maker(
                image[None, ...], mask[None, ...]
            ),
            obj_type="applied_mask_without_transform",
        )
        # TODO(Alex | 26.10.2024): keep dims order inside apply_mask
        applied_mask = np.transpose(applied_mask, (1, 2, 0))
        applied_mask = transform(applied_mask)
        return applied_mask

    # (
    #     source_image_path,
    #     classification_label,
    #     image_to_label,
    #     main_object_label,
    #     mask_path,
    #     mask_value,
    #     bbox_path
    # ) = csv_row

    image_path = source_image_path
    label = classification_label
    is_main_object = main_object_label
    if apply_mask:
        image = open_pil_image(source_image_path)
    else:
        image = read_image(
            source_image_path, format="BGR"
        )  # https://github.com/facebookresearch/detectron2/blob/c69939aa85460e8135f40bce908a6cddaa73065f/detectron2/data/detection_utils.py#L166
        image = image[
            :, :, ::-1
        ]  # TODO(Alex | 09.12.2024): can we read directly to RGB?
        # image.flags.writeable = True # to avoid warnings
        image = np.copy(image)  # to avoid warnings about non-writeable arrays
        # image = image / 255 # uint8 -> float32

    # if not isinstance(bbox_path, str) and np.isnan(bbox_path):
    if bbox_path is None or (
        not isinstance(bbox_path, str) and np.isnan(bbox_path)
    ):
        assert mask_path is not None
        bbox = None
    else:
        bbox = try_to_get_from_cache(
            current_cache_path=current_cache_path,
            idx=idx,
            make_func=prepare_bbox_maker(bbox_path),
            obj_type="bbox",
        )
        bbox = bbox.permute(1, 2, 0).numpy()

        assert bbox.shape[2] == 1
        if bbox.shape[:2] != image.shape[:2]:
            if (
                image.shape[0] == bbox.shape[1]
                and image.shape[1] == bbox.shape[0]
            ):
                # sometimes read_image from detectron2 rotates image to surpass pillow bug
                # see "_apply_exif_orientation" here: https://detectron2.readthedocs.io/en/latest/_modules/detectron2/data/detection_utils.html
                image = image.transpose(1, 0, 2)
            else:
                if allow_bbox_shape_mismatch:
                    print(
                        f"Bbox shape mismatch of {bbox.shape[:2]} (bbox.shape) "
                        f"vs {image.shape[:2]} (image.shape) "
                        f"for {source_image_path}"
                    )
                    bbox = np.ones_like(image)[:, :, 0][..., None]

                else:
                    raise ValueError("bbox shape mismatch")

    # if mask_path is not None:
    if not isinstance(mask_path, str) and np.isnan(mask_path):
        # treat bbox as mask
        assert bbox_path is not None
        assert bbox is not None
        all_masks = None
        mask = bbox
    else:
        # all_masks = torch.load(mask_path, weights_only=False)
        # assert all_masks.max() >= mask_value, (
        #     f"mask_value: {mask_value} is greater than the maximum mask "
        #     f"value: {all_masks.max()} for {mask_path}"
        # )
        # mask = all_masks == mask_value
        mask, all_masks = load_mask(mask_path, mask_value)
        if len(mask.shape) == 3:
            mask = mask[
                0
            ]  # extract first channel as we will duplicate channels later
        assert len(mask.shape) == 2
        mask = mask[..., None]
        # mask = np.transpose(mask, (1, 2, 0))
        assert mask.shape[2] == 1
        assert mask.shape[:2] == image.shape[:2]

    if bbox is None:
        bbox = np.ones_like(mask)

    if mask.max() == 0:
        print(
            f"Mask.max() is 0 "
            f"for {source_image_path}. Using whole image as mask instead.\n"
            f"mask_path: {mask_path}\n"
            f"mask_value: {mask_value}\n"
        )
        mask = np.ones_like(mask)

    if dataset_task == "classification":
        if apply_mask:
            applied_mask = transform_after_mask_apply(
                image,
                mask,
                full_transform,
                current_cache_path=current_cache_path,
            )
        # else:
        #     applied_mask = (image, mask)
        # if extended_output:
        #     # image = torch.Tensor(image)
        #     if mask is not None:
        #         mask = torch.Tensor(mask)
        #     if bbox is not None:
        #         bbox = torch.Tensor(bbox)

    else:
        image, mask, bbox, applied_mask = transform_before_mask_apply(
            idx,
            image,
            mask,
            bbox,
            mask_transform=mask_transform,
            image_transform=image_transform,
            current_cache_path=current_cache_path,
        )

    # applied_mask = applied_mask.squeeze(0)

    if extended_output or not apply_mask:
        if dataset_task == "classification":
            # image = image_transform(mask_transform(image))
            # mask = mask_transform(mask)
            # bbox = mask_transform(bbox)
            image, mask, bbox = transform_image_mask_bbox(
                image, mask, bbox, image_transform, mask_transform
            )

    if not apply_mask:
        applied_mask = (image, mask)

    if extended_output:
        # TODO(Alex | 07.10.2024): compute it only once in init
        # intersection_info = compute_bbox_fit_score(
        #     mask,
        #     bbox,
        #     extended_output=True
        # )
        if metadata is None:
            metadata = {}
        else:
            assert isinstance(metadata, dict)

        metadata["mask_value"] = mask_value

        # metadata = str(metadata)

        # resize for stacking in batches

        all_masks = mask_transform(all_masks)
        # if dataset_task == "classification":
        #     image = image_transform(mask_transform(image))
        #     mask = mask_transform(mask)
        #     bbox = mask_transform(bbox)

        # print("image.shape", image.shape)
        # print("mask.shape", mask.shape)
        # print("bbox.shape", bbox.shape)
        # print("applied.shape", applied_mask.shape)
        # print("all_masks.shape", all_masks.shape) # tmp 6

        return_tuple = (
            idx,
            image,
            label,
            bbox,
            mask,
            applied_mask,
            is_main_object,
            image_path,
            all_masks,
            # intersection_info
            metadata,
        )
    else:
        if dataset_task == "classification":
            return_tuple = (applied_mask, label)
        else:
            return_tuple = (
                image,
                label,
                bbox,
                mask,
                applied_mask,
                is_main_object,
                image_path,
            )
    return return_tuple


# def subpath(path, k, sep=""):
#     return sep.join(path.split(os.sep)[-k:])


def pathprefix(path, k, sep=""):
    return sep.join(path.split(os.sep)[:-k])


def get_mask_id_prefix(masks, k=2):
    assert len(masks)
    first_key = list(masks.keys())[0]
    return pathprefix(first_key, k, sep=os.sep)


class ImageNetBBoxAnnotations(Dataset):
    def __init__(
        self,
        base_dataset,
        bboxes_folder,
        masks,  # can be either dict or path to pickle
        mapping_dict_path,
        mask_per_value_folder,
        image_transform,
        mask_transform,
        dataset_task="detection",
        extended_output=False,
        images_list=None,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.bboxes_folder = bboxes_folder
        self.masks = masks
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.extended_output = extended_output
        self.images_list = images_list

        if isinstance(self.masks, str) and self.extended_output:
            self.masks = load_from_pickle(self.masks)

        assert dataset_task in ["detection", "classification"]
        self.dataset_task = dataset_task

        if os.path.basename(self.bboxes_folder) == "val":
            self.bboxes_type = "val"
        else:
            self.bboxes_type = "train"

        self.mapping_dict = self._make_mapping_dict(
            mapping_dict_path, mask_per_value_folder
        )
        if "mismatch" in self.mapping_dict:
            mismatch_info = self.mapping_dict.pop("mismatch")
            print(f"Mismatch info: {mismatch_info}")
        if dataset_task == "detection":
            self.active_indices = list(self.mapping_dict.keys())
        else:
            self.active_indices = []
            # for classification keep only image with target object
            for idx in self.mapping_dict.keys():
                item = self.mapping_dict[idx]
                if item[-1] == 1:
                    self.active_indices.append(idx)

        self.filter_images_list()

        self.targets = []
        for idx in self.active_indices:
            item = self.mapping_dict[idx]
            base_id = item[0]
            _, target = self.base_dataset.samples[base_id]
            self.targets.append(target)

    # keep only images from the images_list, basenames are given
    def filter_images_list(self):
        if self.images_list is not None:
            new_indices = []
            for idx in self.active_indices:
                item = self.mapping_dict[idx]
                image_path = os.path.basename(item[1])
                if image_path in self.images_list:
                    new_indices.append(idx)
            self.active_indices = new_indices

    def __len__(self):
        return len(self.active_indices)

    def _make_mapping_dict(
        self, mapping_dict_path, mask_per_value_folder, save_every=100
    ):
        raise NotImplementedError(
            'Not implemented, see "make_df_with_foreground_scores"'
        )

        # def is_wnid(class_id):
        #     is_wnid = True
        #     if not len(class_id) == 9:
        #         is_wnid = False

        #     if not class_id[0] == "n":
        #         is_wnid = False

        #     if not class_id[1:].isalnum():
        #         is_wnid = False

        #     return is_wnid

        # def process_masks(
        #     all_masks,
        #     image_path,
        #     mask_id,
        #     mapping_dict,
        #     bbox_path
        # ):

        #     def update_max_and_key(cur_value, cur_key, cache, cache_key):
        #         # update max value
        #         max_value, max_key = cache[cache_key]
        #         if cur_value > max_value:
        #             cache[cache_key] = [cur_value, cur_key]

        #     cur_masks = all_masks[mask_id]['mask']
        #     mask_values = np.unique(cur_masks).tolist()

        #     # variables to find mask with the best bbox_fit_score
        #     cache_for_max_bbox_fit_score = {
        #         "all": [np.inf * -1, None],
        #         "non-bg": [np.inf * -1, None]
        #     }

        #     key_per_mask_value_max = None
        #     # key_per_mask_value_no_bg = None
        #     # bbox = make_bbox(bbox_path).numpy()
        #     bbox = make_bbox(bbox_path)
        #     bbox = (bbox == 1)

        #     mask_shape = cur_masks.shape[-2:]
        #     bbox_shape = bbox.shape[-2:]
        #     if mask_shape != bbox_shape:
        #         mismatch_info = (
        #             mask_shape,
        #             bbox_shape,
        #             image_path,
        #             mask_id,
        #             bbox_path
        #         )
        #         if "mismatch" not in mapping_dict:
        #             mapping_dict["mismatch"] = [mismatch_info]
        #         else:
        #             mapping_dict["mismatch"].append(mismatch_info)
        #         print(mismatch_info)
        #         return

        #     cur_mapping_dict = {}

        #     for mask_value in mask_values:
        #         key_per_mask_value = (mask_id + f"_{mask_value}").replace("/", "_")

        #         if key_per_mask_value in mapping_dict:
        #             continue

        #         mask_per_value = (cur_masks == mask_value)
        #         if mask_per_value.sum() < MIN_NON_ZERO_PIXELS:
        #             continue
        #         mask_per_value = np.concatenate(
        #             [mask_per_value[None, ...]] * bbox.shape[0],
        #             axis=0
        #         )

        #         mask_per_value_path = os.path.join(
        #             mask_per_value_folder,
        #             key_per_mask_value + ".pt"
        #         )

        #         torch.save(mask_per_value, mask_per_value_path)
        #         is_main_object = 0

        #         cur_mapping_dict[key_per_mask_value] = [
        #             base_id,
        #             image_path,
        #             bbox_path,
        #             mask_id,
        #             mask_per_value_path,
        #             is_main_object
        #         ]

        #         mask_per_value = (torch.Tensor(mask_per_value) == 1)
        #         # bbox = (torch.Tensor(bbox) == 1)

        #         bbox_fit_score = compute_bbox_fit_score(mask_per_value, bbox)

        #         update_max_and_key(
        #             bbox_fit_score,
        #             key_per_mask_value,
        #             cache_for_max_bbox_fit_score,
        #             "all"
        #         )
        #         if not is_background(
        #             mask_per_value[0],
        #             threshold=NUM_CORNER_PIXELS_FOR_BG,
        #             to_extract=False
        #         ):
        #             update_max_and_key(
        #                 bbox_fit_score,
        #                 key_per_mask_value,
        #                 cache_for_max_bbox_fit_score,
        #                 "non-bg"
        #             )

        #     key_per_mask_value_max = cache_for_max_bbox_fit_score["non-bg"][1]
        #     if key_per_mask_value_max is None:
        #         key_per_mask_value_max = cache_for_max_bbox_fit_score["all"][1]

        #     assert key_per_mask_value_max is not None
        #     item_to_update = cur_mapping_dict[key_per_mask_value_max]
        #     item_to_update[-1] = 1
        #     cur_mapping_dict[key_per_mask_value_max] = item_to_update

        #     mapping_dict |= cur_mapping_dict

        # tmp_mapping_dict_path = mapping_dict_path + ".tmp"

        # if not os.path.exists(mapping_dict_path):
        #     os.makedirs(os.path.dirname(mapping_dict_path), exist_ok=True)

        #     if isinstance(self.masks, str):
        #         self.masks = load_from_pickle(self.masks)

        #     mask_id_prefix = get_mask_id_prefix(self.masks)

        #     if os.path.exists(tmp_mapping_dict_path):
        #         print("Loading mapping dict from tmp file")
        #         mapping_dict = torch.load(tmp_mapping_dict_path)
        #     else:
        #         mapping_dict = {}

        #     cnt = 0
        #     os.makedirs(mask_per_value_folder, exist_ok=True)

        #     for base_id in tqdm(range(len(self.base_dataset.samples))):

        #         path_target = self.base_dataset.samples[base_id]
        #         path = path_target[0]

        #         image_name = os.path.basename(path)

        #         class_id = os.path.basename(os.path.dirname(path))

        #         # if is_wnid(class_id):
        #         if self.bboxes_type == "val":
        #             folder_path = self.bboxes_folder
        #         else:

        #             folder_path = os.path.join(
        #                 self.bboxes_folder,
        #                 class_id
        #             )

        #         bbox_path = os.path.join(
        #             folder_path,
        #             image_name.replace(".JPEG", ".xml")
        #         )

        #         if not os.path.exists(bbox_path):
        #             continue

        #         mask_id = os.path.join(mask_id_prefix, class_id, image_name)
        #         if not mask_id in self.masks:
        #             continue

        #         process_masks(self.masks, path, mask_id, mapping_dict, bbox_path)
        #         cnt += 1
        #         if cnt == len(self.masks):
        #             break

        #         if cnt % save_every == 0:
        #             torch.save(mapping_dict, tmp_mapping_dict_path)
        # else:
        #     mapping_dict = torch.load(mapping_dict_path)

        # torch.save(mapping_dict, mapping_dict_path)
        # if os.path.exists(tmp_mapping_dict_path):
        #     os.remove(tmp_mapping_dict_path)
        # return mapping_dict

    def __getitem__(self, idx):
        active_idx = self.active_indices[idx]
        (
            base_id,
            image_path,
            bbox_path,
            mask_id,
            mask_per_value_path,
            is_main_object,
        ) = self.mapping_dict[active_idx]
        path, label = self.base_dataset.samples[base_id]
        image = self.base_dataset.loader(path)

        bbox = make_bbox(bbox_path)

        mask = torch.load(mask_per_value_path)

        bbox = bbox.permute(1, 2, 0).numpy()
        mask = np.transpose(mask, (1, 2, 0))

        # old transforms
        seed = torch.randint(0, 2**32, (1,)).item()

        apply_random_seed(seed)
        image = self.image_transform(image)

        apply_random_seed(seed)
        bbox = self.mask_transform(bbox)
        apply_random_seed(seed)
        mask = self.mask_transform(mask)

        if mask.max() == 0:  # avoid empty masks
            applied_mask = torch.ones_like(image) * 0.5  # gray image
        else:
            applied_mask = apply_visual_prompts(
                image.unsqueeze(0),
                mask.unsqueeze(0),
                visual_prompt_type=("naive_gray", "rectangle_crop_resize"),
            )

        applied_mask = applied_mask.squeeze(0)

        if self.extended_output:
            # TODO(Alex | 07.10.2024): compute it only once in init
            intersection_info = compute_bbox_fit_score(
                mask, bbox, extended_output=True
            )
            all_masks = self.masks[mask_id]["mask"]
            return_tuple = (
                idx,
                image,
                label,
                bbox,
                mask,
                applied_mask,
                is_main_object,
                image_path,
                all_masks,
                intersection_info,
            )
        else:
            if self.dataset_task == "classification":
                return_tuple = (applied_mask, label)
            else:
                return_tuple = (
                    image,
                    label,
                    bbox,
                    mask,
                    applied_mask,
                    is_main_object,
                    image_path,
                )
        return return_tuple


def make_image_mask_transforms(transform_config):
    if transform_config is None:
        return lambda x: x, lambda x: x

    mask_transform_config = copy.deepcopy(transform_config)
    mask_transform_config["transforms_list"] = [
        transform_name
        for transform_name in mask_transform_config["transforms_list"]
        if not "Normalize" in transform_name
    ]

    transform = make_transforms(transform_config)

    # masks = load_from_pickle(masks_path)
    # masks = masks_path

    mask_transform = make_transforms(mask_transform_config)
    return transform, mask_transform


# split full transform into specific transform (Normalize) and common transform (everything else)
# first, common transform is applied to both image and mask
# then, specific transform is applied only to image
def make_common_and_specific_transforms(transform_config):
    def is_rgb_convert(transform_name):
        if isinstance(transform_name, torchvision.transforms.transforms.Lambda):
            lambda_code = transform_name.__dict__["lambd"].__code__
            return (
                "RGB" in lambda_code.co_consts
                and "convert" in lambda_code.co_names
            )
        return "to_rgb" in str(transform_name)

    def flatten_compose(all_transforms):
        flattened_transforms = []
        for transform_name in all_transforms:
            if isinstance(
                transform_name, torchvision.transforms.transforms.Compose
            ):
                flattened_compose = flatten_compose(transform_name.transforms)
                flattened_transforms.extend(flattened_compose)
                # all_transforms.remove(transform_name)
            else:
                flattened_transforms.append(transform_name)
        return flattened_transforms

    def split_in_common_and_specific(all_list):
        common_list = []
        specific_list = []
        insert_in_the_beginning = []

        all_list = flatten_compose(all_list)

        for transform_name in all_list:
            if "Normalize" in str(transform_name):
                specific_list.append(transform_name)
            elif "ToTensor" in str(transform_name):
                insert_in_the_beginning.append(transform_name)
            # elif "to_rgb" in str(transform_name):
            elif is_rgb_convert(transform_name):
                optional_convert = (
                    lambda x: transform_name(x)
                    if isinstance(x, PIL.Image.Image)
                    else x
                )
                insert_in_the_beginning.append(optional_convert)
            else:
                common_list.append(transform_name)
        common_list = insert_in_the_beginning + common_list
        return common_list, specific_list

    if transform_config is None:
        return lambda x: x, lambda x: x

    common_transform = copy.deepcopy(transform_config)
    specific_transform = copy.deepcopy(transform_config)

    if isinstance(transform_config, torchvision.transforms.transforms.Compose):
        # return transform_config, lambda x: x
        common_list, specific_list = split_in_common_and_specific(
            transform_config.transforms
        )

        # common_transform = copy.deepcopy(transform_config)
        # specific_transform = copy.deepcopy(transform_config)
        common_transform.transforms = common_list
        specific_transform.transforms = specific_list

    else:
        assert isinstance(transform_config, dict)

        # return common_transform, specific_transform

        # common_transform_config = copy.deepcopy(transform_config)
        # specific_transform_config = copy.deepcopy(transform_config)
        common_list, specific_list = split_in_common_and_specific(
            transform_config["transforms_list"]
        )
        common_transform["transforms_list"] = common_list
        specific_transform["transforms_list"] = specific_list

        # common_transform_config["transforms_list"] = []
        # for transform_name in transform_config["transforms_list"]:
        #     if "Normalize" in transform_name:
        #         # transform_name = transform_name.replace("Random", "Common")
        #         specific_transform_config["transforms_list"] = [transform_name]
        #     else:
        #         common_transform_config["transforms_list"].append(transform_name)

        common_transform = make_transforms(common_transform)
        specific_transform = make_transforms(specific_transform)

    # if specific_transform is None:
    #     specific_transform = lambda x: x

    return common_transform, specific_transform


def make_bboxed_dataset_v2(
    csv_path,
    transform_config,
    dataset_task,
    extended_output,
    cache_path,
    images_list,
    foreground_keyword,
    apply_mask,
    filter_keyword,
):
    # transform, mask_transform = make_image_mask_transforms(transform_config)
    common_transform, specific_transform = make_common_and_specific_transforms(
        transform_config
    )
    return ImageNetBBoxAnnotationsV2(
        csv_path=csv_path,
        image_transform=specific_transform,
        mask_transform=common_transform,
        dataset_task=dataset_task,
        extended_output=extended_output,
        cache_path=cache_path,
        images_list=images_list,
        foreground_keyword=foreground_keyword,
        apply_mask=apply_mask,
        filter_keyword=filter_keyword,
    )


def make_bboxed_dataset(
    base_dataset_config,
    split_for_base_dataset,
    transform_config,
    masks_path,
    bboxes_folder,
    mapping_dict_path,
    mask_per_value_folder,
    dataset_task="detection",
    extended_output=False,
    images_list=None,
):
    # mask_transform_config = copy.deepcopy(transform_config)
    # mask_transform_config["transforms_list"] = [
    #     transform_name
    #         for transform_name
    #             in mask_transform_config["transforms_list"]
    #                 if not "Normalize" in transform_name
    # ]

    # transform = make_transforms(transform_config)

    # # masks = load_from_pickle(masks_path)
    # # masks = masks_path

    # mask_transform = make_transforms(mask_transform_config)

    transform, mask_transform = make_image_mask_transforms(transform_config)

    # make base dataset
    dataset = get_imagenet_dataset(
        base_dataset_config,
        split=split_for_base_dataset,
        transform=None,
        num_samples=0,
        subset_indices=None,
        reverse_indices=False,
    )

    bboxed_dataset = ImageNetBBoxAnnotations(
        base_dataset=dataset,
        bboxes_folder=bboxes_folder,
        # masks=masks,
        masks=masks_path,
        mapping_dict_path=mapping_dict_path,
        mask_per_value_folder=mask_per_value_folder,
        image_transform=transform,
        mask_transform=mask_transform,
        dataset_task=dataset_task,
        extended_output=extended_output,
        images_list=images_list,
    )
    return bboxed_dataset


def make_bboxed_dataset_from_config(bboxed_dataset_config, transform_type):
    # train_transform_config = bboxed_dataset_config.get("train_transform")
    if transform_type == "train":
        train_transform_config = get_with_assert(
            bboxed_dataset_config, "train_transform"
        )
        transform_config = train_transform_config
    else:
        assert transform_type == "eval"
        eval_transform_config = get_with_assert(
            bboxed_dataset_config, "eval_transform"
        )
        transform_config = eval_transform_config

    dataset_task = get_with_assert(bboxed_dataset_config, "dataset_task")
    extended_output = bboxed_dataset_config.get("extended_output", False)
    cache_path = bboxed_dataset_config.get("cache_path")
    images_list = bboxed_dataset_config.get("images_list")

    csv_path = bboxed_dataset_config.get("csv_path")
    if csv_path is None:
        base_dataset_config = get_with_assert(
            bboxed_dataset_config, "base_dataset_config"
        )
        split_for_base_dataset = get_with_assert(
            bboxed_dataset_config, "split_for_base_dataset"
        )
        masks_path = get_with_assert(bboxed_dataset_config, "masks_path")
        bboxes_folder = get_with_assert(bboxed_dataset_config, "bboxes_folder")
        mapping_dict_path = get_with_assert(
            bboxed_dataset_config, "mapping_dict_path"
        )
        mask_per_value_folder = get_with_assert(
            bboxed_dataset_config, "mask_per_value_folder"
        )

        bboxed_dataset = make_bboxed_dataset(
            base_dataset_config,
            split_for_base_dataset,
            transform_config,
            masks_path,
            bboxes_folder,
            mapping_dict_path,
            mask_per_value_folder,
            dataset_task=dataset_task,
            extended_output=extended_output,
            images_list=images_list,
        )
    else:
        foreground_keyword = bboxed_dataset_config.get("foreground_keyword")
        apply_mask = bboxed_dataset_config.get("apply_mask", True)
        filter_keyword = bboxed_dataset_config.get("filter_keyword")
        bboxed_dataset = make_bboxed_dataset_v2(
            csv_path=csv_path,
            transform_config=transform_config,
            dataset_task=dataset_task,
            extended_output=extended_output,
            cache_path=cache_path,
            images_list=images_list,
            foreground_keyword=foreground_keyword,
            apply_mask=apply_mask,
            filter_keyword=filter_keyword,
        )
    return bboxed_dataset


def get_bboxed_dataloaders(
    bboxed_dataset_config,
    train_batch_size,
    eval_batch_size,
    num_workers,
    active_dataset_type="bboxed_dataset",
):
    train_val_split = get_with_assert(bboxed_dataset_config, "train_val_split")

    if train_val_split == 1:
        # assert eval_batch_size == 0 and train_batch_size > 0
        assert train_batch_size > 0
        eval_batch_size = 0
    elif train_val_split > 0 and train_val_split < 1:
        assert eval_batch_size > 0 and train_batch_size > 0
    if train_val_split == 0:
        # assert train_batch_size == 0 and eval_batch_size > 0
        assert eval_batch_size > 0
        train_batch_size = 0

    if train_batch_size > 0:
        train_bbox_dataset = make_bboxed_dataset_from_config(
            bboxed_dataset_config, transform_type="train"
        )

    if eval_batch_size > 0:
        val_bbox_dataset = make_bboxed_dataset_from_config(
            bboxed_dataset_config, transform_type="eval"
        )

    if train_val_split == 1:
        train_dataset = train_bbox_dataset
        val_dataset = None
    elif train_val_split == 0:
        train_dataset = None
        val_dataset = val_bbox_dataset
    else:
        all_idx = range(len(train_bbox_dataset))
        assert len(train_bbox_dataset) == len(val_bbox_dataset)
        if hasattr(train_bbox_dataset, "targets"):
            assert train_bbox_dataset.targets == val_bbox_dataset.targets
            stratify = train_bbox_dataset.targets
        else:
            stratify = None
        train_idx, val_idx = train_test_split(
            all_idx,
            test_size=(1 - train_val_split),
            stratify=stratify,
            # random_state=42
        )

        train_dataset = Subset(train_bbox_dataset, train_idx)
        val_dataset = Subset(val_bbox_dataset, val_idx)

    loaders = {}

    # dataset_type = "bboxed_dataset"

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        loaders[f"{active_dataset_type}_train"] = train_loader

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        loaders[f"{active_dataset_type}_val"] = val_loader

    return loaders


def make_bbox_dl_from_csv(
    csv_path,
    batch_size=128,
    extended_output=False,
    dataset_task="classification",
    num_workers=4,
    fg_keyword=None,
    eval_transform=None,
    apply_mask=True,
    images_list=None,
    filter_keyword=None,
):
    if eval_transform is None:
        eval_transform = EVAL_TRANSFORM_APPLIED_MASK_CONFIG

    bboxed_dataset_config = {
        "csv_path": csv_path,
        "dataset_task": dataset_task,
        "train_val_split": 0.0,
        "eval_transform": eval_transform,
        "extended_output": extended_output,
        "cache_path": None,
        "apply_mask": apply_mask,
        "images_list": images_list,
    }
    if fg_keyword is not None:
        bboxed_dataset_config["foreground_keyword"] = fg_keyword
    if filter_keyword is not None:
        bboxed_dataset_config["filter_keyword"] = filter_keyword

    dataset = make_bboxed_dataset_from_config(
        bboxed_dataset_config, transform_type="eval"
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # shuffle=True, # tmp
        num_workers=num_workers,
    )
