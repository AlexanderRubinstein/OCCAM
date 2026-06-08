import os
import sys
import torch
import pandas as pd
import numpy as np
import torchvision
import PIL
import yaml
from stuned.utility.utils import (
    get_project_root_path,
    get_with_assert,
    get_hash,
)
from torch.utils.data import DataLoader, Subset, Dataset
import copy
from sklearn.model_selection import train_test_split
from stuned.utility.utils import (
    apply_random_seed,
)
from stuned.local_datasets.imagenet1k import get_imagenet_dataset
from stuned.local_datasets.transforms import (
    make_transforms,
)


# local modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("")), "src"))
from occam.datasets.utils import open_pil_image, open_pil_image_uint8, subpath, load_xml
from occam.robust_classification.masking import (
    DEFAULT_VISUAL_PROMPT_TYPE,
    apply_visual_prompts,
)

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


# Technically it is not only for ImageNet and not necessarily requires bounding boxes,
# but we keep the name for legacy reasons
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
        if csv_path[-8:] == ".parquet":
            self.csv = pd.read_parquet(csv_path)
        else:
            assert csv_path[-4:] == ".csv"
            self.csv = pd.read_csv(csv_path, na_values=None)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
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
        else:
            self.current_cache_path = None
        self.allow_bbox_shape_mismatch = allow_bbox_shape_mismatch

        # filter by image names
        if images_list is not None:
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
                    applied_mask = np.transpose(applied_mask, (1, 2, 0))
                    applied_mask = self.full_transform(applied_mask)
                    classification_label = csv_row.iloc[1]
                    return_tuple = (applied_mask, classification_label)

        if return_tuple is None:
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
                    metadata = yaml.safe_load(metadata)

                    if isinstance(metadata, dict) and len(metadata) == 1:
                        metadata = None  # avoid collate issues when dicts have different set of keys

            else:
                metadata = None

            return_tuple = get_return_tuple(
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
            visual_prompt_type=DEFAULT_VISUAL_PROMPT_TYPE,
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
            res = make_func()
            torch.save(res, cache_path)
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

        image, mask, bbox = transform_image_mask_bbox(
            image, mask, bbox, image_transform, mask_transform
        )

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
        applied_mask = np.transpose(applied_mask, (1, 2, 0))
        applied_mask = transform(applied_mask)
        return applied_mask

    image_path = source_image_path
    label = classification_label
    is_main_object = main_object_label
    if apply_mask:
        image = open_pil_image(source_image_path)
    else:
        image = open_pil_image_uint8(source_image_path)

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
                # EXIF orientation can swap width/height relative to stored bbox masks
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

    if not isinstance(mask_path, str) and np.isnan(mask_path):
        # treat bbox as mask
        assert bbox_path is not None
        assert bbox is not None
        all_masks = None
        mask = bbox
    else:
        mask, all_masks = load_mask(mask_path, mask_value)
        if len(mask.shape) == 3:
            mask = mask[
                0
            ]  # extract first channel as we will duplicate channels later
        assert len(mask.shape) == 2
        mask = mask[..., None]
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

    if extended_output or not apply_mask:
        if dataset_task == "classification":
            image, mask, bbox = transform_image_mask_bbox(
                image, mask, bbox, image_transform, mask_transform
            )

    if not apply_mask:
        applied_mask = (image, mask)

    if extended_output:
        if metadata is None:
            metadata = {}
        else:
            assert isinstance(metadata, dict)

        metadata["mask_value"] = mask_value

        all_masks = mask_transform(all_masks)

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


def pathprefix(path, k, sep=""):
    return sep.join(path.split(os.sep)[:-k])


def get_mask_id_prefix(masks, k=2):
    assert len(masks)
    first_key = list(masks.keys())[0]
    return pathprefix(first_key, k, sep=os.sep)


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
        common_list, specific_list = split_in_common_and_specific(
            transform_config.transforms
        )
        common_transform.transforms = common_list
        specific_transform.transforms = specific_list

    else:
        assert isinstance(transform_config, dict)

        common_list, specific_list = split_in_common_and_specific(
            transform_config["transforms_list"]
        )
        common_transform["transforms_list"] = common_list
        specific_transform["transforms_list"] = specific_list

        common_transform = make_transforms(common_transform)
        specific_transform = make_transforms(specific_transform)

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


def make_bboxed_dataset_from_config(bboxed_dataset_config, transform_type):
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

    assert transform_config is not None, "transform_config is required"

    dataset_task = get_with_assert(bboxed_dataset_config, "dataset_task")
    extended_output = bboxed_dataset_config.get("extended_output", False)
    cache_path = bboxed_dataset_config.get("cache_path")
    images_list = bboxed_dataset_config.get("images_list")

    csv_path = bboxed_dataset_config.get("csv_path")
    if csv_path is None:
        raise ValueError("csv_path is required")
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
        assert train_batch_size > 0
        eval_batch_size = 0
    elif train_val_split > 0 and train_val_split < 1:
        assert eval_batch_size > 0 and train_batch_size > 0
    if train_val_split == 0:
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
        )

        train_dataset = Subset(train_bbox_dataset, train_idx)
        val_dataset = Subset(val_bbox_dataset, val_idx)

    loaders = {}

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
        num_workers=num_workers,
    )
