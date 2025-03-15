# import wget
import os
import sys
# import shutil
import torch
from tqdm import tqdm
# import xml.etree.ElementTree as ET
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
    # PrecisionRecallDisplay,
    # roc_auc_score,
    # roc_curve
# )
import matplotlib.pyplot as plt
# import pandas as pd
# from PIL import Image
# import open_clip
from stuned.utility.utils import (
    get_project_root_path,
    load_from_pickle
)
from stuned.local_datasets.imagenet1k import (
    IMAGENET2012_CLASSES_LIST,
    # IMAGENET2012_CLASSES,
    # get_imagenet_dataloaders
)


sys.path.insert(
    0,
    os.path.join(
        get_project_root_path()
    )
)
# import occam
from occam.datasets.utils import (
    open_pil_image,
    make_custom_folder_path2label
)
from occam.datasets.imagenet_classes import get_in_classes_prompts
from occam.datasets.bboxed_dataset import (
    make_bboxed_dataset_from_config,
    get_mask_id_prefix,
    make_bbox,
    compute_bbox_fit_score,
    subpath
)
from occam.datasets.utils import (
    DATASETS_PATH,
    # make_to_classes_mapping,
    make_model_classes_wrapper,
    # torch_max_func,
)
# from densifier.eval_clip.eval import (
#     COUNTER_ANIMAL_CLASSES, # tmp - jsut sort and store here
#     IMAGE_NORMALIZATION_CONST,
#     # CustomImageFolder,
#     apply_visual_prompts,
#     _build_timm_model,
#     is_background,
#     make_dataloader,
#     get_imagenet_prompts,
#     get_text_probs
# )
sys.path.pop(0)


COUNTER_ANIMAL_CLASSES_LIST = [9, 10, 16, 20, 23, 30, 33, 37, 39, 41, 42, 49, 54, 56, 57, 58, 70, 71, 76, 79, 80, 81, 83, 89, 100, 102, 128, 130, 133, 144, 150, 275, 276, 277, 279, 290, 291, 293, 296, 305, 316, 337, 349, 357, 360]
COUNTER_ANIMAL_DATASET_PATH = os.path.join(DATASETS_PATH, "CounterAnimal")
COUNTER_PATH = os.path.join(COUNTER_ANIMAL_DATASET_PATH, "counter")
COMMON_PATH = os.path.join(COUNTER_ANIMAL_DATASET_PATH, "common")


def make_path2label_counter_animal(dataset_path): # for counter animal
    # # dataset_path = "/home/oh/arubinstein17/github/densification/data/CounterAnimal/symlinked/counter"
    # transform = None
    # return_path = True
    # masks = None
    # mask_transform = None

    # dataset = CustomImageFolder(
    #     dataset_path,
    #     transform=transform,
    #     return_path=return_path,
    #     masks=masks,
    #     mask_transform=mask_transform
    # )

    # # res = []
    # # for item in tqdm(dataset):
    # #     res.append([item[2], item[1]])
    # # return res
    # return dataset.samples
    return make_custom_folder_path2label(dataset_path) # TODO(Alex | 22.12.2024): check that it works with counter_animal


def make_mask_name_from_path(path):
    return path.replace(os.sep, "@") + ".mask"

# mapping dict for CounterAnimal
def make_mapping_dict_from_folder(
    path2label,
    masks_path,
    separate_masks_folder,
    bboxes_path,
    assert_shape=False
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
    # def subpath(path, k):
    #     return "".join(path.split(os.sep)[-k:])

    res = {}
    masks = load_from_pickle(masks_path)
    renamed_masks = {}
    for key, value in masks.items():
        renamed_masks[subpath(key, 2)] = value

    masks = renamed_masks
    # assert bboxes_path is None, "Not implemented"
    os.makedirs(separate_masks_folder, exist_ok=True)
    for path, label in tqdm(path2label):
        # mask_id = path[1:]
        mask_id = subpath(path, 2)

        # print(mask_id)
        # print(masks.keys())

        assert mask_id in masks
        # key = path

        mask = masks[mask_id]["mask"]

        if assert_shape:
            image = open_pil_image(path)
            # print("image:", image.shape)
            # print("mask:", mask.shape)

            assert image.shape[:-1] == mask.shape

        if bboxes_path is None:
            bbox_path = None
        else:
            assert False, "Not implemented"

        mask_path = os.path.join(
            separate_masks_folder,
            # os.path.basename(path).split(".")[0] + ".mask"
            make_mask_name_from_path(path)
        )

        torch.save(mask, mask_path)
        res[path] = (mask_path, bbox_path, label)

    return res


def make_counter_animal_categories():
    in_classes_prompts = get_in_classes_prompts()
    # counter_animal_classes = sorted(COUNTER_ANIMAL_CLASSES)
    return [in_classes_prompts[i] for i in COUNTER_ANIMAL_CLASSES_LIST]


def make_counter_animal_clip_mapper():

    def mapper(logits):
        logits_shape = list(logits.shape)
        new_logits_shape = logits_shape[:-1] + [len(IMAGENET2012_CLASSES_LIST)]
        # for IN classes that are not counter animal we have -inf
        new_logits = (torch.ones(new_logits_shape) * -float("inf")).to(logits.device).to(logits.dtype)
        # for counter animal classes we have the original logits
        new_logits[:, COUNTER_ANIMAL_CLASSES_LIST] = logits
        return new_logits
    # should return correct order of classes after self.mapper(logits)
    return mapper


def make_counter_animal_clip_wrapper(model):
    return make_model_classes_wrapper(model, make_counter_animal_clip_mapper)


def make_mapping_dict_counter_animal_(images_folder, masks_path, separate_masks_folder):
    path2label = make_path2label_counter_animal(images_folder)
    # path2label_counter = make_path2label_counter_animal("/home/oh/arubinstein17/github/densification/data/CounterAnimal/symlinked/counter_mislabeled_siglip")

    # mapping_dict_counter = make_mapping_dict(
    mapping_dict = make_mapping_dict_from_folder(
        path2label=path2label,
        masks_path=masks_path,
        separate_masks_folder=separate_masks_folder,
        bboxes_path=None,
        # assert_shape=True
    )
    return mapping_dict
