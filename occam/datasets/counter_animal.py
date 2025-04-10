import os
import sys
import torch
from stuned.utility.utils import get_project_root_path, load_from_pickle
from stuned.local_datasets.imagenet1k import (
    IMAGENET2012_CLASSES_LIST,
)


sys.path.insert(0, os.path.join(get_project_root_path()))
from occam.datasets.utils import open_pil_image, make_custom_folder_path2label
from occam.datasets.imagenet_classes import get_in_classes_prompts
from occam.datasets.utils import (
    DATASETS_PATH,
    make_model_classes_wrapper,
    make_mapping_dict_from_folder,
)
sys.path.pop(0)


COUNTER_ANIMAL_CLASSES_LIST = [
    9,
    10,
    16,
    20,
    23,
    30,
    33,
    37,
    39,
    41,
    42,
    49,
    54,
    56,
    57,
    58,
    70,
    71,
    76,
    79,
    80,
    81,
    83,
    89,
    100,
    102,
    128,
    130,
    133,
    144,
    150,
    275,
    276,
    277,
    279,
    290,
    291,
    293,
    296,
    305,
    316,
    337,
    349,
    357,
    360,
]
COUNTER_ANIMAL_DATASET_PATH = os.path.join(DATASETS_PATH, "CounterAnimal")
COUNTER_PATH = os.path.join(COUNTER_ANIMAL_DATASET_PATH, "counter")
COMMON_PATH = os.path.join(COUNTER_ANIMAL_DATASET_PATH, "common")


def make_path2label_counter_animal(dataset_path):  # for counter animal

    return make_custom_folder_path2label(
        dataset_path
    )


def make_mask_name_from_path(path):
    return path.replace(os.sep, "@") + ".mask"


def make_counter_animal_categories():
    in_classes_prompts = get_in_classes_prompts()
    return [in_classes_prompts[i] for i in COUNTER_ANIMAL_CLASSES_LIST]


def make_counter_animal_clip_mapper():
    def mapper(logits):
        logits_shape = list(logits.shape)
        new_logits_shape = logits_shape[:-1] + [len(IMAGENET2012_CLASSES_LIST)]
        # for IN classes that are not counter animal we have -inf
        new_logits = (
            (torch.ones(new_logits_shape) * -float("inf"))
            .to(logits.device)
            .to(logits.dtype)
        )
        # for counter animal classes we have the original logits
        new_logits[:, COUNTER_ANIMAL_CLASSES_LIST] = logits
        return new_logits

    # should return correct order of classes after self.mapper(logits)
    return mapper


def make_counter_animal_clip_wrapper(model):
    return make_model_classes_wrapper(model, make_counter_animal_clip_mapper)


def make_mapping_dict_counter_animal_(
    images_folder, masks_path, bboxes_path, separate_masks_folder
):
    path2label = make_path2label_counter_animal(images_folder)

    mapping_dict = make_mapping_dict_from_folder(
        path2label=path2label,
        masks_path=masks_path,
        separate_masks_folder=separate_masks_folder,
        bboxes_path=None,
    )
    return mapping_dict
