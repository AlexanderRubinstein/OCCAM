import sys
import os
from stuned.utility.utils import (
    get_project_root_path,
    get_with_assert,
    raise_unknown,
)
from stuned.local_datasets.utils import (
    chain_dataloaders,
    randomly_subsampled_dataloader,
)
from stuned.local_datasets.imagenet1k import (
    IMAGENET2012_CLASSES_LIST,
    get_imagenet_dataloaders,
)
from stuned.local_datasets.transforms import make_transforms


sys.path.insert(0, os.path.join(get_project_root_path()))
from occam.datasets.bboxed_dataset import get_bboxed_dataloaders
from occam.datasets.utils import make_custom_folder_dataloader
from occam.datasets.imagenet_d import get_imagenet_d_dataloaders
from occam.datasets.imagenet_9 import get_imagenet_9_dataloaders

sys.path.pop(0)


TRAIN_DATA_PERCENT_FOR_EVAL = 0.1
EVAL_ON_TRAIN_LOGS_NAME = ("random ({})-fraction" " of train samples").format(
    TRAIN_DATA_PERCENT_FOR_EVAL
)
TRAIN_SPLIT = "train"


def get_dataloaders_for_type(
    active_dataset_type, active_dataset_config, dataloader_config, logger=None
):
    def add_to_loaders(active_dataset_type, loaders, trainloader, testloaders):
        loaders[f"{active_dataset_type}_train"] = trainloader
        for key, value in testloaders.items():
            loaders[f"{active_dataset_type}_{key}"] = value

    loaders = {}

    train_batch_size = get_with_assert(dataloader_config, "train_batch_size")
    eval_batch_size = get_with_assert(dataloader_config, "eval_batch_size")
    num_workers = get_with_assert(dataloader_config, "num_workers")

    if active_dataset_type == "in1k":

        train_transform = make_transforms(
            active_dataset_config.pop("train_transform", None)
        )
        eval_transform = make_transforms(
            active_dataset_config.pop("eval_transform", None)
        )
        return_index = active_dataset_config.pop("return_index", False)
        trainloader, testloaders = get_imagenet_dataloaders(
            train_batch_size,
            eval_batch_size,
            active_dataset_config,
            train_transform=train_transform,
            eval_transform=eval_transform,
            return_index=return_index,
            num_workers=num_workers,
        )
        add_to_loaders(active_dataset_type, loaders, trainloader, testloaders)

    elif "bboxed_dataset" in active_dataset_type:
        loaders = get_bboxed_dataloaders(
            active_dataset_config,
            train_batch_size,
            eval_batch_size,
            num_workers,
            active_dataset_type,
        )
    else:
        raise_unknown(
            active_dataset_type, "active_dataset_type", "datasets_config"
        )
    return loaders


def get_dataloaders(data_config, logger=None):

    dataloader_config = get_with_assert(data_config, "dataloader")
    dataset_configs = get_with_assert(data_config, "dataset_configs")
    active_datasets = get_with_assert(data_config, "active_datasets")
    train_loaders = get_with_assert(data_config, "train_loaders")
    excluded_loaders = data_config.get("exclude_loaders", [])

    list_of_train_loaders = []

    testloaders = {}

    all_loaders = {}
    for active_dataset_type in active_datasets:
        active_dataset_config = get_with_assert(
            dataset_configs, active_dataset_type
        )
        all_loaders |= get_dataloaders_for_type(
            active_dataset_type,
            active_dataset_config,
            dataloader_config,
            logger
        )

    for excluded_loader_name in excluded_loaders:
        assert excluded_loader_name in all_loaders, (
            f"Excluded loader {excluded_loader_name} not found "
            f"in all_loaders: {all_loaders}"
        )
        assert excluded_loader_name not in train_loaders, (
            f"Excluded loader {excluded_loader_name} found "
            f"in train_loaders: {train_loaders}"
        )
        all_loaders.pop(excluded_loader_name)

    for train_loader_name in train_loaders:
        list_of_train_loaders.append(all_loaders.pop(train_loader_name))

    testloaders = all_loaders

    if len(list_of_train_loaders) > 1:
        trainloader = chain_dataloaders(
            list_of_train_loaders, random_order=True
        )
    else:
        trainloader = list_of_train_loaders[0]

    if data_config.get("use_train_for_eval", True):
        testloaders[EVAL_ON_TRAIN_LOGS_NAME] = randomly_subsampled_dataloader(
            trainloader,
            TRAIN_DATA_PERCENT_FOR_EVAL,
            batch_size=get_with_assert(dataloader_config, "eval_batch_size"),
        )

    return trainloader, testloaders


def make_dataloader(
    dataset_path,
    transform,
    batch_size=128,
    num_workers=4,
    return_path=False,
    masks=None,
    mask_transform=None,
    dataloader_type="counter_animal",
    **kwargs,
):
    if (
        dataloader_type == "counter_animal"
        or dataloader_type == "clean_counter"
        or dataloader_type == "clean_common"
    ):
        dataloader = make_custom_folder_dataloader(
            dataset_path,
            transform,
            batch_size=batch_size,
            num_workers=num_workers,
            return_path=return_path,
            masks=masks,
            mask_transform=mask_transform,
        )
    elif (
        "waterbirds" in dataloader_type
        or dataloader_type == "urban_cars"
        or dataloader_type == "in_val"
    ):
        if "waterbirds" in dataloader_type:
            assert "group" in dataloader_type  # expect "waterbirds_group_<i>"

        dataloader = make_custom_folder_dataloader(
            dataset_path,
            transform,
            batch_size=batch_size,
            num_workers=num_workers,
            return_path=return_path,
        )
    elif dataloader_type == "imagenet_9":
        dataset_config = {
            "data_dir": dataset_path,
        }
        dl, dl_name = get_imagenet_9_dataloaders(
            eval_batch_size=batch_size,
            dataset_config=dataset_config,
            num_workers=num_workers,
            eval_transform=transform,
        )
        dataloader = dl
    else:
        assert dataloader_type == "imagenet_d_bg"
        dataset_config = {"data_dir": dataset_path, "ind_types": ["background"]}
        dls = get_imagenet_d_dataloaders(
            eval_batch_size=batch_size,
            dataset_config=dataset_config,
            num_workers=num_workers,
            eval_transform=transform,
            to_map_labels=kwargs.get("to_map_labels", True),
        )
        dataloader = dls["background"]

    return dataloader


def get_imagenet_prompts():

    text = [f"A photo of {label[1]}" for label in IMAGENET2012_CLASSES_LIST]
    return text
