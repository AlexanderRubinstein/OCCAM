import sys
import os


from stuned.utility.utils import (
    get_project_root_path,
    get_with_assert,
    raise_unknown
)
from stuned.utility.logger import (
    log_or_print,
    try_to_log_in_csv,
    try_to_log_in_wandb
)
from stuned.local_datasets.utils import (
    make_default_cache_path,
    chain_dataloaders,
    randomly_subsampled_dataloader,
    # make_or_load_from_cache
)
from stuned.local_datasets.imagenet1k import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    get_imagenet_dataloaders
)
from stuned.local_datasets.transforms import (
    make_transforms
)


# local modules
sys.path.insert(
    0,
    os.path.join(
        get_project_root_path(), "src"
    )
)
import densifier
from densifier.datasets.imagenet_classes import get_in_classes_prompts
from densifier.utility.utils_for_notebooks import (
    visualize_images_side_by_side,
    tensor_for_matplotlib,
    # unnormalize,
    load_data
)
from densifier.datasets.bboxed_dataset import (
    get_bboxed_dataloaders
)
sys.path.pop(0)


TRAIN_DATA_PERCENT_FOR_EVAL = 0.1
EVAL_ON_TRAIN_LOGS_NAME = (
    "random ({})-fraction"
    " of train samples").format(
        TRAIN_DATA_PERCENT_FOR_EVAL
    )
# WATERBIRDS_KEY = "waterbirds"
# CAMELYON_17 = "camelyon17"
# WILDS_DATASETS = (WATERBIRDS_KEY, CAMELYON_17)
# UNLABELED_DATASET_KEY = "unlabeled_dataset"
TRAIN_SPLIT = "train"
# DEFAULT_MEAN = [0.485, 0.456, 0.406]
# DEFAULT_STD = [0.229, 0.224, 0.225]


def get_dataloaders_for_type(
    active_dataset_type,
    active_dataset_config,
    dataloader_config,
    logger=None
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
        # TODO(Alex | 25.09.2024): have only list of transform names here;
        # add transforms definitions dict and bootstrap transforms configs from there
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
            num_workers=num_workers
        )
        add_to_loaders(active_dataset_type, loaders, trainloader, testloaders)
        # loaders[f"{active_dataset_type}_train"] = trainloader
        # for key, value in testloaders.items():
        #     loaders[f"{active_dataset_type}_{key}"] = value
    # elif active_dataset_type == "bboxed_dataset":
    elif "bboxed_dataset" in active_dataset_type:
        loaders = get_bboxed_dataloaders(
            active_dataset_config,
            train_batch_size,
            eval_batch_size,
            num_workers,
            active_dataset_type
        )
    else:
        raise_unknown(
            active_dataset_type,
            "active_dataset_type",
            "datasets_config"
        )
    return loaders


# TODO(Alex | 25.09.2024): move this to stuned,
# ideally we need to edit only dataloader creation part "get_dataloaders_for_type"
def get_dataloaders(data_config, logger=None):

    # data_config = experiment_config["data"]
    # params_config = experiment_config["params"]
    # dataset_config = data_config["dataset"]
    # unlabeled_dataset_config = data_config.get(UNLABELED_DATASET_KEY)
    # cache_path = experiment_config["cache_path"]
    # num_readers = data_config["num_data_readers"]

    dataloader_config = get_with_assert(data_config, "dataloader")
    dataset_configs = get_with_assert(data_config, "dataset_configs")
    active_datasets = get_with_assert(data_config, "active_datasets")
    train_loaders = get_with_assert(data_config, "train_loaders")
    # excluded_loaders = get_with_assert(data_config, "exclude_loaders")
    excluded_loaders = data_config.get("exclude_loaders", [])

    # main_dataset_type = dataset_config["type"]

    # eval_only_dataset_types = dataset_config.get("eval_only_types", [])
    # additional_train_types = dataset_config.get("additional_train_types", [])

    list_of_train_loaders = []

    testloaders = {}
    # trainloader = None

    # ?? train bboxed
    # ?? eval 10% bboxed, val bboxed, val imagenet

    # ?bootstrap list of train loaders from all loaders

    all_loaders = {}
    for active_dataset_type in active_datasets:
        active_dataset_config = get_with_assert(
            dataset_configs,
            active_dataset_type
        )
        all_loaders |= get_dataloaders_for_type(
            active_dataset_type,
            active_dataset_config,
            dataloader_config,
            logger
            # dataloader_config,
            # num_readers,
            # cache_path,
            # logger,
            # eval_only=False,
            # train_only=False
        )

    for excluded_loader_name in excluded_loaders:
        assert excluded_loader_name in all_loaders, \
            f"Excluded loader {excluded_loader_name} not found " \
            f"in all_loaders: {all_loaders}"
        assert excluded_loader_name not in train_loaders, \
            f"Excluded loader {excluded_loader_name} found " \
            f"in train_loaders: {train_loaders}"
        all_loaders.pop(excluded_loader_name)

    for train_loader_name in train_loaders:
        list_of_train_loaders.append(all_loaders.pop(train_loader_name))

    testloaders = all_loaders

    if len(list_of_train_loaders) > 1:
        trainloader = chain_dataloaders(
            list_of_train_loaders,
            random_order=True
        )
    else:
        trainloader = list_of_train_loaders[0]

    if data_config.get("use_train_for_eval", True):
        testloaders[EVAL_ON_TRAIN_LOGS_NAME] \
            = randomly_subsampled_dataloader(
                trainloader,
                TRAIN_DATA_PERCENT_FOR_EVAL,
                batch_size=get_with_assert(
                    dataloader_config,
                    "eval_batch_size"
                )
            )

    return trainloader, testloaders

    ################ For future TODO(Alex | 25.09.2024):

    # "use_train_for_eval"
    # dataloader
    # dataset_configs (transforms are separately for each dataset)

    # trainloader = chain_dataloaders(
    #     list_of_train_loaders,
    #     random_order=True
    # )

    # if data_config.get("use_train_for_eval", True):
    #     testloaders[EVAL_ON_TRAIN_LOGS_NAME] \
    #         = randomly_subsampled_dataloader(
    #             trainloader,
    #             TRAIN_DATA_PERCENT_FOR_EVAL,
    #             batch_size=eval_batch_size
    #         )

    ##################

    # for cur_dataset_type in (
    #     [main_dataset_type] + additional_train_types + eval_only_dataset_types
    # ):

    #     cur_trainloader, cur_testloaders = get_dataloaders_for_type(
    #         cur_dataset_type,
    #         dataset_config,
    #         params_config,
    #         num_readers,
    #         cache_path,
    #         logger,
    #         eval_only=(cur_dataset_type in eval_only_dataset_types),
    #         train_only=(cur_dataset_type in additional_train_types)
    #     )

    #     if cur_trainloader is not None:
    #         list_of_train_loaders.append(cur_trainloader)

    #     if cur_testloaders is not None:
    #         testloaders |= cur_testloaders

    # if len(list_of_train_loaders) == 0:
    #     trainloader = None
    # elif len(list_of_train_loaders) == 1:
    #     trainloader = list_of_train_loaders[0]
    # else:
    #     trainloader = chain_dataloaders(
    #         list_of_train_loaders,
    #         random_order=True
    #     )

    # # add train subset into test dataloaders
    # if trainloader is not None:

    #     eval_batch_size = trainloader.batch_size

    #     if len(testloaders) > 0:
    #         testloader = next(iter(testloaders.values()))
    #         if testloader is not None:
    #             eval_batch_size = testloader.batch_size

    #     if dataset_config.get("use_train_for_eval", True):
    #         testloaders[EVAL_ON_TRAIN_LOGS_NAME] \
    #             = randomly_subsampled_dataloader(
    #                 trainloader,
    #                 TRAIN_DATA_PERCENT_FOR_EVAL,
    #                 batch_size=eval_batch_size
    #             )

    # assert trainloader or testloaders, \
    #     "Both trainloader and testloaders are None"

    # if unlabeled_dataset_config is None:
    #     unlabeled_loaders = None
    # else:
    #     unlabeled_dataset_type = get_with_assert(
    #         unlabeled_dataset_config,
    #         "type"
    #     )
    #     unlabeled_dataset_split = get_with_assert(
    #         unlabeled_dataset_config,
    #         "split"
    #     )

    #     split_is_train = (unlabeled_dataset_split == TRAIN_SPLIT)

    #     unlabeled_loaders = {}
    #     unlabeled_trainloader, unlabeled_testloaders = get_dataloaders_for_type(
    #         unlabeled_dataset_type,
    #         unlabeled_dataset_config,
    #         params_config,
    #         num_readers,
    #         cache_path,
    #         logger,
    #         eval_only=(not split_is_train),
    #         train_only=split_is_train,
    #         unlabeled=True
    #     )
    #     if unlabeled_dataset_split == TRAIN_SPLIT:
    #         unlabeled_loaders[unlabeled_dataset_split] = unlabeled_trainloader
    #     else:
    #         unlabeled_loaders[unlabeled_dataset_split] = get_with_assert(
    #             unlabeled_testloaders,
    #             unlabeled_dataset_split
    #         )

    # return trainloader, testloaders, unlabeled_loaders


# def unnormalize_in1k(image):
#     return unnormalize(
#         image,
#         DEFAULT_MEAN,
#         DEFAULT_STD
#     )
