import os
import sys
import argparse
import torch
import torchvision
import types
import timm
import pandas as pd


sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "src"
    )
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "notebooks"
    )
)
from debiasing import (
    COUNTER_PATH,
    COMMON_PATH,
    VIT_L_EVAL_TRANSFORM_CONFIG,
    make_df_with_foreground_scores,
    eval_models,
    load_model,
    add_lle_model
)
from densifier.datasets.imagenet_d import get_in_d_category_list
from densifier.datasets.imagenet_classes import get_in_classes_prompts
from densifier.eval_clip.eval import (
    add_openai_clip_model,
    add_alpha_clip_model,
)
from densifier.datasets.waterbirds import (
    get_clip_wb_category_list
)
from densifier.datasets.urban_cars import (
    get_clip_uc_category_list
)
from densifier.datasets.counter_animal import (
    make_counter_animal_categories
)
sys.path.pop(0)

from densifier.datasets.imagenet_9 import (
    get_in_9_category_list
)
sys.path.pop(0)


from stuned.utility.utils import (
    # show_images,
    # load_from_pickle,
    # append_dict,
    get_project_root_path,
    raise_unknown
    # get_with_assert
)


CACHE_PATH = "/mnt/lustre/work/oh/arubinstein17/cache"


def get_parser():
    parser = argparse.ArgumentParser(description="add background scores and eval on Urban Cars")
    parser.add_argument(
        "--result_path",
        default="/home/oh/arubinstein17/github/densification/data/results/for_runner_eval_uc.pth",
        help="where to save the results"
    )
    parser.add_argument(
        "--recompute_all",
        action="store_true",
        help="recompute all results"
    )
    parser.add_argument(
        "--dataset_name",
        default="urban_cars",
        help="dataset name"
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="use clip"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size"
    )
    parser.add_argument(
        "--mask_source",
        default="cropformer",
        help="mask source",
        choices=["cropformer", "dino_ft"]
    )
    return parser


def add_clip_models(models_dict, category_list):
    add_openai_clip_model('ViT-L/14', category_list, models_dict)
    add_alpha_clip_model("ViT-L/14", category_list, models_dict)


def main():

    args = get_parser().parse_args()

    _models_dict = {}

    parquets = {}

    _parquet_kwargs = {}

    split_images_masks = []

    _category_list = None

    batch_size = args.batch_size

    data_path = os.path.join(get_project_root_path(), "data")

    parquets_base_dir = os.path.join(data_path, "csvs")

    masks_base_dir = os.path.join(data_path, "masks")

    separate_masks_base_dir = os.path.join(data_path, "separate_masks")

    clean_dataloader_kwargs = {
        "clean_type": args.dataset_name,
    }

    if args.dataset_name == "urban_cars":

        if args.clip:

            _category_list = get_clip_uc_category_list()

            _parquet_kwargs["mapper"] = "urban_cars_clip"
        else:
            raise NotImplementedError("This function was not tested yet")
            add_lle_model(
                arch='resnet50',
                models_dict=_models_dict,
                ckpt_fpath="/home/oh/arubinstein17/github/Whac-A-Mole/exp/urbancars/lle_es_both_urbancars/seed_0/best.pth"
            )

        parquets = {}

        _df_path = os.path.join(parquets_base_dir, f"source_urban_cars_{args.mask_source}.parquet")

        _images_path = os.path.join(CACHE_PATH, "UrbanCars", "test")
        _masks_path = os.path.join(masks_base_dir, f"UrbanCars_test_masks_{args.mask_source}.pkl")

        _separate_masks_folder = os.path.join(separate_masks_base_dir, args.mask_source, "UrbanCars", "test")
        split_images_masks.append((args.dataset_name, _images_path, _masks_path, _separate_masks_folder))
        parquets[args.dataset_name] = (_df_path, _parquet_kwargs)

    elif args.dataset_name == "counter_animal":
        _separate_masks_folder = os.path.join(separate_masks_base_dir, args.mask_source, "CounterAnimal")
        if args.clip:

            _category_list = get_in_classes_prompts()
            # _category_list = make_counter_animal_categories() # should we use this or original IN-classes?
            # _parquet_kwargs["mapper"] = "counter_animal_clip" # comment this out when use original IN-classes

        else:
            raise NotImplementedError()

        for split in ["counter", "common"]:
            if split == "counter":
                _images_path = COUNTER_PATH

            elif split == "common":
                _images_path = COMMON_PATH

            _masks_path = os.path.join(masks_base_dir, f"{split}_masks_{args.mask_source}.pkl")
            _separate_masks_folder = os.path.join(_separate_masks_folder, split)
            _df_path = os.path.join(parquets_base_dir, f"source_counter_animal_{split}_{args.mask_source}.parquet")
            parquets[split] = (_df_path, _parquet_kwargs)
            split_images_masks.append((split, _images_path, _masks_path, _separate_masks_folder))

    elif args.dataset_name == "waterbirds":
        if args.clip:
            _category_list = get_clip_wb_category_list()
            _parquet_kwargs["mapper"] = "waterbirds_clip"
        else:
            raise NotImplementedError()

        for group_id in range(4):
            parquet_name = f"waterbirds_group_{group_id}"
            parquets[parquet_name] = (
                os.path.join(parquets_base_dir, f"source_waterbirds_group_{group_id}_{args.mask_source}.parquet"),
                _parquet_kwargs
            )
            split_images_masks.append(
                (
                    parquet_name,
                    os.path.join(CACHE_PATH, "Waterbirds", "test_split", f"group_{group_id}"),
                    os.path.join(masks_base_dir, f"Waterbirds_test_group{group_id}_masks_{args.mask_source}.pkl"),
                    os.path.join(separate_masks_base_dir, "waterbirds", f"group_{group_id}")
                )
            )

    elif args.dataset_name == "imagenet_9":
        parquet_name = "imagenet_9_mix_rand"
        if args.clip:
            _category_list = get_in_classes_prompts()
            _parquet_kwargs["mapper"] = "in9"
            clean_dataloader_kwargs["clean_type"] = parquet_name
            # clean_dataloader_kwargs |= _parquet_kwargs
        else:
            raise NotImplementedError()

        split_images_masks.append(
            (
                parquet_name,
                os.path.join(CACHE_PATH, "background_challenge", "bg_challenge", "mixed_rand", "val"),
                os.path.join(masks_base_dir, f"ImageNet9_mixed_random_masks_{args.mask_source}.pkl"),
                os.path.join(separate_masks_base_dir, "imagenet_9", "mixed_rand")
            )
        )
        parquets[parquet_name] = (
            os.path.join(parquets_base_dir, f"source_imagenet_9_mix_rand_{args.mask_source}.parquet"),
            _parquet_kwargs
        )

    elif args.dataset_name == "imagenet_d":

        parquet_name = "imagenet_d_bg"
        if args.clip:
            _category_list, _, _ = get_in_d_category_list()
            clean_dataloader_kwargs["clean_type"] = parquet_name
        else:
            raise NotImplementedError()
        clean_dataloader_kwargs["to_map_labels"] = False

        split_images_masks.append(
            (
                parquet_name,
                (os.path.join(CACHE_PATH, "ImageNet-D", "ImageNet-D", "background"), {"to_map_labels": False}),
                os.path.join(masks_base_dir, f"ImageNetD_masks_{args.mask_source}.pkl"),
                os.path.join(separate_masks_base_dir, "imagenet_d", "background")
            )
        )
        parquets[parquet_name] = (
            os.path.join(parquets_base_dir, f"source_imagenet_d_bg_{args.mask_source}.parquet"),
            _parquet_kwargs
        )

    else:
        raise raise_unknown("dataset_name", args.dataset_name, "")

    clean_dataloader_kwargs |= _parquet_kwargs # add mappers to clean_dataloader_kwargs

    if args.clip:
        assert _category_list is not None
        add_clip_models(_models_dict, _category_list)

    for split, images_path, masks_path, separate_masks_folder in split_images_masks:

        assert os.path.exists(masks_path), f"masks path {masks_path} does not exist"

        make_df_with_foreground_scores(
            source_df_path=parquets[split], # attention to mappers
            images_path=images_path,
            masks_path=masks_path,
            separate_masks_folder=separate_masks_folder,
            models_dict=_models_dict,
            foreground_detectors=("oracle", "max_prob"),
            batch_size=args.batch_size,
            dataset_name=args.dataset_name,
            recompute_all=args.recompute_all
        )

    eval_models(
        parquets=parquets,
        models=_models_dict,
        fg_detectors=["oracle", "max_prob", None],
        full_res_save_path=args.result_path,
        batch_size=batch_size,
        clean_dataloader_kwargs=clean_dataloader_kwargs, # attention to mappers
        recompute_all=args.recompute_all
    )

    df = convert_to_table(args.result_path, args.dataset_name)
    print(df)


def convert_to_table(result_path, dataset_name):
    res_dict = torch.load(result_path)

    sort_by = None
    column_names = None

    if dataset_name == "urban_cars":
        df_dict = {}
        for key, value in res_dict.items():
            df_dict[key] = [value["test_both_worst_group_acc"].item()]
        column_names = ["both_worst_group_acc"]

    elif dataset_name in ["counter_animal", "imagenet_d", "imagenet_9"]:

        df_dict = {key: [value] for key, value in res_dict.items()}

        if dataset_name == "counter_animal":
            for key, value in df_dict.items():
                if "common" in key:
                    dataset_type = "common"
                else:
                    assert "counter" in key
                    dataset_type = "counter"
                df_dict[key] = {"acc": value[0], "dataset_type": dataset_type}

            sort_by = "dataset_type"
        else:
            column_names = ["acc"]

    elif dataset_name == "waterbirds":

        df_dict = {}
        for key, value in res_dict.items():
            split = key.split("group_")
            new_key = f"{split[0]}{split[1][1:]}"
            value = float(value)
            if new_key not in df_dict:
                df_dict[new_key] = [value]
            elif value < df_dict[new_key][0]:
                df_dict[new_key] = [value]

        column_names = ["worst_group_acc"]
    else:
        raise_unknown("dataset_name", dataset_name, "")

    df = pd.DataFrame(df_dict).T
    if column_names is not None:
        df.columns = column_names
    if sort_by is not None:
        df = df.sort_values(by=sort_by)
    return df

if __name__ == "__main__":
    main()
