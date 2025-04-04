import os
import sys
import argparse
import torch

# import torchvision
# import types
# import timm
import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
from occam.robust_classification.eval import (
    ENCODED_NAME_SEP,
    make_df_with_foreground_scores,
    eval_models,
)
from occam.robust_classification.models import (
    ModelBuilder,
    add_openai_clip_model,
    add_alpha_clip_model,
    add_openclip_model,
    # make_clip_ensemble,
    get_clip_ensemble_builder,
)
from occam.datasets.imagenet_d import (
    IN_D_PATH,
    get_in_d_category_list,
)
from occam.datasets.imagenet_classes import get_in_classes_prompts
from occam.datasets.waterbirds import (
    WATERBIRDS_PATHS,
    get_clip_wb_category_list,
)
from occam.datasets.urban_cars import URBAN_CARS_PATH, get_clip_uc_category_list
from occam.datasets.counter_animal import (
    COUNTER_PATH,
    COMMON_PATH,
    # make_counter_animal_categories
)
from occam.datasets.imagenet_9 import (
    IMAGENET_9_PATH,
    # get_in_9_category_list
)

sys.path.pop(0)


from stuned.utility.utils import (
    # show_images,
    # load_from_pickle,
    # append_dict,
    get_project_root_path,
    raise_unknown,
    # get_with_assert
)


MAX_COL_WIDTH = 1000


def get_parser():
    parser = argparse.ArgumentParser(
        description="add background scores and eval on spurious backgrounds datasets"
    )
    parser.add_argument(
        "--result_path",
        default="/home/oh/arubinstein17/github/densification/data/results/for_runner_eval_uc.pth",
        help="where to save the results",
    )
    parser.add_argument(
        "--recompute_all", action="store_true", help="recompute all results"
    )
    parser.add_argument(
        "--dataset_name", default="urban_cars", help="dataset name"
    )
    parser.add_argument("--clip", action="store_true", help="use clip")
    parser.add_argument("--siglip", action="store_true", help="use siglip")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size"
    )
    parser.add_argument(
        "--mask_source",
        default="cropformer",
        help="mask source",
        choices=["cropformer", "dino_ft"],
    )
    parser.add_argument(
        "--ens_entropy",
        action="store_true",
        help="use ens entropy as foreground score",
    )
    parser.add_argument(
        "--filter_keyword",
        default=None,
        help="filter keyword for filtering masks",
    )
    return parser


def add_clip_models(models_dict, category_list, dataset_name, siglip=False):
    if dataset_name == "counter_animal_gap":
        add_openai_clip_model("RN50", category_list, models_dict)
        add_openai_clip_model("RN101", category_list, models_dict)
        add_openai_clip_model("RN50x4", category_list, models_dict)
        add_openai_clip_model("RN50x16", category_list, models_dict)
        add_openai_clip_model("RN50x64", category_list, models_dict)
        add_openai_clip_model("ViT-B/32", category_list, models_dict)
        add_openai_clip_model("ViT-B/16", category_list, models_dict)
        add_openai_clip_model("ViT-L/14", category_list, models_dict)
        add_openai_clip_model("ViT-L/14@336px", category_list, models_dict)

        # 'ViT-B-16'
        add_openclip_model(
            model_id="ViT-B-16",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="laion400m_e32",
        )
        add_openclip_model(
            model_id="ViT-B-16",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="datacomp_l_s1b_b8k",
        )
        add_openclip_model(
            model_id="ViT-B-16",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="laion2b_s34b_b88k",
        )
        add_openclip_model(
            model_id="ViT-B-16",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="dfn2b",
        )

        # 'ViT-B-32'
        add_openclip_model(
            model_id="ViT-B-32",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="laion400m_e32",
        )
        add_openclip_model(
            model_id="ViT-B-32",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="datacomp_s_s13m_b4k",
        )
        add_openclip_model(
            model_id="ViT-B-32",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="laion2b_s34b_b79k",
        )
        add_openclip_model(
            model_id="ViT-B-32-256",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="datacomp_s34b_b86k",
        )
        # add_openclip_model(
        #     model_id='ViT-B-32',
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained='dfn2b'
        # )

        # 'ViT-L-14'
        add_openclip_model(
            model_id="ViT-L-14",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="laion400m_e32",
        )
        add_openclip_model(
            model_id="ViT-L-14",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="datacomp_xl_s13b_b90k",
        )
        add_openclip_model(
            model_id="ViT-L-14",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="laion2b_s32b_b82k",
        )
        add_openclip_model(
            model_id="ViT-L-14-quickgelu",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="dfn2b",
        )

        # 'ViT-H-14'
        add_openclip_model(
            model_id="ViT-H-14",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="laion2b_s32b_b79k",
        )
        add_openclip_model(
            model_id="ViT-H-14-quickgelu",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="dfn5b",
        )
        add_openclip_model(
            model_id="ViT-H-14-378-quickgelu",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="dfn5b",
        )

        # 'ViT-G-14'
        add_openclip_model(
            model_id="ViT-g-14",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="laion2b_s34b_b88k",
        )
        add_openclip_model(
            model_id="ViT-bigG-14",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="laion2b_s39b_b160k",
        )

        # 'ConvNext-B'
        add_openclip_model(
            model_id="convnext_base",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="laion400m_s13b_b51k",
        )
        add_openclip_model(
            model_id="convnext_base_w",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="laion2b_s13b_b82k",
        )

    else:
        add_openai_clip_model("ViT-L/14", category_list, models_dict)
        add_alpha_clip_model("ViT-L/14", category_list, models_dict)
        add_openai_clip_model("RN50", category_list, models_dict)
        # print("Uncomment 2 above pls")
    if siglip:
        add_openclip_model(
            model_id="ViT-SO400M-14-SigLIP-384",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="webli",
        )
        # add_openclip_model(
        #     model_id='nllb-clip-base-siglip',
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained='v1'
        # )
        # add_openclip_model(
        #     model_id='nllb-clip-base-siglip',
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained='mrl'
        # )
        # add_openclip_model(
        #     model_id='nllb-clip-large-siglip',
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained='v1'
        # )
        # add_openclip_model(
        #     model_id='nllb-clip-large-siglip',
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained='mrl'
        # )
        # add_openclip_model(
        #     model_id='ViT-bigG-14',
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained='laion2b_s39b_b160k'
        # )
        add_openclip_model(
            model_id="ViT-B-16-SigLIP",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="webli",
        )
        add_openclip_model(
            model_id="ViT-B-16-SigLIP-256",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="webli",
        )
        add_openclip_model(
            model_id="ViT-B-16-SigLIP-i18n-256",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="webli",
        )
        add_openclip_model(
            model_id="ViT-B-16-SigLIP-384",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="webli",
        )
        # add_openclip_model(
        #     model_id='ViT-B-16-SigLIP-512',
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained='webli'
        # )
        add_openclip_model(
            model_id="ViT-L-16-SigLIP-256",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="webli",
        )
        add_openclip_model(
            model_id="ViT-L-16-SigLIP-384",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="webli",
        )
        add_openclip_model(
            model_id="ViT-SO400M-14-SigLIP",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="webli",
        )
        add_openclip_model(
            model_id="ViT-SO400M-14-SigLIP-384",
            category_list=category_list,
            models_dict=models_dict,
            pretrained="webli",
        )


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

    if args.dataset_name == "counter_animal_gap":
        fg_detectors = ["oracle"]
    elif args.dataset_name == "in_val_with_bboxes":
        fg_detectors = ["oracle", "max_prob", "bbox_iou"]
    else:
        fg_detectors = ["oracle", "max_prob"]

    if args.ens_entropy:
        assert args.clip
        fg_detectors += ["ens_entropy"]

    clean_dataloader_kwargs = {
        "clean_type": args.dataset_name,
    }

    if args.siglip:
        assert args.clip

    if args.dataset_name == "urban_cars":
        if args.clip:
            _category_list = get_clip_uc_category_list()

            _parquet_kwargs["mapper"] = "urban_cars_clip"
        else:
            raise NotImplementedError("This function was not tested yet")
            # add_lle_model(
            #     arch='resnet50',
            #     models_dict=_models_dict,
            #     ckpt_fpath="/home/oh/arubinstein17/github/Whac-A-Mole/exp/urbancars/lle_es_both_urbancars/seed_0/best.pth"
            # )

        parquets = {}

        _df_path = os.path.join(
            parquets_base_dir,
            args.mask_source,
            f"source_urban_cars_{args.mask_source}_{args.filter_keyword}.parquet",
        )

        _images_path = URBAN_CARS_PATH
        _masks_path = os.path.join(
            masks_base_dir, args.mask_source, f"UC_masks.pkl"
        )

        _separate_masks_folder = os.path.join(
            separate_masks_base_dir, args.mask_source, "UrbanCars", "test"
        )
        split_images_masks.append(
            (
                args.dataset_name,
                _images_path,
                _masks_path,
                _separate_masks_folder,
            )
        )
        parquets[args.dataset_name] = (_df_path, _parquet_kwargs)

    elif args.dataset_name == "in_val_with_bboxes":
        if args.clip:
            _category_list = get_in_classes_prompts()

        else:
            raise NotImplementedError()

        clean_dataloader_kwargs["clean_type"] = "in_val"

        _separate_masks_folder = os.path.join(
            separate_masks_base_dir, args.mask_source, "in_val_with_bboxes"
        )
        _images_path = "/mnt/lustre/datasets/ImageNet2012/val/"
        _masks_path = os.path.join(
            masks_base_dir, f"in_val_masks_{args.mask_source}.pkl"
        )
        _bboxes_path = os.path.join(data_path, "bboxes_annotations", "val")
        _masks_path = (_masks_path, _bboxes_path)
        split_images_masks.append(
            (
                args.dataset_name,
                _images_path,
                _masks_path,
                _separate_masks_folder,
            )
        )
        _df_path = os.path.join(
            parquets_base_dir,
            args.mask_source,
            f"source_in_val_with_bboxes_{args.mask_source}_{args.filter_keyword}.parquet",
        )
        parquets[args.dataset_name] = (_df_path, _parquet_kwargs)

    elif (
        args.dataset_name == "counter_animal"
        or args.dataset_name == "counter_animal_gap"
    ):
        _separate_masks_folder = os.path.join(
            separate_masks_base_dir, args.mask_source, "CounterAnimal"
        )
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

            _masks_path = os.path.join(
                masks_base_dir, args.mask_source, f"CA_{split}_masks.pkl"
            )
            _separate_masks_folder = os.path.join(_separate_masks_folder, split)
            _df_path = os.path.join(
                parquets_base_dir,
                args.mask_source,
                f"source_counter_animal_{split}_{args.filter_keyword}.parquet",
            )
            parquets[split] = (_df_path, _parquet_kwargs)
            split_images_masks.append(
                (split, _images_path, _masks_path, _separate_masks_folder)
            )

    elif args.dataset_name == "waterbirds":
        if args.clip:
            _category_list = get_clip_wb_category_list()
            _parquet_kwargs["mapper"] = "waterbirds_clip"
        else:
            raise NotImplementedError()

        for group_id in range(len(WATERBIRDS_PATHS)):
            parquet_name = f"waterbirds_group_{group_id}"
            parquets[parquet_name] = (
                os.path.join(
                    parquets_base_dir,
                    args.mask_source,
                    f"source_waterbirds_group_{group_id}_{args.filter_keyword}.parquet",
                ),
                _parquet_kwargs,
            )
            split_images_masks.append(
                (
                    parquet_name,
                    WATERBIRDS_PATHS[group_id],
                    os.path.join(
                        masks_base_dir,
                        args.mask_source,
                        f"WB_group_{group_id}_masks.pkl",
                    ),
                    os.path.join(
                        separate_masks_base_dir,
                        args.mask_source,
                        "waterbirds",
                        f"group_{group_id}",
                    ),
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
                # os.path.join(CACHE_PATH, "background_challenge", "bg_challenge", "mixed_rand", "val"),
                IMAGENET_9_PATH,
                os.path.join(
                    masks_base_dir,
                    args.mask_source,
                    f"IN_9_masks.pkl",
                ),
                os.path.join(
                    separate_masks_base_dir,
                    args.mask_source,
                    "imagenet_9",
                    "mixed_rand",
                ),
            )
        )
        parquets[parquet_name] = (
            os.path.join(
                parquets_base_dir,
                args.mask_source,
                f"source_imagenet_9_mix_rand_{args.mask_source}_{args.filter_keyword}.parquet",
            ),
            _parquet_kwargs,
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
                (
                    os.path.join(IN_D_PATH, "background"),
                    {"to_map_labels": False},
                ),
                os.path.join(
                    masks_base_dir,
                    args.mask_source,
                    f"IN_D_background_masks.pkl",
                ),
                os.path.join(
                    separate_masks_base_dir,
                    args.mask_source,
                    "imagenet_d",
                    "background",
                ),
            )
        )
        parquets[parquet_name] = (
            os.path.join(
                parquets_base_dir,
                args.mask_source,
                f"source_imagenet_d_bg_{args.mask_source}_{args.filter_keyword}.parquet",
            ),
            _parquet_kwargs,
        )

    else:
        raise raise_unknown("dataset_name", args.dataset_name, "")

    clean_dataloader_kwargs |= (
        _parquet_kwargs  # add mappers to clean_dataloader_kwargs
    )

    if args.clip:
        assert _category_list is not None
        add_clip_models(
            models_dict=_models_dict,
            category_list=_category_list,
            dataset_name=args.dataset_name,
            siglip=args.siglip,
        )

    if args.ens_entropy:
        assert args.clip
        # clip_ensemble = make_clip_ensemble(_category_list)
        # _models_dict["clip_ensemble"] = (
        #     clip_ensemble,
        #     clip_ensemble.preprocess,
        # )
        _models_dict["clip_ensemble"] = ModelBuilder(
            get_clip_ensemble_builder(_category_list)
        )
        # raise NotImplementedError("Do we return preprocess as second arg for others?")

    for (
        split,
        images_path,
        masks_path,
        separate_masks_folder,
    ) in split_images_masks:
        if isinstance(masks_path, tuple):
            real_masks_path, real_bboxes_path = masks_path
        else:
            real_masks_path = masks_path
            real_bboxes_path = None

        assert os.path.exists(
            real_masks_path
        ), f"masks path {real_masks_path} does not exist"
        if real_bboxes_path is not None:
            assert os.path.exists(
                real_bboxes_path
            ), f"bboxes path {real_bboxes_path} does not exist"

        make_df_with_foreground_scores(
            source_df_path=parquets[split],  # attention to mappers
            images_path=images_path,
            masks_path=real_masks_path,
            bboxes_path=real_bboxes_path,
            separate_masks_folder=separate_masks_folder,
            models_dict=_models_dict,
            foreground_detectors=fg_detectors,
            batch_size=args.batch_size,
            dataset_name=args.dataset_name,
            recompute_all=args.recompute_all,
            filter_keyword=args.filter_keyword,
        )

    if args.ens_entropy:
        _models_dict.pop(
            "clip_ensemble"
        )  # was needed only to compute ens_entropy scores

    eval_models(
        parquets=parquets,
        models=_models_dict,
        fg_detectors=fg_detectors + [None],
        full_res_save_path=args.result_path,
        batch_size=batch_size,
        clean_dataloader_kwargs=clean_dataloader_kwargs,  # attention to mappers
        recompute_all=args.recompute_all,
        filter_keyword=args.filter_keyword,
    )

    df = convert_to_table(args.result_path, args.dataset_name)
    pd.set_option(
        "display.max_colwidth", MAX_COL_WIDTH
    )  # to see long model names
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

    elif dataset_name in [
        "counter_animal",
        "imagenet_d",
        "imagenet_9",
        "in_val_with_bboxes",
    ]:
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

    elif dataset_name == "counter_animal_gap":
        df_dict = {}
        for key, value in res_dict.items():
            key_split = key.split(ENCODED_NAME_SEP)
            assert len(key_split) == 3
            model_id = key_split[-1]
            fg_detector = key_split[1]
            new_key = model_id
            subset_type = key_split[0].replace("clean_", "")
            col_name = f"{subset_type}{ENCODED_NAME_SEP}{fg_detector}"
            cur_value = df_dict.get(new_key, {})
            cur_value[col_name] = value
            df_dict[new_key] = cur_value
        for key, value in df_dict.items():
            for fg_detector in ["oracle", "None"]:
                common_key = f"common{ENCODED_NAME_SEP}{fg_detector}"
                common_acc = value[common_key]
                counter_key = f"counter{ENCODED_NAME_SEP}{fg_detector}"
                counter_acc = value[counter_key]
                gap_key = f"{fg_detector}{ENCODED_NAME_SEP}gap"
                value[gap_key] = common_acc - counter_acc
        pd.set_option("display.max_columns", None)

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
