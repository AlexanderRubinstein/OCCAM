import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import (
    PrecisionRecallDisplay,
    # roc_auc_score,
    roc_curve,
)

# import scikitplot as skplt
import matplotlib.pyplot as plt

# import einops
# from stuned.utility.utils import get_project_root_path, load_from_pickle


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
)
# import densifier  # for setting the variable with project root

# sys.path.insert(0, os.path.join(get_project_root_path(), "notebooks"))
# from debiasing import make_models, get_uncertainty_score, optionally_make_dir

# sys.path.pop(0)
# import densifier
# from densifier.datasets.bboxed_dataset import make_bboxed_dataset_from_config
# from densifier.eval_clip.eval import (
#     add_openai_clip_model,
#     add_alpha_clip_model,
#     add_openclip_model,
# )
# from densifier.datasets.imagenet_classes import get_in_classes_prompts
# from densifier.eval_clip.eval import (
#     apply_visual_prompts,
#     _build_timm_model,
#     is_background,
# )
# from densifier.utility.utils_for_notebooks import (
#     # visualize_images_side_by_side,
#     # show_image_and_mask,
#     # load_data,
#     # evaluate_model,
#     make_symlink_cmd,
#     tensor_for_matplotlib,
#     # unnormalize,
#     show_in_rows,
#     batch_elements,
# )
# from densifier.detection.uncertainty_scores import (
#     div_continous_unique_per_sample,
#     average_energy_per_sample,
#     ens_entropy_per_sample,
#     entropy,
#     ens_conf_per_sample,
#     get_probs,
# )
from occam.ood_detection.uncertainty_scores import (
    div_continous_unique_per_sample,
    average_energy_per_sample,
    ens_entropy_per_sample,
    entropy,
    ens_conf_per_sample,
    get_probs,
)
from occam.datasets.bboxed_dataset import make_bboxed_dataset_from_config
from occam.datasets.imagenet_classes import get_in_classes_prompts
from occam.robust_classification.models import clip_models_with_same_preprocess

sys.path.pop(0)

# from stuned.local_datasets.transforms import (
#     DEFAULT_MEAN_IN,
#     DEFAULT_STD_IN,
#     make_transforms
# )

from stuned.utility.utils import (
    show_images,
    load_from_pickle,
    append_dict,
    optionally_make_dir,
)
from stuned.local_datasets.imagenet1k import get_imagenet_dataset
from stuned.local_datasets.transforms import (
    DEFAULT_RESIZE_IN,
    DEFAULT_SIZE_IN,
    make_transforms,
    make_default_test_transforms_imagenet,
    make_default_train_transforms_imagenet,
)

CUR_DIR = os.path.abspath("")
DATA_PATH = os.path.join(os.path.dirname(CUR_DIR), "data")

# DEFAULT_MEAN = [0.485, 0.456, 0.406]
# DEFAULT_STD = [0.229, 0.224, 0.225]
MIN_NON_ZERO_PIXELS = 100
NUM_CORNER_PIXELS_FOR_BG = 5


def plot_multiple_pr_curves(labels_scores, title, pr=True):
    plt.figure(figsize=(8, 6))  # Create a new figure
    ax = plt.gca()  # Get the current axis
    plot_chance_level = False
    i = 0

    # Iterate through the provided label and score pairs
    for labels, scores, name in labels_scores:
        if pr:
            if i + 1 == len(labels_scores):
                plot_chance_level = True
            display = PrecisionRecallDisplay.from_predictions(
                labels,
                scores,
                name=name,
                ax=ax,
                plot_chance_level=plot_chance_level,
            )
            i += 1
        else:
            # skplt.metrics.plot_roc_curve(labels, scores)
            # # plt.show()
            fpr, tpr, _ = roc_curve(labels, scores)
            # auc = metrics.roc_auc_score(y_test, y_pred_proba)
            # plt.plot(fpr, tpr, label=name)
            ax.plot(fpr, tpr, label=name)
            # plt.legend(loc=4)
            # plt.show()

    ax.set_title(title)
    plt.legend(loc="best")  # Add a legend
    plt.show()


def eval_model_list_on_dataloader(
    model_list, dataloader, device, max_samples=None
):
    res = {}
    res["is_main_object"] = []

    for model_id, model in model_list:
        model.eval()
        model.to(device)
        res[model_id] = []

    # res[unc_type] = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if max_samples is not None and i == max_samples:
                break
            # images, labels = batch[0], batch[1]
            assert batch[0].shape[0] == 1, "Implemented only for batch size 1"
            applied_mask, is_main_object = batch[4], batch[5]
            # image, label, bbox, mask, applied_mask, is_main_object
            applied_mask = applied_mask.to(device)
            is_main_object = is_main_object.to(device)
            for model_id, model in model_list:
                logits = model(applied_mask)
                res[model_id].append(logits)

            # uncertainty_score = get_uncertainty_score(logits, unc_type=unc_type)
            # res.append((uncertainty_score, is_main_object))
            # res[unc_type].append(uncertainty_score.item())
            res["is_main_object"].append(is_main_object.item())
    return res


# TODO(Alex | 23.09.2024): do this batchwise
def compute_unc(res, label_key):
    unc_scores = {}

    model_ids = []
    for model_id in res.keys():
        if model_id == label_key:
            continue
        # unc_scores[model_id] = {}
        model_ids.append(model_id)

    labels = res[label_key]
    unc_scores[label_key] = labels
    num_samples = len(labels)
    for sample_id in tqdm(range(num_samples)):
        stacked_logits = None
        for model_id in model_ids:
            logits = res[model_id][sample_id]
            append_dict(
                unc_scores,
                {model_id: {"conf": get_probs(logits).max().item()}},
                allow_new_keys=True,
            )
            # print(unc_scores)
            append_dict(
                unc_scores,
                {model_id: {"entropy": [entropy(logits).item()]}},
                allow_new_keys=True,
            )

            if stacked_logits is None:
                stacked_logits = logits.unsqueeze(0)
            else:
                stacked_logits = torch.cat(
                    [stacked_logits, logits.unsqueeze(0)], dim=0
                )
        append_dict(
            unc_scores,
            {
                "ensemble": {
                    "PDS": [
                        1
                        - div_continous_unique_per_sample(stacked_logits).item()
                    ]
                }
            },
            allow_new_keys=True,
        )
        append_dict(
            unc_scores,
            {
                "ensemble": {
                    "energy": [
                        1 - average_energy_per_sample(stacked_logits).item()
                    ]
                }
            },
            allow_new_keys=True,
        )
        append_dict(
            unc_scores,
            {
                "ensemble": {
                    "ens_ent": [
                        1 - ens_entropy_per_sample(stacked_logits).item()
                    ]
                }
            },
            allow_new_keys=True,
        )
        append_dict(
            unc_scores,
            {
                "ensemble": {
                    "ens_conf": [ens_conf_per_sample(stacked_logits).item()]
                }
            },
            allow_new_keys=True,
        )
        # average_energy_per_sample
        # unc_scores["PDS"] = div_continous_unique_per_sample(stacked_logits).item()
    return unc_scores


def get_parser():
    parser = argparse.ArgumentParser(description="compute uncertainty scores")
    parser.add_argument(
        "--result_path",
        # default="/home/oh/arubinstein17/github/densification/data/results/for_runner_eval_uc.pth",
        # default="/home/oh/arubinstein17/github/densification/data/results/uncertainty_scores/5models_ensemble.pkl",
        help="where to save the results",
    )
    # parser.add_argument(
    #     "--recompute_all",
    #     action="store_true",
    #     help="recompute all results"
    # )
    # parser.add_argument(
    #     "--dataset_name",
    #     default="urban_cars",
    #     help="dataset name"
    # )
    # parser.add_argument(
    #     "--clip",
    #     action="store_true",
    #     help="use clip"
    # )
    parser.add_argument("--clips", action="store_true", help="use clip models")
    parser.add_argument(
        "--filter_keyword",
        default=None,
        help="filter keyword for filtering masks",
    )
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=128,
    #     help="batch size"
    # )
    # parser.add_argument(
    #     "--mask_source",
    #     default="cropformer",
    #     help="mask source",
    #     choices=["cropformer", "dino_ft"]
    # )
    return parser


def main():
    args = get_parser().parse_args()

    # model
    # assert dataloader is not None

    # eval

    transform = None

    # ('ViT-L-14', 'openai')
    # ('ViT-L-14', 'laion400m_e31')
    # ('ViT-L-14', 'laion400m_e32')
    # ('ViT-L-14', 'laion2b_s32b_b82k')
    # ('ViT-L-14', 'datacomp_xl_s13b_b90k')
    # ('ViT-L-14', 'commonpool_xl_clip_s13b_b90k')
    # ('ViT-L-14', 'commonpool_xl_laion_s13b_b90k')
    # ('ViT-L-14', 'commonpool_xl_s13b_b90k')
    # ('ViT-L-14-quickgelu', 'metaclip_400m')
    # ('ViT-L-14-quickgelu', 'metaclip_fullcc')
    # ('ViT-L-14-quickgelu', 'dfn2b')
    # ('ViT-L-14-336', 'openai')
    # ('coca_ViT-L-14', 'laion2b_s13b_b90k')
    # ('coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k')
    # ('ViT-L-14-CLIPA', 'datacomp1b')
    # ('ViT-L-14-CLIPA-336', 'datacomp1b')
    # ('ViT-L-14', 'openai')
    # ('ViT-L-14', 'laion400m_e31')
    # ('ViT-L-14', 'laion400m_e32')
    # ('ViT-L-14', 'laion2b_s32b_b82k')
    # ('ViT-L-14', 'datacomp_xl_s13b_b90k')
    # ('ViT-L-14', 'commonpool_xl_clip_s13b_b90k')
    # ('ViT-L-14', 'commonpool_xl_laion_s13b_b90k')
    # ('ViT-L-14', 'commonpool_xl_s13b_b90k')
    # ('ViT-L-14-quickgelu', 'metaclip_400m')
    # ('ViT-L-14-quickgelu', 'metaclip_fullcc')
    # ('ViT-L-14-quickgelu', 'dfn2b')
    # ('ViT-L-14-336', 'openai')
    # ('coca_ViT-L-14', 'laion2b_s13b_b90k')
    # ('coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k')
    # ('ViT-L-14-CLIPA', 'datacomp1b')
    # ('ViT-L-14-CLIPA-336', 'datacomp1b')
    if args.clips:
        # raise NotImplementedError(
        #     "use make_clip_ensemble instead of the code below"
        # )
        # models_dict = {}
        category_list = get_in_classes_prompts()
        # # add_openai_clip_model('ViT-L/14', category_list, models_dict)
        # # add_openclip_model(
        # #     model_id='ViT-L-16-SigLIP-256',
        # #     category_list=category_list,
        # #     models_dict=models_dict,
        # #     pretrained='webli'
        # # )
        # # add_openclip_model(
        # #     model_id='ViT-L-14',
        # #     category_list=category_list,
        # #     models_dict=models_dict,
        # #     pretrained='laion400m_e32'
        # # )
        # add_openclip_model(
        #     model_id="ViT-L-14",
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained="datacomp_xl_s13b_b90k",
        # )
        # # add_openclip_model(
        # #     model_id='ViT-L-14',
        # #     category_list=category_list,
        # #     models_dict=models_dict,
        # #     pretrained='laion2b_s32b_b82k'
        # # ) # has normalize 0.5, 0.5, 0.5
        # add_openclip_model(
        #     model_id="ViT-L-14-quickgelu",
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained="dfn2b",
        # )
        # add_openclip_model(
        #     model_id="ViT-L-14",
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained="openai",
        # )
        # add_openclip_model(
        #     model_id="ViT-L-14",
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained="laion400m_e31",
        # )
        # # add_openclip_model(
        # #     model_id='ViT-L-14',
        # #     category_list=category_list,
        # #     models_dict=models_dict,
        # #     pretrained='laion400m_e31'
        # # )
        # add_openclip_model(
        #     model_id="ViT-L-14",
        #     category_list=category_list,
        #     models_dict=models_dict,
        #     pretrained="laion400m_e32",
        # )

        # final model name: clip_openclip_<pretrained>_ + <model_id>
        # clip_openclip_datacomp_xl_s13b_b90k_ViT-L-14
        # clip_openclip_dfn2b_ViT-L-14-quickgelu
        # clip_openclip_openai_ViT-L-14
        # clip_openclip_laion400m_e31_ViT-L-14
        # clip_openclip_laion400m_e32_ViT-L-14

        # model_list = []
        # for model_id, (model, preprocess) in models_dict.items():
        #     if transform is None:
        #         transform = preprocess
        #     else:
        #         # or at least normalization and cropping the same?
        #         assert str(transform) == str(
        #             preprocess
        #         ), "transforms must be the same"
        #     model_list.append((model_id, model))

        model_list, transform, names_list = clip_models_with_same_preprocess(
            category_list, return_names=True
        )
        model_list = list(zip(names_list, model_list))
    else:
        # model_list = make_models(
        #     [
        #         "resnet50.a1_in1k",
        #         "resnet18.a1_in1k",
        #         "vit_base_patch8_224.augreg2_in21k_ft_in1k",
        #         "tf_efficientnet_b1.ns_jft_in1k",
        #         "efficientnet_lite0.ra_in1k",
        #     ]
        # )
        raise NotImplementedError()

    if transform is None:
        transform = make_default_test_transforms_imagenet()

    bboxed_dataset_config = {
        "csv_path": os.path.join(
            DATA_PATH,
            "csvs",
            "cropformer",
            "source_in_val_cropformer.parquet",
        ),
        #   "csv_path": "/home/oh/arubinstein17/github/densification/data/csvs/cropformer/source_urban_cars_cropformer.parquet", # to speed up debug
        # csv_path: /home/oh/arubinstein17/github/densification/data/csvs/counter_debug.parquet
        "dataset_task": "detection",
        "eval_transform": transform,
        "train_transform": None,
        "train_val_split": 0.0,
        "foreground_keyword": "bbox_iou",  # is_main_object is decided based on bbox_iou
        # "foreground_keyword": "oracle---alpha_clip_ViT-L/14" # for faster debug
        "filter_keyword": args.filter_keyword,
    }

    bboxed_dataset = make_bboxed_dataset_from_config(
        bboxed_dataset_config=bboxed_dataset_config, transform_type="eval"
    )
    # 3m30s when loading in_val data
    # if False:

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        bboxed_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        # prefetch_factor=6
    )

    res_path = args.result_path
    if not os.path.exists(res_path):
        res = eval_model_list_on_dataloader(
            model_list,
            dataloader,
            device=torch.device("cuda:0"),
            # max_samples=100
        )
        optionally_make_dir(res_path)
        torch.save(res, res_path)
    else:
        res = torch.load(res_path)

    unc_scores = compute_unc(res, label_key="is_main_object")
    unc_scores_path = res_path.split(".")[0] + "_unc_scores.pkl"
    optionally_make_dir(unc_scores_path)
    torch.save(unc_scores, unc_scores_path)


if __name__ == "__main__":
    main()
