import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import (
    PrecisionRecallDisplay,
    roc_curve,
)
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
from occam.ood_detection.uncertainty_scores import (
    div_continous_unique_per_sample,
    average_energy_per_sample,
    ens_entropy_per_sample,
    entropy,
    ens_conf_per_sample,
    get_probs,
)
from occam.datasets.utils import DATA_PATH
from occam.datasets.bboxed_dataset import make_bboxed_dataset_from_config
from occam.datasets.imagenet_classes import get_in_classes_prompts
from occam.robust_classification.models import clip_models_with_same_preprocess

sys.path.pop(0)


from stuned.utility.utils import (
    append_dict,
    optionally_make_dir,
)
from stuned.local_datasets.imagenet1k import get_imagenet_dataset
from stuned.local_datasets.transforms import (
    make_default_test_transforms_imagenet,
)


def plot_multiple_pr_curves(labels_scores, title, pr=True):
    """
    plot multiple precision-recall curves
    labels_scores: list of tuples (labels, scores, name)
    title: str, title of the plot
    pr: bool, if True, plot precision-recall curves, otherwise plot ROC curves
    """
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
            fpr, tpr, _ = roc_curve(labels, scores)
            ax.plot(fpr, tpr, label=name)


    ax.set_title(title)
    plt.legend(loc="best")  # Add a legend
    plt.show()


def eval_model_list_on_dataloader(
    model_list, dataloader, device, max_samples=None
):
    """
    evaluate a model list on a dataloader
    model_list: list of tuples (model_id, model)
    dataloader: dataloader
    device: device
    max_samples: max number of samples to evaluate
    """
    res = {}
    res["is_main_object"] = []
    for model_id, model in model_list:
        model.eval()
        model.to(device)
        res[model_id] = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if max_samples is not None and i == max_samples:
                break
            assert batch[0].shape[0] == 1, "Implemented only for batch size 1"
            applied_mask, is_main_object = batch[4], batch[5]
            # image, label, bbox, mask, applied_mask, is_main_object
            applied_mask = applied_mask.to(device)
            is_main_object = is_main_object.to(device)
            for model_id, model in model_list:
                logits = model(applied_mask)
                res[model_id].append(logits)

            res["is_main_object"].append(is_main_object.item())
    return res


def compute_unc(res, label_key):
    """
    compute uncertainty scores for a given model list on a given dataset
    res: dict of lists of logits
    label_key: str, key of the label in res
    """
    unc_scores = {}

    model_ids = []
    for model_id in res.keys():
        if model_id == label_key:
            continue
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
    return unc_scores


def get_parser():
    parser = argparse.ArgumentParser(description="compute uncertainty scores")
    parser.add_argument(
        "--result_path",
        default="./data/results/uncertainty_scores/for_runner_eval_unc.pth",
        help="where to save the results",
    )
    parser.add_argument("--clips", action="store_true", help="use clip models")
    parser.add_argument(
        "--filter_keyword",
        default=None,
        help="filter keyword for filtering masks",
    )
    return parser


def main():
    """
    compute uncertainty scores for a given model list on a given dataset
    """
    args = get_parser().parse_args()

    transform = None

    if args.clips:
        category_list = get_in_classes_prompts()

        model_list, transform, names_list = clip_models_with_same_preprocess(
            category_list, return_names=True
        )
        model_list = list(zip(names_list, model_list))
    else:
        raise NotImplementedError()

    if transform is None:
        transform = make_default_test_transforms_imagenet()

    bboxed_dataset_config = {
        "csv_path": os.path.join(
            DATA_PATH,
            "csvs",
            "cropformer",
            f"source_in_val_{args.filter_keyword}.parquet",
        ),
        "dataset_task": "detection",
        "eval_transform": transform,
        "train_transform": None,
        "train_val_split": 0.0,
        "foreground_keyword": "bbox_iou",  # is_main_object is decided based on bbox_iou
        "filter_keyword": args.filter_keyword,
    }

    bboxed_dataset = make_bboxed_dataset_from_config(
        bboxed_dataset_config=bboxed_dataset_config, transform_type="eval"
    )
    # 3m30s when loading in_val data

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        bboxed_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
    )

    res_path = args.result_path
    if not os.path.exists(res_path):
        res = eval_model_list_on_dataloader(
            model_list,
            dataloader,
            device=torch.device("cuda:0"),
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
