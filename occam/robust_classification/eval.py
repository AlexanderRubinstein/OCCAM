import os
import sys
import torch
from tqdm import tqdm
import pandas as pd
import torchvision
import numpy as np
from scipy import ndimage


from stuned.utility.utils import (
    get_project_root_path,
    raise_unknown,
    optionally_make_dir,
)
from stuned.local_datasets.transforms import (
    # DEFAULT_RESIZE_IN,
    # DEFAULT_SIZE_IN,
    make_transforms,
    # make_default_test_transforms_imagenet
)


sys.path.insert(0, get_project_root_path())
from occam.datasets.counter_animal import (
    COUNTER_PATH,
    COMMON_PATH,
    make_mapping_dict_counter_animal_,
    make_counter_animal_clip_wrapper,
)
from occam.datasets.imagenet_9 import (
    IMAGENET_9_PATH,
    make_mapping_dict_imagenet_9,
    make_in9_wrapper,
)
from occam.datasets.waterbirds import (
    WATERBIRDS_PATHS,
    WATERBIRDS_ONLY_FG_PATHS,
    make_mapping_dict_waterbirds,
    make_waterbirds_clip_wrapper,
)
from occam.datasets.imagenet_d import (
    IN_D_PATH,
    make_mapping_dict_imagenet_d,
    # make_mapping_dict_imagenet_d_bg
)
from occam.datasets.urban_cars import (
    URBAN_CARS_PATH,
    URBAN_CARS_BG_RATIO,
    URBAN_CARS_CO_OCCUR_OBJ_RATIO,
    URBAN_CARS_NUM_CLASS,
    convert_urban_cars_label,
    make_uc_clip_wrapper,
    decode_urban_cars_target,
)
from occam.datasets.utils import (
    make_mapping_dict_generic_from_folder,
    make_source_df,
    to_parquet,
)
from occam.datasets.bboxed_dataset import (
    make_bbox_dl_from_csv,
    compute_bbox_fit_score,
    load_mask,
)
from occam.robust_classification.utils import get_probs
from occam.robust_classification.masking import (
    mask2chw,
    is_background,
)
from occam.datasets.common import make_dataloader
from occam.robust_classification.models import (
    ClipEnsemble,
    ModelBuilder,
)
from occam.ood_detection.uncertainty_scores import (
    ens_entropy_per_sample,
)

# from occam.datasets.imagenet_classes import (
#     get_in_classes_prompts
# )

sys.path.pop(0)


ENCODED_NAME_SEP = "---"
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
ROUND_DECIMALS = 3
AREA_THRESHOLD = 0.001
BORDER_TOUCH_THRESHOLD = 200
BORDER_VS_AREA_RATIO = 0.01
NUM_CONNECTED_COMPONENTS_THRESHOLD = 30
ASPECT_RATIO_THRESHOLD = 6
NEW_MASK_VALUE = 0.5
BACKGROUND_THRESHOLD = 6


def extract_el_if_tuple(obj, i=0):
    if isinstance(obj, (tuple, list)):
        return obj[i]
    return obj


def make_df_with_foreground_scores(
    source_df_path,
    images_path,
    masks_path,
    bboxes_path,
    separate_masks_folder,
    models_dict,
    foreground_detectors,
    dataset_name="counter_animal",
    batch_size=128,
    num_workers=12,
    recompute_all=False,
    filter_keyword=None,
):
    # counter
    # counter_source_df_path = "/home/oh/arubinstein17/github/densification/data/csvs/source_counter.parquet"
    # make_source_df(
    #     images_path="/home/oh/arubinstein17/github/densification/data/CounterAnimal/symlinked/counter",
    #     masks_path="/home/oh/arubinstein17/github/densification/data/masks/counter_masks_1907.pkl",
    #     separate_masks_folder=os.path.join(get_project_root_path(), "data", "separate_masks", "counter"),
    #     source_df_path=counter_source_df_path
    # )
    # counter_source_df_path = "/home/oh/arubinstein17/github/densification/data/csvs/source_counter.parquet"

    # we might pass additional arguments in this variable by making it a tuple
    # if isinstance(source_df_path, (tuple, list)):
    #     extracted_source_df_path = source_df_path[0]
    # else:
    #     extracted_source_df_path = source_df_path

    extracted_source_df_path = extract_el_if_tuple(source_df_path, i=0)
    if not os.path.exists(extracted_source_df_path):
        if dataset_name == "counter_animal":
            make_mapping_dict_func = make_mapping_dict_counter_animal_
        elif dataset_name == "imagenet_9":
            make_mapping_dict_func = make_mapping_dict_imagenet_9
        elif dataset_name == "waterbirds":
            make_mapping_dict_func = make_mapping_dict_waterbirds
        elif dataset_name == "urban_cars":
            make_mapping_dict_func = make_mapping_dict_generic_from_folder
        elif dataset_name == "in_val":
            make_mapping_dict_func = make_mapping_dict_generic_from_folder
        else:
            assert (
                dataset_name == "imagenet_d"
            )  # it was renamed from imagenet-d to imagenet_d for consistency
            make_mapping_dict_func = make_mapping_dict_imagenet_d
        make_source_df_per_dataset(
            images_path=images_path,
            masks_path=masks_path,
            bboxes_path=bboxes_path,
            separate_masks_folder=separate_masks_folder,
            source_df_path=source_df_path,
            make_mapping_dict_func=make_mapping_dict_func,
            filter_keyword=filter_keyword,
        )
    else:
        print(f"Source df already exists: {source_df_path}")
    # for foreground_detector in foreground_detectors:

    if dataset_name == "urban_cars":
        convert_label = convert_urban_cars_label
    else:
        convert_label = None

    for model_name in models_dict.keys():
        add_foreground_score(
            source_df_path,
            foreground_detectors,
            model_name,
            models_dict,
            device="cuda",
            batch_size=batch_size,
            num_workers=num_workers,
            convert_label=convert_label,
            recompute_all=recompute_all,
            filter_keyword=filter_keyword,
        )


def make_source_df_per_dataset(
    images_path,
    masks_path,
    bboxes_path,
    separate_masks_folder,
    source_df_path,
    make_mapping_dict_func=make_mapping_dict_counter_animal_,
    filter_keyword=None,
):
    print("making mapping dict")
    mapping_dict = make_mapping_dict_func(
        images_path, masks_path, bboxes_path, separate_masks_folder
    )
    print("making source df")
    source_df = make_source_df(mapping_dict)

    extracted_source_df_path = extract_el_if_tuple(source_df_path, i=0)

    if filter_keyword is not None:
        print(
            f"Filtering df: {extracted_source_df_path} based on {filter_keyword}"
        )
        source_df = filter_df(source_df, filter_keyword)
    to_parquet(source_df, extracted_source_df_path)


def count_number_connected_components(mask):
    """
    Count the number of connected components in a binary mask.

    Parameters:
    -----------
    mask : numpy.ndarray
        Binary mask with values 0 or 1.

    Returns:
    --------
    int
        Number of connected components in the mask.
    """
    # Ensure the mask is binary (0s and 1s)
    binary_mask = (mask > 0).astype(np.uint8)

    # Label connected components
    labeled_array, num_features = ndimage.label(binary_mask)

    return num_features


# def filter_df(df_path, filter_keyword):
def filter_df(df, filter_keyword):
    def populate_top_areas_dict(
        top_areas_dict, source_image_path, all_masks, top_k
    ):
        cur_areas = {}
        for cur_mask_value in np.unique(all_masks):
            # mask, _ = load_mask(mask_path, mask_value)
            cur_mask_value = int(cur_mask_value)
            mask_area = (all_masks == cur_mask_value).sum()
            cur_areas[cur_mask_value] = mask_area
        # Get top k mask values by area
        top_k_mask_values = sorted(
            cur_areas.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        top_k_mask_values = {k for k, v in top_k_mask_values}
        top_areas_dict[source_image_path] = top_k_mask_values

    def add_all_zeros_mask(
        row, filter_keyword, all_masks, mask_path, rows_to_add, filtered_count
    ):
        all_zeros_masks = np.zeros_like(all_masks) + NEW_MASK_VALUE
        assert mask_path.endswith(
            ".mask"
        ), f"mask_path does end with .mask: {mask_path}"
        all_zeros_masks_path = mask_path.replace(".mask", "_all_zeros.mask")
        torch.save(all_zeros_masks, all_zeros_masks_path)
        row_to_add = row.copy()
        row_to_add["mask_path"] = all_zeros_masks_path
        row_to_add["mask_value"] = NEW_MASK_VALUE
        label_keys = [key for key in row.keys() if "_label" in key]
        for label_key in label_keys:
            if label_key == "classification_label":
                continue
            row_to_add[label_key] = 1
        row_to_add[filter_keyword] = 1
        rows_to_add.append(row_to_add)
        filtered_count -= 1
        return filtered_count

    if filter_keyword in df.columns:
        print(f"{filter_keyword} is already in df, skipping it.")
        return
    df.loc[:, filter_keyword] = 1  # column of 1s
    filtered_count = 0

    filter_names = filter_keyword.split("+")

    top_areas_dict = None
    top_k = None

    rows_to_add = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # if not "mDbAS" in row["source_image_path"]:
        #     continue
        # if not (row["mask_value"] == 12 or row["mask_value"] == 2):
        # if not (row["mask_value"] == 0):
        #     continue
        # print("Please remove above")
        mask_path = row["mask_path"]
        mask, all_masks = load_mask(mask_path, row["mask_value"])
        source_image_path = row["source_image_path"]

        assert (
            len(mask.shape) == 2
        ), f"Mask shape: {mask.shape} for {row['mask_path']} but should be 2D"

        to_filter = False

        border_touch = compute_border_touch(mask)
        mask_area = mask.sum() / (mask.shape[0] * mask.shape[1])
        for filter_name in filter_names:
            if to_filter:
                continue

            if filter_name == "by_mask_size":
                if mask_area < AREA_THRESHOLD:
                    to_filter = True
            elif filter_name == "by_background":
                if is_background(
                    mask, to_extract=False, threshold=BACKGROUND_THRESHOLD
                ):
                    to_filter = True
            elif filter_name == "by_aspect_ratio":
                _, h, w = mask2chw(mask[..., None], enforce_square_shape=False)
                if max(h / w, w / h) > ASPECT_RATIO_THRESHOLD:
                    to_filter = True
            elif filter_name == "by_border_touch":
                if border_touch > BORDER_TOUCH_THRESHOLD:
                    to_filter = True

            elif filter_name == "by_border_vs_area":
                if border_touch / mask_area > BORDER_VS_AREA_RATIO:
                    to_filter = True
            elif filter_name == "by_num_connected_components":
                num_connected_components = count_number_connected_components(
                    mask
                )
                if (
                    num_connected_components
                    > NUM_CONNECTED_COMPONENTS_THRESHOLD
                ):
                    to_filter = True
            elif "by_top_area" in filter_name:
                if top_areas_dict is None:
                    top_areas_dict = {}
                if top_k is None:
                    assert (
                        "@" in filter_name
                    ), f"filter_name must contain @: {filter_name}"
                    top_k = int(filter_name.split("@")[1])

                if source_image_path not in top_areas_dict:
                    populate_top_areas_dict(
                        top_areas_dict, source_image_path, all_masks, top_k
                    )

                if row["mask_value"] not in top_areas_dict[source_image_path]:
                    to_filter = True
            else:
                raise_unknown("Unknown filter name", filter_name, "filter_df")

        if to_filter:
            filtered_count += 1
            df.loc[idx, filter_keyword] = 0
            if (
                sum(
                    df[df["source_image_path"] == source_image_path][
                        filter_keyword
                    ]
                )
                == 0
            ):
                print(
                    f"All masks are filtered for {filter_keyword} for {source_image_path} "
                    f"using the whole image as mask for it"
                )
                filtered_count = add_all_zeros_mask(
                    row,
                    filter_keyword,
                    all_masks,
                    mask_path,
                    rows_to_add,
                    filtered_count,
                )

    for row in rows_to_add:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    print(f"Filtered {filtered_count} / {len(df)} masks")
    return df
    # return df


def compute_border_touch(mask):
    # compute border touch
    # mask is a binary mask
    # return the number of pixels on the border
    return mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum()


def make_keyword(foreground_detector, model_name):
    keyword = f"{foreground_detector}"
    # if not foreground_detector == "bbox_iou":
    if not foreground_detector in ("bbox_iou", "ens_entropy"):
        keyword += f"{ENCODED_NAME_SEP}{model_name}"
    label_key = f"{keyword}_label"
    metadata_key = f"{keyword}_metadata"
    return keyword, label_key, metadata_key


def add_foreground_score(
    source_df_path,
    foreground_detectors,
    model_name,
    models_dict,
    device="cuda",
    batch_size=128,
    num_workers=12,
    convert_label=None,
    recompute_all=False,
    filter_keyword=None,
):
    model = models_dict[model_name]

    model, apply_mask, transform = check_model_specific_options(
        model, model_name
    )

    model, original_model, source_df_path = apply_dataset_specific_options(
        model, source_df_path
    )

    # read df because we need info about its columns
    df = pd.read_parquet(source_df_path)
    # TODO(Alex | 09.11.2024) - don't read df twice, take it from dataset.csv;
    # or read only column names

    print(
        f"Adding foreground scores for {model_name} and detectors=({foreground_detectors})\n df: {source_df_path}"
    )
    print("Existing columns:", df.columns)

    model.eval()
    model.to(device)

    dataloader = make_bbox_dl_from_csv(
        source_df_path,
        extended_output=True,
        batch_size=batch_size,
        num_workers=num_workers,
        eval_transform=transform,
        apply_mask=apply_mask,
        filter_keyword=filter_keyword,
    )
    res_per_fg_score = {}
    foreground_detectors_to_process = (
        []
    )  # to avoid running eval for fg_detectors that are already in df

    for fg_detector in foreground_detectors:
        keyword, label_key, metadata_key = make_keyword(fg_detector, model_name)

        if (fg_detector == "ens_entropy") != isinstance(model, ClipEnsemble):
            continue

        # don't run eval if already in parquet
        if (
            metadata_key in df.columns
            and not df[metadata_key].isnull().values.any()
        ):
            if recompute_all:
                df.drop(columns=[metadata_key, label_key], inplace=True)
            else:
                print(f"{keyword} is already in df, skipping it.")
                continue
        foreground_detectors_to_process.append(fg_detector)
        res_per_fg_score[fg_detector] = {
            "source_image_path": [],
            "mask_value": [],
            "score_for_label": [],
            metadata_key: [],
        }

    if len(res_per_fg_score) == 0:
        return

    for batch in tqdm(dataloader):
        (
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
        ) = batch

        # e.g. for urban cars, when we need to convert [obj_label, bg_label and co-occur_label] to obj_label
        if convert_label is not None:
            label = convert_label(label)

        batch_size = len(idx)

        if not isinstance(
            applied_mask, (list, tuple)
        ):  # do nothing when mask is not applied yet
            applied_mask = applied_mask.to(device)
        else:
            assert len(applied_mask) == 2  # (image, mask)
            applied_mask = (
                applied_mask[0].to(device),
                applied_mask[1].to(device),
            )

        with torch.no_grad():
            model_outputs = model(applied_mask)

        for i in range(batch_size):
            mask_value = metadata["mask_value"][i].item()

            for foreground_detector in foreground_detectors_to_process:
                res = res_per_fg_score[foreground_detector]

                process_outputs(
                    i,
                    foreground_detector,
                    model_outputs,
                    label,
                    mask_value,
                    res,
                    image_path,
                    metadata_key,
                    model_name,
                    bbox,
                    mask,
                )

    model = original_model
    model.to("cpu")

    for fg_detector in foreground_detectors_to_process:
        keyword, label_key, metadata_key = make_keyword(fg_detector, model_name)
        res = res_per_fg_score[fg_detector]
        df = merge_dfs(df, res, label_key=label_key)

    to_parquet(df, source_df_path)
    return df


def process_outputs(
    i,
    foreground_detector,
    # model_inputs,
    model_outputs,
    label,
    mask_value,
    res,
    image_path,
    metadata_key,
    model_name,
    bbox,
    mask,
    # uncertainty_estimators_dict
):
    keyword, label_key, metadata_key = make_keyword(
        foreground_detector, model_name
    )

    # label_key = f"{keyword}_label"
    # metadata_key = f"{keyword}_metadata"
    foreground_score, (
        max_prob,
        max_class,
        gt_prob,
        gt_label,
    ) = compute_foreground_score(
        # model_inputs[i],
        model_outputs[i],
        label[i],
        foreground_detector,
        bbox[i],
        mask[i],
        # uncertainty_estimators_dict=uncertainty_estimators_dict
    )

    # if foreground_detector in ("oracle", "max_prob"):
    #     foreground_score, max_prob_class = foreground_score
    #     max_prob, max_class = max_prob_class
    cur_metadata = {
        str(mask_value): {
            "foreground_score": round(foreground_score, ROUND_DECIMALS),
            "max_prob": round(max_prob, ROUND_DECIMALS),
            "max_class": max_class,
            "gt_class": gt_label.item(),
            "gt_prob": round(gt_prob, ROUND_DECIMALS),
        }
    }

    res["source_image_path"].append(image_path[i])
    res["mask_value"].append(mask_value)
    res["score_for_label"].append(foreground_score)
    res[metadata_key].append(cur_metadata)


def compute_foreground_score(
    # model_inputs,
    model_outputs,
    label,
    foreground_detector,
    bbox=None,
    mask=None,
    # uncertainty_estimators_dict=None
):
    assert foreground_detector in (
        "bbox_iou",
        "ens_entropy",
        "oracle",
        "max_prob",
    )
    probs = get_probs(model_outputs)
    if foreground_detector == "ens_entropy":
        probs = probs.mean(dim=0)  # average over num_models
    max_class = probs.argmax(dim=-1).item()
    max_prob = probs[..., max_class].item()
    gt_prob = probs[..., label].item()

    metadata = (max_prob, max_class, gt_prob, label)
    if foreground_detector == "oracle":
        # print("Probs:", probs.shape)
        score = gt_prob
        # return gt_prob, metadata
    elif foreground_detector == "max_prob":
        score = max_prob
        # return max_prob, metadata
    elif foreground_detector == "ens_entropy":
        # assert uncertainty_estimators_dict is not None
        # ens_output = uncertainty_estimators_dict[foreground_detector](model_inputs)

        # assert ClipEnsemble output
        # assert isinstance(model_outputs, (list))
        assert len(model_outputs.shape) == 2  # [num_models, num_classes]

        score = compute_ens_entropy(model_outputs)

    else:
        assert foreground_detector == "bbox_iou"
        bbox_iou = compute_bbox_fit_score(mask, bbox)
        score = bbox_iou
        # return bbox_iou, metadata
    return score, metadata


def compute_ens_entropy(model_outputs):
    # stacked_logits = None
    # for logits in model_outputs:
    #     # logits = res[model_id][sample_id]
    #     # append_dict(
    #     #     unc_scores,
    #     #     {model_id: {"conf": get_probs(logits).max().item()}},
    #     #     allow_new_keys=True
    #     # )
    #     # # print(unc_scores)
    #     # append_dict(
    #     #     unc_scores,
    #     #     {model_id: {"entropy": [entropy(logits).item()]}},
    #     #     allow_new_keys=True
    #     # )
    #     # print(unc_scores)
    #     # unc_scores[model_id]["conf"].append
    #     # unc_scores[model_id]["entropy"] = entropy(logits).item()
    #     if stacked_logits is None:
    #         stacked_logits = logits.unsqueeze(0)
    #     else:
    #         stacked_logits = torch.cat([stacked_logits, logits.unsqueeze(0)], dim=0)
    # stacked_logits = torch.cat(model_outputs, dim=0)
    # return entropy(stacked_logits).item()
    # raise NotImplementedError("Does 1 - ens_entropy help?")
    return 1 - ens_entropy_per_sample(model_outputs).item()


def merge_dfs(df, res, label_key):
    df_with_score = pd.DataFrame(res)

    # compute is_main_object as max score
    df_with_score["max_score"] = df_with_score.groupby("source_image_path")[
        "score_for_label"
    ].transform("max")
    df_with_score[label_key] = (
        df_with_score["score_for_label"] == df_with_score["max_score"]
    ).astype(int)
    df_with_score.pop("score_for_label")
    df_with_score.pop("max_score")

    # concat dfs
    df = pd.merge(
        df, df_with_score, on=["source_image_path", "mask_value"], how="inner"
    )

    return df


def check_model_specific_options(model, model_name):
    # is_clip = "lip" in model_name.lower()
    apply_mask = True
    # if is_clip:

    if isinstance(model, ModelBuilder):
        model = model.build()

    if isinstance(model, (tuple, list)):
        assert len(model) == 2
        model, transform = model
        model_name_str = model_name.lower()
        if ("maft" in model_name_str) or ("alpha_clip" in model_name_str):
            apply_mask = False
    else:
        transform = EVAL_TRANSFORM_APPLIED_MASK_CONFIG
    return model, apply_mask, transform


def apply_dataset_specific_options(model, dataset_path):
    original_model = model
    if isinstance(dataset_path, (tuple, list)):
        dataset_path, dataset_options = dataset_path
        mapper = dataset_options.get("mapper", None)

        # model can be None when we just want to parse dataset path
        if mapper is not None and model is not None:
            # model = mapper(model)
            if isinstance(model, ClipEnsemble):
                model = model  # don't wrap clip ensemble because we need it only for foreground score
            else:
                model = wrap_model(model, mapper)
    return model, original_model, dataset_path


def wrap_model(model, wrapper_type):
    if wrapper_type is None:
        return model

    assert isinstance(wrapper_type, str)

    # if is_ensemble(model):
    #     model = copy.deepcopy(model)
    #     for i in range(len(model.submodels)):
    #         model.submodels[i] = wrap_model(
    #             model.submodels[i],
    #             wrapper_type
    #         )
    #     if hasattr(model, "soup") and model.soup is not None:
    #         model.soup = wrap_model(model.soup, wrapper_type)
    # else:
    #     # if wrapper_type == "mvh":
    #     #     model = make_mvh_model_wrapper(model)
    if wrapper_type == "in9":
        model = make_in9_wrapper(model)
    elif wrapper_type == "waterbirds_clip":
        model = make_waterbirds_clip_wrapper(model)
    elif wrapper_type == "counter_animal_clip":
        model = make_counter_animal_clip_wrapper(model)
    elif wrapper_type == "urban_cars_clip":
        model = make_uc_clip_wrapper(model)
    else:
        raise_unknown("wrapper type", wrapper_type, "wrapper config")

    return model


def eval_models(
    parquets,
    models,
    fg_detectors,
    full_res_save_path,
    device="cuda",
    batch_size=128,
    clean_dataloader_kwargs={"clean_type": "counter_animal"},
    recompute_all=False,
    filter_keyword=None,
):
    """
    Evaluates machine learning models using specified datasets, transformations, and foreground (FG) detectors,
    saving the results for further analysis.

    Args:
        parquets (dict): A dictionary where keys are dataset names and values are paths to parquet files containing
            bounding box annotations.
        models (dict): A dictionary of model names (str) mapped to model objects or tuples containing the model and
            associated preprocessing functions (e.g., CLIP models).
        fg_detectors (list): A list of foreground detectors to use during evaluation. Use `None` for no detector.
        full_res_save_path (str): Path to save or load the results dictionary. If the file exists, results will be
            loaded from it.
        device (str, optional): The device to use for evaluation (e.g., 'cuda' for GPU or 'cpu' for CPU). Defaults to 'cuda'.
        clean_dataloader_kwargs (dict): Dict with additional arguments for dataloader creation. Defaults to {"clean_type": "counter_animal"}.

    Returns:
        dict: A dictionary containing evaluation results, where keys represent evaluation configurations
            (e.g., dataset and model combinations) and values contain the evaluation metrics or outcomes.

    Notes:
        - The function supports both traditional models and CLIP-based models, with appropriate preprocessing.
        - If the `full_res_save_path` file exists, results are loaded from it to avoid redundant computation.
        - Results are saved back to `full_res_save_path` after evaluation.
        - The function integrates with various data loaders (`make_dataloader`, `make_bbox_dl_from_csv`) and supports
          custom dataset transformations.
    """
    if os.path.exists(full_res_save_path):
        print("Loading full res from", full_res_save_path)
        full_res = torch.load(full_res_save_path)
    else:
        if full_res_save_path is not None:
            optionally_make_dir(full_res_save_path)
        full_res = {}
    # default_transform = make_default_test_transforms_imagenet()

    # clean_type = get_with_assert(clean_dataloader_kwargs, "clean_type")
    clean_type = clean_dataloader_kwargs.pop("clean_type")
    eval_mode = None

    if clean_type == "imagenet_d_bg":
        # dataset_name_path_list = [("clean_in_d_bg", IN_D_PATH)]
        dataset_name_path_list = [
            ("imagenet_d_bg", (IN_D_PATH, clean_dataloader_kwargs))
        ]
    elif clean_type == "in_val":
        raise NotImplementedError("In-val is not supported yet")
        # dataset_name_path_list = [("in_val", (IN_VAL_PATH, clean_dataloader_kwargs))]
    elif clean_type == "urban_cars":
        # raise NotImplementedError("Urban cars are not supported yet")
        dataset_name_path_list = [
            ("urban_cars", (URBAN_CARS_PATH, clean_dataloader_kwargs))
        ]
        eval_mode = "urban_cars"
    elif clean_type == "waterbirds":
        # raise NotImplementedError("Waterbirds are not supported yet")
        standard_wb_group_paths = [
            (
                f"waterbirds_group_{group_id}",
                (WATERBIRDS_PATHS[group_id], clean_dataloader_kwargs),
            )
            for group_id in range(len(WATERBIRDS_PATHS))
            # if group_id == 3 or group_id == 2
        ]

        only_fg_wb_group_paths = [
            (
                f"waterbirds_group_{group_id}_only_fg",
                (WATERBIRDS_ONLY_FG_PATHS[group_id], clean_dataloader_kwargs),
            )
            for group_id in range(len(WATERBIRDS_ONLY_FG_PATHS))
            # if group_id == 3 or group_id == 2
        ]
        # print(f"DEBUG: Uncomment above to run for group if group_id != 3 and group_id != 2")

        dataset_name_path_list = (
            standard_wb_group_paths + only_fg_wb_group_paths
        )
        # dataset_name_path_list = [
        #     ("waterbirds_group_0", (WB_GROUP_0_PATH, clean_dataloader_kwargs)),
        #     ("waterbirds_group_1", (WB_GROUP_1_PATH, clean_dataloader_kwargs)),
        #     ("waterbirds_group_2", (WB_GROUP_2_PATH, clean_dataloader_kwargs)),
        #     ("waterbirds_group_3", (WB_GROUP_3_PATH, clean_dataloader_kwargs)),
        # # fg
        #     ("waterbirds_group_0_only_fg", (WB_GROUP_0_ONLY_FG_PATH, clean_dataloader_kwargs)),
        #     ("waterbirds_group_1_only_fg", (WB_GROUP_1_ONLY_FG_PATH, clean_dataloader_kwargs)),
        #     ("waterbirds_group_2_only_fg", (WB_GROUP_2_ONLY_FG_PATH, clean_dataloader_kwargs)),
        #     ("waterbirds_group_3_only_fg", (WB_GROUP_3_ONLY_FG_PATH, clean_dataloader_kwargs)),
        # # bg
        # ("waterbirds_group_0_only_bg", (WB_GROUP_0_ONLY_BG_PATH, clean_dataloader_kwargs)),
        # ("waterbirds_group_1_only_bg", (WB_GROUP_1_ONLY_BG_PATH, clean_dataloader_kwargs)),
        # ("waterbirds_group_2_only_bg", (WB_GROUP_2_ONLY_BG_PATH, clean_dataloader_kwargs)),
        # ("waterbirds_group_3_only_bg", (WB_GROUP_3_ONLY_BG_PATH, clean_dataloader_kwargs))
        # ]
    elif "imagenet_9_mix_rand" in clean_type:
        # raise NotImplementedError("Imagenet-9 mix rand is not supported yet")

        if clean_type == "imagenet_9_mix_rand":
            clean_dataloader_kwargs = {"mapper": "in9"}
            assert (
                "mapper" in clean_dataloader_kwargs
            )  # give mapper in kwargs when calling eval_models
            assert clean_dataloader_kwargs["mapper"] == "in9"
        else:
            assert clean_type == "imagenet_9_mix_rand_without_mapper"
            assert len(clean_dataloader_kwargs) == 0
            clean_dataloader_kwargs = {}
        dataset_name_path_list = [
            ("imagenet_9", (IMAGENET_9_PATH, clean_dataloader_kwargs))
        ]
    else:
        # raise NotImplementedError("counter_animal is not supported yet")
        assert (
            clean_type == "counter_animal" or clean_type == "counter_animal_gap"
        )
        eval_mode = "counter_animal"
        dataset_name_path_list = [
            ("clean_counter", (COUNTER_PATH, clean_dataloader_kwargs)),
            ("clean_common", (COMMON_PATH, clean_dataloader_kwargs)),
        ]

    apply_mask = True

    for model_name, model in models.items():
        model, apply_mask, transform_config = check_model_specific_options(
            model, model_name
        )

        for fg_detector in fg_detectors:
            if fg_detector is None:
                # fg_keyword = make_fg_keyword("None", model_name)
                fg_keyword, _, _ = make_keyword("None", model_name)
                for (
                    clean_dataset_name,
                    clean_dataset_path,
                ) in dataset_name_path_list:
                    transform = resolve_transform(transform_config)

                    (
                        model,
                        original_model,
                        clean_dataset_path,
                    ) = apply_dataset_specific_options(
                        model, clean_dataset_path
                    )

                    dl = make_dataloader(
                        clean_dataset_path,
                        transform=transform,
                        batch_size=batch_size,
                        return_path=True,
                        dataloader_type=clean_dataset_name,  # TODO(Alex | 19.12.2024): check that it does not fail for CounterAnimal and ImageNet-D
                        **clean_dataloader_kwargs,
                    )

                    full_keyword = (
                        f"{clean_dataset_name}{ENCODED_NAME_SEP}{fg_keyword}"
                    )

                    eval_on_dl(
                        full_res,
                        full_keyword,
                        dl,
                        model,
                        device=device,
                        full_res_save_path=full_res_save_path,
                        mode=eval_mode,
                        recompute_all=recompute_all,
                    )

                    model = original_model
            else:
                # fg_keyword = make_fg_keyword(fg_detector, model_name)
                fg_keyword, _, _ = make_keyword(fg_detector, model_name)
                for parquet_name, parquet_path in parquets.items():
                    (
                        model,
                        original_model,
                        parquet_path,
                    ) = apply_dataset_specific_options(model, parquet_path)

                    dl = make_bbox_dl_from_csv(
                        csv_path=parquet_path,
                        # transform=default_transform,
                        batch_size=batch_size,
                        extended_output=False,
                        dataset_task="classification",
                        num_workers=4,
                        # return_path=True,
                        fg_keyword=fg_keyword,
                        apply_mask=apply_mask,
                        eval_transform=transform_config,
                        filter_keyword=filter_keyword,
                    )

                    full_keyword = f"{parquet_name}_{fg_keyword}@detector_{model_name}@model"
                    # if apply_mask:
                    #     images_extractor = None
                    # else:
                    #     images_extractor =
                    # print("evaluating", full_keyword)
                    eval_on_dl(
                        full_res,
                        full_keyword,
                        dl,
                        model,
                        device=device,
                        full_res_save_path=full_res_save_path,
                        mode=eval_mode,
                        recompute_all=recompute_all,
                    )

                    model = original_model

                # detailed_res = predict_with_model(model, dl, device='cuda')

                # acc = compute_acc_from_detailed_res(detailed_res)
                # full_keyword = f"{parquet_name}{ENCODED_NAME_SEP}{fg_keyword}"
                # full_res[full_keyword] = acc
    print(full_res)
    # os.makedirs(os.path.dirname(full_res_save_path), exist_ok=True)
    # torch.save(full_res, full_res_save_path)
    return full_res


# def make_fg_keyword(fg_detector, model_name):
#     return f"{fg_detector}{ENCODED_NAME_SEP}{model_name}"


def resolve_transform(transform_config):
    if isinstance(transform_config, torchvision.transforms.transforms.Compose):
        return transform_config
    else:
        assert isinstance(transform_config, dict)
        return make_transforms(transform_config)


def eval_on_dl(
    full_res,
    full_keyword,
    dl,
    model,
    device="cuda",
    full_res_save_path=None,
    mode=None,
    recompute_all=False,
):
    if not recompute_all and full_keyword in full_res:
        print(f"{full_keyword} already in results")
        return
    print("evaluating", full_keyword)
    if mode in [None, "counter_animal"]:
        detailed_res = predict_with_model(model, dl, device=device)

        if mode == "counter_animal":
            accs_dict, _, _ = compute_mean_per_class_acc_from_detailed_res(
                detailed_res
            )
            acc = sum(accs_dict.values()) / len(accs_dict)
        else:
            acc = compute_acc_from_detailed_res(detailed_res)
    else:
        assert mode == "urban_cars"
        acc = urban_cars_eval_split(loader=dl, model=model, device=device)
    # full_keyword = f"{parquet_name}{ENCODED_NAME_SEP}{fg_keyword}"
    full_res[full_keyword] = acc
    # print("evaluation result:", full_res[full_keyword])
    if full_res_save_path is not None:
        optionally_make_dir(full_res_save_path)
        torch.save(full_res, full_res_save_path)


def predict_with_model(
    model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Set model to evaluation mode
    model.eval()

    # Initialize a dictionary to store paths, predictions, and labels
    results = {"image_paths": [], "predictions": [], "labels": []}

    # Transfer the model to the device (CPU or GPU)
    # model = model.to(device)
    model.to(device)

    # No need to compute gradients during inference
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Assuming the dataloader returns a tuple: (images, labels, paths)
            images = batch[0]
            labels = batch[1]

            # print(images.mean()) # tmp

            paths = None
            if len(batch) > 2:
                paths = batch[2]
            # images, labels, paths = batch

            # Transfer images and labels to the device
            if torch.is_tensor(
                images
            ):  # sometimes images are list of tensors (when masks are separate from images)
                images = images.to(device)
            else:
                assert len(images) == 2
                images = (images[0].to(device), images[1].to(device))

            labels = labels.to(device)

            # Forward pass to get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Store the results
            if paths is not None:
                results["image_paths"].extend(paths)
            results["predictions"].extend(predicted.cpu().numpy())
            results["labels"].extend(labels.cpu().numpy())

    model.to("cpu")

    return results


def compute_mean_per_class_acc_from_detailed_res(detailed_res):
    """
    detailed_res is a dict with keys 'image_paths', 'predictions', 'labels'
    we want to compute the mean accuracy per class
    output tuple of dicts: {class_name: acc}, {class_name: num_correct}, {class_name: total_num_samples}
    """
    preds = detailed_res["predictions"]
    labels = detailed_res["labels"]
    correct = {}
    total = {}
    accs = {}
    for pred, label in zip(preds, labels):
        if pred == label:
            correct[label] = correct.get(label, 0) + 1
        total[label] = total.get(label, 0) + 1
    for label in correct:
        accs[label] = correct[label] / total[label]
    return accs, correct, total


def compute_acc_from_detailed_res(detailed_res):
    total = len(detailed_res["labels"])
    correct = 0
    for i in range(total):
        if detailed_res["predictions"][i] == detailed_res["labels"][i]:
            correct += 1
    return correct / total


# copied from here: https://github.com/facebookresearch/Whac-A-Mole/blob/c91b097043246bc726fccd70f9860873e28d6122/urbancars_trainers/base_trainer.py#L295
@torch.no_grad()
def urban_cars_eval_split(
    loader,
    model,
    device="cuda",
    split="test",
    num_class=URBAN_CARS_NUM_CLASS,
    enable_amp=True,
    bg_ratio=URBAN_CARS_BG_RATIO,
    co_occur_obj_ratio=URBAN_CARS_CO_OCCUR_OBJ_RATIO,
):
    meter = MultiDimAverageMeter((num_class, num_class, num_class))
    total_correct = []
    total_bg_correct = []
    total_co_occur_obj_correct = []
    total_shortcut_conflict_mask = []

    model.eval()
    model.to(device)
    pbar = tqdm(loader, dynamic_ncols=True)
    for batch in pbar:
        # image, target = data_dict["image"], data_dict["label"]
        image, target = batch[0], batch[1]
        target = decode_urban_cars_target(target)
        if torch.is_tensor(image):
            image = image.to(device, non_blocking=True)
        else:
            assert isinstance(image, (tuple, list))
            image = [el.to(device, non_blocking=True) for el in image]
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=enable_amp):
            output = model(image)

        pred = output.argmax(dim=1)

        obj_label = target[:, 0]
        bg_label = target[:, 1]
        co_occur_obj_label = target[:, 2]

        shortcut_conflict_mask = bg_label != co_occur_obj_label
        total_shortcut_conflict_mask.append(shortcut_conflict_mask.cpu())

        correct = pred == obj_label
        meter.add(correct.cpu(), target.cpu())
        total_correct.append(correct.cpu())

        bg_correct = pred == bg_label
        total_bg_correct.append(bg_correct.cpu())

        co_occur_obj_correct = pred == co_occur_obj_label
        total_co_occur_obj_correct.append(co_occur_obj_correct.cpu())

    model.to("cpu")

    num_correct = meter.cum.reshape(*meter.dims)
    cnt = meter.cnt.reshape(*meter.dims)
    multi_dim_color_acc = num_correct / cnt
    log_dict = {}
    absent_present_str_list = ["absent", "present"]
    absent_present_bg_ratio_list = [1 - bg_ratio, bg_ratio]
    absent_present_co_occur_obj_ratio_list = [
        1 - co_occur_obj_ratio,
        co_occur_obj_ratio,
    ]

    weighted_group_acc = 0
    for bg_shortcut in range(len(absent_present_str_list)):
        for second_shortcut in range(len(absent_present_str_list)):
            first_shortcut_mask = (meter.eye_tsr == bg_shortcut).unsqueeze(2)
            co_occur_obj_shortcut_mask = (
                meter.eye_tsr == second_shortcut
            ).unsqueeze(1)
            mask = first_shortcut_mask * co_occur_obj_shortcut_mask
            acc = multi_dim_color_acc[mask].mean().item()
            bg_shortcut_str = absent_present_str_list[bg_shortcut]
            co_occur_obj_shortcut_str = absent_present_str_list[second_shortcut]
            log_dict[
                f"{split}_bg_{bg_shortcut_str}"
                f"_co_occur_obj_{co_occur_obj_shortcut_str}_acc"
            ] = acc
            cur_group_bg_ratio = absent_present_bg_ratio_list[bg_shortcut]
            cur_group_co_occur_obj_ratio = (
                absent_present_co_occur_obj_ratio_list[second_shortcut]
            )
            cur_group_ratio = cur_group_bg_ratio * cur_group_co_occur_obj_ratio
            weighted_group_acc += acc * cur_group_ratio

    bg_gap = (
        log_dict[f"{split}_bg_absent_co_occur_obj_present_acc"]
        - weighted_group_acc
    )
    co_occur_obj_gap = (
        log_dict[f"{split}_bg_present_co_occur_obj_absent_acc"]
        - weighted_group_acc
    )
    both_gap = (
        log_dict[f"{split}_bg_absent_co_occur_obj_absent_acc"]
        - weighted_group_acc
    )

    log_dict.update(
        {
            f"{split}_id_acc": weighted_group_acc,
            f"{split}_bg_gap": bg_gap,
            f"{split}_co_occur_obj_gap": co_occur_obj_gap,
            f"{split}_both_gap": both_gap,
        }
    )

    total_bg_correct = torch.cat(total_bg_correct, dim=0)
    total_co_occur_obj_correct = torch.cat(total_co_occur_obj_correct, dim=0)
    total_correct = torch.cat(total_correct, dim=0)

    (
        bg_worst_group_acc,
        co_occur_obj_worst_group_acc,
        both_worst_group_acc,
    ) = meter.get_worst_group_acc()

    log_dict.update(
        {
            f"{split}_bg_worst_group_acc": bg_worst_group_acc,
            f"{split}_co_occur_obj_worst_group_acc": co_occur_obj_worst_group_acc,
            f"{split}_both_worst_group_acc": both_worst_group_acc,
        }
    )

    # if method == "erm":
    #     # evaluate cue preference for ERM
    #     obj_acc = total_correct.float().mean().item()
    #     bg_acc = total_bg_correct.float().mean().item()
    #     co_occur_obj_acc = total_co_occur_obj_correct.float().mean().item()

    #     log_dict.update(
    #         {
    #             f"{split}_cue_obj_acc": obj_acc,
    #             f"{split}_cue_bg_acc": bg_acc,
    #             f"{split}_cue_co_occur_obj_acc": co_occur_obj_acc,
    #         }
    #     )

    # self.log_to_wandb(log_dict)

    return log_dict


class MultiDimAverageMeter:
    # reference: https://github.com/alinlab/LfF/blob/master/util.py

    def __init__(self, dims):
        self.dims = dims
        self.eye_tsr = torch.eye(dims[0]).long()
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )

    def get_worst_group_acc(self):
        num_correct = self.cum.reshape(*self.dims)
        cnt = self.cnt.reshape(*self.dims)

        first_shortcut_worst_group_acc = (
            num_correct.sum(dim=2) / cnt.sum(dim=2)
        ).min()
        second_shortcut_worst_group_acc = (
            num_correct.sum(dim=1) / cnt.sum(dim=1)
        ).min()
        both_worst_group_acc = (num_correct / cnt).min()

        return (
            first_shortcut_worst_group_acc,
            second_shortcut_worst_group_acc,
            both_worst_group_acc,
        )
