from flask import Flask, render_template, url_for, redirect
import os
import argparse
import sys
import matplotlib.pyplot as plt
import torch  # for loading torch .pth files
import numpy as np


# local modules
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    ),
)
from occam.datasets.bboxed_dataset import (
    EXTENDED_BBOXED_DATASET_ITEM_LEN,
    make_bboxed_dataset_from_config,
)

sys.path.pop(0)


from stuned.utility.utils import (
    get_with_assert,
    read_yaml,
    pretty_json,
    log_or_print,
    randomly_subsample_indices_uniformly,
)
from stuned.utility.configs import prepare_config
from stuned.utility.logger import make_logger
from stuned.local_datasets.imagenet1k import unnormalize_in1k, in_class_name
from stuned.local_datasets.utils import (
    tensor_for_matplotlib,
)

app = Flask(__name__)

# Define the folder containing images
WEB_APP_ROOT = os.path.dirname(__file__)
IMAGE_FOLDER = os.path.join(WEB_APP_ROOT, "static", "images")
NUM_ITEMS_TO_SHOW = 6
NUM_ITEMS_TO_SHOW_UNICORN = 4
FIG_SIZE = (15, 5)
MAX_LINE_LENGTH = 20


@app.route("/")
def show_images():
    images = [
        f
        for f in os.listdir(IMAGE_FOLDER)
        if os.path.isfile(os.path.join(IMAGE_FOLDER, f))
    ]
    # Create the URLs for the images
    image_urls = [url_for("static", filename=f"images/{img}") for img in images]
    return render_template("index.html", images=image_urls)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start web interface for showing images from dataset."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="path to config file with dataset description",
    )
    parser.add_argument(
        "--path_within_config",
        type=str,
        required=False,
        help="path to the data description within the config file",
    )
    parser.add_argument(
        "--images_list",
        type=str,
        required=False,
        help="List of image names to show; comma separated",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        help="data split to show",
        default="train",
    )
    parser.add_argument(
        "--n_images", type=int, help="Number of images to show", default=10
    )
    parser.add_argument(
        "--data_kwargs",
        type=str,
        help="Additional kwargs for data_config",
    )
    return parser.parse_args()


@app.route("/process-images", methods=["POST"])
def process_images():
    """Apply an operation to each image in the folder."""
    clean_image_folder(logger=None)
    sample_images_from_dataset()
    return redirect("/")


def make_dataset(dataset_config, split, logger):
    log_or_print("Making dataset", logger)
    dataset = make_bboxed_dataset_from_config(
        dataset_config, transform_type=split
    )
    log_or_print("Dataset is ready!", logger)
    return dataset


def prepare_images(dataset_config, split, logger):
    global DATASET

    DATASET = (dataset_config, split, logger)

    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    existing_images = os.listdir(IMAGE_FOLDER)
    if not len(existing_images) == NUM_IMAGES_TO_SHOW:
        clean_image_folder(logger)
        sample_images_from_dataset()


def clean_image_folder(logger):
    log_or_print("Cleaning image folder", logger)
    existing_images = os.listdir(IMAGE_FOLDER)
    for image in existing_images:
        os.remove(os.path.join(IMAGE_FOLDER, image))


def func_from_dict(d):
    return lambda x: d[x]


def sample_images_from_dataset():
    global DATASET
    global NUM_IMAGES_TO_SHOW

    # when dataset has not been created yet, creation args are stored in DATASET
    if isinstance(DATASET, tuple):
        dataset_config, split, logger = DATASET
        DATASET = make_dataset(dataset_config, split, logger)
        label_converter = dataset_config.get("label_converter", None)
        if label_converter is not None:
            DATASET.label_converter = func_from_dict(
                torch.load(label_converter)
            )
        else:
            DATASET.label_converter = None
        DATASET.mode = dataset_config.get("visualization_mode")
        NUM_IMAGES_TO_SHOW = min(len(DATASET), NUM_IMAGES_TO_SHOW)

    assert len(DATASET) > 0, "Dataset is empty"
    subsampled_indices = (
        randomly_subsample_indices_uniformly(len(DATASET), NUM_IMAGES_TO_SHOW)
        .numpy()
        .tolist()
    )
    print(
        "Subsampled indices (out of {}):".format(len(DATASET)),
        subsampled_indices,
    )

    for idx in subsampled_indices:
        save_bbox_image(
            DATASET[idx],
            label_converter=DATASET.label_converter,
            mode=DATASET.mode,
        )


def save_bbox_image(dataset_item, label_converter, mode=None):
    if mode is None:
        ncols = NUM_ITEMS_TO_SHOW
    else:
        ncols = NUM_ITEMS_TO_SHOW_UNICORN
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=FIG_SIZE)
    assert len(dataset_item) == EXTENDED_BBOXED_DATASET_ITEM_LEN
    idx = dataset_item[0]
    image_path = dataset_item[-3]
    show_bbox_image(
        axes, dataset_item, label_converter=label_converter, mode=mode
    )
    plt.tight_layout()
    savepath = os.path.join(
        IMAGE_FOLDER,
        # add mask_value to name to avoid image overrides:
        f"from_{os.path.basename(image_path)}".lower().replace(".", f"_{idx}."),
    )
    plt.savefig(savepath)
    plt.close()


def show_bbox_image(axes, item, label_converter=None, mode=None):
    def update_metadata_dict(metadata_dict):
        new_dict = {}
        for k, v in metadata_dict.items():
            if k == "max_class":
                key = "prediction"
                value = f"{label_converter(int(v))} ({int(v)})"
            elif k == "gt_class":
                continue
            else:
                key, value = k, v
            new_dict[key] = value
        return new_dict

    def prepare_image(image):
        return tensor_for_matplotlib(
            unnormalize_in1k(image[None, ...]).squeeze(0)
        )

    if label_converter is None:
        label_converter = in_class_name

    all_masks = None
    intersection_info = None
    if len(item) + 3 == EXTENDED_BBOXED_DATASET_ITEM_LEN:
        (
            image,
            label,
            bbox,
            mask,
            applied_mask,
            is_main_object,
            image_path,
        ) = item
    else:
        assert len(item) == EXTENDED_BBOXED_DATASET_ITEM_LEN
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
        ) = item

    # when dataset task is classification, image is np.ndarray without transforms
    if torch.is_tensor(image):
        image = prepare_image(image)

    applied_mask = prepare_image(applied_mask)

    mask = mask.to(torch.float32)

    bbox_caption = f"Bbox"
    applied_mask_caption = f"Applied mask"

    if mode is None:
        mask_caption = f"Mask of class {is_main_object}"

        if metadata is not None:
            if isinstance(metadata, (list, tuple)) and len(metadata) == 3:
                fit_score, intersection, union = intersection_info
                bbox_caption += (
                    f"\nIntersection info: "
                    f"\nfit_score: {fit_score}"
                    f"\nintersection: {intersection}"
                    f"\nunion: {union}"
                )
            else:
                assert isinstance(metadata, dict)

                # EXAMPLE: metadata is filled like below:
                # cur_metadata = {
                #     str(mask_value): (
                #         round(foreground_score, 5),
                #         round(max_prob, 5),
                #         max_class
                #     )
                # }
                for key, value in metadata.items():
                    # when loading from parquet can have Nones:
                    # like: {'0': array([5.50e-04, 4.66e-03, 5.39e+02]), '1': None, '2': None, '3': None, '4': None, '5': None, '6': None, '7': None}
                    if value is None:
                        continue
                    applied_mask_caption += f"\n{key}: {value}"

        image_caption_list = [
            (
                image,
                split_in_lines(
                    f"Image of class {label_converter(label)} ({label})"
                    f"\nPath: {os.path.join(os.path.basename(os.path.dirname(image_path)), os.path.basename(image_path))}",
                    max_line_length=MAX_LINE_LENGTH,
                    separator="",
                ),
            ),
            (tensor_for_matplotlib(mask), mask_caption),
            (applied_mask, applied_mask_caption),
            (tensor_for_matplotlib(bbox), bbox_caption),
        ]

        if all_masks is not None:
            if len(all_masks.shape) == 2:
                all_masks = all_masks[None, ...]
                all_masks = np.concatenate([all_masks] * 3, axis=0)
            all_masks = (
                all_masks / all_masks.max()
            )  # TODO(Alex | 05.11.2024): allow for multicolor
            image_caption_list.append(
                (tensor_for_matplotlib(all_masks), f"All masks")
            )
            # show bbox on mask overlay
            assert bbox.shape == mask.shape
            overlay_shape = [3] + list(bbox.shape[1:])
            overlay_bbox_on_mask = torch.zeros(overlay_shape)
            overlay_bbox_on_mask[0] = bbox[0]
            overlay_bbox_on_mask[1] = mask[0]
            image_caption_list.append(
                (
                    tensor_for_matplotlib(overlay_bbox_on_mask),
                    f"Overlay bbox on mask",
                )
            )
    else:
        assert mode == "unicorn"

        if metadata is not None:
            assert isinstance(metadata, dict)

            # EXAMPLE: metadata is filled like below:
            # cur_metadata = {
            #     str(mask_value): (
            #         round(foreground_score, 5),
            #         round(max_prob, 5),
            #         max_class
            #     )
            # }
            for key, value in metadata.items():
                # when loading from parquet can have Nones:
                # like: {'0': array([5.50e-04, 4.66e-03, 5.39e+02]), '1': None, '2': None, '3': None, '4': None, '5': None, '6': None, '7': None}
                if value is None:
                    continue

                if isinstance(value, (dict)):
                    value = update_metadata_dict(value)

                if key == "mask_value":
                    continue

                applied_mask_caption += f"\n{key}: {value}"

            applied_mask_caption += (
                f"\ngt label: {label_converter(label)} ({label})"
            )

        image_caption_list = [
            (image, "Image"),
            (tensor_for_matplotlib(mask), "Mask"),
            (applied_mask, applied_mask_caption),
        ]

        if all_masks is not None:
            if len(all_masks.shape) == 2:
                all_masks = all_masks[None, ...]
                all_masks = np.concatenate([all_masks] * 3, axis=0)
            all_masks = (
                all_masks / all_masks.max()
            )  # TODO(Alex | 05.11.2024): allow for multicolor
            image_caption_list.append(
                (tensor_for_matplotlib(all_masks), f"All masks")
            )

    for j, (img, title) in enumerate(image_caption_list):
        axes[j].imshow(img)
        axes[j].set_title(title)

    # Hide axes for a cleaner look
    for ax in axes:
        ax.axis("off")


def main():
    global NUM_IMAGES_TO_SHOW

    args = parse_args()

    NUM_IMAGES_TO_SHOW = args.n_images

    logger = make_logger()

    config = read_yaml(args.config_path)
    config = prepare_config(config)
    if args.path_within_config is None:
        data_config = config
    else:
        data_config = get_with_assert(
            config, args.path_within_config.split("/")
        )

    log_or_print(pretty_json(data_config), logger)

    data_config["extended_output"] = True
    if args.images_list is not None:
        data_config["images_list"] = args.images_list.split(",")

    if args.data_kwargs:
        kwargs = args.data_kwargs.split(";")
        for kwarg in kwargs:
            if kwarg == "":
                continue
            key, value = kwarg.split("=")
            data_config[key] = value

    prepare_images(data_config, args.split, logger)
    app.run(debug=True)


def split_in_lines(text, max_line_length, separator=" "):
    return insert_char_before_max_width_v2(
        text, max_line_length, separator=separator
    )


def insert_char_before_max_width_v2(
    input_string, max_width, char="\n", separator=" ", indent=" "
):
    if len(input_string) == 0 or max_width == 0:
        return input_string
    current_line = ""
    result = ""
    if separator == "":
        words = list(input_string)
    else:
        words = input_string.split(separator)
    for word in words:
        if current_line == "":
            current_line = word
        elif len(current_line) + len(word) <= max_width:
            current_line = current_line + separator + word
        else:
            result += current_line + char
            current_line = indent + word
    result += current_line
    return result


if __name__ == "__main__":
    main()
