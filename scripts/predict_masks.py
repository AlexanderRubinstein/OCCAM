from torch.utils.data import DataLoader
import argparse
import pickle
import os
import sys
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from ftdinosaur_inference import build_dinosaur


# local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from occam.submodules.dino_ft_wrapper import (
    load_model as load_dino_ft_model,
    get_masks_as_image,
)
from occam.get_segments.run_cropformer import EntityNetV2
from occam.datasets.utils import CustomImageFolder

sys.path.pop(0)


from stuned.utility.utils import (
    optionally_make_dir,
    get_project_root_path,
    create_tar_from_folder,
    remove_file_or_folder,
)


TAR_FOLDER = os.path.join(get_project_root_path(), "data", "tars")


def get_parser():
    parser = argparse.ArgumentParser(description="Predict masks with mask generators")
    parser.add_argument("--model_id", help="Dino-FT model id")
    parser.add_argument("--model_path", help="Mask generator checkpoint path")
    parser.add_argument("--input_folder", help="Input folder")
    parser.add_argument("--num_slots", help="Number of slots", type=int)
    parser.add_argument("--output", help="Output file")
    parser.add_argument(
        "--num_workers", help="Number of workers", type=int, default=12
    )
    parser.add_argument(
        "--mask_generator_type",
        help="Mask generator type",
        type=str,
        choices=["dino-ft", "dino-v1", "slotdiffusion", "cropformer"],
    )
    parser.add_argument(
        "--range", help="Range of images to process", default=None
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=None,
        help="Minimum score for instance predictions of CropFormer to be shown",
    )
    parser.add_argument("--config_file", help="path to config file")
    return parser


def should_be_none(arg, arg_name, mask_generator_type):
    assert arg is None, f"{arg_name} should be None for {mask_generator_type}"


def should_be_provided(arg, arg_name, mask_generator_type):
    assert (
        arg is not None
    ), f"{arg_name} should be provided for {mask_generator_type}"


def get_slotdiffusion_mask_as_image(masks, height, width):
    """Convert SlotDiffusion soft masks [1, K, H, W] to an image-sized hard mask."""
    assert len(masks.shape) == 4, f"masks shape is {masks.shape}"
    assert masks.shape[0] == 1, f"masks shape is {masks.shape}"
    masks = torch.nn.functional.interpolate(
        masks.float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return masks.argmax(dim=1).squeeze(0)


def predict_slot_masks(model, dataset, num_workers, masks_to_image_fn, model_forward_fn):
    masks_to_pickle = {}

    dl = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for sample, _, image_path in tqdm(dl):
        assert len(image_path) == 1

        image_path = image_path[0]  # extract from tuple
        image = Image.open(image_path)  # needed for height and width

        with torch.no_grad():
            inp = sample.to(device)
            outp = model_forward_fn(model, inp)
            masks_as_image = masks_to_image_fn(
                outp["masks"], height=image.height, width=image.width
            )
            mask_id = image_path[1:]  # legacy
            masks_to_pickle[mask_id] = {
                "mask": masks_as_image.cpu().numpy().astype(np.uint8),
            }

    assert len(masks_to_pickle) == len(dl)
    return masks_to_pickle


if __name__ == "__main__":
    """
    predict masks with mask generators
    """
    args = get_parser().parse_args()

    optionally_make_dir(args.output)

    if args.mask_generator_type == "cropformer":
        should_be_provided(args.config_file, "config_file", "cropformer")
        should_be_provided(args.model_path, "model_path", "cropformer")
        should_be_provided(
            args.confidence_threshold, "confidence_threshold", "cropformer"
        )

        should_be_none(args.model_id, "model_id", "cropformer")
        should_be_none(args.num_slots, "num_slots", "cropformer")

        args.opts = ("MODEL.WEIGHTS " + args.model_path).split()
        if args.range == "None":
            args.range = None

        # CropFormer requires a tar dataset
        tar_name = (
            os.path.basename(os.path.dirname(args.input_folder))
            + "_"
            + os.path.basename(args.input_folder)
            + ".tar.gz"
        )
        tar_path = os.path.join(TAR_FOLDER, tar_name)

        if not os.path.exists(tar_path):
            print(f"Tar file {tar_path} does not exist, creating it...")
            optionally_make_dir(tar_path)
            create_tar_from_folder(tar_path, args.input_folder)

        if args.range is None:
            images_range = None
        else:
            start, end = args.range.split(":")
            images_range = [int(start), int(end)]

        # Extract Segments
        args.input = tar_path
        net = EntityNetV2(args)
        output = net.run(range=images_range)

        # Save Segments
        pickle.dump(output, open(args.output, "wb"))
        remove_file_or_folder(tar_path)

    elif args.mask_generator_type == "dino-ft":
        should_be_provided(args.model_id, "model_id", "dino-ft")
        should_be_provided(args.num_slots, "num_slots", "dino-ft")

        should_be_none(args.config_file, "config_file", "dino-ft")
        should_be_none(args.model_path, "model_path", "dino-ft")
        should_be_none(args.range, "range", "dino-ft")
        should_be_none(
            args.confidence_threshold, "confidence_threshold", "dino-ft"
        )
        assert (
            args.confidence_threshold is None
        ), "confidence_threshold is not implemented for dino-ft"

        model = load_dino_ft_model(args.model_id)
        preproc = build_dinosaur.build_preprocessing(args.model_id)

        dataset = CustomImageFolder(
            root=args.input_folder, return_path=True, transform=preproc
        )

        masks_to_pickle = predict_slot_masks(
            model,
            dataset,
            args.num_workers,
            get_masks_as_image,
            lambda cur_model, inp: cur_model(inp, num_slots=args.num_slots),
        )
        pickle.dump(masks_to_pickle, open(args.output, "wb"))

    elif args.mask_generator_type == "dino-v1":
        should_be_provided(args.model_path, "model_path", "dino-v1")
        should_be_provided(args.num_slots, "num_slots", "dino-v1")

        should_be_none(args.config_file, "config_file", "dino-v1")
        should_be_none(args.model_id, "model_id", "dino-v1")
        should_be_none(args.range, "range", "dino-v1")
        should_be_none(
            args.confidence_threshold, "confidence_threshold", "dino-v1"
        )

        from occam.submodules.dino_basic_wrapper import (
            build_preprocessing as build_dino_v1_preprocessing,
            load_model as load_dino_v1_model,
        )

        model = load_dino_v1_model(args.model_path)
        preproc = build_dino_v1_preprocessing()

        dataset = CustomImageFolder(
            root=args.input_folder, return_path=True, transform=preproc
        )

        masks_to_pickle = predict_slot_masks(
            model,
            dataset,
            args.num_workers,
            get_masks_as_image,
            lambda cur_model, inp: cur_model(inp, num_slots=args.num_slots),
        )
        pickle.dump(masks_to_pickle, open(args.output, "wb"))

    elif args.mask_generator_type == "slotdiffusion":
        should_be_provided(args.model_path, "model_path", "slotdiffusion")

        should_be_none(args.config_file, "config_file", "slotdiffusion")
        should_be_none(args.model_id, "model_id", "slotdiffusion")
        should_be_none(args.num_slots, "num_slots", "slotdiffusion")
        should_be_none(args.range, "range", "slotdiffusion")
        should_be_none(
            args.confidence_threshold, "confidence_threshold", "slotdiffusion"
        )

        from occam.submodules.slotdiffusion_wrapper import (
            build_preprocessing as build_slotdiffusion_preprocessing,
            load_model as load_slotdiffusion_model,
        )

        model = load_slotdiffusion_model(args.model_path)
        preproc = build_slotdiffusion_preprocessing()

        dataset = CustomImageFolder(
            root=args.input_folder, return_path=True, transform=preproc
        )

        masks_to_pickle = predict_slot_masks(
            model,
            dataset,
            args.num_workers,
            get_slotdiffusion_mask_as_image,
            lambda cur_model, inp: cur_model(inp),
        )
        pickle.dump(masks_to_pickle, open(args.output, "wb"))
