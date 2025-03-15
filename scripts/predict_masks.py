from torch.utils.data import DataLoader
import argparse
import pickle
import os
import sys
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from ftdinosaur_inference import (
    build_dinosaur
)
from torchvision.datasets import ImageFolder


# local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from occam.submodules.dino_ft_wrapper import (
    load_model,
    get_masks_as_image
)
from occam.datasets.tardataset import TarDataset
from occam.get_segments.run_cropformer import EntityNetV2
from occam.datasets.utils import CustomImageFolder
sys.path.pop(0)


from stuned.utility.utils import optionally_make_dir


def get_parser():
    parser = argparse.ArgumentParser(description="Predict masks with Dino-FT")
    parser.add_argument("--model_id", help="Dino-FT model id")
    parser.add_argument("--input_folder", help="Input folder")
    parser.add_argument("--num_slots", help="Number of slots", type=int, default=5)
    parser.add_argument("--output", help="Output file")
    # parser.add_argument("--batch_size", help="Batch size", type=int, default=1)
    parser.add_argument("--num_workers", help="Number of workers", type=int, default=12)
    parser.add_argument("--mask_generator_type", help="Mask generator type", type=str, choices=["dino-ft", "cropformer"])
    parser.add_argument("--range", help="Range of images to process", default=None)
    # parser.add_argument("--output", help="A directory to save output predictions dump.")
    parser.add_argument("--confidence-threshold",type=float,default=0.5,help="Minimum score for instance predictions to be shown")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER
    )
    return parser


def pop_arg_from_opts(args, arg_name):
    cutoff_i = None
    output = None
    for i in range(len(args.opts)):
        if args.opts[i] == arg_name:
            output = args.opts[i+1]
            cutoff_i = i
            break
    assert cutoff_i is not None
    args.opts = args.opts[:cutoff_i] + args.opts[cutoff_i+2:]
    return output


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.mask_generator_type == "cropformer":
        assert args.num_slots is None, "num_slots should be None for cropformer"
        if args.output is None:
            args.output = pop_arg_from_opts(args, "--output")

        if args.range is None:
            args.range = pop_arg_from_opts(args, "--range")
        elif args.range == "None":
            args.range = None

        print("Arguments: " + str(args))

        # Load Dataset
        input_path = args.input
        if input_path.endswith('.tar') or input_path.endswith('.tar.gz'):
            dataset = TarDataset(input_path, transform=None)
        else:
            dataset = ImageFolder(input_path, transform=None)
        if args.range is None:
            images_range = None
        else:
            # assert ":" == args.range[0]
            # assert ":" == args.range[1]
            # args.range = args.range[1:-1]
            # images_range = parse_list_from_string(args.range, list_separators=":")
            start, end = args.range.split(':')
            images_range = [int(start), int(end)]

        # Extract Segments
        net = EntityNetV2(args)
        output = net.run(range=images_range)

        # Save Segments
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        pickle.dump(output, open(args.output, 'wb'))

    elif args.mask_generator_type == "dino-ft":

        assert args.range is None, "range is not implemented for dino-ft"
        assert args.opts is None, "opts should be None for dino-ft"
        assert args.confidence_threshold is None, \
            "confidence_threshold is not implemented for dino-ft"

        model = load_model(args.model_id)
        preproc = build_dinosaur.build_preprocessing(args.model_id)

        masks_to_pickle = {}

        dataset = CustomImageFolder(
            root=args.input_folder,
            return_path=True,
            transform=preproc
        )

        dl = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)

        for (sample, _, image_path) in tqdm(dl):

            assert len(image_path) == 1

            image_path = image_path[0] # extract from tuple

            image = Image.open(image_path) # needed for height and width

            with torch.no_grad():
                # inp = preproc(image).unsqueeze(0)
                # inp = sample.unsqueeze(0)
                inp = sample.to(device)
                outp = model(inp, num_slots=args.num_slots)
                masks_as_image = get_masks_as_image(
                    outp["masks"],
                    height=image.height,
                    width=image.width
                )
                mask_id = image_path[1:] # legacy
                masks_to_pickle[mask_id] = {
                    "mask": masks_as_image.cpu().numpy().astype(np.uint8),
                    # "scores": None
                }
        assert len(masks_to_pickle) == len(dl)
        optionally_make_dir(args.output)
        pickle.dump(masks_to_pickle, open(args.output, "wb"))
