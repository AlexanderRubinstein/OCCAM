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


# local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from occam.submodules.dino_ft_wrapper import (
    load_model,
    get_masks_as_image
)
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
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    model = load_model(args.model_id)
    preproc = build_dinosaur.build_preprocessing(args.model_id)

    masks_to_pickle = {}

    dataset = CustomImageFolder(root=args.input_folder, return_path=True, transform=preproc)

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
