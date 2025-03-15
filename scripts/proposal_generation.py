from torch.utils.data import DataLoader
import argparse
import pickle
from torchvision.datasets import ImageFolder
import os
import sys
# from stuned.utility.utils import (
#     get_project_root_path
# )
from stuned.utility.utils import parse_list_from_string


# local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# import densifier
from occam.datasets.tardataset import TarDataset
from occam.get_segments.run_cropformer import EntityNetV2
sys.path.pop(0)


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


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument("--config-file", default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml", help="path to config file")
    parser.add_argument("--input", help="Path to a tar file")
    parser.add_argument("--range", help="Range of images to process", default=None)
    parser.add_argument("--output", help="A directory to save output predictions dump.")
    parser.add_argument("--confidence-threshold",type=float,default=0.5,help="Minimum score for instance predictions to be shown")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER)
    return parser

if __name__ == "__main__":
    # Loaded Arguments
    args = get_parser().parse_args()

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
