import os
import sys
import shutil
from tqdm import tqdm
import argparse


# local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import occam
from occam.datasets.utils import DATA_PATH

sys.path.pop(0)


from stuned.utility.utils import (
    get_project_root_path,
    optionally_make_dir,
    download_and_extract_tar,
    remove_file_or_folder,
    download_file,
)


IMAGENET_D_URL = (
    "https://drive.google.com/uc?id=11zTXmg5yNjZwi8bwc541M1h5tPAVGeQc"
)
IMAGENET_9_URL = "https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz"
# For WB and CA use original images, restructure folder structure;
# for UC generate from scratch using scripts from https://arxiv.org/abs/2212.04825
UC_WB_CA_URL = (
    "https://drive.google.com/uc?id=1fRirj3s7ndY-cj25pqvHKCLwYfxr4_Ks"
)
DATASETS_FOLDER = os.path.join(get_project_root_path(), "data", "datasets")
IMAGENET_VAL_KAGGLE_URL = (
    "https://www.kaggle.com/code/joaoparana/download-imagenet-validation-set"
)
# copied from here: https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg
CHECKPOINTS_FOLDER = os.path.join(get_project_root_path(), "checkpoints")
CROPFORMER_CHECKPOINT_URL = (
    "https://drive.google.com/uc?id=1BckHBDdxPDO9yJ9x1gzBiTsqD_AChklZ"
)
ALPHA_CLIP_CHECKPOINT_URL = (
    "https://drive.google.com/uc?id=1WykuBYWePriCVeW5lOwBsgxgeBMzb1nd"
)
OOD_RESULTS_URL = (
    "https://drive.google.com/uc?id=1Iri4pQcv2S6A_y38o1gCUVwnhJ52t0GB"
)
BBOXES_URL = "https://drive.google.com/uc?id=1mcH4bximxJ0cEz44PhgNarwlrLMr0_6A"


def download_datasets(datasets_folder, ignore_image_net_val):
    """
    download datasets
    datasets_folder: path to the datasets folder
    ignore_image_net_val: if True, ignore ImageNet-val
    """
    # UrbanCars, Waterbirds, CounterAnimals
    if not os.path.exists(os.path.join(datasets_folder, "UrbanCars")):
        assert not os.path.exists(
            os.path.join(datasets_folder, "Waterbirds")
        ), "Waterbirds already exists while UrbanCars is missing, please delete Waterbirds and try again"
        assert not os.path.exists(
            os.path.join(datasets_folder, "CounterAnimals")
        ), "CounterAnimals already exists while UrbanCars is missing, please delete CounterAnimals and try again"
        print("Downloading UrbanCars, Waterbirds, CounterAnimals")
        download_and_extract_tar(DATA_PATH, UC_WB_CA_URL, extension=".tar")

    if not ignore_image_net_val:
        # ImageNet validation
        imagenet_val_folder = os.path.join(datasets_folder, "ImageNet-val")
        if not os.path.exists(os.path.join(imagenet_val_folder)):
            raise ValueError(
                f"ImageNet-val folder does not exist in {imagenet_val_folder}. "
                f"Please manually download it from e.g. {IMAGENET_VAL_KAGGLE_URL}. "
                f"If it is already downloaded, please make a symlink to it, "
                f"e.g. `ln -s <path_to_imagenet_val_folder> {imagenet_val_folder}`."
            )

    # ImageNet-D
    imagenet_d_folder = os.path.join(datasets_folder, "ImageNet-D")
    if not os.path.exists(os.path.join(imagenet_d_folder, "background")):
        print("Downloading ImageNet-D")
        download_and_extract_tar(
            datasets_folder, IMAGENET_D_URL, extension=".tar"
        )
        for subset in ["material", "questions", "texture"]:
            remove_file_or_folder(os.path.join(imagenet_d_folder, subset))

    # ImageNet-9 - 4.83GB
    imagenet_9_folder = os.path.join(datasets_folder, "ImageNet-9")
    if not os.path.exists(os.path.join(imagenet_9_folder, "mixed_rand")):
        print("Downloading ImageNet-9")
        download_and_extract_tar(
            datasets_folder, IMAGENET_9_URL, extension=".tar"
        )
        shutil.move(
            os.path.join(datasets_folder, "bg_challenge"),
            os.path.join(imagenet_9_folder),
        )
        for subset in tqdm(os.listdir(imagenet_9_folder)):
            print("Removing unused subsets of ImageNet-9")
            if subset != "mixed_rand":
                remove_file_or_folder(os.path.join(imagenet_9_folder, subset))

    # bboxes for OOD detection
    annotations_path = os.path.join(DATA_PATH, "bboxes_annotations")
    if not os.path.exists(os.path.join(annotations_path, "val")):
        print(
            "Downloading ground truth bboxes for OOD detection on ImageNet-val"
        )
        download_and_extract_tar(DATA_PATH, BBOXES_URL, extension=".tar")

    print("All datasets downloaded!")


def download_checkpoints(checkpoints_folder):
    """
    download checkpoints
    checkpoints_folder: path to the checkpoints folder
    """
    optionally_make_dir(checkpoints_folder, call_dirname=False)

    # Cropformer
    cropformer_checkpoint_path = os.path.join(
        CHECKPOINTS_FOLDER, "CropFormer_hornet_3x_03823a.pth"
    )
    if not os.path.exists(cropformer_checkpoint_path):
        print(
            f"Downloading Cropformer checkpoint to {cropformer_checkpoint_path}"
        )
        download_file(cropformer_checkpoint_path, CROPFORMER_CHECKPOINT_URL)

    # AlphaClip
    alpha_clip_checkpoint_path = os.path.join(
        CHECKPOINTS_FOLDER, "clip_l14_grit20m_fultune_2xe.pth"
    )
    if not os.path.exists(alpha_clip_checkpoint_path):
        print(
            f"Downloading AlphaClip checkpoint to {alpha_clip_checkpoint_path}"
        )
        download_file(alpha_clip_checkpoint_path, ALPHA_CLIP_CHECKPOINT_URL)

    # results for OOD detection
    results_path = os.path.join(DATA_PATH, "results")
    if not os.path.exists(os.path.join(results_path, "ood_detection")):
        assert not os.path.exists(
            os.path.join(results_path, "uncertainty_scores")
        ), (
            "OOD detection results already exist while uncertainty scores are missing, "
            "please delete OOD detection results and try again"
        )
        print("Downloading results checkpoints for OOD detection")
        download_and_extract_tar(DATA_PATH, OOD_RESULTS_URL, extension=".tar")

    print("All checkpoints are downloaded!")


def main():
    """
    download datasets and checkpoints
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore_image_net_val", action="store_true")
    args = parser.parse_args()

    download_datasets(DATASETS_FOLDER, args.ignore_image_net_val)
    download_checkpoints(CHECKPOINTS_FOLDER)


if __name__ == "__main__":
    main()
