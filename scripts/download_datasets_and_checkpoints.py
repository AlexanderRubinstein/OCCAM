import os
import sys


# local imports
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(__file__))
)
import occam
sys.path.pop(0)


from stuned.utility.utils import (
    get_project_root_path,
    optionally_make_dir,
    download_and_extract_tar,
    remove_file_or_folder,
    download_file
)


IMAGENET_D_URL = "https://drive.google.com/uc?id=11zTXmg5yNjZwi8bwc541M1h5tPAVGeQc"
DATASETS_FOLDER = os.path.join(
    get_project_root_path(),
    "data",
    "datasets"
)
IMAGENET_VAL_KAGGLE_URL = "https://www.kaggle.com/code/joaoparana/download-imagenet-validation-set"
# copied from here: https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg
CHECKPOINTS_FOLDER = os.path.join(get_project_root_path(), "checkpoints")
CROPFORMER_CHECKPOINT_URL = "https://drive.google.com/uc?id=1BckHBDdxPDO9yJ9x1gzBiTsqD_AChklZ"
ALPHA_CLIP_CHECKPOINT_URL = "https://drive.google.com/uc?id=1WykuBYWePriCVeW5lOwBsgxgeBMzb1nd"

def download_datasets(datasets_folder):

    optionally_make_dir(datasets_folder, call_dirname=False)

    # ImageNet validation
    imagenet_val_folder = os.path.join(DATASETS_FOLDER, "ImageNet-val")
    if not os.path.exists(os.path.join(imagenet_val_folder)):
        raise ValueError(
            f"ImageNet-val folder does not exist in {imagenet_val_folder}. "
            f"Please manually download it from e.g. {IMAGENET_VAL_KAGGLE_URL}. "
            f"If it is already downloaded, please make a symlink to it, "
            f"e.g. `ln -s <path_to_imagenet_val_folder> {imagenet_val_folder}`."
        )

    # ImageNet-D
    imagenet_d_folder = os.path.join(DATASETS_FOLDER, "ImageNet-D")
    if not os.path.exists(os.path.join(imagenet_d_folder, "background")):
        download_and_extract_tar(DATASETS_FOLDER, IMAGENET_D_URL, extension=".tar")
        for subset in ["material", "questions", "texture"]:
            remove_file_or_folder(os.path.join(imagenet_d_folder, subset))
    print("All datasets downloaded!")


def download_checkpoints(checkpoints_folder):

    optionally_make_dir(checkpoints_folder, call_dirname=False)

    # Cropformer
    cropformer_checkpoint_path = os.path.join(
        CHECKPOINTS_FOLDER,
        "CropFormer_hornet_3x_03823a.pth"
    )
    if not os.path.exists(cropformer_checkpoint_path):
        print(f"Downloading Cropformer checkpoint to {cropformer_checkpoint_path}")
        download_file(
            cropformer_checkpoint_path,
            CROPFORMER_CHECKPOINT_URL
        )

    # AlphaClip
    alpha_clip_checkpoint_path = os.path.join(
        CHECKPOINTS_FOLDER,
        "clip_l14_grit20m_fultune_2xe.pth"
    )
    if not os.path.exists(alpha_clip_checkpoint_path):
        print(f"Downloading AlphaClip checkpoint to {alpha_clip_checkpoint_path}")
        download_file(alpha_clip_checkpoint_path, ALPHA_CLIP_CHECKPOINT_URL)
    print("All checkpoints are downloaded!")


def main():
    download_datasets(DATASETS_FOLDER)
    download_checkpoints(CHECKPOINTS_FOLDER)


if __name__ == "__main__":
    main()
