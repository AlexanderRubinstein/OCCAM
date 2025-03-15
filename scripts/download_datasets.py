import os
import sys


# local imports
sys.path.insert(
    0,
    # get_project_root_path()
    os.path.dirname(os.path.dirname(__file__))
)
# from occam.utils import (
#     download_and_extract_tar
# )
import occam
sys.path.pop(0)


from stuned.utility.utils import (
    get_project_root_path,
    optionally_make_dir,
    download_and_extract_tar,
    remove_file_or_folder
)


DATASETS_URL = "<fill in the url>"
IMAGENET_D_URL = "https://drive.google.com/uc?id=11zTXmg5yNjZwi8bwc541M1h5tPAVGeQc"
DATASETS_FOLDER = os.path.join(
    get_project_root_path(),
    "data",
    "datasets"
)
IMAGENET_VAL__KAGGLE_URL = "https://www.kaggle.com/code/joaoparana/download-imagenet-validation-set"


def main():

    optionally_make_dir(DATASETS_FOLDER, call_dirname=False)

    # ImageNet validation
    imagenet_val_folder = os.path.join(DATASETS_FOLDER, "ImageNet-val")
    if not os.path.exists(os.path.join(imagenet_val_folder)):
        raise ValueError(
            f"ImageNet-val folder does not exist in {imagenet_val_folder}. "
            f"Please manually download it from e.g. {IMAGENET_VAL__KAGGLE_URL}. "
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


if __name__ == "__main__":
    main()
