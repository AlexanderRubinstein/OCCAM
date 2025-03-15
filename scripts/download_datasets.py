import os
import sys
from stuned.utility.utils import (
    get_project_root_path,
    optionally_make_dir,
    download_and_extract_tar
)


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


DATASETS_URL = "<fill in the url>"
DATASETS_FOLDER = os.path.join(
    get_project_root_path(),
    "data",
    "datasets"
)


def main():
    # parent_folder = os.path.dirname(DATASETS_FOLDER)
    # os.makedirs(parent_folder, exist_ok=True)
    optionally_make_dir(DATASETS_FOLDER, call_dirname=False)
    download_and_extract_tar(DATASETS_FOLDER, DATASETS_URL)
    print("All datasets downloaded!")


if __name__ == "__main__":
    main()
