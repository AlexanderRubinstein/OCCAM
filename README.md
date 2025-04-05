# Are We Done with Object-Centric Learning?

- Remove this: gdrive_storage_folder: https://drive.google.com/drive/folders/1bCKbgY29CXsgYSwBHk49weolkA5cBePT?usp=share_link

## Overview

![Are We Done with Object-Centric Learning?](./figures/teaser.png "Are We Done with Object-Centric Learning?")

This is an implementation of the paper Are We Done with Object-Centric Learning? [REF].

[FILL][explanation for OCCAM]

The sections below has the following contents. In the section ["Installation"](#installation) we explain how to create a conda environment with all the necessary libraries. The section ["Download datasets and checkpoints"](#download-datasets-and-checkpoints) describe how to download datasets and models correspondingly. The section ["Evaluate on robust classification"](#evaluate-robust-classification) explains how to evaluate the models and reproduce the results for robust classification reported in the paper. The section ["Note about stuned.run_from_csv.py and .csv files"](#note-about-stunedrun_from_csvpy-and-csv-files) gives additional information about the scripts running pipeline we use in this repository.

Note: All commands are supposed to be run from the root of this repository and all paths are given relatively to it.

## Installation

To create a python environment we use [anaconda](https://www.anaconda.com/).

To create and activate a conda environment with `Python 3.10.0` run the following commands:

```
mkdir ./envs && conda create --yes --prefix ./envs/occam python==3.10.0
conda activate ./envs/occam/
pip install -r requirements.txt
cd occam/get_segments/modeling/pixel_decoder/ops/ && bash make.sh
```

Note: Installation of libraries needed for [High-Quality Entity Segmenta-
tion (HQES)]([FILL][link HQES paper]) requires compiling CUDA kernel, therefore it requires `CUDA_HOME` environment variable to be set.

## Download datasets and checkpoints

To download the datasets ([FILL][GB]) and model checkpoints ([FILL][GB]) needed for evaluation please run the following command (see ["Folder structure"](#folder-structure) for details of the resulting folders structure):

```
python scripts/download_datasets_and_checkpoints.py
```

Note: ImageNet Validation [REF] set is not downloaded automatically by the script above, therefore you should manually download (e.g. [from Kaggle](https://www.kaggle.com/code/joaoparana/download-imagenet-validation-set)) or symlink it to `data/datasets/ImageNet-val`.

### Folder structure

Upon a successful completion of the script `scripts/download_datasets_and_checkpoints.py` `data` folder will be created and will have the following structure:

[FILL][Make relevant structure]
```
📦data
┗ 📂datasets
  ┗ 📂cached
    ┣ 📂Brightness_1
    ┃ ┗ 📜4c905e75df34398dcc32_...50000_samples.hdf5
    ┣ 📂Brightness_5
    ┃ ┗ 📜ef1173603b558d0a45ac_...50000_samples.hdf5
    ...
    ┣ 📂Zoom Blur_5
    ┃ ┗ 📜3f50303a2e87f30cba0c_torch_...50000_samples.hdf5
    ┣ 📜in_a_deit3b_-1_.hdf5
    ┣ 📜in_r_deit3b_-1_.hdf5
    ┣ 📜in_train_deit3b_-1_4_epochs.hdf5
    ┣ 📜in_val_deit3b_-1.hdf5
    ┣ 📜inat_deit3b_-1_.hdf5
    ┗ 📜oi_deit3b_-1_.hdf5

```

In addition to that `checkpoints` folder will also be created and will have the following structure:

[FILL][Make relevant structure]
```
📦data
┗ 📂datasets
  ┗ 📂cached
    ┣ 📂Brightness_1
    ┃ ┗ 📜4c905e75df34398dcc32_...50000_samples.hdf5
    ┣ 📂Brightness_5
    ┃ ┗ 📜ef1173603b558d0a45ac_...50000_samples.hdf5
    ...
    ┣ 📂Zoom Blur_5
    ┃ ┗ 📜3f50303a2e87f30cba0c_torch_...50000_samples.hdf5
    ┣ 📜in_a_deit3b_-1_.hdf5
    ┣ 📜in_r_deit3b_-1_.hdf5
    ┣ 📜in_train_deit3b_-1_4_epochs.hdf5
    ┣ 📜in_val_deit3b_-1.hdf5
    ┣ 📜inat_deit3b_-1_.hdf5
    ┗ 📜oi_deit3b_-1_.hdf5

```

After following the steps from the section ["Generate masks"](#generate-masks) additional folders [FILL][Folder names] will be created inside `data` folder, so that its resulting structure will be the following:

```
📦data
┗ 📂datasets
  ┗ 📂cached
    ┣ 📂Brightness_1
    ┃ ┗ 📜4c905e75df34398dcc32_...50000_samples.hdf5
    ┣ 📂Brightness_5
    ┃ ┗ 📜ef1173603b558d0a45ac_...50000_samples.hdf5
    ...
    ┣ 📂Zoom Blur_5
    ┃ ┗ 📜3f50303a2e87f30cba0c_torch_...50000_samples.hdf5
    ┣ 📜in_a_deit3b_-1_.hdf5
    ┣ 📜in_r_deit3b_-1_.hdf5
    ┣ 📜in_train_deit3b_-1_4_epochs.hdf5
    ┣ 📜in_val_deit3b_-1.hdf5
    ┣ 📜inat_deit3b_-1_.hdf5
    ┗ 📜oi_deit3b_-1_.hdf5

```

[FILL][Make relevant structure]

## Generate masks

In this section we generate masks to compute outputs of the mask generator in OCCAM pipeline. Later we will use them for robust classification in ["Evaluate robust classification"](#evaluate-robust-classification).

[FILL][copy data from filled to non-filled csv, remove de-anonymizing login names + remove all slurm-related fields from the filled csv]

Make sure that `data` and `checkpoints` folders have the structure described in ["Folder structure"](#folder-structure).

To generate the masks run the following command (see ["Note about stuned.run_from_csv.py and .csv files"](#note-about-stunedrun_from_csvpy-and-csv-files) for details):

```
export ROOT=./ && export ENV=$ROOT/envs/occam && export PROJECT_ROOT_PROVIDED_FOR_STUNED=$ROOT && conda activate $ENV && python -m stuned.run_from_csv --conda_env $ENV --csv_path $ROOT/sheets/mask_generation.csv --run_locally --n_groups 1
```

Upon a successful scripts completion `./sheets/mask_generation.csv` will look like `./sheets/mask_generation_filled.csv` and the file subfolders [FILL] will be created in `data` folder.

## Evaluate robust classification

In this section we evaluate OCCAM on out-of-distribution (OOD) image classification with spurious backgrounds.

Make sure that `data` and `checkpoints` folders have the structure described in ["Folder structure"](#folder-structure).

To evaluate the models run the command (see ["Note about stuned.run_from_csv.py and .csv files"](#note-about-stunedrun_from_csvpy-and-csv-files) for details):

```
export ROOT=./ && export ENV=$ROOT/envs/occam && export PROJECT_ROOT_PROVIDED_FOR_STUNED=$ROOT && conda activate $ENV && python -m stuned.run_from_csv --conda_env $ENV --csv_path $ROOT/sheets/robust_classification.csv --run_locally --n_groups 1
```

Upon a successful scripts completion `sheets/robust_classification.csv` will look like `sheets/robust_classification_filled.csv`.

The accuracies can be seen in the end of <run_folder>/stdout.txt file, where <run_folder> are the paths from `run_folder` column in `./sheets/robust_classification.csv` table.

E.g. accuracy of OCCAM with FT-Dinosaur masks on ImageNet-D dataset that reproduce rows from [FILL][TABLE] are the following:

[FILL][TABLE row + copy from stdout]

|              | C-1 | C-5 | iNaturalist | OpenImages  |
|--------------|--------|------------|------------|------|
| ood_det_cov_   |   **0.681** |      **0.894** |   0.932 |   0.912 |
| ood_det_sem_   |   0.662 |   0.879 |   **0.977** |   **0.941** |

Please note that results may differ depending on the [CUDA](https://developer.nvidia.com/cuda-toolkit) version, the results above are computed for CUDA 12.2.

Note: Currently we provide only robust classification evaluation of FT-Dinosaur [REF] masks on ImageNet-D dataset [REF].
Results for other datasets or segmentation models as well as results for segmentation experiments and OOD detection experiments are currently not supported because of the unexpected shutdown of Galvani computing cluster in ML Cloud of University of Tübingen on 10.03.2025. We can provide commands to reproduce other results by request once the cluster is back online. The biggest part of the necessary code for that is already in this repository we just need to slightly adapt and test it.


## Note about stuned.run_from_csv.py and .csv files

.csv files are created for compact scripts running and logs recording using separate repository [STAI-tuned](https://github.com/AlexanderRubinstein/STAI-tuned). To run the scrips from the .csv file it should be submitted by the commands specified in the relevant sections, such as e.g:

```
export ROOT=./ && export ENV=$ROOT/envs/diverse_universe && export PROJECT_ROOT_PROVIDED_FOR_STUNED=$ROOT && conda activate $ENV && python -m stuned.run_from_csv --conda_env $ENV --csv_path $ROOT/result_sheets/evaluation.csv --run_locally --n_groups 1
```

### .csv file structure

- Each line of a .csv file corresponds to one run of the script written in the column "path_to_main".
- The script from the "path_to_main" column is parametrized by the config file specified in the column "path_to_default_config".
- The config from the "path_to_default_config" column is modified by the columns that start with keyword "delta:<...>".

### "n_groups" argument

`--n_groups k` means that `k` lines will be running at a time.

### logs

The logs from running each row (more precisely, the script in this row) are stored in the `stderr.txt` and `stdout.txt` files inside the folder specified in the "run_folder" column which will be automatically generated after submitting a .csv file.

### "whether_to_run" column

"whether_to_run" column can be modified to select the rows to run. After the .csv file submission only the rows that had `1` in this column are executed.
Whenever a script from some row successfully completes the corresponding value in the column "whether_to_run" changes from `1` to `0`. To rerun the same script 0 should be changed to 1 again.

### "status" column

Immediately after the .csv file submission for the rows that are being run a "status" column will be created (if it does not exist) with the value `Submitted` in it. Once corresponding sripts start running the "status" value will change to `Running`. Once the script completes status will become `Complete`. If the script fails its status will be `Fail`.

If something does not allow the script to start the status can be stuck with `Submitted` value. In that case please check the submission log file which is by default in `tmp/tmp_log_for_run_from_csv.out`.
