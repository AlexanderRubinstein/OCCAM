import os
import sys
import argparse
import torch
import numpy as np
import copy
import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import occam

sys.path.pop(0)


from stuned.utility.utils import (
    get_with_assert,
    optionally_make_dir,
)


NUM_LAST_LINES_IN_STDOUT = 10000
MAX_COL_WIDTH = 1000


def get_parser():
    parser = argparse.ArgumentParser(
        description="add background scores and eval on spurious backgrounds datasets"
    )
    parser.add_argument(
        "--csv_with_results",
        default="./sheets/robust_classification.csv",
        help="csv with all results",
    )
    parser.add_argument(
        "--result_folder",
        default="./data/results/robust_classification",
        help="where to save the results",
    )

    return parser


def parse_results_line(line, dataset_name):
    line = line.replace("mix_rand_", "")  # remove for ImageNet-9
    line = line.replace("clean_", "")  # remove for CounterAnimal clean

    if dataset_name == "counter_animal":
        dataset_name = line.split("_")[0]

    line = line.replace(
        dataset_name + "_", ""
    )  # we already know the dataset name
    if line[0] == "_":
        line = line[
            1:
        ]  # in case of Waterbirds we have two underscores at the beginning

    line_parts = line.split()

    if dataset_name in ["common", "counter"]:
        assert len(line_parts) == 3
    else:
        assert len(line_parts) == 2

    full_name = line_parts[0]
    score = float(line_parts[1])

    # Parse the full name
    if "None" in full_name:
        split = full_name.split("---")
        assert len(split) == 3
        dataset, fg_score, model = split
        if "only_fg" in full_name:
            fg_score = "only_fg"
    else:
        fg_score, model_raw = full_name.split(
            "@detector_"
        )  # oracle---clip_openai_ViT-L/14@detector_clip_openai_ViT-L/14@model
        fg_score = fg_score.split("---")[
            0
        ]  # oracle---clip_openai_RN50 -> oracle
        model = model_raw.split("@model")[0]
    return dataset_name, fg_score, model, score


def is_result_line(line, dataset_name):
    if dataset_name == "counter_animal":
        return line.startswith("counter") or line.startswith("common")
    return line.startswith(dataset_name)


def main():
    parser = get_parser()
    args = parser.parse_args()
    csv = pd.read_csv(args.csv_with_results)
    results_df = None
    result_rows = []
    for i, row in csv.iterrows():
        # skip empty rows
        if not np.any(row):
            continue
        dataset_name = get_with_assert(row, "delta:kwargs/dataset_name")
        mask_source = get_with_assert(row, "delta:kwargs/mask_source")
        run_folder = get_with_assert(row, "run_folder")

        stdout_path = os.path.join(run_folder, "stdout.txt")
        with open(stdout_path, "r") as f:
            f.seek(0, os.SEEK_END)
            f.seek(
                f.tell() - min(f.tell(), NUM_LAST_LINES_IN_STDOUT)
            )  # Read last 1000 chars
            stdout = f.read()
        last_lines = stdout.split("\n")
        for line in last_lines:
            line = line.split("(log): ")[
                -1
            ]  # in case log is on the same line as the results
            if is_result_line(line, dataset_name):
                cur_dataset_name, fg_score, model, score = parse_results_line(
                    line, dataset_name
                )

                if fg_score == "None":
                    fg_score = "-"
                    cur_mask_source = "-"
                else:
                    cur_mask_source = mask_source

                if "clip" in model:
                    if "alpha_clip" in model:
                        arch = "AlphaCLIP"
                    else:
                        arch = "CLIP"
                else:
                    raise NotImplementedError(
                        f"Only clip models are supported for now, got {model}"
                    )

                mask_method = None
                if arch == "CLIP":
                    if (
                        cur_mask_source == "cropformer"
                        or cur_mask_source == "dino_ft"
                    ):
                        mask_method = "Gray BG + Crop"
                    else:
                        assert cur_mask_source == "-"
                        mask_method = "-"
                elif arch == "AlphaCLIP":
                    if (
                        cur_mask_source == "cropformer"
                        or cur_mask_source == "dino_ft"
                    ):
                        mask_method = "$\\alpha$-channel"
                    else:
                        assert cur_mask_source == "-"
                        mask_method = "($\\alpha$ = 1)"
                else:
                    raise NotImplementedError(
                        f"Only clip models are supported for now, got {model}"
                    )

                assert mask_method is not None

                result_row = {
                    "arch": arch,
                    "mask_method": mask_method,
                    "dataset": cur_dataset_name,
                    "fg_score": fg_score,
                    "model": model,
                    "score": score,
                    "mask_source": cur_mask_source,
                }

                result_rows.append(result_row)

        exp_finished = False
        if len(last_lines) > 1:
            exp_finished = "Logger context cleaned!" in last_lines[-2]

        if not exp_finished:
            print(
                f"Experiment in row {i} of sheet {args.csv_with_results} not finished!"
            )
            continue

    for result_row in result_rows:
        cur_row = copy.deepcopy(result_row)
        cur_row[cur_row.pop("dataset")] = cur_row.pop("score")
        cur_df = pd.DataFrame([cur_row])

        if results_df is None:
            results_df = cur_df
        else:
            results_df = pd.concat([results_df, cur_df], ignore_index=True)

    # flatten diagonal to horizontal
    groupby_cols = ["arch", "mask_source", "mask_method", "fg_score", "model"]
    dataset_cols = {
        col for col in results_df.columns if col not in groupby_cols
    }

    results_df = results_df.groupby(groupby_cols, as_index=False).agg(
        {**{col: single_non_nan for col in dataset_cols}}
    )

    pd.set_option(
        "display.max_colwidth", MAX_COL_WIDTH
    )  # to see long model names

    optionally_make_dir(args.result_folder, call_dirname=False)
    print(results_df)
    results_df.to_csv(
        os.path.join(args.result_folder, "Table_4.csv"), index=False
    )


def single_non_nan(x):
    non_nan = x.dropna()
    if len(non_nan) == 1:
        return non_nan.iloc[0]
    elif len(non_nan) == 0:
        return "-"
    else:
        assert all(
            non_nan == non_nan.iloc[0]
        ), f"Expected values for identical scenarios to be equal, got {non_nan}"
        return non_nan.iloc[0]


if __name__ == "__main__":
    main()
