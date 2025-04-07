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
    is_number,
)


NUM_LAST_LINES_IN_STDOUT = 100000
MAX_COL_WIDTH = 1000
ALPHA_CHANNEL = "$\\alpha$-channel"
ALPHA_ONE = "($\\alpha$ = 1)"
ALPHA_CLIP = "alpha_clip_ViT-L/14"
CLIP = "clip_openai_ViT-L/14"
SIGLIP = "clip_openclip_webli_ViT-B-16-SigLIP-384"
CLIP_RN50 = "clip_openai_RN50"


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


def format_percentage(x):
    return f"{(100 * float(x)):.1f}"


def format_number(x):
    if x == "-" or not is_number(x):
        return x
    else:
        return format_percentage(x)


def parse_results_line(line, dataset_name):
    # if "clean_" in line:
    #     print("DEBUG:")
    line = line.replace("mix_rand_", "")  # remove for ImageNet-9
    line = line.replace("bg_", "")

    if dataset_name == "counter_animal":
        if "---None---" in line:
            dataset_name = line.split("---")[0]
        else:
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
    line_split = line.split()
    if len(line_split) < 2:
        return False
    if dataset_name == "counter_animal":
        start_match = line.startswith("counter") or line.startswith("common")
        end_match = is_number(line_split[-2])
    else:
        start_match = line.startswith(dataset_name)
        end_match = is_number(line_split[-1])
    return start_match and end_match


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
            line = line.replace("clean_", "")  # remove for CounterAnimal clean
            # if "clean_" in line and "0.658368" in line:
            #     print("DEBUG:")
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
                elif fg_score == "only_fg":
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
                        mask_method = ALPHA_CHANNEL
                    else:
                        assert cur_mask_source == "-"
                        mask_method = ALPHA_ONE
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

    table_2a = make_table_2(results_df, "a")
    table_2b = make_table_2(results_df, "b")
    table_2c = make_table_2(results_df, "c")
    table_2d = make_table_2(results_df, "d")
    table_3 = make_table_3(results_df)
    table_4 = make_table_4(results_df)
    table_5 = make_table_5(results_df)

    # ordered_rows = [
    #     [("arch", "CLIP"), ("mask_source", "-"), ("mask_method", "-"), ("fg_score", "-"), ("model", "clip_openai_ViT-L/14")],
    # ]

    # table_4 = None
    # for row in ordered_rows:
    #     row_4 = results_df
    #     for key, value in row:
    #         row_4 = row_4[row_4[key] == value]
    #     if table_4 is None:
    #         table_4 = row_4
    #     else:
    #         table_4 = pd.concat([table_4, row_4], ignore_index=True)

    pd.set_option(
        "display.max_colwidth", MAX_COL_WIDTH
    )  # to see long model names

    optionally_make_dir(args.result_folder, call_dirname=False)

    table_2a.to_csv(
        os.path.join(args.result_folder, "Table_2a.csv"), index=False
    )
    table_2b.to_csv(
        os.path.join(args.result_folder, "Table_2b.csv"), index=False
    )
    table_2c.to_csv(
        os.path.join(args.result_folder, "Table_2c.csv"), index=False
    )
    table_2d.to_csv(
        os.path.join(args.result_folder, "Table_2d.csv"), index=False
    )
    table_3.to_csv(os.path.join(args.result_folder, "Table_3.csv"), index=False)
    table_4.to_csv(os.path.join(args.result_folder, "Table_4.csv"), index=False)
    table_5.to_csv(os.path.join(args.result_folder, "Table_5.csv"), index=False)
    # print(results_df)
    # results_df.to_csv(
    #     os.path.join(args.result_folder, "Table_4.csv"), index=False
    # )


def filter_table_by_ordered_rows(results_df, ordered_rows, col_names):
    table = None
    for ordered_row in ordered_rows:
        row = results_df
        for key, value in zip(col_names, ordered_row):
            row = row[row[key] == value]
        if table is None:
            table = row
        else:
            table = pd.concat([table, row], ignore_index=True)
    return table


def make_table_2(results_df, section):
    col_names = ["arch", "mask_method", "mask_source", "fg_score", "model"]
    cols_order = [
        "arch",
        "mask_source",
        "mask_method",
        "fg_score",
    ]
    if section == "a":
        cols_order += ["imagenet_d"]
        ordered_rows = [
            ["AlphaCLIP", ALPHA_ONE, "-", "-", ALPHA_CLIP],
            ["CLIP", "Gray BG + Crop", "dino_ft", "oracle", CLIP],
            ["CLIP", "Gray BG + Crop", "cropformer", "oracle", CLIP],
            #
            ["CLIP", "-", "-", "-", SIGLIP],
            ["CLIP", "Gray BG + Crop", "dino_ft", "oracle", SIGLIP],
            ["CLIP", "Gray BG + Crop", "cropformer", "oracle", SIGLIP],
        ]
    elif section in ["b", "c", "d"]:
        if section == "b":
            cols_order += ["urban_cars"]
        elif section == "c":
            cols_order += ["imagenet_9"]
        elif section == "d":
            cols_order += ["waterbirds"]
        ordered_rows = [
            ["CLIP", "-", "-", "-", CLIP],
            ["CLIP", "Gray BG + Crop", "dino_ft", "oracle", CLIP],
            ["CLIP", "Gray BG + Crop", "cropformer", "oracle", CLIP],
            #
            ["CLIP", "-", "-", "-", CLIP_RN50],
            ["CLIP", "Gray BG + Crop", "dino_ft", "oracle", CLIP_RN50],
            ["CLIP", "Gray BG + Crop", "cropformer", "oracle", CLIP_RN50],
        ]
    else:
        raise NotImplementedError(f"Section {section} not implemented")

    # table = None
    # for ordered_row in ordered_rows:
    #     row = results_df
    #     for key, value in zip(col_names, ordered_row):
    #         row = row[row[key] == value]
    #     if table is None:
    #         table = row
    #     else:
    #         table = pd.concat([table, row], ignore_index=True)
    table = filter_table_by_ordered_rows(results_df, ordered_rows, col_names)

    table = table[cols_order]

    table = table.map(format_number)

    # for i, row in table_4.iterrows():
    #     row["model"] = row["model"].split("@")[0]

    return table


def make_table_3(results_df):
    col_names = ["arch", "mask_method", "mask_source", "fg_score", "model"]
    cols_order = [
        "arch",
        "mask_source",
        "mask_method",
        "fg_score",
        "Cmn/Ctr",
        "Cmn - Ctr",
    ]
    ordered_rows = [
        ["AlphaCLIP", ALPHA_ONE, "-", "-", ALPHA_CLIP],
        ["AlphaCLIP", ALPHA_CHANNEL, "dino_ft", "oracle", ALPHA_CLIP],
        ["AlphaCLIP", ALPHA_CHANNEL, "cropformer", "oracle", ALPHA_CLIP],
    ]
    table = filter_table_by_ordered_rows(results_df, ordered_rows, col_names)
    add_delta_column(table, name="Cmn - Ctr")
    table["Cmn/Ctr"] = table.apply(
        lambda row: f"{format_percentage(row['common'])}/{format_percentage(row['counter'])}"
        if row["common"] != "-" and row["counter"] != "-"
        else "-",
        axis=1,
    )
    table = table[cols_order]
    table = table.map(format_number)
    return table


def make_table_4(results_df):
    col_names = ["arch", "mask_method", "mask_source", "fg_score", "model"]
    ordered_rows = [
        ["CLIP", "-", "-", "-", CLIP],
        #
        # ["CLIP", "Gray BG + Crop", "dino_ft", "max_prob", CLIP],
        ["CLIP", "Gray BG + Crop", "dino_ft", "ens_entropy", CLIP],
        ["CLIP", "Gray BG + Crop", "dino_ft", "oracle", CLIP],
        #
        # ["CLIP", "Gray BG + Crop", "cropformer", "max_prob", CLIP],
        ["CLIP", "Gray BG + Crop", "cropformer", "ens_entropy", CLIP],
        ["CLIP", "Gray BG + Crop", "cropformer", "oracle", CLIP],
        #
        ["AlphaCLIP", ALPHA_ONE, "-", "-", ALPHA_CLIP],
        #
        # ["AlphaCLIP", ALPHA_CHANNEL, "dino_ft", "max_prob", ALPHA_CLIP],
        ["AlphaCLIP", ALPHA_CHANNEL, "dino_ft", "ens_entropy", ALPHA_CLIP],
        ["AlphaCLIP", ALPHA_CHANNEL, "dino_ft", "oracle", ALPHA_CLIP],
        #
        # ["AlphaCLIP", ALPHA_CHANNEL, "cropformer", "max_prob", ALPHA_CLIP],
        ["AlphaCLIP", ALPHA_CHANNEL, "cropformer", "ens_entropy", ALPHA_CLIP],
        ["AlphaCLIP", ALPHA_CHANNEL, "cropformer", "oracle", ALPHA_CLIP],
        # [("arch", "CLIP"), ("mask_source", "-"), ("mask_method", "-"), ("fg_score", "-"), ("model", "clip_openai_ViT-L/14")],
    ]

    # table_4 = None
    # for row in ordered_rows:
    #     row_4 = results_df
    #     for key, value in zip(col_names, row):
    #         row_4 = row_4[row_4[key] == value]
    #     if table_4 is None:
    #         table_4 = row_4
    #     else:
    #         table_4 = pd.concat([table_4, row_4], ignore_index=True)
    table = filter_table_by_ordered_rows(results_df, ordered_rows, col_names)

    # Add delta column with difference between common and counter
    add_delta_column(table)

    cols_order = [
        "arch",
        "mask_source",
        "mask_method",
        "fg_score",
        "waterbirds",
        "imagenet_9",
        "imagenet_d",
        "urban_cars",
        "delta",
    ]
    table = table[cols_order]

    table = table.map(format_number)

    # for i, row in table_4.iterrows():
    #     row["model"] = row["model"].split("@")[0]

    return table


def make_table_5(results_df):
    col_names = ["arch", "mask_method", "mask_source", "fg_score", "model"]
    cols_order = [
        "arch",
        "mask_source",
        "mask_method",
        "fg_score",
        "waterbirds",
    ]

    ordered_rows = [
        ["CLIP", "-", "-", "-", CLIP],
        # ["CLIP", "Gray BG + Crop", "dino_ft", "oracle", CLIP],
        ["CLIP", "Gray BG + Crop", "cropformer", "max_prob", CLIP],
        ["CLIP", "Gray BG + Crop", "cropformer", "ens_entropy", CLIP],
        ["CLIP", "Gray BG + Crop", "cropformer", "oracle", CLIP],
        ["CLIP", "-", "-", "only_fg", CLIP],
    ]

    # table = None
    # for ordered_row in ordered_rows:
    #     row = results_df
    #     for key, value in zip(col_names, ordered_row):
    #         row = row[row[key] == value]
    #     if table is None:
    #         table = row
    #     else:
    #         table = pd.concat([table, row], ignore_index=True)
    table = filter_table_by_ordered_rows(results_df, ordered_rows, col_names)

    table = table[cols_order]

    table = table.map(format_number)

    # for i, row in table_4.iterrows():
    #     row["model"] = row["model"].split("@")[0]

    return table


def add_delta_column(table, name="delta"):
    table[name] = table.apply(
        lambda row: float(row["common"]) - float(row["counter"])
        if row["common"] != "-" and row["counter"] != "-"
        else "-",
        axis=1,
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
