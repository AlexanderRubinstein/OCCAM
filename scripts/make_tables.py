import os
import sys
import argparse
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
MISSING_RESULT = "??"
ALPHA_CHANNEL = "$\\alpha$-channel"
ALPHA_ONE = "($\\alpha$ = 1)"
ALPHA_CLIP = "alpha_clip_ViT-L/14"
CLIP = "clip_openai_ViT-L/14"
SIGLIP = "clip_openclip_webli_ViT-B-16-SigLIP-384"
CLIP_RN50 = "clip_openai_RN50"
MASK_SOURCE_DISPLAY_NAMES = {
    "dino_v1": "dino-v1",
}
MASK_SOURCES_WITH_MASKS = {
    "cropformer",
    "dino_ft",
    "dino-v1",
    "slotdiffusion",
}
MASK_SOURCE_TABLE_ORDER = ["dino_ft", "dino-v1", "slotdiffusion", "cropformer"]


def get_parser():
    parser = argparse.ArgumentParser(
        description="make tables from the paper based on the logs"
    )
    parser.add_argument(
        "--csv_with_results",
        default="./sheets/robust_classification_filled.csv",
        help="csv with all results",
    )
    parser.add_argument(
        "--result_folder",
        default="./data/results/robust_classification",
        help="where to save the results",
    )

    return parser


def format_percentage(x):
    """
    format a number as a percentage
    x: number to format
    """
    return f"{(100 * float(x)):.1f}"


def format_number(x):
    """
    format a number as a percentage if it is a number, otherwise return initial value
    x: number to format
    """
    if x == "-" or not is_number(x):
        return x
    else:
        return format_percentage(x)


def is_missing_value(x):
    return pd.isna(x) or x in ["", "?"]


def format_mask_source(mask_source):
    return MASK_SOURCE_DISPLAY_NAMES.get(mask_source, mask_source)


def warn(warnings, message):
    warnings.append(message)
    print(f"WARNING: {message}")


def parse_results_line(line, dataset_name):
    """
    parse a line in logs to get the dataset name, foreground score, model, and score
    line: line to parse
    dataset_name: dataset name
    """
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
    """
    check if a line in logs contains results for a given dataset
    line: line to check
    dataset_name: dataset name
    """
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
    """
    parse experiment logs and make tables from the paper based on the logs
    """
    parser = get_parser()
    args = parser.parse_args()
    csv = pd.read_csv(args.csv_with_results)
    results_df = None
    result_rows = []
    warnings = []
    for i, row in csv.iterrows():
        # skip empty rows
        required_cols = [
            "delta:kwargs/dataset_name",
            "delta:kwargs/mask_source",
            "run_folder",
        ]
        if all(is_missing_value(row.get(col)) for col in required_cols):
            continue
        dataset_name = get_with_assert(row, "delta:kwargs/dataset_name")
        mask_source = format_mask_source(
            get_with_assert(row, "delta:kwargs/mask_source")
        )
        run_folder = get_with_assert(row, "run_folder")

        if is_missing_value(dataset_name) or is_missing_value(mask_source):
            warn(
                warnings,
                f"Skipping row {i} of sheet {args.csv_with_results}: "
                "missing dataset_name or mask_source.",
            )
            continue

        if is_missing_value(run_folder):
            warn(
                warnings,
                f"Experiment in row {i} of sheet {args.csv_with_results} "
                f"({dataset_name}, {mask_source}) has no run folder. "
                f"Using {MISSING_RESULT} in tables.",
            )
            continue

        stdout_path = os.path.join(run_folder, "stdout.txt")
        if not os.path.exists(stdout_path):
            warn(
                warnings,
                f"Experiment in row {i} of sheet {args.csv_with_results} "
                f"({dataset_name}, {mask_source}) has no stdout at "
                f"{stdout_path}. Using {MISSING_RESULT} in tables.",
            )
            continue

        with open(stdout_path, "r") as f:
            f.seek(0, os.SEEK_END)
            f.seek(
                f.tell() - min(f.tell(), NUM_LAST_LINES_IN_STDOUT)
            )  # Read last chars
            stdout = f.read()
        last_lines = stdout.split("\n")
        cur_result_rows = []
        for line in last_lines:
            line = line.replace("clean_", "")  # remove for CounterAnimal clean
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
                    if cur_mask_source in MASK_SOURCES_WITH_MASKS:
                        mask_method = "Gray BG + Crop"
                    else:
                        assert cur_mask_source == "-"
                        mask_method = "-"
                elif arch == "AlphaCLIP":
                    if cur_mask_source in MASK_SOURCES_WITH_MASKS:
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

                cur_result_rows.append(result_row)

        exp_finished = False
        if len(last_lines) > 1:
            exp_finished = "Logger context cleaned!" in last_lines[-2]

        if not exp_finished:
            warn(
                warnings,
                f"Experiment in row {i} of sheet {args.csv_with_results} "
                f"({dataset_name}, {mask_source}) is not finished. "
                f"Using {MISSING_RESULT} in tables.",
            )
            continue
        result_rows.extend(cur_result_rows)

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
    if results_df is None:
        results_df = pd.DataFrame(columns=groupby_cols)

    dataset_cols = {
        col for col in results_df.columns if col not in groupby_cols
    }

    if len(results_df) > 0:
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

    pd.set_option(
        "display.max_colwidth", MAX_COL_WIDTH
    )  # to see long model names

    optionally_make_dir(args.result_folder, call_dirname=False)

    warnings_path = os.path.join(args.result_folder, "warnings.txt")
    with open(warnings_path, "w") as f:
        if warnings:
            f.write("\n".join(warnings) + "\n")
        else:
            f.write("No warnings.\n")

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


def filter_table_by_ordered_rows(results_df, ordered_rows, col_names):
    """
    filter the results dataframe by the ordered rows
    results_df: results dataframe
    ordered_rows: ordered rows to filter by; each row is a list of values,
        where each value is the value to filter by
        for the corresponding column in col_names
    col_names: column names to filter by
    """
    table = None
    for ordered_row in ordered_rows:
        row = results_df
        for key, value in zip(col_names, ordered_row):
            row = row[row[key] == value]
        if len(row) == 0:
            row = pd.DataFrame(
                [{key: value for key, value in zip(col_names, ordered_row)}]
            )
        if table is None:
            table = row
        else:
            table = pd.concat([table, row], ignore_index=True)
    return table


def ensure_columns(table, cols, fill_value=MISSING_RESULT):
    for col in cols:
        if col not in table.columns:
            table[col] = fill_value
        else:
            table[col] = table[col].fillna(fill_value)
    return table


def make_table_2(results_df, section):
    """
    make table 2
    results_df: results dataframe
    section: section of the table to make (a, b, c, d)
    """
    col_names = ["arch", "mask_method", "mask_source", "fg_score", "model"]
    cols_order = [
        "arch",
        "mask_source",
        "mask_method",
        "fg_score",
    ]
    if section == "a":
        cols_order += ["imagenet_d"]
        ordered_rows = [["AlphaCLIP", ALPHA_ONE, "-", "-", ALPHA_CLIP]]
        ordered_rows += [
            ["CLIP", "Gray BG + Crop", mask_source, "oracle", CLIP]
            for mask_source in MASK_SOURCE_TABLE_ORDER
        ]
        ordered_rows += [["CLIP", "-", "-", "-", SIGLIP]]
        ordered_rows += [
            ["CLIP", "Gray BG + Crop", mask_source, "oracle", SIGLIP]
            for mask_source in MASK_SOURCE_TABLE_ORDER
        ]
    elif section in ["b", "c", "d"]:
        if section == "b":
            cols_order += ["urban_cars"]
        elif section == "c":
            cols_order += ["imagenet_9"]
        elif section == "d":
            cols_order += ["waterbirds"]
        ordered_rows = [["CLIP", "-", "-", "-", CLIP]]
        ordered_rows += [
            ["CLIP", "Gray BG + Crop", mask_source, "oracle", CLIP]
            for mask_source in MASK_SOURCE_TABLE_ORDER
        ]
        ordered_rows += [["CLIP", "-", "-", "-", CLIP_RN50]]
        ordered_rows += [
            ["CLIP", "Gray BG + Crop", mask_source, "oracle", CLIP_RN50]
            for mask_source in MASK_SOURCE_TABLE_ORDER
        ]
    else:
        raise NotImplementedError(f"Section {section} not implemented")

    table = filter_table_by_ordered_rows(results_df, ordered_rows, col_names)

    table = ensure_columns(table, cols_order)
    table = table[cols_order]

    table = table.map(format_number)

    return table


def make_table_3(results_df):
    """
    make table 3
    results_df: results dataframe
    """
    col_names = ["arch", "mask_method", "mask_source", "fg_score", "model"]
    cols_order = [
        "arch",
        "mask_source",
        "mask_method",
        "fg_score",
        "Cmn/Ctr",
        "Cmn - Ctr",
    ]
    ordered_rows = [["AlphaCLIP", ALPHA_ONE, "-", "-", ALPHA_CLIP]]
    ordered_rows += [
        ["AlphaCLIP", ALPHA_CHANNEL, mask_source, "oracle", ALPHA_CLIP]
        for mask_source in MASK_SOURCE_TABLE_ORDER
    ]
    table = filter_table_by_ordered_rows(results_df, ordered_rows, col_names)
    table = ensure_columns(table, ["common", "counter"])
    add_delta_column(table, name="Cmn - Ctr")
    table["Cmn/Ctr"] = table.apply(
        lambda row: f"{format_percentage(row['common'])}/{format_percentage(row['counter'])}"
        if is_number(row["common"]) and is_number(row["counter"])
        else MISSING_RESULT,
        axis=1,
    )
    table = ensure_columns(table, cols_order)
    table = table[cols_order]
    table = table.map(format_number)
    return table


def make_table_4(results_df):
    """
    make table 4
    results_df: results dataframe
    """
    col_names = ["arch", "mask_method", "mask_source", "fg_score", "model"]
    ordered_rows = [["CLIP", "-", "-", "-", CLIP]]
    ordered_rows += [
        ["CLIP", "Gray BG + Crop", mask_source, fg_score, CLIP]
        for mask_source in MASK_SOURCE_TABLE_ORDER
        for fg_score in ["ens_entropy", "oracle"]
    ]
    ordered_rows += [["AlphaCLIP", ALPHA_ONE, "-", "-", ALPHA_CLIP]]
    ordered_rows += [
        ["AlphaCLIP", ALPHA_CHANNEL, mask_source, fg_score, ALPHA_CLIP]
        for mask_source in MASK_SOURCE_TABLE_ORDER
        for fg_score in ["ens_entropy", "oracle"]
    ]

    table = filter_table_by_ordered_rows(results_df, ordered_rows, col_names)
    table = ensure_columns(
        table,
        [
            "waterbirds",
            "imagenet_9",
            "imagenet_d",
            "urban_cars",
            "common",
            "counter",
        ],
    )

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
    table = ensure_columns(table, cols_order)
    table = table[cols_order]

    table = table.map(format_number)

    return table


def make_table_5(results_df):
    """
    make table 5
    results_df: results dataframe
    """
    col_names = ["arch", "mask_method", "mask_source", "fg_score", "model"]
    cols_order = [
        "arch",
        "mask_source",
        "mask_method",
        "fg_score",
        "waterbirds",
    ]

    ordered_rows = [["CLIP", "-", "-", "-", CLIP]]
    ordered_rows += [
        ["CLIP", "Gray BG + Crop", mask_source, fg_score, CLIP]
        for mask_source in MASK_SOURCE_TABLE_ORDER
        for fg_score in ["max_prob", "ens_entropy", "oracle"]
    ]
    ordered_rows += [["CLIP", "-", "-", "only_fg", CLIP]]

    table = filter_table_by_ordered_rows(results_df, ordered_rows, col_names)

    table = ensure_columns(table, cols_order)
    table = table[cols_order]

    table = table.map(format_number)

    return table


def add_delta_column(table, name="delta"):
    """
    add a column with the difference between common and counter for CounterAnimal
    table: table to add the column to
    name: name of the column to add
    """
    table[name] = table.apply(
        lambda row: float(row["common"]) - float(row["counter"])
        if is_number(row["common"]) and is_number(row["counter"])
        else MISSING_RESULT,
        axis=1,
    )


def single_non_nan(x):
    """
    return the only non-nan value in the series,
    assert that there is only one non-nan value (needed when aggregating rows after grouping)
    """
    non_nan = x.dropna()
    if len(non_nan) == 1:
        return non_nan.iloc[0]
    elif len(non_nan) == 0:
        return MISSING_RESULT
    else:
        assert all(
            non_nan == non_nan.iloc[0]
        ), f"Expected values for identical scenarios to be equal, got {non_nan}"
        return non_nan.iloc[0]


if __name__ == "__main__":
    main()
