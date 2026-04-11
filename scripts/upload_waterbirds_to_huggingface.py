#!/usr/bin/env python3
"""
Create / update a Hugging Face *dataset* repository with the local Waterbirds tree
and a dataset card (README.md).

Layout on the Hub (two splits, named groups):

  - ``FG_plus_BG`` — foreground + background (original images).
  - ``FG`` — foreground-only crops.

Under each: ``landbird_on_land``, ``landbird_on_water``, ``waterbird_on_land``,
``waterbird_on_water`` (class subfolders ``0`` / ``1`` unchanged).

Requires: huggingface_hub, and ``huggingface-cli login`` or HF_TOKEN / --token.

Full uploads use ``upload_large_folder``. ``--debug`` uploads a tiny subset via
``upload_folder``. ``--wipe-remote`` deletes the remote dataset repo and recreates it
before uploading (clean slate).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from occam.datasets.utils import DATASETS_PATH
from occam.datasets.waterbirds_layout import (
    GROUP_SUBDIRS,
    REQUIRED_TOP_LEVEL,
    SPLIT_FG_ONLY,
    SPLIT_FG_PLUS_BG,
    assert_uploadable_waterbirds,
    assert_waterbirds_layout,
    detect_layout,
    migrate_legacy_tar_extract_to_hub_layout,
)

import waterbirds_hf_common


OCCAM_REPO_README = (
    "https://github.com/AlexanderRubinstein/OCCAM/blob/main/README.md"
)
OCCAM_PAPER_HF = "https://huggingface.co/papers/2504.07092"
OCCAM_PAPER_ARXIV = "https://arxiv.org/abs/2504.07092"
WATERBIRDS_ORIGINAL = "https://arxiv.org/abs/1911.08731"
DFR_PAPER = "https://arxiv.org/abs/2204.02937"

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def _is_image_file(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(s) for s in _IMAGE_SUFFIXES)


def _list_images_in_group(group_dir: str) -> list[str]:
    out: list[str] = []
    for root, _, files in os.walk(group_dir):
        for fn in sorted(files):
            if _is_image_file(fn):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def _resolve_group_dir(
    src_wb: str, layout: str, split_hub: str, group_id: int
) -> str:
    """Absolute path to one group folder (hub or legacy source)."""
    gname = GROUP_SUBDIRS[group_id]
    if layout == "hub":
        return os.path.join(src_wb, split_hub, gname)
    if split_hub == SPLIT_FG_PLUS_BG:
        return os.path.join(src_wb, "test_split", f"group_{group_id}")
    return os.path.join(src_wb, "FG-Only", "test_split", f"group_{group_id}")


def build_debug_waterbirds_staging(
    src_wb: str, dest_root: str, per_group: int
) -> tuple[int, list[str]]:
    """
    Copy up to ``per_group`` images per group into ``dest_root`` using Hub split names
    (``FG_plus_BG`` / ``FG`` + named group folders). Source may be hub or legacy.
    """
    layout = detect_layout(src_wb)
    rel_uploaded: list[str] = []
    n = 0
    for split_hub in (SPLIT_FG_PLUS_BG, SPLIT_FG_ONLY):
        for gid in range(4):
            gdir = _resolve_group_dir(src_wb, layout, split_hub, gid)
            if not os.path.isdir(gdir):
                continue
            picks = _list_images_in_group(gdir)[:per_group]
            for src_path in picks:
                rel_within = os.path.relpath(src_path, gdir)
                dst_path = os.path.join(
                    dest_root, split_hub, GROUP_SUBDIRS[gid], rel_within
                )
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                rel_uploaded.append(
                    os.path.relpath(dst_path, dest_root).replace(os.sep, "/")
                )
                n += 1
    return n, rel_uploaded


def dataset_readme_body() -> str:
    return f"""# Waterbirds (OCCAM layout)

This repository hosts the **Waterbirds** image files used in the
[OCCAM]({OCCAM_PAPER_HF}) codebase ([arXiv]({OCCAM_PAPER_ARXIV})),
laid out for experiments on **subpopulation / group shifts** and **foreground-only**
evaluation.

## Original data and credit

The images come from the **Waterbirds** benchmark introduced with group
distributionally robust optimization in:

> Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang,
> *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization*,
> [arXiv:1911.08731]({WATERBIRDS_ORIGINAL}).

Please cite that work when using the original benchmark. Licensing and redistribution
terms of the underlying images follow the original dataset / WILDS release; refer to
the paper and official sources for details.

## Folder layout (two splits, named groups)

On the Hugging Face **Files** tab you should see two top-level splits:

1. **`{SPLIT_FG_PLUS_BG}`** — original images (**foreground + background**).
2. **`{SPLIT_FG_ONLY}`** — **foreground-only** crops (background removed).

Under each split, four group folders (same semantics as ``group_0`` … ``group_3`` in older OCCAM drops):

| Folder | Meaning |
|--------|---------|
| `{GROUP_SUBDIRS[0]}` | Landbird on land (`[0,0,*]`) |
| `{GROUP_SUBDIRS[1]}` | Landbird on water (`[1,0,*]`) |
| `{GROUP_SUBDIRS[2]}` | Waterbird on land (`[0,1,*]`) |
| `{GROUP_SUBDIRS[3]}` | Waterbird on water (`[1,1,*]`) |

Class subfolders **`0`** / **`1`** follow the OCCAM code (see repository below).

Foreground-only images follow the **deep feature reweighting** setting; extraction follows
Kirichenko, Izmailov & Wilson,
*Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*
([arXiv:2204.02937]({DFR_PAPER}); [code](https://github.com/PolinaKirichenko/deep_feature_reweighting)).

## OCCAM codebase

Download scripts, configs, and full experiment documentation live in the OCCAM repo:

- [{OCCAM_REPO_README}]({OCCAM_REPO_README})

The canonical download path in the codebase is `scripts/download_datasets_and_checkpoints.py`,
which fetches this dataset from the Hub after installing UrbanCars / CounterAnimals from
the shared Google Drive archive.

## Citation (OCCAM)

If you use this **exact packaging** together with OCCAM, please also cite the OCCAM paper
([HF paper page]({OCCAM_PAPER_HF}), [arXiv]({OCCAM_PAPER_ARXIV})).
"""


def full_readme_md(*, debug: bool) -> str:
    if debug:
        yaml_header = """---
license: other
pretty_name: Waterbirds (OCCAM) — DEBUG sample only
tags:
- image
- image-classification
- robustness
- spurious-correlation
- waterbirds
- occam
task_categories:
- image-classification
size_categories: n<1K
---

"""
        debug_banner = (
            "> **Debug upload:** This revision contains only a handful of sample images "
            "(for testing the upload pipeline). **Do not use for experiments.** "
            "Re-upload without `--debug` for the full dataset.\n\n"
        )
    else:
        yaml_header = """---
license: other
pretty_name: Waterbirds (OCCAM — FG+BG / FG-only splits, named groups)
tags:
- image
- image-classification
- robustness
- spurious-correlation
- waterbirds
- occam
task_categories:
- image-classification
size_categories: 10K<n<100K
---

"""
        debug_banner = ""
    return yaml_header + debug_banner + dataset_readme_body()


def main():
    parser = argparse.ArgumentParser(
        description="Upload local Waterbirds folder to a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        default=waterbirds_hf_common.default_hf_repo_id(),
        help="HF dataset id (e.g. org/name)",
    )
    parser.add_argument(
        "--folder",
        default=os.path.join(DATASETS_PATH, "Waterbirds"),
        help=(
            "Local Waterbirds root (Hub layout: FG_plus_BG/ and FG/; "
            "legacy test_split/ is migrated in place before full upload)"
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (else env HF_TOKEN or cached login)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private (only applies when creating new repo)",
    )
    parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Skip create_repo (repo must already exist)",
    )
    parser.add_argument(
        "--wipe-remote",
        action="store_true",
        help=(
            "Delete the remote dataset repository (if it exists), then create it again "
            "and upload. Destructive; cannot be combined with --no-create-repo."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Parallel workers for upload_large_folder "
            "(default: huggingface_hub chooses from CPU count)"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Upload only N images per group per split (see --debug-per-group) "
            "via upload_folder (fast sanity check)"
        ),
    )
    parser.add_argument(
        "--debug-per-group",
        type=int,
        default=2,
        metavar="N",
        help="With --debug, copy this many images per group per split (default: 2)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    if args.wipe_remote and args.no_create_repo:
        raise SystemExit("--wipe-remote cannot be used with --no-create-repo")

    wb = os.path.abspath(args.folder)
    assert_uploadable_waterbirds(wb)

    if args.debug and args.debug_per_group < 1:
        raise SystemExit("--debug-per-group must be >= 1")

    api = HfApi(token=args.token)

    if args.wipe_remote:
        print(f"Deleting remote dataset {args.repo_id!r} (if it exists) …")
        api.delete_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            token=args.token,
            missing_ok=True,
        )
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=False,
        )
    elif not args.no_create_repo:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

    readme = full_readme_md(debug=args.debug).encode("utf-8")
    api.upload_file(
        path_or_fileobj=readme,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Dataset card (README)"
        + (" (debug)" if args.debug else ""),
    )

    if args.debug:
        per = args.debug_per_group
        with tempfile.TemporaryDirectory(prefix="waterbirds_hf_debug_") as tmp:
            n, rels = build_debug_waterbirds_staging(wb, tmp, per_group=per)
            if n == 0:
                raise SystemExit(
                    "Debug staging is empty (no images found for the configured splits/groups)."
                )
            max_expected = per * 4 * 2
            print(
                f"Debug mode: staging {n} files "
                f"(≤{max_expected} = {per}×4 groups×2 splits) …"
            )
            for r in rels[:48]:
                print(f"  {r}")
            if len(rels) > 48:
                print(f"  … and {len(rels) - 48} more")
            for sub in REQUIRED_TOP_LEVEL:
                sub_path = os.path.join(tmp, sub)
                if not os.path.isdir(sub_path):
                    continue
                print(f"Uploading {sub!r} (debug subset) …")
                api.upload_folder(
                    folder_path=sub_path,
                    path_in_repo=sub,
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    commit_message=f"Debug subset: {sub}",
                )
    else:
        if detect_layout(wb) == "legacy":
            print(
                "Local folder uses legacy layout (test_split / FG-Only). "
                "Migrating in place to FG_plus_BG / FG …"
            )
            migrate_legacy_tar_extract_to_hub_layout(wb)
        assert_waterbirds_layout(wb)

        print("Uploading FG_plus_BG/ and FG/ via upload_large_folder …")
        api.upload_large_folder(
            repo_id=args.repo_id,
            folder_path=wb,
            repo_type="dataset",
            allow_patterns=[f"{SPLIT_FG_PLUS_BG}/*", f"{SPLIT_FG_ONLY}/*"],
            num_workers=args.num_workers,
            print_report=True,
        )

    print(f"Done. Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
