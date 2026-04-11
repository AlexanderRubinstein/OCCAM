#!/usr/bin/env python3
"""
Create / update a Hugging Face *dataset* repository with the local Waterbirds tree
and a dataset card (README.md).

Hub layout uses **eight top-level subscenario folders** (each with class subdirs ``0``/``1``):

  ``landbird_on_land``, ``landbird_on_land_fg_only``, ``landbird_on_water``, …

The dataset card YAML includes a ``configs`` block (one ``data_dir`` per subscenario) so
the Hub Dataset Viewer shows a **Subset** per subscenario without a Python loading script
(compatible with ``datasets`` 4.x).

Requires: huggingface_hub, and ``huggingface-cli login`` or HF_TOKEN / ``--token``.

Full uploads use ``upload_large_folder``. ``--debug`` uploads a tiny subset via
``upload_folder``. ``--wipe-remote`` deletes the remote dataset repo and recreates it
before uploading (clean slate).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from occam.datasets.utils import DATASETS_PATH
from occam.datasets.waterbirds_layout import (
    GROUP_SUBDIRS,
    REQUIRED_TOP_LEVEL,
    assert_uploadable_waterbirds,
    assert_waterbirds_layout,
    detect_layout,
    migrate_legacy_tar_extract_to_hub_layout,
    migrate_split_layout_to_subscenarios,
    source_dir_for_subscenario,
)

import waterbirds_hf_common

_HUB_DATASET_BUILDER_TEMPLATE = os.path.join(
    _SCRIPT_DIR, "waterbirds_hf_dataset_builder.py"
)


def _default_dataset_script_filename(repo_id: str) -> str:
    base = repo_id.split("/")[-1]
    slug = re.sub(r"[^0-9a-zA-Z_]+", "_", base).strip("_") or "waterbirds"
    if slug[0].isdigit():
        slug = "d_" + slug
    return f"{slug}.py"


def _upload_dataset_loading_script(
    api, repo_id: str, token: Optional[str], script_filename: str
) -> None:
    with open(_HUB_DATASET_BUILDER_TEMPLATE, "rb") as f:
        body = f.read()
    print(
        f"Uploading Hugging Face ``datasets`` loading script as {script_filename!r} …"
    )
    api.upload_file(
        path_or_fileobj=body,
        path_in_repo=script_filename,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add datasets loading script (Subset = subscenario)",
    )


def _try_delete_remote_dataset_script(
    api, repo_id: str, token: Optional[str], script_filename: str
) -> None:
    """Remove a legacy Hub ``*.py`` loading script (``datasets`` 4.x no longer loads these)."""
    try:
        api.delete_file(
            path_in_repo=script_filename,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Remove legacy datasets loading script (use README configs instead)",
        )
        print(
            f"Removed remote loading script {script_filename!r} (replaced by README ``configs``)."
        )
    except Exception as e:
        print(f"(No remote loading script removed: {e})")


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


def build_debug_waterbirds_staging(
    src_wb: str, dest_root: str, per_group: int
) -> tuple[int, list[str]]:
    """
    Copy up to ``per_group`` images per **subscenario** folder into ``dest_root``.
    Source may be hub (eight folders), ``FG_plus_BG``/``FG`` split, or legacy.
    """
    layout = detect_layout(src_wb)
    rel_uploaded: list[str] = []
    n = 0
    for name in REQUIRED_TOP_LEVEL:
        gdir = source_dir_for_subscenario(src_wb, layout, name)
        if not os.path.isdir(gdir):
            continue
        picks = _list_images_in_group(gdir)[:per_group]
        for src_path in picks:
            rel_within = os.path.relpath(src_path, gdir)
            dst_path = os.path.join(dest_root, name, rel_within)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            rel_uploaded.append(
                os.path.relpath(dst_path, dest_root).replace(os.sep, "/")
            )
            n += 1
    return n, rel_uploaded


def dataset_readme_body() -> str:
    table_header = "| Folder | Description |\n|--------|-------------|"
    sub_rows = "\n".join(
        f"| `{g}` | Original image (foreground + **background**); "
        f"same spurious-cue group as historical ``group_{i}`` |"
        for i, g in enumerate(GROUP_SUBDIRS)
    )
    sub_rows_fg = "\n".join(
        f"| `{g}_fg_only` | Foreground-only crop for the same group as `{g}` |"
        for g in GROUP_SUBDIRS
    )
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

## Folder layout (eight subscenarios)

On the Hugging Face **Files** tab you should see **eight** top-level folders. Each
``*_fg_only`` folder matches the **same spurious-cue group** as the folder without the
suffix (historical ``group_0`` … ``group_3``):

{table_header}
{sub_rows}
{sub_rows_fg}

### Class labels inside ``0/`` and ``1/``

Each subscenario folder contains subfolders **`0`** and **`1`**, which are the **binary
coarse bird-type labels** used by OCCAM configs and ``ImageFolder``-style loaders:

- **`1`** → **landbird**
- **`0`** → **waterbird**

(These are *not* the 200 fine-grained species names; they are the two high-level types
for the Waterbirds classification head in this benchmark.)

Foreground-only crops follow the **deep feature reweighting** setting; extraction follows
Kirichenko, Izmailov & Wilson,
*Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*
([arXiv:2204.02937]({DFR_PAPER}); [code](https://github.com/PolinaKirichenko/deep_feature_reweighting)).

## Hub Dataset Viewer (Subset = subscenario)

The dataset card YAML declares **[``configs``](https://huggingface.co/docs/datasets/repository_structure#define-your-splits-and-subsets-in-yaml)**
with one entry per subscenario. Each config sets ``data_dir`` to that folder so the Hub
uses the built-in **ImageFolder** loader: one **train** split per subset, columns ``image``
and ``label`` (folder names ``0`` / ``1``; see above for bird-type meaning).

Example:

```python
from datasets import load_dataset

ds = load_dataset("YOUR_ORG/waterbirds", "landbird_on_land_fg_only", split="train")
```

No ``trust_remote_code`` is required (``datasets`` 4.x does not load Hub Python dataset scripts).

If the viewer still shows a single **default** subset after updating the card, delete any stale
auto-generated **`data/`** folder on the Hub **Files** tab (leftover from an older layout) and
refresh the page.

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


def _dataset_card_yaml_frontmatter(*, debug: bool) -> str:
    """README YAML including ``configs`` for Hub Subset = subscenario (ImageFolder)."""
    if debug:
        lines = [
            "license: other",
            "pretty_name: Waterbirds (OCCAM) — DEBUG sample only",
            "tags:",
            "- image",
            "- image-classification",
            "- robustness",
            "- spurious-correlation",
            "- waterbirds",
            "- occam",
            "task_categories:",
            "- image-classification",
            "size_categories: n<1K",
        ]
    else:
        lines = [
            "license: other",
            "pretty_name: Waterbirds (OCCAM — eight subscenarios + class labels)",
            "tags:",
            "- image",
            "- image-classification",
            "- robustness",
            "- spurious-correlation",
            "- waterbirds",
            "- occam",
            "task_categories:",
            "- image-classification",
            "size_categories: 10K<n<100K",
        ]
    lines.append("configs:")
    for i, name in enumerate(REQUIRED_TOP_LEVEL):
        lines.append(f"- config_name: {name}")
        lines.append(f"  data_dir: {name}")
        if i == 0:
            lines.append("  default: true")
    return "---\n" + "\n".join(lines) + "\n---\n\n"


def full_readme_md(*, debug: bool) -> str:
    yaml_header = _dataset_card_yaml_frontmatter(debug=debug)
    if debug:
        debug_banner = (
            "> **Debug upload:** This revision contains only a handful of sample images "
            "(for testing the upload pipeline). **Do not use for experiments.** "
            "Re-upload without `--debug` for the full dataset.\n\n"
        )
    else:
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
            "Local Waterbirds root (eight subscenario folders after migration; "
            "legacy ``test_split/`` is migrated in place before full upload)"
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
            "Upload only N images per subscenario folder (see --debug-per-group) "
            "via upload_folder (fast sanity check)"
        ),
    )
    parser.add_argument(
        "--debug-per-group",
        type=int,
        default=2,
        metavar="N",
        help="With --debug, copy up to N images per subscenario folder (default: 2)",
    )
    parser.add_argument(
        "--dataset-script",
        action="store_true",
        help=(
            "Upload legacy ``*.py`` Hub dataset builder (breaks ``load_dataset`` on "
            "``datasets``≥4; not needed for the viewer — README ``configs`` are used instead)"
        ),
    )
    parser.add_argument(
        "--dataset-script-filename",
        default=None,
        metavar="NAME.py",
        help=(
            "With ``--dataset-script``, filename on the Hub (default: ``<repo_basename>.py``)"
        ),
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
    if args.dataset_script_filename and not args.dataset_script:
        raise SystemExit("--dataset-script-filename requires --dataset-script")

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

    script_fn = (
        args.dataset_script_filename
        if args.dataset_script_filename
        else _default_dataset_script_filename(args.repo_id)
    )
    if args.dataset_script:
        print(
            "Warning: Hub ``*.py`` dataset scripts are not supported by ``datasets`` 4.x; "
            "prefer README ``configs`` only."
        )
        _upload_dataset_loading_script(api, args.repo_id, args.token, script_fn)
    else:
        _try_delete_remote_dataset_script(
            api, args.repo_id, args.token, script_fn
        )

    if args.debug:
        per = args.debug_per_group
        with tempfile.TemporaryDirectory(prefix="waterbirds_hf_debug_") as tmp:
            n, rels = build_debug_waterbirds_staging(wb, tmp, per_group=per)
            if n == 0:
                raise SystemExit(
                    "Debug staging is empty (no images found for subscenario folders)."
                )
            max_expected = per * len(REQUIRED_TOP_LEVEL)
            print(
                f"Debug mode: staging {n} files "
                f"(≤{max_expected} = {per}×{len(REQUIRED_TOP_LEVEL)} subscenarios) …"
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
                "Migrating in place to eight subscenario folders …"
            )
            migrate_legacy_tar_extract_to_hub_layout(wb)
        migrate_split_layout_to_subscenarios(wb)
        assert_waterbirds_layout(wb)

        patterns = [f"{name}/*" for name in REQUIRED_TOP_LEVEL]
        print(
            f"Uploading {len(patterns)} subscenario trees via upload_large_folder …"
        )
        api.upload_large_folder(
            repo_id=args.repo_id,
            folder_path=wb,
            repo_type="dataset",
            allow_patterns=patterns,
            num_workers=args.num_workers,
            print_report=True,
        )

    print(f"Done. Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
