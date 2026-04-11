#!/usr/bin/env python3
"""
Verify that Waterbirds extracted from the UC+WB+CA Google Drive tar matches the
Hugging Face dataset snapshot (same relative paths and byte-identical files).

Downloads the tar (unless --tar-path), extracts only the Waterbirds subtree,
downloads the HF snapshot into a temporary directory, then compares SHA-256 per file.
README.md and other Hub-only root files are ignored on the HF side.

Legacy folders (``test_split/``) or older ``FG_plus_BG``/``FG`` splits are normalized
to the core eight-folder Hub layout in a temp copy when needed, without mutating
the originals. By default, ``*_bg_only`` trees and auxiliary ``bg_only/`` are ignored when
comparing the Google Drive tar to the Hub snapshot (see ``compare_waterbirds_trees``).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from stuned.utility.utils import download_file, remove_file_or_folder

import waterbirds_hf_common

# Same URL as scripts/download_datasets_and_checkpoints.py
UC_WB_CA_URL = (
    "https://drive.google.com/uc?id=1fRirj3s7ndY-cj25pqvHKCLwYfxr4_Ks"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tar-path",
        default=None,
        help="Existing .tar file from the Google Drive bundle (skips download if set)",
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="HF dataset id (default: env or waterbirds_hf_common default)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token for private datasets",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Print temp directories and do not delete them (for debugging)",
    )
    parser.add_argument(
        "--strict-bg-only-compare",
        action="store_true",
        help=(
            "Include ``*_bg_only`` and ``bg_only/`` in the byte-wise tree comparison "
            "(default skips them so the Drive tar matches a Hub tree with extra bg-only crops)"
        ),
    )
    parser.add_argument(
        "--from-gdrive-tree",
        default=None,
        help=(
            "Use this existing Waterbirds folder (e.g. extracted from the tar) "
            "instead of downloading the archive"
        ),
    )
    parser.add_argument(
        "--from-hf-tree",
        default=None,
        help=(
            "Use this existing folder (e.g. snapshot_download output) "
            "instead of downloading from the Hub"
        ),
    )
    args = parser.parse_args()

    hf_repo = args.hf_repo or waterbirds_hf_common.default_hf_repo_id()

    tmp_tar = None
    tmp_root = None
    tar_path = args.tar_path
    rc = 1
    legacy_compare_dirs: list[str] = []
    try:
        if args.from_gdrive_tree and args.from_hf_tree:
            gdrive_wb = os.path.abspath(args.from_gdrive_tree)
            hf_wb = os.path.abspath(args.from_hf_tree)
            g_cmp, tg = waterbirds_hf_common.prepare_tree_for_compare(gdrive_wb)
            h_cmp, th = waterbirds_hf_common.prepare_tree_for_compare(hf_wb)
            if tg:
                legacy_compare_dirs.append(tg)
            if th:
                legacy_compare_dirs.append(th)
            waterbirds_hf_common.assert_waterbirds_layout(g_cmp)
            waterbirds_hf_common.assert_waterbirds_layout(h_cmp)
            gdrive_side, hf_side = g_cmp, h_cmp
        else:
            if args.from_gdrive_tree or args.from_hf_tree:
                parser.error(
                    "Pass both --from-gdrive-tree and --from-hf-tree, or neither."
                )
            if tar_path is None:
                tmp_tar = tempfile.NamedTemporaryFile(
                    suffix=".tar", delete=False
                ).name
                print(f"Downloading Google Drive archive to {tmp_tar} …")
                download_file(tmp_tar, UC_WB_CA_URL)
                tar_path = tmp_tar

            tmp_root = tempfile.mkdtemp(prefix="waterbirds_cmp_")
            gdrive_wb = os.path.join(tmp_root, "from_gdrive")
            hf_wb = os.path.join(tmp_root, "from_hf")

            print(f"Extracting Waterbirds from tar → {gdrive_wb} …")
            waterbirds_hf_common.extract_waterbirds_from_uc_wb_ca_tar(
                tar_path, gdrive_wb
            )
            waterbirds_hf_common.assert_waterbirds_layout(gdrive_wb)

            print(f"Downloading Hugging Face snapshot ({hf_repo}) → {hf_wb} …")
            waterbirds_hf_common.download_waterbirds_snapshot(
                hf_wb, repo_id=hf_repo, token=args.hf_token
            )
            waterbirds_hf_common.assert_waterbirds_layout(hf_wb)
            gdrive_side, hf_side = gdrive_wb, hf_wb

        ignore = {"README.md", ".gitattributes"}
        ok, msgs = waterbirds_hf_common.compare_waterbirds_trees(
            gdrive_side,
            hf_side,
            ignore_top_files=ignore,
            ignore_bg_only_subtrees=not args.strict_bg_only_compare,
        )
        if ok:
            print(
                "OK: Google Drive (tar) and Hugging Face trees are identical."
            )
            rc = 0
        else:
            print("MISMATCH:")
            for m in msgs[:200]:
                print(" ", m)
            if len(msgs) > 200:
                print(f" … and {len(msgs) - 200} more")
            rc = 1

        if args.keep:
            print(f"Kept temp root: {tmp_root}")
    finally:
        for d in legacy_compare_dirs:
            if os.path.isdir(d) and not args.keep:
                remove_file_or_folder(d)
        if (
            tmp_root
            and os.path.isdir(tmp_root)
            and not args.keep
            and not (args.from_gdrive_tree and args.from_hf_tree)
        ):
            remove_file_or_folder(tmp_root)
        if tmp_tar and os.path.isfile(tmp_tar):
            remove_file_or_folder(tmp_tar)

    sys.exit(rc)


if __name__ == "__main__":
    main()
