"""
Shared helpers for Hugging Face Waterbirds snapshots, UC-WB-CA tar subset extraction,
and byte-identical tree comparisons.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
from typing import Iterable, Iterator, List, Optional, Tuple

# Scripts may import this module with only ``scripts/`` on sys.path; ensure repo root.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from occam.datasets.waterbirds_layout import (
    AUX_BG_ONLY_ROOT,
    REQUIRED_TOP_LEVEL,
    assert_waterbirds_layout,
    migrate_legacy_tar_extract_to_hub_layout,
    subscenario_background_only,
)

# Default dataset repo once published; override with OCCAM_WATERBIRDS_HF_DATASET or CLI.
DEFAULT_WATERBIRDS_HF_DATASET = "AlexanderRubinstein/OCCAM-Waterbirds"


def default_hf_repo_id() -> str:
    return os.environ.get(
        "OCCAM_WATERBIRDS_HF_DATASET", DEFAULT_WATERBIRDS_HF_DATASET
    )


def waterbirds_relative_path_from_tar_member(member_name: str) -> Optional[str]:
    """
    Map a tar member name to a path relative to the Waterbirds/ root.
    Handles prefixes like datasets/Waterbirds/... or ./datasets/Waterbirds/...
    """
    n = member_name.replace("\\", "/").lstrip("./")
    marker = "Waterbirds/"
    idx = n.find(marker)
    if idx == -1:
        return None
    return n[idx + len(marker) :]


def extract_waterbirds_from_uc_wb_ca_tar(
    tar_path: str, dest_waterbirds_root: str
) -> None:
    """
    Extract only files under .../Waterbirds/ from the UrbanCars+Waterbirds+CounterAnimals
    Google Drive archive into dest_waterbirds_root.

    The archive uses legacy paths (``test_split/group_*``, ``FG-Only/test_split/group_*``);
    this rewrites them in-place to eight core top-level subscenario folders (e.g.
    ``landbird_on_land``, ``landbird_on_land_fg_only``, …). If the archive also contains
    ``bg_only/test_split/group_*``, leave it in place; use
    :func:`occam.datasets.waterbirds_layout.materialize_bg_only_subscenarios` to populate
    ``*_bg_only`` folders before a full Hub upload.
    """
    os.makedirs(dest_waterbirds_root, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            rel = waterbirds_relative_path_from_tar_member(member.name)
            if rel is None or rel.endswith("/"):
                continue
            target = os.path.join(dest_waterbirds_root, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with tf.extractfile(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    migrate_legacy_tar_extract_to_hub_layout(dest_waterbirds_root)


def download_waterbirds_snapshot(
    dest_waterbirds_root: str,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    from huggingface_hub import snapshot_download

    repo_id = repo_id or default_hf_repo_id()
    os.makedirs(os.path.dirname(dest_waterbirds_root) or ".", exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dest_waterbirds_root,
        token=token,
        local_dir_use_symlinks=False,
    )


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _waterbirds_compare_ignore_top_dirs(
    *,
    ignore_bg_only_subtrees: bool,
    extra: Optional[Iterable[str]],
) -> set[str]:
    dirs = set(extra or [])
    if ignore_bg_only_subtrees:
        dirs.add(AUX_BG_ONLY_ROOT)
        dirs.update(subscenario_background_only(gid) for gid in range(4))
    return dirs


def iter_compare_files(
    root: str,
    ignore_top_files: Optional[Iterable[str]] = None,
    ignore_top_level_dirs: Optional[Iterable[str]] = None,
) -> Iterator[Tuple[str, str]]:
    """
    Yields (relative_path_posix, sha256) for every file under root, sorted by path.
    Skips symlink targets as separate logic (none expected).

    ``ignore_top_level_dirs``: do not descend into these directory names directly under
    ``root`` (useful to skip auxiliary ``bg_only/`` or ``*_bg_only`` trees when comparing
    tar-derived trees to Hub snapshots).
    """
    ignore = set(ignore_top_files or [])
    ignore_dirs = set(ignore_top_level_dirs or [])
    root = os.path.abspath(root)
    out: List[Tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.abspath(dirpath) == root:
            dirnames[:] = [d for d in sorted(dirnames) if d not in ignore_dirs]
        # stable walk
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            if dirpath == root and name in ignore:
                continue
            full = os.path.join(dirpath, name)
            if not os.path.isfile(full):
                continue
            rel = os.path.relpath(full, root)
            rel_posix = rel.replace(os.sep, "/")
            out.append((rel_posix, file_sha256(full)))
    out.sort(key=lambda x: x[0])
    for item in out:
        yield item


def compare_waterbirds_trees(
    dir_a: str,
    dir_b: str,
    ignore_top_files: Optional[Iterable[str]] = None,
    ignore_top_level_dirs: Optional[Iterable[str]] = None,
    ignore_bg_only_subtrees: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Return (ok, messages). Compares relative paths and SHA-256 of every file.

    When ``ignore_bg_only_subtrees`` is True (default), skips the auxiliary ``bg_only/``
    tree and top-level ``*_bg_only`` folders so a Google Drive tar (eight core folders) can
    match a Hub tree that also ships background-only crops. Pass ``False`` for a strict
    comparison of all twelve subscenario trees.
    """
    merged_dirs = _waterbirds_compare_ignore_top_dirs(
        ignore_bg_only_subtrees=ignore_bg_only_subtrees,
        extra=ignore_top_level_dirs,
    )
    a = list(
        iter_compare_files(
            dir_a,
            ignore_top_files=ignore_top_files,
            ignore_top_level_dirs=merged_dirs,
        )
    )
    b = list(
        iter_compare_files(
            dir_b,
            ignore_top_files=ignore_top_files,
            ignore_top_level_dirs=merged_dirs,
        )
    )
    msgs: List[str] = []
    if len(a) != len(b):
        msgs.append(f"File count differs: {len(a)} vs {len(b)}")
    am = dict(a)
    bm = dict(b)
    all_paths = sorted(set(am.keys()) | set(bm.keys()))
    for p in all_paths:
        if p not in am:
            msgs.append(f"Missing in first tree: {p}")
            continue
        if p not in bm:
            msgs.append(f"Missing in second tree: {p}")
            continue
        if am[p] != bm[p]:
            msgs.append(f"Content differs: {p}")
    return (len(msgs) == 0, msgs)


def prepare_tree_for_compare(waterbirds_root: str) -> tuple[str, Optional[str]]:
    """
    If ``waterbirds_root`` is already the core Hub layout (eight folders), return it unchanged.
    Otherwise copy into a temp dir in that layout (legacy or ``FG_plus_BG``/``FG`` split)
    and return ``(temp_path, temp_path)`` for cleanup.
    """
    import tempfile

    from occam.datasets.waterbirds_layout import (
        detect_layout,
        materialize_hub_layout_copy,
    )

    root = os.path.abspath(waterbirds_root)
    kind = detect_layout(root)
    if kind == "hub":
        return root, None
    tmp = tempfile.mkdtemp(prefix="waterbirds_compare_")
    materialize_hub_layout_copy(root, tmp, layout=kind)
    return tmp, tmp
