"""
Waterbirds directory layout on disk and on the Hugging Face dataset viewer.

Two top-level *splits* (for clear browsing on the Hub):
  - ``FG_plus_BG`` — original images (foreground + background).
  - ``FG`` — foreground-only crops (same grouping).

Under each split, four *group* folders use descriptive names (indices match ``group_i``):
  0 landbird_on_land, 1 landbird_on_water, 2 waterbird_on_land, 3 waterbird_on_water.

Legacy layout (Google Drive tar / older OCCAM checkouts) is still recognized for
migration and upload scripts:

  ``test_split/group_{0..3}/`` and ``FG-Only/test_split/group_{0..3}/``.
"""

from __future__ import annotations

import os
import shutil
from typing import Literal, Optional

# Hub / current layout — two splits, named groups (class subdirs 0/ and 1/ unchanged).
SPLIT_FG_PLUS_BG = "FG_plus_BG"
# Short folder name so the Hub "Files" view matches FG + BG vs FG.
SPLIT_FG_ONLY = "FG"

GROUP_SUBDIRS = (
    "landbird_on_land",
    "landbird_on_water",
    "waterbird_on_land",
    "waterbird_on_water",
)

REQUIRED_TOP_LEVEL = (SPLIT_FG_PLUS_BG, SPLIT_FG_ONLY)

LayoutKind = Literal["hub", "legacy"]


def detect_layout(waterbirds_root: str) -> LayoutKind:
    hub_bg = os.path.join(waterbirds_root, SPLIT_FG_PLUS_BG)
    hub_fg = os.path.join(waterbirds_root, SPLIT_FG_ONLY)
    if os.path.isdir(hub_bg) and os.path.isdir(hub_fg):
        return "hub"
    legacy = os.path.join(waterbirds_root, "test_split")
    if os.path.isdir(legacy):
        return "legacy"
    raise FileNotFoundError(
        f"No recognized Waterbirds layout under {waterbirds_root!r}: "
        f"expected either {SPLIT_FG_PLUS_BG!r}/ and {SPLIT_FG_ONLY!r}/ "
        f"or legacy {legacy!r}/."
    )


def assert_waterbirds_layout(waterbirds_root: str) -> None:
    """Require current Hub layout (after download or migration)."""
    kind = detect_layout(waterbirds_root)
    if kind != "hub":
        raise FileNotFoundError(
            f"Expected Hub layout under {waterbirds_root!r} "
            f"({SPLIT_FG_PLUS_BG!r}, {SPLIT_FG_ONLY!r} with named group folders). "
            f"Found legacy layout; migrate or re-download from Hugging Face."
        )
    for split in REQUIRED_TOP_LEVEL:
        sp = os.path.join(waterbirds_root, split)
        if not os.path.isdir(sp):
            raise FileNotFoundError(f"Missing split directory: {sp!r}")
        for gname in GROUP_SUBDIRS:
            gp = os.path.join(sp, gname)
            if not os.path.isdir(gp):
                raise FileNotFoundError(f"Missing group directory: {gp!r}")


def assert_uploadable_waterbirds(waterbirds_root: str) -> LayoutKind:
    """Hub or legacy tree with at least one image group (for upload script)."""
    kind = detect_layout(waterbirds_root)
    if kind == "hub":
        assert_waterbirds_layout(waterbirds_root)
        return kind
    for gid in range(4):
        if os.path.isdir(
            os.path.join(waterbirds_root, "test_split", f"group_{gid}")
        ):
            return kind
    raise FileNotFoundError(
        f"Legacy Waterbirds under {waterbirds_root!r} has no test_split/group_* folders."
    )


def path_with_background(base: str, group_id: int) -> str:
    return os.path.join(base, SPLIT_FG_PLUS_BG, GROUP_SUBDIRS[group_id])


def path_foreground_only(base: str, group_id: int) -> str:
    return os.path.join(base, SPLIT_FG_ONLY, GROUP_SUBDIRS[group_id])


def materialize_hub_layout_copy(
    src_root: str, dest_root: str, *, layout: Optional[LayoutKind] = None
) -> None:
    """
    Copy images into ``dest_root`` using the Hub layout. Source may be hub (subset copy)
    or legacy (full tree). Does not delete ``src_root``.
    """
    layout = layout or detect_layout(src_root)
    os.makedirs(dest_root, exist_ok=True)
    if layout == "hub":
        for split in REQUIRED_TOP_LEVEL:
            for gname in GROUP_SUBDIRS:
                src_g = os.path.join(src_root, split, gname)
                if not os.path.isdir(src_g):
                    continue
                dst_g = os.path.join(dest_root, split, gname)
                shutil.copytree(src_g, dst_g, dirs_exist_ok=True)
        return

    # legacy -> hub
    for gid, gname in enumerate(GROUP_SUBDIRS):
        src_g = os.path.join(src_root, "test_split", f"group_{gid}")
        if os.path.isdir(src_g):
            dst_g = os.path.join(dest_root, SPLIT_FG_PLUS_BG, gname)
            shutil.copytree(src_g, dst_g, dirs_exist_ok=True)
        src_fg = os.path.join(src_root, "FG-Only", "test_split", f"group_{gid}")
        if os.path.isdir(src_fg):
            dst_fg = os.path.join(dest_root, SPLIT_FG_ONLY, gname)
            shutil.copytree(src_fg, dst_fg, dirs_exist_ok=True)


def migrate_legacy_tar_extract_to_hub_layout(waterbirds_root: str) -> None:
    """
    In-place: rename legacy ``test_split/group_*`` and ``FG-Only/test_split/group_*``
    to ``FG_plus_BG/<name>/`` and ``FG/<name>/``. Removes empty legacy dirs.
    """
    if detect_layout(waterbirds_root) != "legacy":
        return
    ts = os.path.join(waterbirds_root, "test_split")
    for gid, _ in enumerate(GROUP_SUBDIRS):
        src = os.path.join(ts, f"group_{gid}")
        if not os.path.isdir(src):
            continue
        dst = path_with_background(waterbirds_root, gid)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
    if os.path.isdir(ts) and not os.listdir(ts):
        os.rmdir(ts)

    fg_ts = os.path.join(waterbirds_root, "FG-Only", "test_split")
    for gid, _ in enumerate(GROUP_SUBDIRS):
        src = os.path.join(fg_ts, f"group_{gid}")
        if not os.path.isdir(src):
            continue
        dst = path_foreground_only(waterbirds_root, gid)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
    fg_only = os.path.join(waterbirds_root, "FG-Only")
    if os.path.isdir(fg_ts) and not os.listdir(fg_ts):
        os.rmdir(fg_ts)
    if os.path.isdir(fg_only) and not os.listdir(fg_only):
        os.rmdir(fg_only)
