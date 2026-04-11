"""
Waterbirds directory layout on disk and on the Hugging Face dataset viewer.

Eight top-level *subscenario* folders (each holds class subfolders ``0`` and ``1``):

  - ``<group>`` — original image (foreground + background) for that group.
  - ``<group>_fg_only`` — foreground-only crop for the same group.

Where ``<group>`` is one of:
  ``landbird_on_land``, ``landbird_on_water``, ``waterbird_on_land``, ``waterbird_on_water``.

Legacy layout (Google Drive tar / older OCCAM checkouts):

  ``test_split/group_{0..3}/`` and ``FG-Only/test_split/group_{0..3}/``.

Intermediate layout (older Hub drops, still supported for one-step migration):

  ``FG_plus_BG/<group>/`` and ``FG/<group>/``.
"""

from __future__ import annotations

import os
import shutil
from typing import Literal, Optional, Tuple

# Older two-split layout (migration source only).
SPLIT_FG_PLUS_BG = "FG_plus_BG"
SPLIT_FG_ONLY = "FG"

# Four groups (same order as historical group_0 … group_3).
GROUP_SUBDIRS = (
    "landbird_on_land",
    "landbird_on_water",
    "waterbird_on_land",
    "waterbird_on_water",
)

SUBSCENARIO_SUFFIX_FG_ONLY = "_fg_only"


def subscenario_with_background(group_id: int) -> str:
    return GROUP_SUBDIRS[group_id]


def subscenario_foreground_only(group_id: int) -> str:
    return f"{GROUP_SUBDIRS[group_id]}{SUBSCENARIO_SUFFIX_FG_ONLY}"


def all_subscenario_dir_names() -> Tuple[str, ...]:
    names: list[str] = []
    for g in GROUP_SUBDIRS:
        names.append(g)
        names.append(f"{g}{SUBSCENARIO_SUFFIX_FG_ONLY}")
    return tuple(names)


REQUIRED_TOP_LEVEL = all_subscenario_dir_names()

LayoutKind = Literal["hub", "legacy", "hub_split"]


def _has_split_layout(waterbirds_root: str) -> bool:
    return os.path.isdir(
        os.path.join(waterbirds_root, SPLIT_FG_PLUS_BG)
    ) and os.path.isdir(os.path.join(waterbirds_root, SPLIT_FG_ONLY))


def _has_subscenario_layout(waterbirds_root: str) -> bool:
    for name in REQUIRED_TOP_LEVEL:
        if not os.path.isdir(os.path.join(waterbirds_root, name)):
            return False
    return True


def detect_layout(waterbirds_root: str) -> LayoutKind:
    if _has_subscenario_layout(waterbirds_root):
        return "hub"
    if _has_split_layout(waterbirds_root):
        return "hub_split"
    legacy = os.path.join(waterbirds_root, "test_split")
    if os.path.isdir(legacy):
        return "legacy"
    raise FileNotFoundError(
        f"No recognized Waterbirds layout under {waterbirds_root!r}: "
        f"expected subscenario dirs ({subscenario_with_background(0)!r}, …), "
        f"split dirs {SPLIT_FG_PLUS_BG!r}/{SPLIT_FG_ONLY!r}, or legacy {legacy!r}/."
    )


def migrate_split_layout_to_subscenarios(waterbirds_root: str) -> None:
    """``FG_plus_BG/<g>/`` + ``FG/<g>/`` → eight top-level subscenario folders."""
    if detect_layout(waterbirds_root) != "hub_split":
        return
    for gid, gname in enumerate(GROUP_SUBDIRS):
        src = os.path.join(waterbirds_root, SPLIT_FG_PLUS_BG, gname)
        dst = os.path.join(waterbirds_root, subscenario_with_background(gid))
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
        src_fg = os.path.join(waterbirds_root, SPLIT_FG_ONLY, gname)
        dst_fg = os.path.join(waterbirds_root, subscenario_foreground_only(gid))
        if os.path.isdir(src_fg):
            if os.path.exists(dst_fg):
                shutil.rmtree(dst_fg)
            shutil.move(src_fg, dst_fg)
    for split in (SPLIT_FG_PLUS_BG, SPLIT_FG_ONLY):
        p = os.path.join(waterbirds_root, split)
        if os.path.isdir(p) and not os.listdir(p):
            os.rmdir(p)


def assert_waterbirds_layout(waterbirds_root: str) -> None:
    """Require current Hub subscenario layout (eight top-level folders)."""
    migrate_split_layout_to_subscenarios(waterbirds_root)
    if not _has_subscenario_layout(waterbirds_root):
        kind = detect_layout(waterbirds_root)
        raise FileNotFoundError(
            f"Expected eight Waterbirds subscenario folders under {waterbirds_root!r} "
            f"(e.g. {subscenario_with_background(0)!r}, {subscenario_foreground_only(0)!r}, …). "
            f"After split migration, layout is {kind!r}. Run legacy migration or re-download."
        )
    for name in REQUIRED_TOP_LEVEL:
        p = os.path.join(waterbirds_root, name)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Missing subscenario directory: {p!r}")


def assert_uploadable_waterbirds(waterbirds_root: str) -> LayoutKind:
    """Hub, split-hub, or legacy tree usable for upload."""
    kind = detect_layout(waterbirds_root)
    if kind == "hub_split":
        migrate_split_layout_to_subscenarios(waterbirds_root)
        kind = detect_layout(waterbirds_root)
    if kind == "hub":
        assert_waterbirds_layout(waterbirds_root)
        return "hub"
    if kind == "legacy":
        for gid in range(4):
            if os.path.isdir(
                os.path.join(waterbirds_root, "test_split", f"group_{gid}")
            ):
                return "legacy"
    raise FileNotFoundError(
        f"Legacy Waterbirds under {waterbirds_root!r} has no test_split/group_* folders."
    )


def path_with_background(base: str, group_id: int) -> str:
    return os.path.join(base, subscenario_with_background(group_id))


def path_foreground_only(base: str, group_id: int) -> str:
    return os.path.join(base, subscenario_foreground_only(group_id))


def group_id_from_subscenario_folder(subdir: str) -> int:
    if subdir.endswith(SUBSCENARIO_SUFFIX_FG_ONLY):
        base = subdir[: -len(SUBSCENARIO_SUFFIX_FG_ONLY)]
    else:
        base = subdir
    return GROUP_SUBDIRS.index(base)


def is_fg_only_subscenario_folder(subdir: str) -> bool:
    return subdir.endswith(SUBSCENARIO_SUFFIX_FG_ONLY)


def source_dir_for_subscenario(
    src_root: str, layout: LayoutKind, subdir: str
) -> str:
    """Where images for one subscenario folder live under ``src_root``."""
    gid = group_id_from_subscenario_folder(subdir)
    gname = GROUP_SUBDIRS[gid]
    if layout == "hub":
        return os.path.join(src_root, subdir)
    if layout == "hub_split":
        if is_fg_only_subscenario_folder(subdir):
            return os.path.join(src_root, SPLIT_FG_ONLY, gname)
        return os.path.join(src_root, SPLIT_FG_PLUS_BG, gname)
    # legacy
    if is_fg_only_subscenario_folder(subdir):
        return os.path.join(src_root, "FG-Only", "test_split", f"group_{gid}")
    return os.path.join(src_root, "test_split", f"group_{gid}")


def materialize_hub_layout_copy(
    src_root: str, dest_root: str, *, layout: Optional[LayoutKind] = None
) -> None:
    """
    Copy images into ``dest_root`` using the subscenario Hub layout (eight folders).
    Does not modify ``src_root``. Source may be hub, hub_split, or legacy.
    """
    layout = layout or detect_layout(src_root)
    os.makedirs(dest_root, exist_ok=True)
    if layout == "hub":
        for name in REQUIRED_TOP_LEVEL:
            src_g = os.path.join(src_root, name)
            if not os.path.isdir(src_g):
                continue
            dst_g = os.path.join(dest_root, name)
            shutil.copytree(src_g, dst_g, dirs_exist_ok=True)
        return
    if layout == "hub_split":
        for name in REQUIRED_TOP_LEVEL:
            src_g = source_dir_for_subscenario(src_root, "hub_split", name)
            if not os.path.isdir(src_g):
                continue
            dst_g = os.path.join(dest_root, name)
            shutil.copytree(src_g, dst_g, dirs_exist_ok=True)
        return

    # legacy -> hub subscenarios
    for name in REQUIRED_TOP_LEVEL:
        src_g = source_dir_for_subscenario(src_root, "legacy", name)
        if not os.path.isdir(src_g):
            continue
        dst_g = os.path.join(dest_root, name)
        shutil.copytree(src_g, dst_g, dirs_exist_ok=True)


def migrate_legacy_tar_extract_to_hub_layout(waterbirds_root: str) -> None:
    """
    In-place: legacy ``test_split/group_*`` / ``FG-Only/test_split/group_*``, or
    ``hub_split`` (FG_plus_BG/FG), → eight subscenario folders.
    """
    migrate_split_layout_to_subscenarios(waterbirds_root)
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
