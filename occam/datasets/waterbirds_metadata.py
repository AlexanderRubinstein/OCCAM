"""
Waterbirds ``metadata.csv`` helpers (WILDS-style columns) for pairing fg+bg and bg-only files.

Columns used: ``img_id``, ``img_filename``, ``y``, ``split``, ``place``, ``place_filename``.

On disk, coarse class folders match ``y`` (``0`` = landbird, ``1`` = waterbird in metadata).
Spurious-cue groups match ``(y, place)`` → ``group_0`` … ``group_3`` / Hub subscenario names.
"""

from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Set, Tuple

from occam.datasets.waterbirds_layout import (
    GROUP_SUBDIRS,
    path_background_only,
    path_with_background,
)

METADATA_CSV_FILENAME = "metadata.csv"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Requested source of pure backgrounds for ``*_bg_only`` synthesis.
DEFAULT_PLACES_ROOT = os.path.join(
    _PROJECT_ROOT, "data", "dataset", "places", "data_256_standard"
)

# WILDS Waterbirds test split id in metadata.csv (matches on-disk fg+bg trees in OCCAM).
METADATA_SPLIT_TEST = "2"


def occam_image_basename(img_filename: str) -> str:
    """
    Basename used under ``<subscenario>/{y}/`` for the composite (fg+bg) image.

    Example: ``004.Groove_billed_Ani/Groove_Billed_Ani_0005_1750.jpg`` →
    ``004.Groove_billed_Ani_Groove_Billed_Ani_0005_1750.jpg.jpg``.
    """
    base = img_filename.replace("\\", "/").strip("/").replace("/", "_")
    if not base.lower().endswith(".jpg"):
        base = f"{base}.jpg"
    if not base.lower().endswith(".jpg.jpg"):
        base = f"{base}.jpg"
    return base


def group_id_from_metadata(y: int | str, place: int | str) -> int:
    """Map metadata ``(y, place)`` to historical ``group_0`` … ``group_3`` / Hub subscenario index."""
    y_i, place_i = int(y), int(place)
    if y_i == 0 and place_i == 0:
        return 0
    if y_i == 0 and place_i == 1:
        return 1
    if y_i == 1 and place_i == 0:
        return 2
    if y_i == 1 and place_i == 1:
        return 3
    raise ValueError(
        f"Invalid Waterbirds metadata (y, place)=({y!r}, {place!r})"
    )


def folder_label_from_metadata(y: int | str) -> str:
    """Class subfolder name under each subscenario (matches ``y`` on disk)."""
    return str(int(y))


def default_metadata_path(waterbirds_root: str) -> str:
    return os.path.join(waterbirds_root, METADATA_CSV_FILENAME)


def iter_metadata_rows(metadata_path: str) -> Iterator[Dict[str, str]]:
    with open(metadata_path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def default_places_root() -> str:
    return DEFAULT_PLACES_ROOT


def places_source_path(places_root: str, place_filename: str) -> str:
    rel = place_filename.replace("\\", "/").lstrip("/")
    return os.path.join(places_root, rel)


def hub_paths_for_metadata_row(
    waterbirds_root: str, places_root: str, row: Dict[str, str]
) -> Tuple[str, str, str, int, str]:
    """
    Return ``(fg_bg_path, bg_only_path, places_bg_src_path, group_id, folder_label)``.
    """
    gid = group_id_from_metadata(row["y"], row["place"])
    lab = folder_label_from_metadata(row["y"])
    basename = occam_image_basename(row["img_filename"])
    fg_bg = os.path.join(
        path_with_background(waterbirds_root, gid), lab, basename
    )
    bg_only = os.path.join(
        path_background_only(waterbirds_root, gid), lab, basename
    )
    place_src = places_source_path(places_root, row["place_filename"])
    return fg_bg, bg_only, place_src, gid, lab


@dataclass
class BgOnlySyncStats:
    copied: int = 0
    missing_aux: int = 0
    missing_fg: int = 0
    pruned: int = 0
    rows_considered: int = 0


def sync_bg_only_subscenarios_from_metadata(
    waterbirds_root: str,
    metadata_path: Optional[str] = None,
    places_root: Optional[str] = None,
    *,
    prune: bool = True,
    require_fg_exists: bool = True,
    splits: Optional[Set[str]] = None,
) -> BgOnlySyncStats:
    """
    Populate ``*_bg_only`` trees so each background image uses the **same basename** as the
    matching fg+bg composite listed in ``metadata.csv``.

    Background pixels are taken from ``places_root`` + ``place_filename`` from metadata.
    Rows without a composite on disk are skipped when ``require_fg_exists`` is True.

    With ``prune=True``, removes ``*_bg_only`` files that are not paired with an on-disk
    fg+bg file for a selected metadata row.
    """
    waterbirds_root = os.path.abspath(waterbirds_root)
    metadata_path = metadata_path or default_metadata_path(waterbirds_root)
    places_root = os.path.abspath(places_root or default_places_root())
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Waterbirds metadata not found: {metadata_path!r} "
            f"(expected {METADATA_CSV_FILENAME} next to the dataset root)."
        )
    if not os.path.isdir(places_root):
        raise FileNotFoundError(
            f"Places root not found: {places_root!r}. "
            "Set places_root explicitly or place data under "
            f"{default_places_root()!r}."
        )

    stats = BgOnlySyncStats()
    expected_bg_paths: Set[str] = set()

    for row in iter_metadata_rows(metadata_path):
        if splits is not None and row.get("split") not in splits:
            continue
        stats.rows_considered += 1
        fg_bg, bg_only, place_src, _gid, _lab = hub_paths_for_metadata_row(
            waterbirds_root, places_root, row
        )
        if require_fg_exists and not os.path.isfile(fg_bg):
            stats.missing_fg += 1
            continue
        expected_bg_paths.add(os.path.abspath(bg_only))
        if not os.path.isfile(place_src):
            stats.missing_aux += 1
            continue
        os.makedirs(os.path.dirname(bg_only), exist_ok=True)
        shutil.copy2(place_src, bg_only)
        stats.copied += 1

    if prune:
        for gid, gname in enumerate(GROUP_SUBDIRS):
            bg_root = path_background_only(waterbirds_root, gid)
            if not os.path.isdir(bg_root):
                continue
            for lab in os.listdir(bg_root):
                lab_dir = os.path.join(bg_root, lab)
                if not os.path.isdir(lab_dir):
                    continue
                for fn in os.listdir(lab_dir):
                    full = os.path.abspath(os.path.join(lab_dir, fn))
                    if not os.path.isfile(full):
                        continue
                    if full not in expected_bg_paths:
                        os.remove(full)
                        stats.pruned += 1

    return stats


def materialize_bg_only_subscenarios(waterbirds_root: str) -> int:
    """
    Legacy entry point: metadata-driven sync (replaces wholesale ``copytree``).

    Returns the number of background files written.
    """
    stats = sync_bg_only_subscenarios_from_metadata(
        waterbirds_root,
        require_fg_exists=True,
        prune=True,
    )
    return stats.copied
