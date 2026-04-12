#!/usr/bin/env python3
"""
Match each ``*_bg_only`` image to the most similar **foreground+background** image in the
paired subscenario (same spurious-cue group and same coarse label folder ``0`` / ``1``),
using **SigLIP** (OpenCLIP) image embeddings and cosine similarity.

With ``--search-places``, instead match each ``*_bg_only`` to the nearest **Places365**
background synthesized the same way as in Polina Kirichenko's Waterbirds generation
notebook (``combine_and_mask`` with an all-zero bird mask on a black canvas = ``crop_and_resize``
of the place image to the bg canvas size). Land subscenarios (``*_on_land``) use the land
place categories; water-background subscenarios (``*_on_water``) use the water categories,
matching the notebook's ``target_places`` split.

Outputs a JSON mapping (and optional CSV) from background-only relative paths to full-scene
relative paths plus similarity scores. Use this when ``*_bg_only`` filenames do not align
with ``<group>/`` filenames.

Requires: ``torch``, ``open_clip``, ``Pillow``, ``tqdm`` (same stack as OCCAM CLIP / SigLIP eval).

Example:

  python scripts/match_waterbirds_bg_only_to_full_siglip.py \\
    --waterbirds-root data/datasets/Waterbirds \\
    --output data/datasets/Waterbirds/bg_only_to_full_siglip.json

  # Prefer pixel-identical backgrounds (after resizing full to bg size), then SigLIP tie-break:
  python scripts/match_waterbirds_bg_only_to_full_siglip.py ... --exact-pixel

  # Only landbird (label 1) folders:
  python scripts/match_waterbirds_bg_only_to_full_siglip.py ... --class 1

  # Nearest Places365 background (same warping as DFR notebook), not Waterbirds full frames:
  python scripts/match_waterbirds_bg_only_to_full_siglip.py ... \\
    --search-places --places-dir /path/to/places4waterbirds

  # Before Places matching, normalize canvases to one size (see
  # scripts/normalize_waterbirds_bg_only_resolution.py: --targets bg_only, full, or both).

  # Nearest warped Places365 scene for each original FG+BG frame (same pools as --search-places):
  python scripts/match_waterbirds_bg_only_to_full_siglip.py ... \\
    --match-to-original --places-dir /path/to/places365_root
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_wb_normalize_resolution_module = None


def _load_wb_normalize_resolution_module():
    """Lazy import of ``normalize_waterbirds_bg_only_resolution`` (square helpers)."""
    global _wb_normalize_resolution_module
    if _wb_normalize_resolution_module is None:
        path = os.path.join(
            _SCRIPT_DIR, "normalize_waterbirds_bg_only_resolution.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_occam_wb_normalize_resolution", path
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _wb_normalize_resolution_module = mod
    return _wb_normalize_resolution_module


def _prepare_places_square_region(
    norm_mod,
    pil: Image.Image,
    x0: int,
    y0: int,
    side: int,
    *,
    apply_same_square: bool,
    drop_black: bool,
) -> Image.Image:
    """Apply ``--drop-black`` (splice out square) and/or ``--apply-same-square`` (black paint)."""
    rgb = pil.convert("RGB")
    if drop_black:
        return norm_mod.cut_square_region_from_rgb(rgb, x0, y0, side)
    if apply_same_square:
        return norm_mod.apply_black_square_coords(rgb, x0, y0, side)
    return rgb


from occam.datasets.utils import DATASETS_PATH
from occam.datasets.waterbirds_layout import (
    GROUP_SUBDIRS,
    path_background_only,
    path_with_background,
)

DEBUG_MATCH_LIMIT = 3


def _save_debug_match_pair(
    row: dict,
    match_index: int,
    waterbirds_root: str,
    places_dir: Optional[str],
    debug_dir: str,
) -> None:
    """
    Write ``{idx:03d}_full*`` and ``{idx:03d}_matched_background*`` under ``debug_dir``
    (copies of source files, preserving extension when possible).
    """
    os.makedirs(debug_dir, exist_ok=True)
    base = os.path.join(debug_dir, f"{match_index:03d}")

    def _copy(src: Optional[str], dest_suffix: str) -> None:
        if not src or not os.path.isfile(src):
            return
        ext = os.path.splitext(src)[1] or ".jpg"
        shutil.copy2(src, f"{base}_{dest_suffix}{ext}")

    gid = GROUP_SUBDIRS.index(str(row["group"]))
    label = str(row["label"])

    full_src: Optional[str] = None
    bg_src: Optional[str] = None
    mt = row.get("match_target")

    if mt == "original_to_places365":
        if row.get("full_relpath"):
            full_src = os.path.join(waterbirds_root, row["full_relpath"])
        if places_dir and row.get("place_relpath"):
            bg_src = os.path.join(places_dir, row["place_relpath"])
    elif mt == "places365" and row.get("bg_only_relpath"):
        bn = os.path.basename(row["bg_only_relpath"])
        full_src = os.path.join(
            path_with_background(waterbirds_root, gid), label, bn
        )
        if places_dir and row.get("place_relpath"):
            bg_src = os.path.join(places_dir, row["place_relpath"])
    else:
        # bg_only → full (Waterbirds pool): paired full vs matched full
        if row.get("bg_only_relpath"):
            bn = os.path.basename(row["bg_only_relpath"])
            paired = os.path.join(
                path_with_background(waterbirds_root, gid), label, bn
            )
            if os.path.isfile(paired):
                full_src = paired
        if row.get("full_relpath"):
            cand = os.path.join(waterbirds_root, row["full_relpath"])
            if full_src is None:
                full_src = cand
            else:
                bg_src = cand
        if bg_src is None and row.get("bg_only_relpath"):
            bg_src = os.path.join(waterbirds_root, row["bg_only_relpath"])

    _copy(full_src, "full")
    _copy(bg_src, "matched_background")


try:
    import open_clip
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "open_clip is required (e.g. pip install open_clip_torch). "
        "OCCAM eval code loads SigLIP the same way.\n"
        f"This process is using Python: {sys.executable}\n"
        "If you started from run_waterbirds_bg_matching_background.sh, set PYTHON or "
        "OCCAM_PYTHON to your OCCAM venv interpreter (e.g. envs/occam/bin/python)."
    ) from e

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

try:
    _RESAMPLE = Image.Resampling.LANCZOS  # Pillow >= 9
except AttributeError:  # pragma: no cover
    _RESAMPLE = Image.LANCZOS  # type: ignore[attr-defined]


# --- Places365 / Waterbirds background warping (from group_DRO dataset_utils) ---------------
# MIT-licensed reference: https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/dataset_utils.py
# Used by generate_waterbirds_fg_bg.ipynb (deep_feature_reweighting).


def crop_and_resize(
    source_img: Image.Image, target_img: Image.Image
) -> Image.Image:
    """Resize/crop ``source_img`` to match ``target_img`` width and height (PIL sizes)."""
    source_width, source_height = source_img.size
    target_width, target_height = target_img.size

    if (source_width < target_width) or (source_height < target_height):
        width_resize = (
            target_width,
            int((target_width / source_width) * source_height),
        )
        if (width_resize[0] >= target_width) and (
            width_resize[1] >= target_height
        ):
            source_resized = source_img.resize(width_resize, _RESAMPLE)
        else:
            height_resize = (
                int((target_height / source_height) * source_width),
                target_height,
            )
            assert (height_resize[0] >= target_width) and (
                height_resize[1] >= target_height
            )
            source_resized = source_img.resize(height_resize, _RESAMPLE)
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)

    source_cropped = source_img.crop(resize)
    return source_cropped.resize((target_width, target_height), _RESAMPLE)


def combine_and_mask(
    img_new: Image.Image, mask: np.ndarray, img_black: Image.Image
) -> Image.Image:
    """Warp ``img_new`` to ``img_black`` size, mask by ``mask``, add onto ``img_black``."""
    img_resized = crop_and_resize(img_new, img_black)
    img_resized_np = np.asarray(img_resized, dtype=np.float64)
    m = mask.astype(np.float64)
    if m.ndim == 2:
        m = m[..., None]
    img_masked_np = np.around(img_resized_np * (1.0 - m)).astype(np.uint8)
    img_combined_np = np.asarray(img_black, dtype=np.uint8) + img_masked_np
    return Image.fromarray(np.clip(img_combined_np, 0, 255))


def place_background_from_file(
    place_path: str, canvas_w: int, canvas_h: int
) -> Image.Image:
    """
    Same as notebook ``place_img``: ``combine_and_mask(place, seg * 0, black_canvas)``
    with ``black_canvas`` the bg canvas size.
    """
    black = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    mask = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    with Image.open(place_path) as place:
        place_rgb = place.convert("RGB")
    return combine_and_mask(place_rgb, mask, black)


# Default category names per deep_feature_reweighting ``generate_waterbirds_fg_bg.ipynb``.
DEFAULT_PLACES_LAND_CATEGORIES = ("bamboo_forest", "forest/broadleaf")
DEFAULT_PLACES_WATER_CATEGORIES = ("ocean", "lake/natural")


def _places_pool_index_for_group(group_name: str) -> int:
    """0 = land place pool, 1 = water place pool (matches ``df['place']`` in the notebook)."""
    if group_name.endswith("_on_land"):
        return 0
    if group_name.endswith("_on_water"):
        return 1
    raise ValueError(f"Unrecognized Waterbirds group name: {group_name!r}")


def list_places365_images_for_categories(
    places_dir: str, categories: Sequence[str]
) -> List[str]:
    """
    List JPEG/PNG images under ``places_dir/{c[0]}/{c}/`` for each category string ``c``,
    as in the notebook (``os.path.join(places_dir, target_place[0], target_place)``).
    """
    out: List[str] = []
    places_dir = os.path.abspath(places_dir)
    for cat in categories:
        cat = cat.strip().strip("/")
        if not cat:
            continue
        sub = os.path.join(places_dir, cat[0], cat)
        if not os.path.isdir(sub):
            continue
        for fn in sorted(os.listdir(sub)):
            if _is_image_file(fn):
                out.append(os.path.join(sub, fn))
    out.sort()
    return out


def _parse_category_list(s: str) -> Tuple[str, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(parts)


def _is_image_file(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(s) for s in _IMAGE_SUFFIXES)


def list_images_recursive(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if _is_image_file(fn):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def _to_rel(path: str, base: str) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


@torch.inference_mode()
def embed_image_paths(
    model: torch.nn.Module,
    preprocess,
    paths: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Return L2-normalized float32 CPU tensor of shape ``[N, D]``."""
    out_chunks: List[torch.Tensor] = []
    for i in tqdm(
        range(0, len(paths), batch_size),
        desc="Embedding",
        unit="batch",
        disable=len(paths) <= batch_size,
    ):
        batch_paths = paths[i : i + batch_size]
        tensors: List[torch.Tensor] = []
        for p in batch_paths:
            with Image.open(p) as im:
                im = im.convert("RGB")
            tensors.append(preprocess(im))
        batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                z = model.encode_image(batch)
        else:
            z = model.encode_image(batch)
        z = F.normalize(z.float(), dim=-1)
        out_chunks.append(z.cpu())
    if not out_chunks:
        return torch.empty(0, 0)
    return torch.cat(out_chunks, dim=0)


@torch.inference_mode()
def embed_pil_images(
    model: torch.nn.Module,
    preprocess,
    images: Sequence[Image.Image],
    *,
    device: torch.device,
    batch_size: int,
    amp_dtype: torch.dtype | None,
    desc: str = "Embedding (Places)",
) -> torch.Tensor:
    """Return L2-normalized float32 CPU tensor ``[N, D]`` for in-memory RGB PIL images."""
    out_chunks: List[torch.Tensor] = []
    n = len(images)
    for i in tqdm(
        range(0, n, batch_size),
        desc=desc,
        unit="batch",
        disable=n <= batch_size,
    ):
        batch_imgs = images[i : i + batch_size]
        tensors = [preprocess(im) for im in batch_imgs]
        batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                z = model.encode_image(batch)
        else:
            z = model.encode_image(batch)
        z = F.normalize(z.float(), dim=-1)
        out_chunks.append(z.cpu())
    if not out_chunks:
        return torch.empty(0, 0)
    return torch.cat(out_chunks, dim=0)


def _exact_pixel_match_counts(
    bg_rgb: np.ndarray, full_paths: Sequence[str]
) -> np.ndarray:
    """
    For one bg image ``bg_rgb`` (H, W, 3) uint8, return per-full-image counts of pixels
    where the full image (resized to H×W) equals ``bg_rgb`` on all three channels.
    """
    h, w = bg_rgb.shape[:2]
    counts = np.zeros(len(full_paths), dtype=np.int64)
    for i, fp in enumerate(full_paths):
        with Image.open(fp) as im:
            full = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if full.shape[0] != h or full.shape[1] != w:
            full = np.asarray(
                Image.fromarray(full).resize((w, h), _RESAMPLE), dtype=np.uint8
            )
        counts[i] = int(np.all(full == bg_rgb, axis=-1).sum())
    return counts


def _match_bg_to_full(
    E_bg: torch.Tensor,
    E_full: torch.Tensor,
    *,
    chunk: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each bg row, argmax cosine over full rows.
    ``E_bg`` / ``E_full`` are L2-normalized on CPU; matmul chunks on ``device``.
    """
    if E_bg.shape[0] == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0)
    E_full = E_full.to(device)
    best_i = []
    best_s = []
    for start in range(0, E_bg.shape[0], chunk):
        block = E_bg[start : start + chunk].to(device)
        sim = block @ E_full.t()
        best_i.append(sim.argmax(dim=1).cpu())
        best_s.append(sim.max(dim=1).values.cpu())
    return torch.cat(best_i), torch.cat(best_s)


def _second_best_margin(
    E_bg: torch.Tensor,
    E_full: torch.Tensor,
    *,
    chunk: int,
    device: torch.device,
) -> torch.Tensor:
    """Per bg row: sim(best) - sim(second best) over full set (same as best if duplicate)."""
    if E_bg.shape[0] == 0:
        return torch.empty(0)
    E_full = E_full.to(device)
    margins = []
    for start in range(0, E_bg.shape[0], chunk):
        block = E_bg[start : start + chunk].to(device)
        sim = block @ E_full.t()
        top2 = torch.topk(sim, k=min(2, sim.shape[1]), dim=1)
        vals = top2.values
        if vals.shape[1] == 1:
            margins.append(vals[:, 0].cpu())
        else:
            margins.append((vals[:, 0] - vals[:, 1]).cpu())
    return torch.cat(margins)


def run_matching(
    waterbirds_root: str,
    *,
    model_id: str,
    pretrained: str,
    batch_size: int,
    matmul_chunk: int,
    device: torch.device,
    use_amp: bool,
    min_margin: float | None,
    exact_pixel: bool,
    class_label: Optional[str],
    debug: bool = False,
    debug_dir: str = "",
) -> Tuple[List[dict], dict]:
    waterbirds_root = os.path.abspath(waterbirds_root)
    amp_dtype = torch.float16 if use_amp and device.type == "cuda" else None
    dbg_out = os.path.abspath(debug_dir or "debug_match") if debug else ""

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_id,
        pretrained=pretrained,
    )
    model = model.to(device).eval()

    matches: List[dict] = []
    meta = {
        "waterbirds_root": waterbirds_root,
        "model_id": model_id,
        "pretrained": pretrained,
        "groups": GROUP_SUBDIRS,
        "exact_pixel_primary_sort": exact_pixel,
        "class_filter": class_label,
    }
    if debug:
        meta["debug"] = True
        meta["debug_match_dir"] = dbg_out

    labels: Tuple[str, ...] = (
        (class_label,) if class_label in ("0", "1") else ("0", "1")
    )

    for gid, group_name in enumerate(GROUP_SUBDIRS):
        full_root = path_with_background(waterbirds_root, gid)
        bg_root = path_background_only(waterbirds_root, gid)
        for label in labels:
            full_dir = os.path.join(full_root, label)
            bg_dir = os.path.join(bg_root, label)
            if not os.path.isdir(full_dir) or not os.path.isdir(bg_dir):
                continue
            full_paths = list_images_recursive(full_dir)
            bg_paths = list_images_recursive(bg_dir)
            if not full_paths or not bg_paths:
                continue

            E_full = embed_image_paths(
                model,
                preprocess,
                full_paths,
                device=device,
                batch_size=batch_size,
                amp_dtype=amp_dtype,
            )
            E_bg = embed_image_paths(
                model,
                preprocess,
                bg_paths,
                device=device,
                batch_size=batch_size,
                amp_dtype=amp_dtype,
            )

            if exact_pixel:
                cos_all = (E_bg @ E_full.t()).cpu().numpy()
                margin = None
                if min_margin is not None:
                    margin = _second_best_margin(
                        E_bg,
                        E_full,
                        chunk=matmul_chunk,
                        device=device,
                    )
                for j, bg_p in enumerate(
                    tqdm(
                        bg_paths,
                        desc=f"Exact-pixel {group_name}/{label}",
                        leave=False,
                    )
                ):
                    with Image.open(bg_p) as im:
                        bg_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                    counts = _exact_pixel_match_counts(bg_rgb, full_paths)
                    cos_vec = cos_all[j].astype(np.float64)
                    # lexsort: last key is primary ascending → largest count last; first key
                    # breaks ties so highest cosine last among equal counts.
                    order = np.lexsort((cos_vec, counts.astype(np.int64)))
                    fi = int(order[-1])
                    sim = float(cos_vec[fi])
                    row = {
                        "group": group_name,
                        "label": label,
                        "bg_only_relpath": _to_rel(bg_p, waterbirds_root),
                        "full_relpath": _to_rel(
                            full_paths[fi], waterbirds_root
                        ),
                        "cosine_similarity": sim,
                        "exact_pixel_matches": int(counts[fi]),
                    }
                    if margin is not None:
                        row["best_minus_second"] = float(margin[j])
                    matches.append(row)
                    if debug:
                        _save_debug_match_pair(
                            matches[-1],
                            len(matches) - 1,
                            waterbirds_root,
                            None,
                            dbg_out,
                        )
                        print(
                            f"--debug: saved pair #{len(matches)} under {dbg_out!r}",
                            file=sys.stderr,
                        )
                    if debug and len(matches) >= DEBUG_MATCH_LIMIT:
                        meta["debug_early_stop"] = True
                        return matches, meta
            else:
                best_idx, best_sim = _match_bg_to_full(
                    E_bg, E_full, chunk=matmul_chunk, device=device
                )
                margin = None
                if min_margin is not None:
                    margin = _second_best_margin(
                        E_bg,
                        E_full,
                        chunk=matmul_chunk,
                        device=device,
                    )

                for j, bg_p in enumerate(bg_paths):
                    fi = int(best_idx[j])
                    sim = float(best_sim[j])
                    row = {
                        "group": group_name,
                        "label": label,
                        "bg_only_relpath": _to_rel(bg_p, waterbirds_root),
                        "full_relpath": _to_rel(
                            full_paths[fi], waterbirds_root
                        ),
                        "cosine_similarity": sim,
                    }
                    if margin is not None:
                        row["best_minus_second"] = float(margin[j])
                    matches.append(row)
                    if debug:
                        _save_debug_match_pair(
                            matches[-1],
                            len(matches) - 1,
                            waterbirds_root,
                            None,
                            dbg_out,
                        )
                        print(
                            f"--debug: saved pair #{len(matches)} under {dbg_out!r}",
                            file=sys.stderr,
                        )
                    if debug and len(matches) >= DEBUG_MATCH_LIMIT:
                        meta["debug_early_stop"] = True
                        return matches, meta

    return matches, meta


def _pixel_match_counts_vs_stack(
    bg_rgb: np.ndarray, warped_stack: np.ndarray
) -> np.ndarray:
    """Per-place exact pixel counts; ``warped_stack`` is ``(M, H, W, 3)`` uint8."""
    return (
        np.all(
            warped_stack == bg_rgb[np.newaxis, ...],
            axis=-1,
        )
        .sum(axis=(1, 2))
        .astype(np.int64)
    )


def run_matching_places(
    waterbirds_root: str,
    places_dir: str,
    *,
    land_categories: Tuple[str, ...],
    water_categories: Tuple[str, ...],
    model_id: str,
    pretrained: str,
    batch_size: int,
    matmul_chunk: int,
    device: torch.device,
    use_amp: bool,
    min_margin: float | None,
    exact_pixel: bool,
    class_label: Optional[str],
    apply_same_square: bool,
    drop_black: bool,
    fg_only_root: str,
    debug: bool = False,
    debug_dir: str = "",
) -> Tuple[List[dict], dict]:
    """
    Match each ``*_bg_only`` to the nearest **warped** Places365 image (notebook-style
    ``place_img``), within the land vs water pool implied by the subscenario name.
    """
    waterbirds_root = os.path.abspath(waterbirds_root)
    places_dir = os.path.abspath(places_dir)
    fg_only_root = os.path.abspath(fg_only_root)
    amp_dtype = torch.float16 if use_amp and device.type == "cuda" else None
    use_square_geom = apply_same_square or drop_black
    dbg_out = os.path.abspath(debug_dir or "debug_match") if debug else ""

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_id,
        pretrained=pretrained,
    )
    model = model.to(device).eval()

    matches: List[dict] = []
    meta = {
        "waterbirds_root": waterbirds_root,
        "places_dir": places_dir,
        "places_land_categories": list(land_categories),
        "places_water_categories": list(water_categories),
        "model_id": model_id,
        "pretrained": pretrained,
        "groups": GROUP_SUBDIRS,
        "exact_pixel_primary_sort": exact_pixel,
        "class_filter": class_label,
        "match_target": "places365",
        "search_places": True,
        "apply_same_square": apply_same_square,
        "drop_black": drop_black,
        "fg_only_root": fg_only_root if use_square_geom else None,
    }
    if debug:
        meta["debug"] = True
        meta["debug_match_dir"] = dbg_out

    labels: Tuple[str, ...] = (
        (class_label,) if class_label in ("0", "1") else ("0", "1")
    )

    # Cache key: ``(pool_idx, w, h)`` or, with per-image square geometry,
    # ``(pool_idx, w, h, x0, y0, side)`` (``--apply-same-square`` / ``--drop-black``).
    pool_emb_cache: Dict[
        Tuple[int, ...], Tuple[torch.Tensor, Optional[np.ndarray]]
    ] = {}

    norm_mod = (
        _load_wb_normalize_resolution_module() if use_square_geom else None
    )

    for gid, group_name in enumerate(GROUP_SUBDIRS):
        pool_idx = _places_pool_index_for_group(group_name)
        cats = land_categories if pool_idx == 0 else water_categories
        place_paths = list_places365_images_for_categories(places_dir, cats)
        if not place_paths:
            print(
                f"WARNING: no Places365 images found for {group_name!r} "
                f"(pool={pool_idx}, categories={cats}) under {places_dir!r}; skipping.",
                file=sys.stderr,
            )
            continue

        bg_root = path_background_only(waterbirds_root, gid)
        for label in labels:
            bg_dir = os.path.join(bg_root, label)
            if not os.path.isdir(bg_dir):
                continue
            bg_paths = list_images_recursive(bg_dir)
            if not bg_paths:
                continue

            buckets: Dict[Tuple[int, int], List[str]] = defaultdict(list)
            for bg_p in bg_paths:
                with Image.open(bg_p) as im:
                    w_sz, h_sz = im.size
                buckets[(w_sz, h_sz)].append(bg_p)

            rows_by_bg: Dict[str, dict] = {}

            for (w_sz, h_sz), bucket_paths in sorted(buckets.items()):
                if use_square_geom:
                    assert norm_mod is not None
                    square_groups: Dict[
                        Tuple[int, int, int], List[str]
                    ] = defaultdict(list)
                    for bg_p in bucket_paths:
                        bn = os.path.basename(bg_p)
                        full_p = os.path.join(
                            path_with_background(waterbirds_root, gid),
                            label,
                            bn,
                        )
                        (
                            x0,
                            y0,
                            side,
                        ) = norm_mod.black_square_params_from_paired_full(
                            w_sz,
                            h_sz,
                            full_path=full_p
                            if os.path.isfile(full_p)
                            else None,
                            waterbirds_root=waterbirds_root,
                            fg_only_root=fg_only_root,
                            gid=gid,
                            label=label,
                            basename=bn,
                            warn_prefix=f"{group_name}/{label}",
                        )
                        square_groups[(x0, y0, side)].append(bg_p)
                    group_iter = list(square_groups.items())
                else:
                    group_iter = [((0, 0, 0), bucket_paths)]

                for (x0, y0, side), gpaths in group_iter:
                    ck: Tuple[int, ...] = (
                        (pool_idx, w_sz, h_sz, x0, y0, side)
                        if use_square_geom
                        else (pool_idx, w_sz, h_sz)
                    )
                    if ck not in pool_emb_cache:
                        warped_pils: List[Image.Image] = []
                        warped_rgbs: List[np.ndarray] = []
                        for pp in tqdm(
                            place_paths,
                            desc=f"Places warp {group_name}/{label} {w_sz}x{h_sz}",
                            leave=False,
                        ):
                            pil = place_background_from_file(pp, w_sz, h_sz)
                            if use_square_geom:
                                assert norm_mod is not None
                                pil = _prepare_places_square_region(
                                    norm_mod,
                                    pil,
                                    x0,
                                    y0,
                                    side,
                                    apply_same_square=apply_same_square,
                                    drop_black=drop_black,
                                )
                            warped_pils.append(pil)
                            if exact_pixel:
                                warped_rgbs.append(
                                    np.asarray(pil, dtype=np.uint8)
                                )
                        E_pl = embed_pil_images(
                            model,
                            preprocess,
                            warped_pils,
                            device=device,
                            batch_size=batch_size,
                            amp_dtype=amp_dtype,
                            desc=f"SigLIP Places {w_sz}x{h_sz}",
                        )
                        w_stack: Optional[np.ndarray] = None
                        if exact_pixel:
                            w_stack = np.stack(warped_rgbs, axis=0)
                        pool_emb_cache[ck] = (E_pl, w_stack)

                    E_places, warped_stack = pool_emb_cache[ck]

                    if use_square_geom:
                        assert norm_mod is not None
                        masked_bgs: List[Image.Image] = []
                        for bg_p in gpaths:
                            with Image.open(bg_p) as im:
                                masked_bgs.append(
                                    _prepare_places_square_region(
                                        norm_mod,
                                        im,
                                        x0,
                                        y0,
                                        side,
                                        apply_same_square=apply_same_square,
                                        drop_black=drop_black,
                                    )
                                )
                        E_bg = embed_pil_images(
                            model,
                            preprocess,
                            masked_bgs,
                            device=device,
                            batch_size=batch_size,
                            amp_dtype=amp_dtype,
                            desc=f"SigLIP bg {w_sz}x{h_sz}",
                        )
                    else:
                        E_bg = embed_image_paths(
                            model,
                            preprocess,
                            gpaths,
                            device=device,
                            batch_size=batch_size,
                            amp_dtype=amp_dtype,
                        )

                    cos_all = (E_bg @ E_places.t()).cpu().numpy()
                    margin = None
                    if min_margin is not None:
                        margin = _second_best_margin(
                            E_bg,
                            E_places,
                            chunk=matmul_chunk,
                            device=device,
                        )

                    if exact_pixel:
                        assert warped_stack is not None
                        for j, bg_p in enumerate(gpaths):
                            with Image.open(bg_p) as im:
                                im_rgb = (
                                    _prepare_places_square_region(
                                        norm_mod,
                                        im,
                                        x0,
                                        y0,
                                        side,
                                        apply_same_square=apply_same_square,
                                        drop_black=drop_black,
                                    )
                                    if use_square_geom
                                    else im.convert("RGB")
                                )
                                bg_rgb = np.asarray(im_rgb, dtype=np.uint8)
                            counts = _pixel_match_counts_vs_stack(
                                bg_rgb, warped_stack
                            )
                            cos_vec = cos_all[j].astype(np.float64)
                            order = np.lexsort(
                                (cos_vec, counts.astype(np.int64))
                            )
                            fi = int(order[-1])
                            sim = float(cos_vec[fi])
                            row = {
                                "group": group_name,
                                "label": label,
                                "bg_only_relpath": _to_rel(
                                    bg_p, waterbirds_root
                                ),
                                "full_relpath": None,
                                "place_relpath": _to_rel(
                                    place_paths[fi], places_dir
                                ),
                                "cosine_similarity": sim,
                                "exact_pixel_matches": int(counts[fi]),
                                "match_target": "places365",
                            }
                            if margin is not None:
                                row["best_minus_second"] = float(margin[j])
                            rows_by_bg[bg_p] = row
                    else:
                        best_idx, best_sim = _match_bg_to_full(
                            E_bg, E_places, chunk=matmul_chunk, device=device
                        )
                        for j, bg_p in enumerate(gpaths):
                            fi = int(best_idx[j])
                            sim = float(best_sim[j])
                            row = {
                                "group": group_name,
                                "label": label,
                                "bg_only_relpath": _to_rel(
                                    bg_p, waterbirds_root
                                ),
                                "full_relpath": None,
                                "place_relpath": _to_rel(
                                    place_paths[fi], places_dir
                                ),
                                "cosine_similarity": sim,
                                "match_target": "places365",
                            }
                            if margin is not None:
                                row["best_minus_second"] = float(margin[j])
                            rows_by_bg[bg_p] = row

            for bg_p in bg_paths:
                if bg_p not in rows_by_bg:
                    continue
                matches.append(rows_by_bg[bg_p])
                if debug:
                    _save_debug_match_pair(
                        matches[-1],
                        len(matches) - 1,
                        waterbirds_root,
                        places_dir,
                        dbg_out,
                    )
                    print(
                        f"--debug: saved pair #{len(matches)} under {dbg_out!r}",
                        file=sys.stderr,
                    )
                if debug and len(matches) >= DEBUG_MATCH_LIMIT:
                    meta["debug_early_stop"] = True
                    return matches, meta

    return matches, meta


def run_matching_original_to_places(
    waterbirds_root: str,
    places_dir: str,
    *,
    land_categories: Tuple[str, ...],
    water_categories: Tuple[str, ...],
    model_id: str,
    pretrained: str,
    batch_size: int,
    matmul_chunk: int,
    device: torch.device,
    use_amp: bool,
    min_margin: float | None,
    exact_pixel: bool,
    class_label: Optional[str],
    apply_same_square: bool,
    drop_black: bool,
    fg_only_root: str,
    debug: bool = False,
    debug_dir: str = "",
) -> Tuple[List[dict], dict]:
    """
    For each **original** FG+BG image (``landbird_on_land/0/*.jpg``, …), find the nearest
    notebook-style warped Places365 frame from the same land/water pool as for
    ``--search-places``. Embeddings compare the full composite to each warped place-only
    canvas (same size as the original).
    """
    waterbirds_root = os.path.abspath(waterbirds_root)
    places_dir = os.path.abspath(places_dir)
    fg_only_root = os.path.abspath(fg_only_root)
    amp_dtype = torch.float16 if use_amp and device.type == "cuda" else None
    use_square_geom = apply_same_square or drop_black
    dbg_out = os.path.abspath(debug_dir or "debug_match") if debug else ""

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_id,
        pretrained=pretrained,
    )
    model = model.to(device).eval()

    matches: List[dict] = []
    meta = {
        "waterbirds_root": waterbirds_root,
        "places_dir": places_dir,
        "places_land_categories": list(land_categories),
        "places_water_categories": list(water_categories),
        "model_id": model_id,
        "pretrained": pretrained,
        "groups": GROUP_SUBDIRS,
        "exact_pixel_primary_sort": exact_pixel,
        "class_filter": class_label,
        "match_target": "original_to_places365",
        "match_to_original": True,
        "apply_same_square": apply_same_square,
        "drop_black": drop_black,
        "fg_only_root": fg_only_root if use_square_geom else None,
    }
    if debug:
        meta["debug"] = True
        meta["debug_match_dir"] = dbg_out

    labels: Tuple[str, ...] = (
        (class_label,) if class_label in ("0", "1") else ("0", "1")
    )

    pool_emb_cache: Dict[
        Tuple[int, ...], Tuple[torch.Tensor, Optional[np.ndarray]]
    ] = {}

    norm_mod = (
        _load_wb_normalize_resolution_module() if use_square_geom else None
    )

    for gid, group_name in enumerate(GROUP_SUBDIRS):
        pool_idx = _places_pool_index_for_group(group_name)
        cats = land_categories if pool_idx == 0 else water_categories
        place_paths = list_places365_images_for_categories(places_dir, cats)
        if not place_paths:
            print(
                f"WARNING: no Places365 images found for {group_name!r} "
                f"(pool={pool_idx}, categories={cats}) under {places_dir!r}; skipping.",
                file=sys.stderr,
            )
            continue

        full_root = path_with_background(waterbirds_root, gid)
        for label in labels:
            full_dir = os.path.join(full_root, label)
            if not os.path.isdir(full_dir):
                continue
            orig_paths = list_images_recursive(full_dir)
            if not orig_paths:
                continue

            buckets: Dict[Tuple[int, int], List[str]] = defaultdict(list)
            for orig_p in orig_paths:
                with Image.open(orig_p) as im:
                    w_sz, h_sz = im.size
                buckets[(w_sz, h_sz)].append(orig_p)

            rows_by_orig: Dict[str, dict] = {}

            for (w_sz, h_sz), bucket_paths in sorted(buckets.items()):
                if use_square_geom:
                    assert norm_mod is not None
                    square_groups: Dict[
                        Tuple[int, int, int], List[str]
                    ] = defaultdict(list)
                    for orig_p in bucket_paths:
                        bn = os.path.basename(orig_p)
                        (
                            x0,
                            y0,
                            side,
                        ) = norm_mod.black_square_params_from_paired_full(
                            w_sz,
                            h_sz,
                            full_path=orig_p,
                            waterbirds_root=waterbirds_root,
                            fg_only_root=fg_only_root,
                            gid=gid,
                            label=label,
                            basename=bn,
                            warn_prefix=f"{group_name}/{label}",
                        )
                        square_groups[(x0, y0, side)].append(orig_p)
                    group_iter = list(square_groups.items())
                else:
                    group_iter = [((0, 0, 0), bucket_paths)]

                for (x0, y0, side), gpaths in group_iter:
                    ck: Tuple[int, ...] = (
                        (pool_idx, w_sz, h_sz, x0, y0, side)
                        if use_square_geom
                        else (pool_idx, w_sz, h_sz)
                    )
                    if ck not in pool_emb_cache:
                        warped_pils: List[Image.Image] = []
                        warped_rgbs: List[np.ndarray] = []
                        for pp in tqdm(
                            place_paths,
                            desc=f"Places warp orig {group_name}/{label} {w_sz}x{h_sz}",
                            leave=False,
                        ):
                            pil = place_background_from_file(pp, w_sz, h_sz)
                            if use_square_geom:
                                assert norm_mod is not None
                                pil = _prepare_places_square_region(
                                    norm_mod,
                                    pil,
                                    x0,
                                    y0,
                                    side,
                                    apply_same_square=apply_same_square,
                                    drop_black=drop_black,
                                )
                            warped_pils.append(pil)
                            if exact_pixel:
                                warped_rgbs.append(
                                    np.asarray(pil, dtype=np.uint8)
                                )
                        E_pl = embed_pil_images(
                            model,
                            preprocess,
                            warped_pils,
                            device=device,
                            batch_size=batch_size,
                            amp_dtype=amp_dtype,
                            desc=f"SigLIP Places orig {w_sz}x{h_sz}",
                        )
                        w_stack: Optional[np.ndarray] = None
                        if exact_pixel:
                            w_stack = np.stack(warped_rgbs, axis=0)
                        pool_emb_cache[ck] = (E_pl, w_stack)

                    E_places, warped_stack = pool_emb_cache[ck]

                    if use_square_geom:
                        assert norm_mod is not None
                        masked_orig: List[Image.Image] = []
                        for orig_p in gpaths:
                            with Image.open(orig_p) as im:
                                masked_orig.append(
                                    _prepare_places_square_region(
                                        norm_mod,
                                        im,
                                        x0,
                                        y0,
                                        side,
                                        apply_same_square=apply_same_square,
                                        drop_black=drop_black,
                                    )
                                )
                        E_orig = embed_pil_images(
                            model,
                            preprocess,
                            masked_orig,
                            device=device,
                            batch_size=batch_size,
                            amp_dtype=amp_dtype,
                            desc=f"SigLIP orig {w_sz}x{h_sz}",
                        )
                    else:
                        E_orig = embed_image_paths(
                            model,
                            preprocess,
                            gpaths,
                            device=device,
                            batch_size=batch_size,
                            amp_dtype=amp_dtype,
                        )

                    cos_all = (E_orig @ E_places.t()).cpu().numpy()
                    margin = None
                    if min_margin is not None:
                        margin = _second_best_margin(
                            E_orig,
                            E_places,
                            chunk=matmul_chunk,
                            device=device,
                        )

                    if exact_pixel:
                        assert warped_stack is not None
                        for j, orig_p in enumerate(gpaths):
                            with Image.open(orig_p) as im:
                                im_rgb = (
                                    _prepare_places_square_region(
                                        norm_mod,
                                        im,
                                        x0,
                                        y0,
                                        side,
                                        apply_same_square=apply_same_square,
                                        drop_black=drop_black,
                                    )
                                    if use_square_geom
                                    else im.convert("RGB")
                                )
                                orig_rgb = np.asarray(im_rgb, dtype=np.uint8)
                            counts = _pixel_match_counts_vs_stack(
                                orig_rgb, warped_stack
                            )
                            cos_vec = cos_all[j].astype(np.float64)
                            order = np.lexsort(
                                (cos_vec, counts.astype(np.int64))
                            )
                            fi = int(order[-1])
                            sim = float(cos_vec[fi])
                            row = {
                                "group": group_name,
                                "label": label,
                                "bg_only_relpath": None,
                                "full_relpath": _to_rel(
                                    orig_p, waterbirds_root
                                ),
                                "place_relpath": _to_rel(
                                    place_paths[fi], places_dir
                                ),
                                "cosine_similarity": sim,
                                "exact_pixel_matches": int(counts[fi]),
                                "match_target": "original_to_places365",
                            }
                            if margin is not None:
                                row["best_minus_second"] = float(margin[j])
                            rows_by_orig[orig_p] = row
                    else:
                        best_idx, best_sim = _match_bg_to_full(
                            E_orig, E_places, chunk=matmul_chunk, device=device
                        )
                        for j, orig_p in enumerate(gpaths):
                            fi = int(best_idx[j])
                            sim = float(best_sim[j])
                            row = {
                                "group": group_name,
                                "label": label,
                                "bg_only_relpath": None,
                                "full_relpath": _to_rel(
                                    orig_p, waterbirds_root
                                ),
                                "place_relpath": _to_rel(
                                    place_paths[fi], places_dir
                                ),
                                "cosine_similarity": sim,
                                "match_target": "original_to_places365",
                            }
                            if margin is not None:
                                row["best_minus_second"] = float(margin[j])
                            rows_by_orig[orig_p] = row

            for orig_p in orig_paths:
                if orig_p not in rows_by_orig:
                    continue
                matches.append(rows_by_orig[orig_p])
                if debug:
                    _save_debug_match_pair(
                        matches[-1],
                        len(matches) - 1,
                        waterbirds_root,
                        places_dir,
                        dbg_out,
                    )
                    print(
                        f"--debug: saved pair #{len(matches)} under {dbg_out!r}",
                        file=sys.stderr,
                    )
                if debug and len(matches) >= DEBUG_MATCH_LIMIT:
                    meta["debug_early_stop"] = True
                    return matches, meta

    return matches, meta


def _summarize(matches: Sequence[dict]) -> dict:
    inv: Dict[str, List[str]] = defaultdict(list)
    for m in matches:
        tgt = m.get("place_relpath") or m.get("full_relpath")
        src = m.get("bg_only_relpath") or m.get("full_relpath")
        if tgt and src:
            inv[str(tgt)].append(str(src))
    collisions = {k: v for k, v in inv.items() if len(v) > 1}
    return {
        "num_matches": len(matches),
        "num_full_targets_with_multiple_bg": len(collisions),
        "example_collisions": dict(list(collisions.items())[:8]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build bg_only → full (FG+BG) image mapping via SigLIP cosine similarity "
            "(per group and coarse label folder)."
        )
    )
    parser.add_argument(
        "--waterbirds-root",
        default=os.path.join(DATASETS_PATH, "Waterbirds"),
        help="Waterbirds root (twelve subscenario folders)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path (mapping list + metadata)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path with the same rows as the JSON ``matches`` list",
    )
    parser.add_argument(
        "--model-id",
        default="ViT-B-16-SigLIP-384",
        help="OpenCLIP model name (default matches OCCAM SigLIP-384 usage)",
    )
    parser.add_argument(
        "--pretrained",
        default="webli",
        help="OpenCLIP pretrained tag",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Image batch size for embedding",
    )
    parser.add_argument(
        "--matmul-chunk",
        type=int,
        default=256,
        help="Bg rows per GPU chunk when multiplying against all full embeddings",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (default: cuda if available)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable autocast float16 on CUDA",
    )
    parser.add_argument(
        "--min-best-minus-second",
        type=float,
        default=None,
        metavar="DELTA",
        help=(
            "If set, drop matches where (top1 cosine − top2 cosine) < DELTA "
            "(ambiguous neighbors)"
        ),
    )
    parser.add_argument(
        "--exact-pixel",
        action="store_true",
        help=(
            "Sort full-image candidates by exact RGB pixel agreement first (full image "
            "resized to each bg_only size, then count matching pixels), then by SigLIP "
            "cosine. Slower (loads every pair in software) but picks true pixel matches "
            "when backgrounds align. Adds field exact_pixel_matches to each row."
        ),
    )
    parser.add_argument(
        "--class",
        dest="class_label",
        choices=("0", "1"),
        default=None,
        metavar="N",
        help=(
            "Run matching only for coarse label folder N: 0=waterbird, 1=landbird "
            "(default: both)."
        ),
    )
    parser.add_argument(
        "--search-places",
        action="store_true",
        help=(
            "Match each bg_only to the nearest Places365 image warped like the DFR "
            "Waterbirds notebook (crop_and_resize / combine_and_mask with an empty bird "
            "mask on a black canvas). Uses --places-dir; land vs water scene pools follow "
            "subscenario name (*_on_land vs *_on_water). Incompatible with --match-to-original."
        ),
    )
    parser.add_argument(
        "--match-to-original",
        action="store_true",
        dest="match_to_original",
        help=(
            "For each original FG+BG image (folders without _bg_only or _fg_only, e.g. "
            "landbird_on_land/0/*.jpg), find the single nearest warped Places365 frame from "
            "the same category pools as --search-places (land vs water by subscenario name). "
            "Uses SigLIP cosine between the full composite and each place warped to that "
            "image's canvas size (notebook place_img geometry). Requires --places-dir. "
            "Incompatible with --search-places."
        ),
    )
    parser.add_argument(
        "--places-dir",
        default=None,
        help=(
            "Root of the Places365 subset laid out as in the notebook, e.g. "
            "…/b/bamboo_forest/*.jpg and …/f/forest/broadleaf/*.jpg."
        ),
    )
    parser.add_argument(
        "--places-land-categories",
        default=",".join(DEFAULT_PLACES_LAND_CATEGORIES),
        help="Comma-separated category names for *_on_land groups (notebook defaults).",
    )
    parser.add_argument(
        "--places-water-categories",
        default=",".join(DEFAULT_PLACES_WATER_CATEGORIES),
        help="Comma-separated category names for *_on_water groups (notebook defaults).",
    )
    parser.add_argument(
        "--apply-same-square",
        action="store_true",
        help=(
            "Places matching only: before SigLIP (and exact-pixel tie-break), paint the same "
            "black square as for the paired full FG+BG frame (FG mask when available, else "
            "centered min(w,h)//2) onto each warped Places crop and each query image."
        ),
    )
    parser.add_argument(
        "--fg-only-root",
        default=None,
        help=(
            "With --apply-same-square or --drop-black: legacy FG-Only tree root "
            "(``test_split/group_<id>/``). Default: ``<--waterbirds-root>/FG-Only``. "
            "Hub ``*_fg_only`` is tried first."
        ),
    )
    parser.add_argument(
        "--drop-black",
        action="store_true",
        help=(
            "Places matching only: before SigLIP (and exact-pixel tie-break), **remove** the "
            "same FG-aligned square region as for the paired full image (splice left/right or "
            "top/bottom strips so masked pixels are not in the tensor). Applies to each "
            "warped Places candidate and each query image. If combined with --apply-same-square, "
            "only the splice is used for embeddings (black paint is skipped)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Stop after three matches. For each match, copy two files into ``debug_match/`` "
            "under the current working directory: ``NNN_full*`` (paired or query FG+BG) and "
            "``NNN_matched_background*`` (matched full from pool, Places crop, or bg_only fallback). "
            "Skips ``--min-best-minus-second`` filtering so all three rows are kept."
        ),
    )
    args = parser.parse_args()

    if args.apply_same_square and not (
        args.search_places or args.match_to_original
    ):
        parser.error(
            "--apply-same-square is only valid with --search-places or --match-to-original"
        )
    if args.drop_black and not (args.search_places or args.match_to_original):
        parser.error(
            "--drop-black is only valid with --search-places or --match-to-original"
        )

    if args.search_places and args.match_to_original:
        parser.error(
            "--search-places and --match-to-original cannot be used together"
        )
    if args.search_places or args.match_to_original:
        if not args.places_dir or not os.path.isdir(args.places_dir):
            parser.error(
                "An existing --places-dir is required with --search-places or "
                "--match-to-original"
            )

    device = torch.device(args.device)
    land_cats = _parse_category_list(args.places_land_categories)
    water_cats = _parse_category_list(args.places_water_categories)
    fg_only_root = os.path.abspath(
        args.fg_only_root
        if args.fg_only_root
        else os.path.join(os.path.abspath(args.waterbirds_root), "FG-Only")
    )
    dbg_dir = os.path.abspath("debug_match") if args.debug else ""

    if args.search_places:
        matches, meta = run_matching_places(
            args.waterbirds_root,
            args.places_dir,
            land_categories=land_cats,
            water_categories=water_cats,
            model_id=args.model_id,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
            matmul_chunk=args.matmul_chunk,
            device=device,
            use_amp=not args.no_amp,
            min_margin=args.min_best_minus_second,
            exact_pixel=args.exact_pixel,
            class_label=args.class_label,
            apply_same_square=args.apply_same_square,
            drop_black=args.drop_black,
            fg_only_root=fg_only_root,
            debug=args.debug,
            debug_dir=dbg_dir,
        )
    elif args.match_to_original:
        matches, meta = run_matching_original_to_places(
            args.waterbirds_root,
            args.places_dir,
            land_categories=land_cats,
            water_categories=water_cats,
            model_id=args.model_id,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
            matmul_chunk=args.matmul_chunk,
            device=device,
            use_amp=not args.no_amp,
            min_margin=args.min_best_minus_second,
            exact_pixel=args.exact_pixel,
            class_label=args.class_label,
            apply_same_square=args.apply_same_square,
            drop_black=args.drop_black,
            fg_only_root=fg_only_root,
            debug=args.debug,
            debug_dir=dbg_dir,
        )
    else:
        matches, meta = run_matching(
            args.waterbirds_root,
            model_id=args.model_id,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
            matmul_chunk=args.matmul_chunk,
            device=device,
            use_amp=not args.no_amp,
            min_margin=args.min_best_minus_second,
            exact_pixel=args.exact_pixel,
            class_label=args.class_label,
            debug=args.debug,
            debug_dir=dbg_dir,
        )

    if args.min_best_minus_second is not None and not args.debug:
        before = len(matches)
        matches = [
            m
            for m in matches
            if m.get("best_minus_second", 1.0) >= args.min_best_minus_second
        ]
        meta["filtered_by_margin"] = {
            "threshold": args.min_best_minus_second,
            "before": before,
            "after": len(matches),
        }

    summary = _summarize(matches)
    mapping = {}
    for m in matches:
        map_key = m.get("bg_only_relpath") or m.get("full_relpath")
        if not map_key:
            continue
        entry: dict = {"cosine_similarity": m["cosine_similarity"]}
        if m.get("full_relpath") is not None:
            entry["full_relpath"] = m["full_relpath"]
        if m.get("place_relpath"):
            entry["place_relpath"] = m["place_relpath"]
        if "exact_pixel_matches" in m:
            entry["exact_pixel_matches"] = m["exact_pixel_matches"]
        mapping[map_key] = entry
    payload = {
        "meta": meta,
        "summary": summary,
        "mapping": mapping,
        "matches": matches,
    }

    os.makedirs(
        os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(matches)} matches to {args.output!r}")
    if args.search_places:
        tail = "matched targets with >1 bg_only pointing to them"
    elif args.match_to_original:
        tail = "Places365 targets with >1 original FG+BG pointing to them"
    else:
        tail = "full-image targets with >1 bg_only pointing to them"
    print(f"{tail}: {summary['num_full_targets_with_multiple_bg']}")

    if args.csv:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True
        )
        if matches:
            fieldnames = list(matches[0].keys())
            with open(args.csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(matches)
        else:
            with open(args.csv, "w", encoding="utf-8") as f:
                f.write("")
        print(f"Wrote CSV to {args.csv!r}")


if __name__ == "__main__":
    main()
