#!/usr/bin/env python3
"""
Resize Waterbirds images to a single width×height (in place) so Places matching uses
one warp size per canvas pool.

- ``--targets bg_only`` (default): only ``*_bg_only/…`` (for ``--search-places``).
- ``--targets full``: only original FG+BG folders (no ``_bg_only`` / ``_fg_only`` suffix),
  e.g. ``landbird_on_land/0/`` (for ``--match-to-original``).
- ``--targets both``: **full first**, then ``*_bg_only``, so originals are normalized before
  backgrounds when you use one command.

Requires Pillow, tqdm, and NumPy. Optional ``--match-fg-size`` uses FG-only images to
position the black square (see ``--fg-only-root``). ``--inpaint`` needs OpenCV
(``opencv-python-headless``).

Examples:

  # Letterbox bg_only to 224×224 (preserve aspect, pad with black):
  python scripts/normalize_waterbirds_bg_only_resolution.py \\
    --waterbirds-root data/datasets/Waterbirds --width 224 --height 224 --fit contain

  # Same size for original FG+BG images only:
  python scripts/normalize_waterbirds_bg_only_resolution.py \\
    --waterbirds-root data/datasets/Waterbirds --width 224 --height 224 --targets full

  # Originals first, then bg_only (recommended if you use both Places modes):
  python scripts/normalize_waterbirds_bg_only_resolution.py \\
    --waterbirds-root data/datasets/Waterbirds --width 224 --height 224 --targets both

  # Black out centered square (side = min(w,h)//2) then resize (e.g. gray placeholder):
  python scripts/normalize_waterbirds_bg_only_resolution.py \\
    --waterbirds-root data/datasets/Waterbirds --width 224 --height 224 --gray-square

  # Square from FG-only mask (requires --gray-square); tries ``*_fg_only`` then FG-Only/…:
  python scripts/normalize_waterbirds_bg_only_resolution.py \\
    --waterbirds-root data/datasets/Waterbirds --width 224 --height 224 \\
    --gray-square --match-fg-size \\
    --fg-only-root data/datasets/Waterbirds/FG-Only

  # Black out the bird silhouette (FG mask), not a rectangle (incompatible with --gray-square):
  python scripts/normalize_waterbirds_bg_only_resolution.py \\
    --waterbirds-root data/datasets/Waterbirds --width 224 --height 224 \\
    --targets full --keep-mask-shape

  # Preview counts only:
  python scripts/normalize_waterbirds_bg_only_resolution.py ... --dry-run

  # Write at most 2 images per label folder under ``<scenario>_debug/<label>/`` (no in-place writes):
  python scripts/normalize_waterbirds_bg_only_resolution.py \\
    --waterbirds-root data/datasets/Waterbirds --width 224 --height 224 --targets full --debug

  # Fill masked regions via OpenCV inpainting instead of black (requires a mask flag):
  python scripts/normalize_waterbirds_bg_only_resolution.py \\
    --waterbirds-root data/datasets/Waterbirds --width 224 --height 224 \\
    --targets full --keep-mask-shape --inpaint --inpaint-radius 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from occam.datasets.utils import DATASETS_PATH
from occam.datasets.waterbirds_layout import (
    GROUP_SUBDIRS,
    path_background_only,
    path_foreground_only,
    path_with_background,
)

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    _RESAMPLE = Image.LANCZOS  # type: ignore[attr-defined]


def _is_image_file(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(s) for s in _IMAGE_SUFFIXES)


def list_images_recursive(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if _is_image_file(fn):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def _resize_stretch(im: Image.Image, tw: int, th: int) -> Image.Image:
    return im.convert("RGB").resize((tw, th), _RESAMPLE)


def _resize_contain(im: Image.Image, tw: int, th: int) -> Image.Image:
    im = im.convert("RGB")
    sw, sh = im.size
    if sw <= 0 or sh <= 0:
        return Image.new("RGB", (tw, th), (0, 0, 0))
    scale = min(tw / sw, th / sh)
    nw = max(1, int(round(sw * scale)))
    nh = max(1, int(round(sh * scale)))
    resized = im.resize((nw, nh), _RESAMPLE)
    out = Image.new("RGB", (tw, th), (0, 0, 0))
    offx = (tw - nw) // 2
    offy = (th - nh) // 2
    out.paste(resized, (offx, offy))
    return out


def _resize_cover(im: Image.Image, tw: int, th: int) -> Image.Image:
    im = im.convert("RGB")
    sw, sh = im.size
    if sw <= 0 or sh <= 0:
        return Image.new("RGB", (tw, th), (0, 0, 0))
    scale = max(tw / sw, th / sh)
    nw = max(1, int(round(sw * scale)))
    nh = max(1, int(round(sh * scale)))
    resized = im.resize((nw, nh), _RESAMPLE)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def resolve_fg_only_image_path(
    waterbirds_root: str,
    fg_only_root: str,
    gid: int,
    label: str,
    basename: str,
) -> str | None:
    """
    Hub layout: ``<group>_fg_only/<label>/<basename>``; legacy: ``FG-Only/test_split/group_<gid>/``.
    """
    hub = os.path.join(
        path_foreground_only(waterbirds_root, gid), label, basename
    )
    if os.path.isfile(hub):
        return hub
    legacy = os.path.join(fg_only_root, "test_split", f"group_{gid}", basename)
    if os.path.isfile(legacy):
        return legacy
    return None


def _foreground_mask_bool(fg: Image.Image) -> np.ndarray:
    """
    ``True`` = likely foreground (bird) pixel; shape ``(H, W)``.

    JPEG and other opaque images become RGBA with **alpha 255 everywhere**; treating
    ``alpha > 8`` as foreground would mark the **whole** image and black it out in debug
    / ``--keep-mask-shape``. We only use alpha when it clearly carries transparency
    (min alpha not fully opaque).
    """
    rgba = np.asarray(fg.convert("RGBA"))
    rgb = rgba[:, :, :3].astype(np.int16)
    a = rgba[:, :, 3].astype(np.int16)

    # Real RGBA segmentation / cutout: some pixels not fully opaque.
    if int(a.max()) > 8 and int(a.min()) < 245:
        return a > 8

    # DFR-style composite: bird on mid-gray (150,150,150). Background shares that plate.
    near_plate = np.abs(rgb - 150).max(axis=-1) <= 28
    frac_plate = float(np.mean(near_plate))
    if frac_plate > 0.2:
        return np.abs(rgb - 150).max(axis=-1) > 28

    # Typical FG-only: bird on near-black background.
    return np.max(rgb, axis=2) > 35


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive minx, miny, maxx, maxy; ``None`` if empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    miny, maxy = int(ys.min()), int(ys.max())
    minx, maxx = int(xs.min()), int(xs.max())
    return minx, miny, maxx, maxy


def _square_covering_bbox(
    minx: int, miny: int, maxx: int, maxy: int, w: int, h: int
) -> tuple[int, int, int]:
    """
    Smallest axis-aligned square that covers the bbox, placed then clipped to ``[0,w)×[0,h)``.
    Returns ``(x0, y0, side)`` with square ``[x0, x0+side) × [y0, y0+side)``.
    """
    bw = maxx - minx + 1
    bh = maxy - miny + 1
    side = max(bw, bh, 1)
    side = min(side, w, h)
    cx = (minx + maxx + 1) / 2.0
    cy = (miny + maxy + 1) / 2.0
    x0 = int(math.floor(cx - side / 2.0))
    y0 = int(math.floor(cy - side / 2.0))
    x0 = max(0, min(x0, w - side))
    y0 = max(0, min(y0, h - side))
    return x0, y0, side


def _square_region_mask(
    w: int, h: int, x0: int, y0: int, side: int
) -> np.ndarray:
    """Boolean mask ``True`` inside the axis-aligned square (same geometry as ``ImageDraw``)."""
    m = np.zeros((h, w), dtype=bool)
    if side < 1:
        return m
    x0c = max(0, min(int(x0), w - 1))
    y0c = max(0, min(int(y0), h - 1))
    side_i = max(0, min(int(side), w - x0c, h - y0c))
    if side_i < 1:
        return m
    m[y0c : y0c + side_i, x0c : x0c + side_i] = True
    return m


def center_half_square_mask_hw(w: int, h: int) -> np.ndarray:
    """Centered square of side ``min(w,h)//2`` (matches ``mask_center_square_half_short_side``)."""
    x0, y0, side = center_black_square_params(w, h)
    return _square_region_mask(w, h, x0, y0, side)


def bird_mask_bool_from_fg_path(
    w: int,
    h: int,
    fg_path: str,
    *,
    warn_prefix: str,
) -> np.ndarray | None:
    """Foreground (bird) mask from FG-only image, or ``None`` if mask is empty."""
    with Image.open(fg_path) as fg_src:
        fg = fg_src.copy()
        if fg.size != (w, h):
            fg = fg.resize((w, h), Image.Resampling.NEAREST)
    mask = _foreground_mask_bool(fg)
    if not bool(np.any(mask)):
        print(
            f"{warn_prefix}: empty foreground mask in {fg_path!r}; skipping bird mask.",
            file=sys.stderr,
        )
        return None
    return mask


def square_mask_from_fg_path(
    w: int,
    h: int,
    fg_path: str,
    *,
    warn_prefix: str,
) -> np.ndarray:
    """Smallest square covering FG mask, else centered half-size square."""
    with Image.open(fg_path) as fg_src:
        fg = fg_src.copy()
        if fg.size != (w, h):
            fg = fg.resize((w, h), Image.Resampling.NEAREST)
    fmask = _foreground_mask_bool(fg)
    bbox = _bbox_from_mask(fmask)
    if bbox is None:
        print(
            f"{warn_prefix}: empty foreground mask in {fg_path!r}; "
            f"falling back to centered half-size square.",
            file=sys.stderr,
        )
        return center_half_square_mask_hw(w, h)
    minx, miny, maxx, maxy = bbox
    x0, y0, side = _square_covering_bbox(minx, miny, maxx, maxy, w, h)
    return _square_region_mask(w, h, x0, y0, side)


def fill_masked(
    im: Image.Image,
    mask: np.ndarray,
    *,
    use_inpaint: bool,
    inpaint_radius: float,
) -> Image.Image:
    """
    For every ``mask`` pixel that is true, either set RGB to black or inpaint from
    surrounding background (OpenCV Telea) when ``use_inpaint`` is true.
    """
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return im.convert("RGB")
    if not use_inpaint:
        out = rgb.copy()
        out[m] = 0
        return Image.fromarray(out)
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "--inpaint requires OpenCV. Install with: pip install opencv-python-headless"
        ) from e
    mask_u8 = m.astype(np.uint8) * 255
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    mask_u8 = np.ascontiguousarray(mask_u8)
    result = cv2.inpaint(
        bgr,
        mask_u8,
        float(inpaint_radius),
        flags=cv2.INPAINT_TELEA,
    )
    rgb_out = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_out)


def mask_square_from_fg_alignment(
    im: Image.Image,
    fg_path: str,
    *,
    warn_prefix: str,
) -> Image.Image:
    """
    Black out the smallest square that fully covers the foreground, using ``fg_path``
    (same canvas size as ``im``, or resized to match).
    """
    w, h = im.size
    mask = square_mask_from_fg_path(w, h, fg_path, warn_prefix=warn_prefix)
    return fill_masked(im, mask, use_inpaint=False, inpaint_radius=0.0)


def center_black_square_params(w: int, h: int) -> tuple[int, int, int]:
    """Return ``(x0, y0, side)`` for the centered ``min(w,h)//2`` square (may have ``side==0``)."""
    side = min(w, h) // 2
    if side < 1:
        return 0, 0, 0
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return x0, y0, side


def fg_black_square_params(
    canvas_w: int,
    canvas_h: int,
    fg_path: str,
    *,
    warn_prefix: str,
) -> tuple[int, int, int]:
    """Smallest covering square from FG-only mask on ``canvas_w×canvas_h``."""
    with Image.open(fg_path) as fg_src:
        fg = fg_src.copy()
        if fg.size != (canvas_w, canvas_h):
            fg = fg.resize((canvas_w, canvas_h), Image.Resampling.NEAREST)
    mask = _foreground_mask_bool(fg)
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        if warn_prefix:
            print(
                f"{warn_prefix}: empty FG mask in {fg_path!r}; using centered square params.",
                file=sys.stderr,
            )
        return center_black_square_params(canvas_w, canvas_h)
    minx, miny, maxx, maxy = bbox
    return _square_covering_bbox(minx, miny, maxx, maxy, canvas_w, canvas_h)


def black_square_params_from_paired_full(
    canvas_w: int,
    canvas_h: int,
    *,
    full_path: str | None,
    waterbirds_root: str,
    fg_only_root: str,
    gid: int,
    label: str,
    basename: str,
    warn_prefix: str = "",
) -> tuple[int, int, int]:
    """
    Square (``x0``, ``y0``, ``side``) aligned with the paired **full** FG+BG frame: FG-only
    mask when available, else centered ``min(w,h)//2`` (optionally informed by ``full_path``
    size after resize to canvas).
    """
    fg_p = resolve_fg_only_image_path(
        waterbirds_root, fg_only_root, gid, label, basename
    )
    if fg_p is not None and os.path.isfile(fg_p):
        return fg_black_square_params(
            canvas_w, canvas_h, fg_p, warn_prefix=warn_prefix
        )
    if full_path is not None and os.path.isfile(full_path):
        with Image.open(full_path) as fim:
            fr = fim.convert("RGB")
            if fr.size != (canvas_w, canvas_h):
                fr = fr.resize((canvas_w, canvas_h), _RESAMPLE)
        # Center square uses canvas geometry (same as masking ``fr`` alone).
        return center_black_square_params(canvas_w, canvas_h)
    return center_black_square_params(canvas_w, canvas_h)


def apply_black_square_coords(
    im: Image.Image, x0: int, y0: int, side: int
) -> Image.Image:
    """Paint black on ``im`` over ``[x0,x0+side)×[y0,y0+side)`` (no-op if ``side < 1``)."""
    rgb = im.convert("RGB")
    if side < 1:
        return rgb
    out = rgb.copy()
    dr = ImageDraw.Draw(out)
    dr.rectangle((x0, y0, x0 + side, y0 + side), fill=(0, 0, 0))
    return out


def cut_square_region_from_rgb(
    im: Image.Image, x0: int, y0: int, side: int
) -> Image.Image:
    """
    Remove the axis-aligned square ``[x0, x0+side) × [y0, y0+side)`` by splicing the
    complementary strips (no padding). Uses a **horizontal** splice (delete a vertical
    band) or **vertical** splice (delete a horizontal band), whichever keeps more pixels
    when both are valid; otherwise uses whichever splice is possible. If neither applies
    (degenerate mask), returns ``im`` unchanged.
    """
    rgb = im.convert("RGB")
    W, H = rgb.size
    if side < 1 or W < 1 or H < 1:
        return rgb
    x0 = max(0, min(int(x0), W - 1))
    y0 = max(0, min(int(y0), H - 1))
    side = int(side)
    side = max(0, min(side, W - x0, H - y0))
    if side < 1:
        return rgb

    def splice_horizontal() -> Image.Image | None:
        if side >= W or x0 + side > W:
            return None
        nw = W - side
        out = Image.new("RGB", (nw, H))
        ox = 0
        if x0 > 0:
            out.paste(rgb.crop((0, 0, x0, H)), (0, 0))
            ox = x0
        rw = W - x0 - side
        if rw > 0:
            out.paste(rgb.crop((x0 + side, 0, W, H)), (ox, 0))
        return out

    def splice_vertical() -> Image.Image | None:
        if side >= H or y0 + side > H:
            return None
        nh = H - side
        out = Image.new("RGB", (W, nh))
        oy = 0
        if y0 > 0:
            out.paste(rgb.crop((0, 0, W, y0)), (0, 0))
            oy = y0
        bh = H - y0 - side
        if bh > 0:
            out.paste(rgb.crop((0, y0 + side, W, H)), (0, oy))
        return out

    ah = (W - side) * H if side < W and x0 + side <= W else -1
    av = W * (H - side) if side < H and y0 + side <= H else -1
    if ah < 0 and av < 0:
        return rgb
    if ah >= av:
        sh = splice_horizontal()
        if sh is not None:
            return sh
        sv = splice_vertical()
        return sv if sv is not None else rgb
    sv = splice_vertical()
    if sv is not None:
        return sv
    sh = splice_horizontal()
    return sh if sh is not None else rgb


def mask_bird_shape_from_fg(
    im: Image.Image,
    fg_path: str,
    *,
    warn_prefix: str,
) -> Image.Image:
    """
    Set pixels to black where the paired FG-only image indicates foreground (bird shape),
    not an axis-aligned bounding square.
    """
    w, h = im.size
    mask = bird_mask_bool_from_fg_path(w, h, fg_path, warn_prefix=warn_prefix)
    if mask is None:
        return im.convert("RGB")
    return fill_masked(im, mask, use_inpaint=False, inpaint_radius=0.0)


def mask_center_square_half_short_side(im: Image.Image) -> Image.Image:
    """
    Replace the centered square of side ``min(w,h) // 2`` with black.

    Intended for composites that have a solid gray plate in the middle (~half the
    shorter image dimension per side of the square); this removes it before resize.
    """
    w, h = im.size
    mask = center_half_square_mask_hw(w, h)
    return fill_masked(im, mask, use_inpaint=False, inpaint_radius=0.0)


def normalize_to_canvas(
    im: Image.Image, tw: int, th: int, fit: str
) -> Image.Image:
    if fit == "stretch":
        return _resize_stretch(im, tw, th)
    if fit == "contain":
        return _resize_contain(im, tw, th)
    if fit == "cover":
        return _resize_cover(im, tw, th)
    raise ValueError(fit)


def _save_image(path: str, im: Image.Image) -> None:
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=ext or ".jpg", prefix=".wb_norm_", dir=d or "."
    )
    os.close(fd)
    try:
        if ext in (".jpg", ".jpeg"):
            im.save(tmp, format="JPEG", quality=95, subsampling=0)
        elif ext == ".png":
            im.save(tmp, format="PNG", optimize=True)
        else:
            im.save(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Resize Waterbirds images under *_bg_only and/or original FG+BG folders to a "
            "fixed width×height (in place) so Places search uses a single warp size per pool."
        )
    )
    p.add_argument(
        "--waterbirds-root",
        default=os.path.join(DATASETS_PATH, "Waterbirds"),
        help="Waterbirds root (subscenario folders)",
    )
    p.add_argument(
        "--targets",
        choices=("bg_only", "full", "both"),
        default="bg_only",
        help=(
            "Which trees to resize: bg_only (*_bg_only), full (original landbird_on_land/…), "
            "or both (full first, then bg_only; default: bg_only)."
        ),
    )
    p.add_argument("--width", type=int, required=True, metavar="W")
    p.add_argument("--height", type=int, required=True, metavar="H")
    p.add_argument(
        "--fit",
        choices=("stretch", "contain", "cover"),
        default="contain",
        help="How to map source aspect ratio to the target canvas (default: contain).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print how many files would be touched per folder",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Process at most 2 images per ``<group>/<label>/`` folder and write results to "
            "``<waterbirds-root>/<scenario_folder_name>_debug/<label>/`` (same basename); "
            "does not modify originals. Incompatible with --dry-run."
        ),
    )
    p.add_argument(
        "--keep-mask-shape",
        action="store_true",
        dest="keep_mask_shape",
        help=(
            "Before resizing, paint black on pixels covered by the **bird-shaped** foreground "
            "from the same-named ``*_fg_only`` file (Hub or legacy FG-Only tree). "
            "Mutually exclusive with --gray-square and --match-fg-size."
        ),
    )
    p.add_argument(
        "--gray-square",
        action="store_true",
        help=(
            "Before resizing, paint a black square: either centered min(w,h)//2 (default), "
            "or, with --match-fg-size, the smallest square that covers the FG-only mask."
        ),
    )
    p.add_argument(
        "--match-fg-size",
        action="store_true",
        dest="match_fg_size",
        help=(
            "Requires --gray-square. For each image, load the same-named file under "
            "``*_fg_only`` (Hub) or ``<fg-only-root>/test_split/group_<id>/`` (legacy), "
            "infer foreground pixels, and black out the smallest axis-aligned square that "
            "fully covers that foreground (aligned to this image size). "
            "If the FG file is missing, falls back to the centered half-size square."
        ),
    )
    p.add_argument(
        "--fg-only-root",
        default=None,
        metavar="DIR",
        help=(
            "Legacy FG-Only tree root (expects ``test_split/group_{0..3}/``). "
            "Default when omitted: ``<--waterbirds-root>/FG-Only``. Hub ``landbird_on_…_fg_only`` "
            "paths are always tried first."
        ),
    )
    p.add_argument(
        "--inpaint",
        action="store_true",
        help=(
            "After building the removal mask (--gray-square, --match-fg-size, or "
            "--keep-mask-shape), fill masked pixels with OpenCV Telea inpainting from "
            "neighboring background instead of black. Requires opencv-python-headless."
        ),
    )
    p.add_argument(
        "--inpaint-radius",
        type=float,
        default=4.0,
        metavar="R",
        help="Radius passed to cv2.inpaint (default: 4). Only used with --inpaint.",
    )
    args = p.parse_args()
    tw, th = args.width, args.height
    if tw < 1 or th < 1:
        p.error("--width and --height must be positive")
    if args.match_fg_size and not args.gray_square:
        p.error("--match-fg-size requires --gray-square")
    if args.keep_mask_shape and (args.gray_square or args.match_fg_size):
        p.error(
            "--keep-mask-shape cannot be combined with --gray-square or --match-fg-size"
        )
    if args.inpaint and not (
        args.gray_square or args.match_fg_size or args.keep_mask_shape
    ):
        p.error(
            "--inpaint requires one of --gray-square, --match-fg-size, or --keep-mask-shape"
        )
    if args.debug and args.dry_run:
        p.error("--debug and --dry-run cannot be used together")

    root = os.path.abspath(args.waterbirds_root)
    fg_only_root = os.path.abspath(
        args.fg_only_root
        if args.fg_only_root
        else os.path.join(root, "FG-Only")
    )
    total_changed = 0
    total_skip_same = 0

    # ``both``: original FG+BG first, then bg_only (for Places pipelines).
    if args.targets == "both":
        kind_order: tuple[str, ...] = ("full", "bg_only")
    else:
        kind_order = (args.targets,)

    for kind in kind_order:
        for gid, group_name in enumerate(GROUP_SUBDIRS):
            if kind == "bg_only":
                scenario_root = path_background_only(root, gid)
            else:
                scenario_root = path_with_background(root, gid)
            if not os.path.isdir(scenario_root):
                continue
            for label in ("0", "1"):
                d = os.path.join(scenario_root, label)
                if not os.path.isdir(d):
                    continue
                paths = list_images_recursive(d)
                if not paths:
                    continue
                if args.debug:
                    paths = paths[:2]
                scenario_folder_name = os.path.basename(
                    os.path.normpath(scenario_root)
                )
                desc = f"{kind} {group_name}/{label}"
                if args.dry_run:
                    n_diff = 0
                    for fp in paths:
                        with Image.open(fp) as im:
                            if im.size != (tw, th):
                                n_diff += 1
                    print(
                        f"{desc}: {len(paths)} images, "
                        f"{n_diff} would be resized to {tw}x{th}"
                    )
                    continue
                for fp in tqdm(paths, desc=desc, leave=False):
                    with Image.open(fp) as src:
                        im = src.convert("RGB")
                        w0, h0 = im.size
                        mask: np.ndarray | None = None
                        if args.keep_mask_shape:
                            bn = os.path.basename(fp)
                            fg_p = resolve_fg_only_image_path(
                                root, fg_only_root, gid, label, bn
                            )
                            if fg_p is None:
                                print(
                                    f"{desc}: no FG-only match for {bn!r} "
                                    f"(tried hub + {fg_only_root!r}); skipping bird mask.",
                                    file=sys.stderr,
                                )
                            else:
                                mask = bird_mask_bool_from_fg_path(
                                    w0, h0, fg_p, warn_prefix=desc
                                )
                        elif args.match_fg_size:
                            bn = os.path.basename(fp)
                            fg_p = resolve_fg_only_image_path(
                                root, fg_only_root, gid, label, bn
                            )
                            if fg_p is None:
                                print(
                                    f"{desc}: no FG-only match for {bn!r} "
                                    f"(tried hub + {fg_only_root!r}); using centered square.",
                                    file=sys.stderr,
                                )
                                mask = center_half_square_mask_hw(w0, h0)
                            else:
                                mask = square_mask_from_fg_path(
                                    w0, h0, fg_p, warn_prefix=desc
                                )
                        elif args.gray_square:
                            mask = center_half_square_mask_hw(w0, h0)

                        if mask is not None:
                            im = fill_masked(
                                im,
                                mask,
                                use_inpaint=args.inpaint,
                                inpaint_radius=args.inpaint_radius,
                            )
                        if not args.debug:
                            if (
                                im.size == (tw, th)
                                and not args.gray_square
                                and not args.match_fg_size
                                and not args.keep_mask_shape
                                and not args.inpaint
                            ):
                                total_skip_same += 1
                                continue
                        out = normalize_to_canvas(im, tw, th, args.fit)
                    if args.debug:
                        debug_dir = os.path.join(
                            root, f"{scenario_folder_name}_debug", label
                        )
                        out_path = os.path.join(debug_dir, os.path.basename(fp))
                        _save_image(out_path, out)
                    else:
                        _save_image(fp, out)
                    total_changed += 1

    if args.dry_run:
        return
    gs = ", --gray-square" if args.gray_square else ""
    mf = ", --match-fg-size" if args.match_fg_size else ""
    km = ", --keep-mask-shape" if args.keep_mask_shape else ""
    ip = ", --inpaint" if args.inpaint else ""
    db = ", --debug" if args.debug else ""
    if args.debug:
        print(
            f"Done (--targets={args.targets}{gs}{mf}{km}{ip}{db}). Wrote {total_changed} images "
            f"under ``*_debug`` subfolders (max 2 per label folder); not in-place."
        )
    else:
        print(
            f"Done (--targets={args.targets}{gs}{mf}{km}{ip}). Resized {total_changed} files; "
            f"skipped {total_skip_same} already {tw}x{th}."
        )


if __name__ == "__main__":
    main()
