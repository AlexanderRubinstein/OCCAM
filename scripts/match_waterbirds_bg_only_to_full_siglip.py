#!/usr/bin/env python3
"""
Match each ``*_bg_only`` image to the most similar **foreground+background** image in the
paired subscenario (same spurious-cue group and same coarse label folder ``0`` / ``1``),
using **SigLIP** (OpenCLIP) image embeddings and cosine similarity.

Outputs a JSON mapping (and optional CSV) from background-only relative paths to full-scene
relative paths plus similarity scores. Use this when ``*_bg_only`` filenames do not align
with ``<group>/`` filenames.

Requires: ``torch``, ``open_clip``, ``Pillow``, ``tqdm`` (same stack as OCCAM CLIP / SigLIP eval).

Example:

  python scripts/match_waterbirds_bg_only_to_full_siglip.py \\
    --waterbirds-root data/datasets/Waterbirds \\
    --output data/datasets/Waterbirds/bg_only_to_full_siglip.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from occam.datasets.utils import DATASETS_PATH
from occam.datasets.waterbirds_layout import (
    GROUP_SUBDIRS,
    path_background_only,
    path_with_background,
)

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
) -> Tuple[List[dict], dict]:
    waterbirds_root = os.path.abspath(waterbirds_root)
    amp_dtype = torch.float16 if use_amp and device.type == "cuda" else None

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
    }

    for gid, group_name in enumerate(GROUP_SUBDIRS):
        full_root = path_with_background(waterbirds_root, gid)
        bg_root = path_background_only(waterbirds_root, gid)
        for label in ("0", "1"):
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
                    "full_relpath": _to_rel(full_paths[fi], waterbirds_root),
                    "cosine_similarity": sim,
                }
                if margin is not None:
                    row["best_minus_second"] = float(margin[j])
                matches.append(row)

    return matches, meta


def _summarize(matches: Sequence[dict]) -> dict:
    inv: Dict[str, List[str]] = defaultdict(list)
    for m in matches:
        inv[m["full_relpath"]].append(m["bg_only_relpath"])
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
    args = parser.parse_args()

    device = torch.device(args.device)
    matches, meta = run_matching(
        args.waterbirds_root,
        model_id=args.model_id,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        matmul_chunk=args.matmul_chunk,
        device=device,
        use_amp=not args.no_amp,
        min_margin=args.min_best_minus_second,
    )

    if args.min_best_minus_second is not None:
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
    mapping = {
        m["bg_only_relpath"]: {
            "full_relpath": m["full_relpath"],
            "cosine_similarity": m["cosine_similarity"],
        }
        for m in matches
    }
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
    print(
        f"Full-image targets with >1 bg match: {summary['num_full_targets_with_multiple_bg']}"
    )

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
