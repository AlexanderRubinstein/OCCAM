# Optional legacy Hugging Face ``datasets`` loading script for the OCCAM Waterbirds layout.
# ``datasets`` 4.x does **not** load Hub ``*.py`` dataset scripts; the supported approach is
# README YAML ``configs`` (see ``upload_waterbirds_to_huggingface.py``). Upload this file
# only with ``--dataset-script`` if you need it for an old ``datasets`` pin.
#
# Copy to the dataset repo root as ``waterbirds.py`` (repo id ``…/waterbirds``) or
# ``<repo_last_segment>.py`` per https://huggingface.co/docs/datasets/dataset_script

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Tuple, Union

import datasets
from datasets import (
    BuilderConfig,
    ClassLabel,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Image,
    Split,
    SplitGenerator,
    Version,
)

_SUBSCENARIOS: Tuple[str, ...] = (
    "landbird_on_land",
    "landbird_on_land_fg_only",
    "landbird_on_water",
    "landbird_on_water_fg_only",
    "waterbird_on_land",
    "waterbird_on_land_fg_only",
    "waterbird_on_water",
    "waterbird_on_water_fg_only",
)

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _is_image_file(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in _IMAGE_EXT


def _iter_image_label_pairs(
    scenario_root: str,
) -> Iterator[Tuple[str, Dict[str, Union[str, int]]]]:
    """Yield (id, example) for ImageFolder-style layout ``scenario_root/{0,1}/*``."""
    n = 0
    for label_str in ("0", "1"):
        class_dir = os.path.join(scenario_root, label_str)
        if not os.path.isdir(class_dir):
            continue
        label = int(label_str)
        for fn in sorted(os.listdir(class_dir)):
            if not _is_image_file(fn):
                continue
            path = os.path.join(class_dir, fn)
            if not os.path.isfile(path):
                continue
            yield f"{label_str}_{n}", {"image": path, "label": label}
            n += 1


class Waterbirds(GeneratorBasedBuilder):
    """OCCAM Waterbirds — one config (Subset) per subscenario folder."""

    VERSION = Version("1.0.0")

    BUILDER_CONFIGS: List[BuilderConfig] = [
        BuilderConfig(
            name=name,
            version=VERSION,
            description=f"Subscenario `{name}/` (coarse labels in 0/=waterbird, 1/=landbird).",
        )
        for name in _SUBSCENARIOS
    ]
    DEFAULT_CONFIG_NAME = _SUBSCENARIOS[0]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description=(
                "Waterbirds (OCCAM layout): eight subscenario directories at repo root, "
                "each with coarse class folders 0 (waterbird) and 1 (landbird)."
            ),
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(names=["waterbird", "landbird"]),
                }
            ),
            supervised_keys=("image", "label"),
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[SplitGenerator]:
        cfg_name = self.config.name
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), cfg_name)
        )
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"scenario_root": base_dir},
            )
        ]

    def _generate_examples(
        self, scenario_root: str
    ) -> Iterator[Tuple[str, Dict[str, Union[str, int]]]]:
        if not os.path.isdir(scenario_root):
            return
        yield from _iter_image_label_pairs(scenario_root)
