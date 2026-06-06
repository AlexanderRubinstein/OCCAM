"""Inference helpers for basic DINOSAUR (OCLF / Bridging-the-Gap paper).

Basic DINOSAUR uses a frozen DINO ViT-B/16 encoder, slot attention, and an MLP
patch decoder. Checkpoints are PyTorch Lightning `.ckpt` files from the
Object-Centric Learning Framework (OCLF), e.g. the COCO model from AdaSlot:
https://github.com/amazon-science/AdaSlot
"""

from __future__ import annotations

import os
from functools import partial
from typing import Optional, Union

import torch
from torch import nn

from ftdinosaur_inference import utils as dino_utils
from ftdinosaur_inference.modules.decoding import PatchDecoder
from ftdinosaur_inference.modules.dinosaur import DINOSAUR
from ftdinosaur_inference.modules.helpers import build_mlp, build_two_layer_mlp
from ftdinosaur_inference.modules.slot_attention import (
    RandomSlotInitialization,
    SlotAttentionGrouping,
)

from stuned.utility.utils import get_project_root_path

# AdaSlot release: fixed-slot DINOSAUR trained on COCO (OCLF experiment
# `coco_feat_rec_dino_base16`).
DEFAULT_CHECKPOINT_URL = (
    "https://drive.google.com/uc?id=1NaGR0n25Y3Sn6zQKESst3uvjuI0F443C"
)
DEFAULT_CHECKPOINT_NAME = "dinosaur_coco_dino_base16.ckpt"

SLOT_DIM = 256
FEATURE_DIM = 768
NUM_SLOTS = 7
NUM_PATCHES = 196
INPUT_SIZE = 224


class DinoV1Encoder(nn.Module):
    """Frozen DINO ViT-B/16 patch encoder (timm / OCLF-compatible keys)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb16",
            pretrained=pretrained,
        )
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # OCLF uses block-12 features (last block output, no final norm).
        return self.backbone.get_intermediate_layers(images, n=1)[0][:, 1:]


def build_model(pretrained_encoder: bool = True) -> DINOSAUR:
    """Build DINOSAUR with the COCO ViT-B/16 + MLP decoder configuration."""
    return DINOSAUR(
        encoder=DinoV1Encoder(pretrained=pretrained_encoder),
        slot_init=RandomSlotInitialization(SLOT_DIM, NUM_SLOTS),
        slot_attention=SlotAttentionGrouping(
            SLOT_DIM,
            SLOT_DIM,
            ff_mlp=build_two_layer_mlp(
                SLOT_DIM,
                SLOT_DIM,
                hidden_dim=4 * SLOT_DIM,
                initial_layer_norm=True,
                residual=True,
            ),
            feature_transform=build_two_layer_mlp(
                FEATURE_DIM,
                SLOT_DIM,
                hidden_dim=FEATURE_DIM,
                initial_layer_norm=True,
            ),
        ),
        decoder=PatchDecoder(
            SLOT_DIM,
            FEATURE_DIM,
            NUM_PATCHES,
            decoder=partial(build_mlp, features=[2048, 2048, 2048]),
            top_k=None,
        ),
    )


def default_checkpoint_path() -> str:
    return os.path.join(get_project_root_path(), "checkpoints", DEFAULT_CHECKPOINT_NAME)


def download_checkpoint(
    destination: Optional[str] = None,
    url: str = DEFAULT_CHECKPOINT_URL,
) -> str:
    """Download the default COCO DINOSAUR checkpoint if missing."""
    destination = destination or default_checkpoint_path()
    if os.path.exists(destination):
        return destination

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Install gdown to auto-download the DINOSAUR checkpoint: pip install gdown"
        ) from exc

    gdown.download(url, destination, quiet=False)
    return destination


def load_oclf_checkpoint(model: DINOSAUR, checkpoint_path: str) -> DINOSAUR:
    """Load slot-attention weights from an OCLF / PyTorch Lightning checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading OCLF checkpoints requires `omegaconf` "
            "(pip install omegaconf)."
        ) from exc
    state_dict = dino_utils.convert_checkpoint_from_oclf(checkpoint)

    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    decoder_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("encoder.") and "gumbel" not in key
    }

    # OCLF drops the final ViT norm when extracting block-12 features.
    model.encoder.backbone.load_state_dict(encoder_state, strict=False)
    model.load_state_dict(decoder_state, strict=False)
    model.eval()
    return model


def load_model(
    checkpoint_path: Optional[str] = None,
    download: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> DINOSAUR:
    """Build basic DINOSAUR and load COCO checkpoint weights."""
    if checkpoint_path is None:
        checkpoint_path = default_checkpoint_path()
    if download and not os.path.exists(checkpoint_path):
        download_checkpoint(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"DINOSAUR checkpoint not found at {checkpoint_path}. "
            "Download COCO.ckpt from the AdaSlot checkpoint release and save it there, "
            "or call download_checkpoint()."
        )

    model = build_model(pretrained_encoder=False)
    load_oclf_checkpoint(model, checkpoint_path)
    return model.to(device)


def build_preprocessing() -> nn.Module:
    """224x224 center-crop preprocessing used by OCLF COCO DINOSAUR."""
    return dino_utils.build_preprocessing(
        INPUT_SIZE,
        dino_utils.IMAGENET_MEAN,
        dino_utils.IMAGENET_STD,
        center_crop=True,
    )


@torch.no_grad()
def predict(
    model: DINOSAUR,
    image,
    num_slots: int = NUM_SLOTS,
    device: Union[str, torch.device] = "cpu",
):
    """Run basic DINOSAUR on a PIL image and return the model output dict."""
    preproc = build_preprocessing()
    if not isinstance(image, torch.Tensor):
        tensor = preproc(image).unsqueeze(0)
    else:
        tensor = image

    model = model.to(device)
    tensor = tensor.to(device)
    return model(tensor, num_slots=num_slots)
