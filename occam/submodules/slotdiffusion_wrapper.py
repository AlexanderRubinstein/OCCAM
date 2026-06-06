"""Inference helpers for SlotDiffusion unsupervised segmentation (COCO + DINO ViT-S/8).

SlotDiffusion evaluates segmentation by running the slot encoder with testing mode,
which returns soft masks from the last slot-attention iteration without invoking the
latent diffusion decoder. See https://arxiv.org/abs/2305.11281 and the official
repo: https://github.com/Wuziyi616/SlotDiffusion
"""

from __future__ import annotations

import os
import zipfile
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from transformers import ViTModel

from stuned.utility.utils import get_project_root_path

# Config: slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_coco_params-res224.py
RESOLUTION = (224, 224)
NUM_SLOTS = 7
SLOT_SIZE = 256
SLOT_MLP_SIZE = 512
NUM_ITERATIONS = 3
ENC_OUT_CHANNELS = 256
PATCH_SIZE = 8
VISUAL_RESOLUTION = (28, 28)
VISUAL_CHANNELS = 384
NORM_MEAN = 0.5
NORM_STD = 0.5
EPS = 1e-6

DEFAULT_CHECKPOINT_NAME = "sa_ldm_dino_coco_params-res224.pth"
DEFAULT_CHECKPOINT_URL = (
    "https://drive.google.com/uc?id=1PSElX2ucqqLuCjjl2_skM-7-qjwb2hWh"
)
ENCODER_PREFIXES = (
    "init_latents",
    "slot_attention.",
    "encoder.",
    "encoder_pos_embedding.",
    "encoder_out_layer.",
)

_DINO_ATTN_SUFFIX_MAP = {
    "attention.attention.query": "attention.q_proj",
    "attention.attention.key": "attention.k_proj",
    "attention.attention.value": "attention.v_proj",
    "attention.output.dense": "attention.o_proj",
    "intermediate.dense": "mlp.fc1",
    "output.dense": "mlp.fc2",
}


def _remap_dino_checkpoint_key(key: str) -> Optional[str]:
    """Map SlotDiffusion ViT keys to current HuggingFace ViTModel names."""
    prefix = "encoder.dino."
    if not key.startswith(prefix):
        return key

    relative = key[len(prefix):]
    if relative.startswith("embeddings.") or relative.startswith("layernorm."):
        return prefix + relative

    if relative.startswith("encoder.layer."):
        layer_idx, remainder = relative[len("encoder.layer.") :].split(".", 1)
        param = remainder.rsplit(".", 1)[-1]
        for old_suffix, new_suffix in _DINO_ATTN_SUFFIX_MAP.items():
            if remainder.startswith(old_suffix):
                return f"{prefix}layers.{layer_idx}.{new_suffix}.{param}"
        if remainder.startswith("layernorm_before.") or remainder.startswith("layernorm_after."):
            return f"{prefix}layers.{layer_idx}.{remainder}"
    return None


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2).contiguous()
        return inputs + emb_proj


class DINOEncoder(nn.Module):
    """Frozen DINO ViT-S/8 encoder (HuggingFace ViTModel, SlotDiffusion-compatible)."""

    def __init__(self, resolution: int = 28, patch_size: int = 8, small_size: bool = True):
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        version = "s" if small_size else "b"
        self.dino = ViTModel.from_pretrained(f"facebook/dino-vit{version}{patch_size}")
        for param in self.dino.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.dino(x).last_hidden_state[:, 1:, :]
        out = out.reshape(out.shape[0], self.resolution, self.resolution, out.shape[-1])
        return out.permute(0, 3, 1, 2)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self


class SlotAttentionWMask(nn.Module):
    """Slot attention that returns segmentation masks from the last iteration."""

    def __init__(
        self,
        in_features,
        num_iterations,
        num_slots,
        slot_size,
        mlp_hidden_size,
        eps=1e-6,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.eps = eps
        self.attn_scale = slot_size**-0.5

        self.norm_inputs = nn.LayerNorm(in_features)
        self.project_q = nn.Sequential(
            nn.LayerNorm(slot_size),
            nn.Linear(slot_size, slot_size, bias=False),
        )
        self.project_k = nn.Linear(in_features, slot_size, bias=False)
        self.project_v = nn.Linear(in_features, slot_size, bias=False)
        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_size),
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size),
        )

    def forward(self, inputs, slots):
        bs, num_inputs, _ = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        for attn_iter in range(self.num_iterations):
            slots_prev = slots
            q = self.project_q(slots)
            attn_logits = self.attn_scale * torch.einsum("bnc,bmc->bnm", k, q)
            attn = F.softmax(attn_logits, dim=-1)
            if attn_iter == self.num_iterations - 1:
                seg_mask = attn.detach().clone().permute(0, 2, 1)

            attn = attn + self.eps
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.einsum("bnm,bnc->bmc", attn, v)
            slots = self.gru(
                updates.view(bs * self.num_slots, self.slot_size),
                slots_prev.view(bs * self.num_slots, self.slot_size),
            )
            slots = slots.view(bs, self.num_slots, self.slot_size)
            slots = slots + self.mlp(slots)

        return slots, seg_mask

    @property
    def dtype(self):
        return self.project_k.weight.dtype


class SlotDiffusionEncoder(nn.Module):
    """Encode-only SlotDiffusion model for unsupervised mask inference."""

    def __init__(self):
        super().__init__()
        self.resolution = RESOLUTION
        self.num_slots = NUM_SLOTS
        self.visual_resolution = VISUAL_RESOLUTION

        self.init_latents = nn.Parameter(torch.empty(1, NUM_SLOTS, SLOT_SIZE))
        nn.init.normal_(self.init_latents)

        self.encoder = DINOEncoder(
            resolution=VISUAL_RESOLUTION[0],
            patch_size=PATCH_SIZE,
            small_size=True,
        )
        self.encoder_pos_embedding = SoftPositionEmbed(VISUAL_CHANNELS, VISUAL_RESOLUTION)
        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(VISUAL_CHANNELS),
            nn.Linear(VISUAL_CHANNELS, ENC_OUT_CHANNELS),
            nn.ReLU(),
            nn.Linear(ENC_OUT_CHANNELS, ENC_OUT_CHANNELS),
        )
        self.slot_attention = SlotAttentionWMask(
            in_features=ENC_OUT_CHANNELS,
            num_iterations=NUM_ITERATIONS,
            num_slots=NUM_SLOTS,
            slot_size=SLOT_SIZE,
            mlp_hidden_size=SLOT_MLP_SIZE,
            eps=EPS,
        )

    @property
    def dtype(self):
        return self.slot_attention.dtype

    def _get_encoder_out(self, img):
        encoder_out = self.encoder(img).type(self.dtype)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        encoder_out = encoder_out.permute(0, 2, 1).contiguous()
        return self.encoder_out_layer(encoder_out)

    def encode(self, img, init_slots=None):
        batch_size = img.shape[0]
        encoder_out = self._get_encoder_out(img)
        if init_slots is None:
            init_slots = self.init_latents.repeat(batch_size, 1, 1)

        slots, masks = self.slot_attention(encoder_out, init_slots)
        masks = masks.unflatten(-1, self.visual_resolution)

        if self.visual_resolution != self.resolution:
            masks = masks.flatten(0, 1).unsqueeze(1)
            masks = F.interpolate(
                masks,
                self.resolution,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).unflatten(0, (batch_size, self.num_slots))

        return slots, masks

    @torch.no_grad()
    def forward(self, img):
        slots, masks = self.encode(img)
        return {"masks": masks, "slots": slots}


def default_checkpoint_path() -> str:
    return os.path.join(get_project_root_path(), "checkpoints", DEFAULT_CHECKPOINT_NAME)


def _extract_checkpoint_from_zip(zip_path: str, destination: str) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        candidates = [
            name for name in archive.namelist()
            if name.endswith(DEFAULT_CHECKPOINT_NAME)
        ]
        if not candidates:
            raise FileNotFoundError(
                f"{DEFAULT_CHECKPOINT_NAME} not found inside {zip_path}"
            )
        archive.extract(candidates[0], os.path.dirname(destination))
        extracted = os.path.join(os.path.dirname(destination), candidates[0])
        if extracted != destination:
            os.replace(extracted, destination)
    return destination


def download_checkpoint(
    destination: Optional[str] = None,
    url: str = DEFAULT_CHECKPOINT_URL,
) -> str:
    """Download the COCO SlotDiffusion checkpoint if missing."""
    destination = destination or default_checkpoint_path()
    if os.path.exists(destination):
        return destination

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Install gdown to auto-download the SlotDiffusion checkpoint: pip install gdown"
        ) from exc

    zip_path = destination + ".zip"
    gdown.download(url, zip_path, quiet=False)
    return _extract_checkpoint_from_zip(zip_path, destination)


def load_checkpoint(model: SlotDiffusionEncoder, checkpoint_path: str) -> SlotDiffusionEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {}
    for key, value in checkpoint.items():
        if not key.startswith(ENCODER_PREFIXES):
            continue
        if key.startswith("encoder.dino.pooler."):
            continue
        remapped_key = _remap_dino_checkpoint_key(key)
        if remapped_key is None:
            continue
        state_dict[remapped_key] = value

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"encoder.dino.pooler.dense.weight", "encoder.dino.pooler.dense.bias"}
    missing = [key for key in missing if key not in allowed_missing]
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading SlotDiffusion checkpoint: {unexpected}"
        )
    if missing:
        raise RuntimeError(f"Missing keys when loading SlotDiffusion checkpoint: {missing}")
    model.eval()
    return model


def build_model(pretrained_dino: bool = True) -> SlotDiffusionEncoder:
    if not pretrained_dino:
        raise ValueError("SlotDiffusion requires a DINO encoder; use pretrained_dino=True.")
    return SlotDiffusionEncoder()


def load_model(
    checkpoint_path: Optional[str] = None,
    download: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> SlotDiffusionEncoder:
    if checkpoint_path is None:
        checkpoint_path = default_checkpoint_path()
    if download and not os.path.exists(checkpoint_path):
        download_checkpoint(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SlotDiffusion checkpoint not found at {checkpoint_path}. "
            "Download pretrained.zip from the SlotDiffusion release and extract "
            f"{DEFAULT_CHECKPOINT_NAME}, or call download_checkpoint()."
        )

    model = build_model()
    load_checkpoint(model, checkpoint_path)
    return model.to(device)


def _resize_min_shape(image: np.ndarray, resolution=RESOLUTION) -> np.ndarray:
    height, width, _ = image.shape
    factor = max(resolution[0] / height, resolution[1] / width)
    resize_h = int(round(height * factor))
    resize_w = int(round(width * factor))
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((resize_w, resize_h), Image.BILINEAR)
    return np.array(pil_image)


def _center_crop(image: np.ndarray, resolution=RESOLUTION) -> np.ndarray:
    height, width, _ = image.shape
    if height == resolution[0]:
        crop_xmin = (width - resolution[0]) // 2
        return image[:, crop_xmin:crop_xmin + resolution[0]]
    crop_ymin = (height - resolution[1]) // 2
    return image[crop_ymin:crop_ymin + resolution[1], :]


def preprocess_image(image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
    """COCO val preprocessing used by SlotDiffusion (ResizeMinShape + CenterCrop)."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    image = _resize_min_shape(image, RESOLUTION)
    image = _center_crop(image, RESOLUTION)
    image = image.astype(np.float32) / 255.0
    image = (image - NORM_MEAN) / NORM_STD
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    return tensor


def build_preprocessing():
    return preprocess_image


@torch.no_grad()
def predict(
    model: SlotDiffusionEncoder,
    image,
    device: Union[str, torch.device] = "cpu",
):
    """Run SlotDiffusion segmentation on a PIL image or preprocessed tensor."""
    preproc = build_preprocessing()
    if not isinstance(image, torch.Tensor):
        tensor = preproc(image).unsqueeze(0)
    else:
        tensor = image if image.ndim == 4 else image.unsqueeze(0)

    model = model.to(device)
    tensor = tensor.to(device)
    return model(tensor)
