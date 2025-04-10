import sys
import os
import torch


from stuned.utility.utils import get_project_root_path
from stuned.local_datasets.transforms import (
    make_transforms,
)


# local modules
sys.path.insert(
    0,
    os.path.join(
       get_project_root_path()
    )
)
from occam.submodules.alpha_clip import alpha_clip
sys.path.pop(0)


# # local modules
# import alpha_clip


CHECKPOINTS_DICT = {
    "ViT-L/14": os.path.join(
        get_project_root_path(),
        "checkpoints",
        "clip_l14_grit20m_fultune_2xe.pth"
    ),
}


ALPHA_CLIP_IMAGE_PREPROCESS_CONFIG = {
    "transforms_list": [
        "from_class-ToTensor",
        "from_class-Resize",
        "from_class-CenterCrop",
        "from_class-Normalize"
    ],
    "from_class-Normalize": {
        "class": "torchvision.transforms.Normalize",
        "kwargs": {
            "std": [
                0.26862954,
                0.26130258,
                0.27577711
            ],
            "mean": [
                0.48145466,
                0.4578275,
                0.40821073
            ]
        }
    },
    "from_class-ToTensor": {
        "class": "torchvision.transforms.ToTensor"
    },
    "from_class-CenterCrop": {
        "class": "torchvision.transforms.CenterCrop",
        "kwargs": {
            "size": 224
        }
    },
    "from_class-Resize": {
        "class": "torchvision.transforms.Resize",
        "kwargs": {
            "size": 224
        }
    }
}


ALPHA_CLIP_MASK_PREPROCESS_CONFIG = {
    "transforms_list": [
        "from_class-Normalize"
    ],
    "from_class-Normalize": {
        "class": "torchvision.transforms.Normalize",
        "kwargs": {
        "std": 0.26,
        "mean": 0.5
        }
    },
}


class AlphaClipWrapper(torch.nn.Module):
    def __init__(self, model, text_features):
        super().__init__()
        self.model = model
        self.text_features = text_features
        # in addition to resize and crop performed in getitem,
        # mask_transform also normalizes the mask
        self.mask_transform = make_transforms(ALPHA_CLIP_MASK_PREPROCESS_CONFIG)

    def forward(self, images_masks):

        if isinstance(images_masks, (tuple, list)):
            assert len(images_masks) == 2
            images = images_masks[0]
            masks = images_masks[1]
        else:
            images = images_masks
            masks_shape = list(images.shape)
            masks_shape[1] = 1
            # if no masks are provided, the mask is considered to cover all image
            masks = torch.ones(masks_shape)

        device = images.device
        images = images.half()
        masks = self.mask_transform(masks.float())
        masks = masks.half().to(device)
        if masks.shape[1] > 1:
            assert torch.isclose(masks[:, 1, ...], masks[:, 0, ...]).all()
            masks = masks[:, 0, ...].unsqueeze(1)
        image_features = self.model.visual(images, masks)

        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # get probs
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        return similarity


def make_alpha_clip_model(
    model_id,
    category_list,
    device='cuda'
):
    model, preprocess = alpha_clip.load(
        model_id,
        alpha_vision_ckpt_pth=CHECKPOINTS_DICT[model_id],
        device=device
    )

    text = alpha_clip.tokenize(category_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return AlphaClipWrapper(model, text_features)
