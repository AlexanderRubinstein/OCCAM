import matplotlib
import numpy as np
import PIL
import torch
from PIL import Image
from torchvision.utils import draw_segmentation_masks
from ftdinosaur_inference import (
    utils,
    build_dinosaur
)


# inference dino
def get_cmap(num_classes, cmap="tab10"):
    cmap = matplotlib.colormaps[cmap].resampled(num_classes)(range(num_classes))
    cmap = [tuple((255 * cl[:3]).astype(int)) for cl in cmap]
    return cmap


def overlay_masks_on_image(
    img: PIL.Image, masks: torch.Tensor, num_masks: int, alpha: float = 0.6
) -> PIL.Image:
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # C x H x W
    height, width = img_tensor.shape[1:]

    # Need to resize masks to image (1 x K x P -> 1 x K x H x W)
    masks_as_image = utils.resize_patches_to_image(masks, size=(height, width))
    masks_as_image = utils.soft_masks_to_one_hot(masks_as_image).squeeze(0)

    # Overlay masks on image
    masks_on_image = draw_segmentation_masks(
        img_tensor, masks_as_image, alpha=alpha, colors=get_cmap(num_masks)
    ) # takes input of shape (Masks, H, W)

    # Convert back to PIL
    masks_on_image = masks_on_image.permute(1, 2, 0).numpy()
    return Image.fromarray(masks_on_image.astype(np.uint8))


def get_masks_as_image(
    masks,
    height,
    width
) -> PIL.Image:

    assert len(masks.shape) == 3, f"masks shape is {masks.shape}" # Batch x Masks x Patches
    assert masks.shape[0] == 1, f"masks shape is {masks.shape}"

    # Need to resize masks to image (1 x K x P -> 1 x K x H x W)
    masks_as_image = utils.resize_patches_to_image(masks, size=(height, width))
    masks_as_image = utils.soft_masks_to_one_hot(masks_as_image).squeeze(0)

    all_masks = torch.zeros((height, width))
    for mask_id in range(masks_as_image.shape[0]):
        all_masks[masks_as_image[mask_id, :, :] == 1] = mask_id
    return all_masks


def load_model(model_name):
    model = build_dinosaur.build(model_name, pretrained=True)
    model.eval()
    return model


def prep_for_display(masks_as_image):
    image = torch.stack([masks_as_image] * 3, dim=0) * (255 / masks_as_image.max())
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image.astype(np.uint8))
