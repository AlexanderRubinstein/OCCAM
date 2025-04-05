import cv2
import numpy as np
import torch
import torchvision
import sys

# import os

from stuned.utility.utils import get_project_root_path


sys.path.insert(0, get_project_root_path())
from occam.datasets.utils import IMAGE_NORMALIZATION_CONST

sys.path.pop(0)


EPS = 1e-6


# taken from Algorithm B here: https://arxiv.org/pdf/2312.07661
def apply_visual_prompts(
    image_array,
    mask,
    visual_prompt_type,
    # visualize=False,
    color=(255, 0, 0),
    thickness=1,
    blur_strength=(15, 15),
    enforce_square_shape=True,
    enforce_channels_last=False,
):
    # print(visual_prompt_type)
    # enforce channels last
    if enforce_channels_last:
        image_last_dim = image_array.shape[-1]
        assert image_last_dim == 3
        assert mask.shape[-1] == image_last_dim

    if len(image_array.shape) == 4:
        assert len(mask.shape) == 4
        image_array = image_array[0]
        mask = mask[0]

    if torch.is_tensor(image_array):
        assert torch.is_tensor(mask)
        device = image_array.device
        image_array = image_array.cpu().numpy()
        mask = mask.cpu().numpy()
        image_array = image_array.transpose(1, 2, 0)
        mask = mask.transpose(1, 2, 0)
        is_torch = True
    else:
        is_torch = False

    assert len(image_array.shape) == 3
    assert len(mask.shape) == 3

    if enforce_square_shape:
        # assert image_array.shape[-1] == image_array.shape[-2]
        assert image_array.shape[-3] == image_array.shape[-2]

    # image_array = image_array.transpose(1, 2, 0)

    # print("mask.shape", mask.shape) # tmp
    # print("image.shape", image_array.shape) # tmp
    # mask = mask.transpose(1, 2, 0).astype(np.uint8)
    mask = mask.astype(np.uint8)

    prompted_image = np.ascontiguousarray(image_array, dtype=np.float32)
    # inv_mask = (1 - mask)[:, :, None] # old
    inv_mask = 1 - mask  # new
    if "blur" in visual_prompt_type:
        # blur the part out side the mask
        # Blur the entire image
        blurred = cv2.GaussianBlur(prompted_image, blur_strength, 0)
        # Get the sharp region using the mask
        # mask = mask[:, :, 0]
        # mask = mask[:, :, 0][..., None]

        # sharp_region = cv2.bitwise_and(
        #     prompted_image,
        #     prompted_image,
        #     mask=np.clip(mask, 0, 255)
        # )
        sharp_region = prompted_image * mask  # simple
        # Get the blurred region using the inverted mask
        blurred_region = blurred * inv_mask
        # Combine the sharp and blurred regions
        prompted_image = cv2.add(sharp_region, blurred_region)
    # if 'gray' in visual_prompt_type:
    #     gray = cv2.cvtColor(
    #         prompted_image, cv2.
    #         COLOR_BGR2GRAY
    #     )
    #     # make gray part 3 channel
    #     gray = np.stack([gray, gray, gray], axis=-1)
    #     # Get the sharp region using the mask
    #     # color_region = cv2.bitwise_and(
    #     #     prompted_image,
    #     #     prompted_image,
    #     #     mask=np.clip(mask, 0, 255)
    #     # )
    #     # # Get the blurred region using the inverted mask
    #     # inv_mask = 1 - mask
    #     # gray_region = (gray * inv_mask)
    #     # # Combine the sharp and blurred regions
    #     # prompted_image = cv2.add(
    #     #     color_region,
    #     #     gray_region
    #     # )

    #     prompted_image = prompted_image * mask + gray * inv_mask

    # if 'black' in visual_prompt_type:
    #     prompted_image = cv2.bitwise_and(
    #         prompted_image,
    #         prompted_image,
    #         mask=np.clip(mask, 0, 255)
    #     )

    if "naive_gray" in visual_prompt_type:
        # print('naive_gray')
        # print("mask", mask.shape)
        # print("inv_mask", inv_mask.shape)
        # print("prompted_image", prompted_image.shape)
        prompted_image = prompted_image * mask + 0.5 * inv_mask
        prompted_image = np.ascontiguousarray(prompted_image, dtype=np.float32)

    if "circle" in visual_prompt_type:
        mask_center, mask_height, mask_width = mask2chw(
            mask, enforce_square_shape=enforce_square_shape
        )
        center_coordinates = (mask_center[1], mask_center[0])
        axes_length = (
            # mask_width // 2,
            # mask_height // 2
            mask_width * 2 // 3,
            mask_height * 2 // 3,
        )

        prompted_image *= IMAGE_NORMALIZATION_CONST
        prompted_image = cv2.ellipse(
            prompted_image,
            center_coordinates,
            axes_length,
            angle=0,
            startAngle=0,
            # endAngle=360,
            endAngle=360.0,
            color=color,
            thickness=thickness,
        )
        prompted_image /= IMAGE_NORMALIZATION_CONST
        # prompted_image = np.clip(prompted_image, 0, 1)

    # if 'rectangle' in visual_prompt_type:
    #     mask_center, mask_height, mask_width = mask2chw(mask)
    #     center_coordinates = (mask_center[1], mask_center[0])
    #     start_point = (
    #         mask_center[1] - mask_width // 2,
    #         mask_center[0] - mask_height // 2
    #     )
    #     end_point = (
    #         mask_center[1] + mask_width // 2,
    #         mask_center[0] + mask_height // 2
    #     )
    #     prompted_image = cv2.rectangle(prompted_image,
    #         start_point,
    #         end_point,
    #         color,
    #         thickness
    #     )

    # if 'contour' in visual_prompt_type:
    #     # Find the contours of the mask
    #     # fill holes for the mask
    #     mask = binary_fill_holes(mask)
    #     contours, hierarchy = cv2.findContours(
    #         mask,
    #         cv2.RETR_TREE,
    #         cv2.CHAIN_APPROX_SIMPLE
    #     )
    #     # Draw the contours on the image
    #     prompted_image = cv2.drawContours(
    #         prompted_image,
    #         contours,
    #         -1,
    #         color,
    #         thickness
    #     )

    if "rectangle_crop_resize" in visual_prompt_type:
        # print('rectangle_crop_resize')
        # print("mask", mask.shape)
        # print("inv_mask", inv_mask.shape)
        # print("prompted_image", prompted_image.shape)

        mask_center, mask_height, mask_width = mask2chw(
            mask, enforce_square_shape=enforce_square_shape
        )
        center_coordinates = (mask_center[1], mask_center[0])
        square_side = max(mask_height, mask_width)
        start_point = (
            mask_center[0] - square_side * 1 // 2,
            mask_center[1] - square_side * 1 // 2,
        )
        # start_point = (
        #     mask_center[0] - mask_height * 1 // 2,
        #     mask_center[1] - mask_width * 1 // 2,
        # )
        # end_point = (
        #     mask_center[1] + mask_width // 2,
        #     mask_center[0] + mask_height // 2
        # )
        # height = start_point[0] - end_point[0]
        # width = start_point[1] - end_point[1]
        # torchvision.transforms.functional.crop(img: Tensor, top: int, left: int, height: int, width: int)\prompted_image
        prompted_image = torch.Tensor(prompted_image.transpose(2, 0, 1))
        resize_transform = torchvision.transforms.Resize(
            prompted_image.shape[-2:],
            # interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT,
        )

        prompted_image = torchvision.transforms.functional.crop(
            prompted_image,
            start_point[0],
            start_point[1],
            # height,
            # width
            # mask_height,
            # mask_width
            square_side,
            square_side,
        )
        # print("prompted_image.shape", prompted_image.shape)
        prompted_image = resize_transform(prompted_image)
        if "naive_gray" in visual_prompt_type:
            prompted_image[prompted_image == 0] = 0.5
            # for i in range(1, prompted_image.shape[1] - 1):
            #     for j in range(1, prompted_image.shape[2] - 1):
            #         if abs(prompted_image[0, i - 1, j] - 0.5) < EPS and abs(prompted_image[0, i + 1, j] - 0.5) < EPS:
            #             prompted_image[:, i, j] = 0.5
            #         if abs(prompted_image[0, i, j - 1] - 0.5) < EPS and abs(prompted_image[0, i, j + 1] - 0.5) < EPS:
            #             prompted_image[:, i, j] = 0.5

        prompted_image = prompted_image.numpy().transpose(1, 2, 0)
        prompted_image = np.ascontiguousarray(prompted_image, dtype=np.float32)
    prompted_image = prompted_image.transpose(2, 0, 1)[None, ...]

    if is_torch:
        return torch.Tensor(prompted_image).to(device)
    else:
        return prompted_image


def mask2chw(mask, enforce_square_shape=True):
    """
    Calculate the center, height, and width of the given mask.

    Parameters:
    mask (numpy.ndarray): Binary mask with values 0 or 1.

    Returns:
    tuple: (center, height, width)
        - center (tuple): Coordinates of the center of the mask (row, col).
        - height (int): Height of the bounding box around the mask.
        - width (int): Width of the bounding box around the mask.
    """
    # Find where the mask is non-zero
    # ignore batch and channel dimensions
    # mask = mask[0, 0, ...]
    assert len(mask.shape) == 3
    if enforce_square_shape:
        assert mask.shape[0] == mask.shape[1]  # cv2 style
    mask = mask[..., 0]
    indices = np.argwhere(mask > 0)
    assert len(indices) > 0

    # Calculate the bounding box of the mask
    top_left = indices.min(axis=0)
    bottom_right = indices.max(axis=0)

    # Calculate the center of the bounding box
    center = ((top_left + bottom_right) // 2).tolist()

    # Calculate height and width of the bounding box
    height = bottom_right[0] - top_left[0] + 1
    width = bottom_right[1] - top_left[1] + 1

    return center, height, width


def is_background(mask, to_extract=True, threshold=2):
    # print(mask.shape)
    if to_extract:
        mask = mask[0, 0]
    # print(mask.shape) # tmp
    assert len(mask.shape) == 2, f"mask.shape: {mask.shape}"
    mid_0 = mask.shape[0] // 2
    mid_1 = mask.shape[1] // 2
    sum = (
        mask[0, 0].item()
        + mask[0, -1].item()
        + mask[-1, 0].item()
        + mask[-1, -1].item()
    )
    sum += (
        mask[0, mid_1].item()
        + mask[mid_0, -1].item()
        + mask[-1, mid_1].item()
        + mask[mid_0, 0].item()
    )
    # print(sum)
    return sum >= threshold


# def is_background_v2(mask, to_extract=True, threshold=2):
#     # print(mask.shape)
#     # if to_extract:
#     #     mask = mask[0, 0]

#     touches_0_all = mask[0, :].sum() > 0
#     touches_last_all = mask[-1, :].sum() > 0
#     touches_all_0 = mask[:, 0].sum() > 0
#     touches_all_last = mask[:, -1].sum() > 0
#     if touches_0_all and touches_1_all and touches_2_all and touches_3_all:
#         return True
#     else:
#         return False

#     # print(mask.shape) # tmp
#     assert len(mask.shape) == 2, f"mask.shape: {mask.shape}"
#     mid_0 = mask.shape[0] // 2
#     mid_1 = mask.shape[1] // 2
#     sum = mask[0, 0].item() + mask[0, -1].item() + mask[-1, 0].item() + mask[-1, -1].item()
#     sum += mask[0, mid_1].item() + mask[mid_0, -1].item() + mask[-1, mid_1].item() + mask[mid_0, 0].item()
#     # print(sum)
#     return sum >= threshold
