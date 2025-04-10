# Copied from https://raw.githubusercontent.com/qqlu/Entity/main/Entityv2/CropFormer/demo_cropformer/demo_from_dirs.py
from PIL import Image
import copy
import torch
import numpy as np
import cv2
import numpy as np
import torch
from tqdm import tqdm
from typing import List
import sys
import os
from stuned.utility.utils import AttrDict


try:
    from detectron2.data import MetadataCatalog
    from detectron2.engine.defaults import DefaultPredictor
    import detectron2.data.transforms as T
    from detectron2.utils.visualizer import ColorMode
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config

    # local imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    sys.path.insert(
        0, (os.path.dirname(os.path.dirname(__file__)))
    )  # to allow importing from get_segments directly
    from occam.get_segments.utils import add_maskformer2_config

    # import occam.get_segments.cropformer_model
    from occam.get_segments.tardataset import TarDataset
    from occam.get_segments.utils import (
        BatchResizeShortestEdge,
        EntityCrop,
        EntityCropTransform,
    )
    sys.path.pop(0)
    sys.path.pop(0)

    class EntityNetV2(DefaultPredictor):
        def __init__(
            self,
            args,
            instance_mode=ColorMode.IMAGE,
            store_dataloader=True,
        ):
            """
            Args:
                cfg (CfgNode):
                instance_mode (ColorMode):
                parallel (bool): whether to run the model in different processes from visualization.
                    Useful since the visualization logic can be slow.
            """

            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_maskformer2_config(cfg)
            cfg.merge_from_file(args.config_file)
            cfg.merge_from_list(args.opts)
            cfg.freeze()
            super().__init__(cfg)
            self.model = self.model.cuda()
            self.metadata = MetadataCatalog.get(
                cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
            )
            self.instance_mode = instance_mode

            # Get dataset loaded
            self.augs, self.crop_augs = self.generate_img_augs(cfg)
            if store_dataloader:
                dataset = TarDataset(
                    args.input, transforms_help=[self.augs, self.crop_augs]
                )
                self.dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    num_workers=2,
                    shuffle=False,
                    collate_fn=None,
                    pin_memory=True,
                )
            else:
                self.dataloader = None
            self.confidence_threshold = args.confidence_threshold

        def preprocess(self, image):  # TODO(Alex | 13.11.2024): optimize it
            assert image.shape[0] == 1
            # image = image[0].to(torch.float32).cpu().numpy()
            image = image[0].cpu().numpy()
            image = np.asarray(image)
            # Apply transforms to the input image.
            image = image[:, :, ::-1]
            height, width = image.shape[:2]
            aug_input_ori = T.AugInput(copy.deepcopy(image))
            aug_input_ori, _ = T.apply_transform_gens(self.augs, aug_input_ori)
            image_ori = aug_input_ori.image
            image_ori = torch.as_tensor(
                image_ori.astype("float32").transpose(2, 0, 1)
            )
            aug_input_crop = T.AugInput(copy.deepcopy(image))
            transforms_crop = self.crop_augs(aug_input_crop)
            image_crop = aug_input_crop.image
            image_crop = torch.as_tensor(
                image_crop.astype("float32").transpose(0, 3, 1, 2)
            )
            for transform_type in transforms_crop:
                if isinstance(transform_type, EntityCropTransform):
                    crop_axises = transform_type.crop_axises
                    crop_indexes = transform_type.crop_indexes
            return {
                "image": image_ori,
                "height": height,
                "width": width,
                "image_crop": image_crop,
                "crop_region": crop_axises,
                "crop_indexes": crop_indexes,
            }

        def generate_img_augs(self, cfg):
            shortest_side = np.random.choice([cfg.INPUT.MIN_SIZE_TEST])
            augs = [
                T.ResizeShortestEdge(
                    (shortest_side,),
                    cfg.INPUT.MAX_SIZE_TEST,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                ),
            ]

            # Build original image augmentation
            crop_augs = []
            entity_crops = EntityCrop(
                cfg.ENTITY.CROP_AREA_RATIO,
                cfg.ENTITY.CROP_STRIDE_RATIO,
                cfg.ENTITY.CROP_SAMPLE_NUM_TEST,
                False,
            )
            crop_augs.append(entity_crops)

            entity_resize = BatchResizeShortestEdge(
                (shortest_side,),
                cfg.INPUT.MAX_SIZE_TEST,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
            crop_augs.append(entity_resize)

            crop_augs = T.AugmentationList(crop_augs)
            return augs, crop_augs

        def run(self, range=None):
            """
            Args:
                image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                    This is the format used by OpenCV.
            Returns:
                predictions (dict): the output of the model.
                vis_output (VisImage): the visualized image output.
            """
            mega_dict = {}
            with torch.inference_mode():
                for i, (inputs, filename) in enumerate(tqdm(self.dataloader)):
                    if range is not None:
                        if i >= range[1]:
                            break
                        if i < range[0]:
                            continue
                    filename = filename[0]
                    inputs["image"], inputs["image_crop"] = inputs["image"].squeeze(
                        0
                    ).cuda(non_blocking=True), inputs["image_crop"].squeeze(0).cuda(
                        non_blocking=True
                    )
                    predictions = self.model([inputs])[0]
                    pred_masks = predictions["instances"].pred_masks
                    pred_scores = predictions["instances"].scores
                    selected_indexes = pred_scores >= self.confidence_threshold
                    selected_scores = pred_scores[selected_indexes]
                    selected_masks = pred_masks[selected_indexes]
                    _, m_H, m_W = selected_masks.shape
                    mask_id = np.zeros((m_H, m_W), dtype=np.uint8)

                    selected_scores, ranks = torch.sort(selected_scores)
                    ranks = ranks + 1

                    for index in ranks:
                        mask_id[
                            (selected_masks[index - 1] == 1).cpu().numpy()
                        ] = int(index)

                    mega_dict[filename] = {
                        "mask": mask_id,
                        "scores": selected_scores,
                    }
            return mega_dict


    class EntitySegDecoder(torch.nn.Module):
        def __init__(
            self, config_path: str, opts: List[str], confidence_threshold: float
        ):
            super().__init__()

            args = AttrDict(
                {
                    "config_file": config_path,
                    "opts": opts,
                    "confidence_threshold": confidence_threshold,
                }
            )

            # import is needed to register architecture
            # requires compiling MultiScaleDeformableAttention CUDA op with the following commands:
            # cd <REPO_ROOT>/occam/get_segments/modeling/pixel_decoder/ops
            # sh make.sh
            import occam.get_segments.cropformer_model

            self.entity_net = EntityNetV2(args=args, store_dataloader=False)

        def forward(self, image):
            """
            Args:
                image (torch.Tensor; dtype=torch.uint8): an image of shape (B, H, W, C) (in BGR order).
            Returns:
                mask_id (torch.Tensor): a mask of shape (B, H, W) with the predicted mask ids.
            """
            with torch.inference_mode():
                inputs = self.entity_net.preprocess(image)

                inputs["image"], inputs["image_crop"] = inputs["image"].squeeze(
                    0
                ).cuda(non_blocking=True), inputs["image_crop"].squeeze(0).cuda(
                    non_blocking=True
                )

                predictions = self.entity_net.model([inputs])[0]
                pred_masks = predictions["instances"].pred_masks
                pred_scores = predictions["instances"].scores
                selected_indexes = (
                    pred_scores >= self.entity_net.confidence_threshold
                )
                selected_scores = pred_scores[selected_indexes]
                selected_masks = pred_masks[selected_indexes]
                _, m_H, m_W = selected_masks.shape
                mask_id = np.zeros((m_H, m_W), dtype=np.uint8)

                selected_scores, ranks = torch.sort(selected_scores)
                ranks = ranks + 1

                for index in ranks:
                    mask_id[(selected_masks[index - 1] == 1).cpu().numpy()] = int(
                        index
                    )

            return mask_id

except ImportError:
    class EntityNetV2:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "detectron2 is not installed. Please install it using the following command: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            )


