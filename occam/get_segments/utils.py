from detectron2.config import CfgNode as CN
import sys
from PIL import Image
import random
from fvcore.transforms.transform import NoOpTransform, Transform
from detectron2.data.transforms.augmentation import Augmentation
import numpy as np
from PIL import Image
import copy

def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    ## For Entity
    cfg.ENTITY = CN()
    cfg.ENTITY.ENABLE = False
    cfg.ENTITY.CROP_AREA_RATIO = 0.7
    cfg.ENTITY.CROP_STRIDE_RATIO = 0.6
    cfg.ENTITY.CROP_SAMPLE_NUM_TRAIN = 1
    cfg.ENTITY.CROP_SAMPLE_NUM_TEST = 4

    ## fuse frame embeddings to batch embedding
    cfg.ENTITY.FUSE_NUM_LAYERS = 1
    cfg.ENTITY.FUSE_ENC_HIDDIEN_DIM = 256
    cfg.ENTITY.FUSE_ENC_NHEADS = 8
    cfg.ENTITY.FUSE_ENC_PRE_NORM = False
    cfg.ENTITY.FUSE_ENC_DIM_FEEDFORWARD = 2048
    cfg.ENTITY.FUSE_ENC_LAST_LAYERS = 1
    cfg.ENTITY.FUSE_DEC_NUM_LAYERS = 3

    ## Hornet backbone
    cfg.MODEL.HORNET = CN()
    cfg.MODEL.HORNET.DEPTHS = [2, 3, 18, 2]
    cfg.MODEL.HORNET.BASE_DIM = 192
    cfg.MODEL.HORNET.GCONV = ['partial(gnconv, order=2, s=1/3)', 'partial(gnconv, order=3, s=1/3)', 'partial(gnconv, order=4, s=1/3, h=24, w=13, gflayer=GlobalLocalFilter)', 'partial(gnconv, order=5, s=1/3, h=12, w=7, gflayer=GlobalLocalFilter)']
    cfg.MODEL.HORNET.DROP_PATH_RATE=0.6
    cfg.MODEL.HORNET.OUT_FEATURES = ["res2", "res3", "res4", "res5"]



class BatchResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, imgs, interp=None):
        dim_num = len(imgs.shape)
        assert dim_num == 4
        interp_method = interp if interp is not None else self.interp
        resized_imgs = []
        for img in imgs:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
            resized_imgs.append(ret)
        resized_imgs = np.stack(resized_imgs)
        return resized_imgs

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords
    
    def apply_box(self, boxes):
        boxes = boxes[0]
        new_boxes = super(BatchResizeTransform, self).apply_box(boxes[:,:4])
        boxes[...,:4] = new_boxes
        return boxes[None]

    def apply_segmentation(self, segmentation):
        if len(segmentation.shape)==3:
            segmentation = segmentation[..., None]
            segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
            segmentation = segmentation[..., 0]
        else:
            segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

class EntityCropTransform(Transform):
    """
    Consectively crop the images
    """
    def __init__(self, crop_axises, crop_indexes):
        super().__init__()
        self._set_attributes(locals())
    
    def apply_image(self, img):
        """
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255]
        returns:
            ndarray: cropped images
        """
        dim_num = len(img.shape)
        imgs = []
        
        for crop_axis in self.crop_axises:
            x0, y0, x1, y1 = crop_axis
            if dim_num <= 3:
                crop_img = img[y0:y1, x0:x1]
            else:
                crop_img = img[..., y0:y1, x0:x1, :]
            imgs.append(crop_img)

        if dim_num <= 3:
            imgs = np.stack(imgs, axis=0)
        else:
            imgs = np.concatenate(imgs, axis=0)
        return imgs
    
    def apply_coords(self, coords: np.ndarray, x0, y0):
        coords[:, 0] -= x0
        coords[:, 1] -= y0
        return coords
    
    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        box: Nx4, [x0, y0, x1, y1]
        """
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        split_boxes = []
        crop_ws, crop_hs = [], []
        for crop_axis in self.crop_axises:
            startw, starth, endw, endh = crop_axis
            coords_new = self.apply_coords(copy.deepcopy(coords), startw, starth).reshape((-1, 4, 2))
            minxy = coords_new.min(axis=1)
            maxxy = coords_new.max(axis=1)
            trans_boxes = np.concatenate((minxy, maxxy), axis=1)
            
            crop_ws.append(endw-startw)
            crop_hs.append(endh-starth)
            split_boxes.append(trans_boxes)
        split_boxes = np.stack(split_boxes, axis=1)
        ### clip to the image boundary
        ## assert each crop size is equal
        for crop_index, (crop_w, crop_h) in enumerate(zip(crop_ws, crop_hs)):
            assert crop_w == crop_ws[0], "crop width is not equal, crop_{}: {}, crop_0: {}".format(crop_index, crop_w, crop_ws[0])
            assert crop_h == crop_hs[0], "crop height is not equal, crop_{}: {}, crop_0: {}".format(crop_index, crop_h, crop_hs[0])
        crop_w = crop_ws[0]
        crop_h = crop_hs[0]
        # pdb.set_trace()
        split_boxes[...,0::2] = np.clip(split_boxes[...,0::2], 0, crop_w)
        split_boxes[...,1::2] = np.clip(split_boxes[...,1::2], 0, crop_h)
        valid_inds = (split_boxes[...,2]>split_boxes[...,0]) & (split_boxes[...,3]>split_boxes[...,1])
        split_infos = np.concatenate((split_boxes, valid_inds[...,None]), axis=-1)
        return split_infos

class BatchResizeShortestEdge(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())

    def get_transform(self, image):
        dim_num = len(image.shape)
        assert dim_num == 4, "the tensor should be in [B, H, W, C]"
        h, w = image.shape[1:3]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return BatchResizeTransform(h, w, newh, neww, self.interp)

class EntityCrop(Augmentation):
    def __init__(self, crop_ratio, stride_ratio, sample_num, is_train):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        crop_axises, crop_indexes = self.get_crop_axises((h, w))
        transform = EntityCropTransform(crop_axises, crop_indexes)
        return transform
    
    def get_crop_axises(self, image_size):
        h, w = image_size
        crop_w = int(self.crop_ratio*w)
        crop_h = int(self.crop_ratio*h)
        # if self.is_train:
        stride_w = int(self.stride_ratio*w)
        stride_h = int(self.stride_ratio*h)
        # pdb.set_trace()

        crop_axises  = []
        for starth in range(0, h, stride_h):
            for startw in range(0, w, stride_w):
                endh = min(starth+crop_h, h)
                endw = min(startw+crop_w, w)
                starth = int(endh-crop_h)
                startw = int(endw-crop_w)
                crop_axises.append([startw, starth, endw, endh])
        if self.is_train:
            crop_indexes = random.sample([i for i in range(len(crop_axises))], self.sample_num)
            crop_axises = [crop_axises[i] for i in crop_indexes]
        else:
            crop_indexes = [i for i in range(self.sample_num)]
        # left_upper   = [0, 0, crop_w, crop_h]
        # right_upper  = [w-crop_w, 0, w, crop_h]
        # left_bottom  = [0, h-crop_h, crop_w, h]
        # right_bottom = [w-crop_w, h-crop_h, w, h]
        
        # crop_axises = [left_upper, right_upper, left_bottom, right_bottom]
        # crop_indexes = [0,1,2,3]
        assert len(crop_axises)==len(crop_indexes)
        return crop_axises, crop_indexes


