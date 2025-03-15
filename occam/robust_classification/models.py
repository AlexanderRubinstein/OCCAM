import sys
import open_clip
import clip # pip install git+https://github.com/openai/CLIP.git
import torch

from stuned.utility.utils import (
    get_project_root_path
)


sys.path.insert(
    0,
    get_project_root_path()
)
# from occam.robust_classification.masking import (
#     make_clip_model,
#     ClipWrapper
# )
from occam.datasets.common import (
    get_imagenet_prompts
)
from occam.submodules.alpha_clip_wrapper import (
    ALPHA_CLIP_IMAGE_PREPROCESS_CONFIG,
    make_alpha_clip_model
)
sys.path.pop(0)


def add_openai_clip_model(model_id, category_list, models_dict, name_prefix="clip_openai_"):
    model, preprocess, text_features = make_clip_model(
        model_id=model_id,
        pretrained=None,
        clip_type="open_ai",
        class_prompts=category_list
    )
    wrapped_model = ClipWrapper(model, text_features)
    models_dict[name_prefix + model_id] = (wrapped_model, preprocess)


def add_openclip_model(model_id, category_list, models_dict, pretrained, name_prefix=None):
    if name_prefix is None:
        name_prefix = f"clip_openclip_{pretrained}_"
    model, preprocess, text_features = make_clip_model(
        model_id=model_id,
        pretrained=pretrained,
        clip_type="open_clip",
        class_prompts=category_list
    )
    wrapped_model = ClipWrapper(model, text_features)
    models_dict[name_prefix + model_id] = (wrapped_model, preprocess)


def make_clip_model(model_id, pretrained, clip_type="open_clip", class_prompts=None):

    if class_prompts is None:
        class_prompts = get_imagenet_prompts()
    if clip_type == "open_clip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_id,
            pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_id)

        text_features = get_text_features(
            model,
            tokenizer,
            class_prompts,
            apply_per_prompt=False
        )
    else:
        assert clip_type == "open_ai"
        assert pretrained is None
        model, preprocess = clip.load(model_id, "cuda", jit=False)

        text_features = get_text_features(
            model,
            clip.tokenize,
            class_prompts,
            apply_per_prompt=True
        )


    return model, preprocess, text_features


class ClipWrapper(torch.nn.Module):

    def __init__(self, model, text_features):
        super().__init__()
        self.model = model
        self.text_features = text_features
        # self.tokenizer = tokenizer
        # self.preprocess = preprocess
    def forward(self, x):
        return get_text_probs(
            self.model,
            x,
            self.text_features,
            subset_tensor=None
        )


def get_text_features(model, tokenizer, class_prompts, apply_per_prompt=False):
    if apply_per_prompt:
        # torch.cat(
        #     [clip.tokenize(class_prompt) for class_prompt in class_prompts]
        # )
        text = torch.cat(
            [tokenizer(class_prompt) for class_prompt in class_prompts]
        )
    else:
        text = tokenizer(class_prompts)
    text = text.cuda()
    model.cuda()
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    model.cpu()
    return text_features


def get_text_probs(model, image, text_features, subset_tensor):
    image_features = model.encode_image(image)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T)
    if subset_tensor is not None:
        text_probs = text_probs * subset_tensor
    return text_probs


def add_alpha_clip_model(model_id, category_list, models_dict):
    model = make_alpha_clip_model(
        model_id=model_id,
        category_list=category_list
    )
    models_dict["alpha_clip_" + model_id] = (model, ALPHA_CLIP_IMAGE_PREPROCESS_CONFIG)
