import sys
import open_clip
import clip  # pip install git+https://github.com/openai/CLIP.git
import torch

from stuned.utility.utils import get_project_root_path


sys.path.insert(0, get_project_root_path())
# from occam.robust_classification.masking import (
#     make_clip_model,
#     ClipWrapper
# )
from occam.datasets.common import get_imagenet_prompts
from occam.submodules.alpha_clip_wrapper import (
    ALPHA_CLIP_IMAGE_PREPROCESS_CONFIG,
    make_alpha_clip_model,
)

sys.path.pop(0)


class ModelBuilder:
    def __init__(self, build_model):
        self.build_model = build_model

    def build(self):
        return self.build_model()


def add_clip_model(
    model_id,
    category_list,
    models_dict,
    # name_prefix="clip_openai_",
    clip_type="open_ai",
    pretrained=None,
):
    if clip_type in ["open_ai", "open_clip"]:
        if clip_type == "open_ai":
            assert pretrained is None
            name_prefix = "clip_openai_"
        else:
            assert clip_type == "open_clip"
            name_prefix = f"clip_openclip_{pretrained}_"

        def model_builder_clip():
            model, preprocess, text_features = make_clip_model(
                model_id=model_id,
                pretrained=pretrained,
                clip_type=clip_type,
                class_prompts=category_list,
            )
            wrapped_model = ClipWrapper(model, text_features)
            return (wrapped_model, preprocess)

        model_builder = model_builder_clip
    else:
        assert clip_type == "alpha_clip"
        assert pretrained is None
        name_prefix = "alpha_clip_"

        def model_builder_alpha_clip():
            wrapped_model = make_alpha_clip_model(
                model_id=model_id, category_list=category_list
            )

            # wrapped_model = model
            preprocess = ALPHA_CLIP_IMAGE_PREPROCESS_CONFIG
            return (wrapped_model, preprocess)

        model_builder = model_builder_alpha_clip
        # models_dict["alpha_clip_" + model_id] = (
        #     model,
        #     ALPHA_CLIP_IMAGE_PREPROCESS_CONFIG,
        # )
    # models_dict[name_prefix + model_id] = (wrapped_model, preprocess)
    models_dict[name_prefix + model_id] = ModelBuilder(model_builder)


def add_openai_clip_model(
    model_id, category_list, models_dict, name_prefix="clip_openai_"
):
    add_clip_model(
        model_id=model_id,
        category_list=category_list,
        models_dict=models_dict,
        clip_type="open_ai",
        pretrained=None,
    )
    # model, preprocess, text_features = make_clip_model(
    #     model_id=model_id,
    #     pretrained=None,
    #     clip_type="open_ai",
    #     class_prompts=category_list,
    # )
    # wrapped_model = ClipWrapper(model, text_features)
    # models_dict[name_prefix + model_id] = (wrapped_model, preprocess)


def add_openclip_model(
    model_id, category_list, models_dict, pretrained, name_prefix=None
):
    add_clip_model(
        model_id=model_id,
        category_list=category_list,
        models_dict=models_dict,
        clip_type="open_clip",
        pretrained=pretrained,
    )
    # if name_prefix is None:
    #     name_prefix = f"clip_openclip_{pretrained}_"
    # model, preprocess, text_features = make_clip_model(
    #     model_id=model_id,
    #     pretrained=pretrained,
    #     clip_type="open_clip",
    #     class_prompts=category_list,
    # )
    # wrapped_model = ClipWrapper(model, text_features)
    # models_dict[name_prefix + model_id] = (wrapped_model, preprocess)


def add_alpha_clip_model(model_id, category_list, models_dict):
    add_clip_model(
        model_id=model_id,
        category_list=category_list,
        models_dict=models_dict,
        clip_type="alpha_clip",
        pretrained=None,
    )
    # model = make_alpha_clip_model(
    #     model_id=model_id, category_list=category_list
    # )
    # models_dict["alpha_clip_" + model_id] = (
    #     model,
    #     ALPHA_CLIP_IMAGE_PREPROCESS_CONFIG,
    # )


def make_clip_model(
    model_id, pretrained, clip_type="open_clip", class_prompts=None
):
    if class_prompts is None:
        class_prompts = get_imagenet_prompts()
    if clip_type == "open_clip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_id)

        text_features = get_text_features(
            model, tokenizer, class_prompts, apply_per_prompt=False
        )
    else:
        assert clip_type == "open_ai"
        assert pretrained is None
        model, preprocess = clip.load(model_id, "cuda", jit=False)

        text_features = get_text_features(
            model, clip.tokenize, class_prompts, apply_per_prompt=True
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
            self.model, x, self.text_features, subset_tensor=None
        )


def clip_models_with_same_preprocess(category_list):
    models_dict = {}
    # category_list = get_in_classes_prompts()
    # add_openai_clip_model('ViT-L/14', category_list, models_dict)
    # add_openclip_model(
    #     model_id='ViT-L-16-SigLIP-256',
    #     category_list=category_list,
    #     models_dict=models_dict,
    #     pretrained='webli'
    # )
    # add_openclip_model(
    #     model_id='ViT-L-14',
    #     category_list=category_list,
    #     models_dict=models_dict,
    #     pretrained='laion400m_e32'
    # )
    add_openclip_model(
        model_id="ViT-L-14",
        category_list=category_list,
        models_dict=models_dict,
        pretrained="datacomp_xl_s13b_b90k",
    )
    # add_openclip_model(
    #     model_id='ViT-L-14',
    #     category_list=category_list,
    #     models_dict=models_dict,
    #     pretrained='laion2b_s32b_b82k'
    # ) # has normalize 0.5, 0.5, 0.5
    add_openclip_model(
        model_id="ViT-L-14-quickgelu",
        category_list=category_list,
        models_dict=models_dict,
        pretrained="dfn2b",
    )
    add_openclip_model(
        model_id="ViT-L-14",
        category_list=category_list,
        models_dict=models_dict,
        pretrained="openai",
    )
    add_openclip_model(
        model_id="ViT-L-14",
        category_list=category_list,
        models_dict=models_dict,
        pretrained="laion400m_e31",
    )
    # add_openclip_model(
    #     model_id='ViT-L-14',
    #     category_list=category_list,
    #     models_dict=models_dict,
    #     pretrained='laion400m_e31'
    # )
    add_openclip_model(
        model_id="ViT-L-14",
        category_list=category_list,
        models_dict=models_dict,
        pretrained="laion400m_e32",
    )

    # final model name: clip_openclip_<pretrained>_ + <model_id>
    # clip_openclip_datacomp_xl_s13b_b90k_ViT-L-14
    # clip_openclip_dfn2b_ViT-L-14-quickgelu
    # clip_openclip_openai_ViT-L-14
    # clip_openclip_laion400m_e31_ViT-L-14
    # clip_openclip_laion400m_e32_ViT-L-14

    models_list = []
    transform = None
    for model_id, builder in models_dict.items():
        assert isinstance(builder, ModelBuilder)

        model, preprocess = builder.build()
        # if isinstance(builder, ModelBuilder):
        #     model, preprocess = builder.build()
        # else:
        #     model, preprocess = builder

        if transform is None:
            transform = preprocess
        else:
            # or at least normalization and cropping the same?
            assert str(transform) == str(
                preprocess
            ), "transforms must be the same"
        models_list.append(model)
    assert transform is not None
    return models_list, transform


class ClipEnsemble(torch.nn.Module):
    def __init__(self, category_list):
        super().__init__()
        models_list, preprocess = clip_models_with_same_preprocess(
            category_list
        )
        self.preprocess = preprocess
        self.models = torch.nn.ModuleList(models_list)

    def forward(self, x):
        outputs_list = [model(x) for model in self.models]
        outputs_tensor = torch.stack(outputs_list, dim=0)
        outputs_tensor = outputs_tensor.transpose(
            0, 1
        )  # (batch_size, num_models, ...)
        return outputs_tensor


def make_clip_ensemble(category_list):
    clip_ensemble = ClipEnsemble(category_list)
    return clip_ensemble


def get_clip_ensemble_builder(category_list):
    def clip_ensemble_builder():
        clip_ensemble = make_clip_ensemble(category_list)
        return clip_ensemble, clip_ensemble.preprocess

    return clip_ensemble_builder


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
    text_probs = 100.0 * image_features @ text_features.T
    if subset_tensor is not None:
        text_probs = text_probs * subset_tensor
    return text_probs


# def add_alpha_clip_model(model_id, category_list, models_dict):
#     model = make_alpha_clip_model(
#         model_id=model_id, category_list=category_list
#     )
#     models_dict["alpha_clip_" + model_id] = (
#         model,
#         ALPHA_CLIP_IMAGE_PREPROCESS_CONFIG,
#     )
