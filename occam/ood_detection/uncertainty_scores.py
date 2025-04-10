# import gdown
# import pickle
import os
import sys
# import einops
# from stuned.utility.utils import (
#     get_project_root_path,
#     load_from_pickle
# )
# import tqdm
import torch
# import PIL
# import shutil
# import open_clip
# import torchvision
# from torchvision.datasets import ImageFolder
# from torchvision.models import resnet50
# import torchvision.transforms as transforms
# from stuned.utility.utils import (
#     get_with_assert,
#     get_project_root_path,
#     parse_list_from_string
# )
# from stuned.utility.logger import (
#     log_or_print,
#     try_to_log_in_csv,
#     try_to_log_in_wandb
# )
# from stuned.local_datasets.imagenet1k import (
#     IMAGENET2012_CLASSES_LIST,
#     IMAGENET2012_CLASSES,
#     get_imagenet_dataloaders
# )
# from stuned.local_datasets.transforms import (
#     # DEFAULT_MEAN_IN,
#     # DEFAULT_STD_IN,
#     make_transforms
# )
# import wandb
# import timm
import torch.nn.functional as F
from stuned.utility.utils import (
    get_project_root_path
)


# local modules
sys.path.insert(
    0,
    get_project_root_path()
)
# import densifier
from occam.robust_classification.utils import (
    get_probs
)
# from densifier.datasets.imagenet_classes import get_in_classes_prompts
# from densifier.utility.utils_for_notebooks import (
#     visualize_images_side_by_side,
#     tensor_for_matplotlib,
#     unnormalize,
#     load_data
# )
sys.path.pop(0)


# taken from <diverse-universe-public> repo
# def are_probs(logits):
#     if (
#             logits.min() >= 0
#         and
#             logits.max() <= 1
#         # don't check sums to one for the cases
#         # like IN_A where masking drops some probs

#         # and
#         #     abs(logits.sum(-1)[0][0] - 1) > EPS
#     ):
#         return True
#     return False


# def get_probs(logits):
#     if are_probs(logits):
#         probs = logits
#     else:
#         probs = F.softmax(logits, dim=-1)
#     return probs


def entropy(probs: torch.Tensor, dim=-1):
    "Calcuate the entropy of a categorical probability distribution."
    log_probs = probs.log()
    ent = (-probs*log_probs).sum(dim=dim)
    return ent


def ens_entropy_per_sample(logits, models_dim=0):
    probs = get_probs(logits)
    av_probs = probs.mean(dim=models_dim)
    ent = entropy(av_probs)
    return ent


def ens_entropy(logits, models_dim=0):
    return ens_entropy_per_sample(logits, models_dim=models_dim).mean()


def pairwise_distances(stacked_vectors, p=2):
    distances = torch.cdist(stacked_vectors, stacked_vectors, p=p)

    # Step 2: Calculate the average pairwise distance
    # We only need the upper triangle of the matrix (excluding the diagonal) to avoid duplicate distances
    # and to exclude the distance of vectors with themselves (which is zero)
    # triu_indices = torch.triu_indices(distances.shape[0], distances.shape[1], offset=1)
    # pairwise_distances = distances[triu_indices[0], triu_indices[1]]

    # Compute the mean of these distances
    average_distance = distances.mean((-1, -2))
    return average_distance


def similarity_between_models_per_sample(logits):

    probs = get_probs(logits)
    probs_batch_first = probs.transpose(0, 1)
    similarity = pairwise_distances(probs_batch_first)
    return similarity


def unreduced_entropy_per_sample(logits):
    return average_entropy_per_sample(logits, models_dim=None).transpose(0, 1)


def average_entropy_per_sample(logits, models_dim=0):
    probs = get_probs(logits)
    # av_probs = probs.mean(dim=1)
    # ent = entropy(av_probs)
    # conf = av_probs.max(dim=-1).values

    av_ent = entropy(probs, dim=-1)
    # average over ensemble dim
    if models_dim is not None:
        av_ent = av_ent.mean(dim=models_dim)

    return av_ent


def average_entropy(logits, models_dim=0):
    return average_entropy_per_sample(logits, models_dim=models_dim).mean()


def mutual_information_per_sample(logits, models_dim=0):
    probs = get_probs(logits)
    av_probs = probs.mean(dim=models_dim)
    ent = entropy(av_probs)
    # conf = av_probs.max(dim=-1).values
    # ent =

    av_ent = average_entropy(logits)

    # average over ensemble dim
    # av_ent = entropy(probs, dim=-1).mean(dim=1)
    mutual_information = ent - av_ent
    return mutual_information


# def a2d_score(logits):
#     return a2d_score_per_sample(logits).mean()


# # input shape: [n_models, batch_size, num_classes]
# def a2d_score_per_sample(logits):
#     probs = get_probs(logits)
#     # m_idx = torch.randint(0, probs.shape[0], (1,)).item()
#     total_score = None

#     # for symmetricity treat each model as p2
#     for m_idx in range(probs.shape[0]):
#         a2d_loss = a2d_loss_impl(
#             probs,
#             m_idx,
#             dbat_loss_type='v1',
#             reduction='none'
#         )
#         inv_a2d_loss = -a2d_loss
#         if total_score is None:
#             total_score = inv_a2d_loss.unsqueeze(0)
#         else:
#             total_score = torch.cat([total_score, inv_a2d_loss.unsqueeze(0)], dim=0)

#     return total_score.mean(0)
#     # return a2d_loss_impl(probs, m_idx, dbat_loss_type='v1', reduction='mean')


def mutual_information(logits):
    return mutual_information_per_sample(logits).mean()


def average_energy_per_sample(logits, models_dim=0):
    return -torch.logsumexp(logits, dim=-1).mean(dim=models_dim)


def average_energy(logits, models_dim=0):
    return average_energy_per_sample(logits, models_dim=models_dim).mean()


def average_max_logit_per_sample(logits, models_dim=0):
    return -(logits.max(dim=-1).values.mean(dim=models_dim))


def average_max_logit(logits, models_dim=0):
    return average_max_logit_per_sample(logits, models_dim=models_dim).mean()


def ens_conf_per_sample(logits, models_dim=0):
    probs = get_probs(logits)
    av_probs = probs.mean(dim=models_dim)
    conf = av_probs.max(dim=-1).values
    return conf


def ens_conf(logits, models_dim=0):
    return ens_conf_per_sample(logits, models_dim=models_dim).mean()


def div_different_preds(logits):

    return div_different_preds_per_sample(logits).mean()


def div_continous_unique(logits):

    return div_continous_unique_per_sample(logits).mean()


def div_continous_unique_per_sample(logits):
    probs = get_probs(logits).clone()

    max_by_model = probs.max(0).values
    sum_over_classes = max_by_model.sum(-1)

    # probs_batch_first = probs.transpose(0, 1)
    # num_models, batch_size, num_classes = probs.shape
    # res = []
    # for sample_probs in probs_batch_first:
    #     score = 0
    #     for i in range(num_models):

    #         max_prob_classes = sample_probs.max(-1)
    #         max_prob_models = max_prob_classes.values.max(-1)
    #         m_star = max_prob_models.indices
    #         k_star = max_prob_classes.indices[m_star]
    #         max_prob = max_prob_models.values

    #         score += max_prob
    #         sample_probs[m_star, k_star] = -1
    #     res.append(score)

    return torch.Tensor(sum_over_classes)


def div_different_preds_per_sample(logits):

    probs = get_probs(logits)
    preds = probs.argmax(-1).t()
    res = []

    for per_sample_preds in preds:
        res.append(len(per_sample_preds.unique()))

    return torch.Tensor(res)
