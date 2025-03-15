import torch.nn.functional as F


def get_probs(logits):
    if are_probs(logits):
        probs = logits
    else:
        probs = F.softmax(logits, dim=-1)
    return probs


def are_probs(logits):
    if (
            logits.min() >= 0
        and
            logits.max() <= 1
        # don't check sums to one for the cases
        # like IN_A where masking drops some probs

        # and
        #     abs(logits.sum(-1)[0][0] - 1) > EPS
    ):
        return True
    return False
