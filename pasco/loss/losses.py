import torch
import torch.nn.functional as F
from torch import nn
from pasco.models.misc import prune_outside_coords
import MinkowskiEngine as ME
from pasco.loss.lovasz import lovasz_softmax_flat
import numpy as np


def CE_ssc_loss(pred_F, target_sparse, class_weight):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """

    criterion = nn.CrossEntropyLoss(
        weight=class_weight.type_as(pred_F),
        ignore_index=0,
        reduction="mean",  # NOTE: ignore empty class
    )

    loss = criterion(pred_F, target_sparse.long())

    return loss


def dice_loss(inputs, targets, is_inputs_logit=True):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if is_inputs_logit:
        inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(0)
    denominator = inputs.sum(0) + targets.sum(0)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)  # [N, num_masks]

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


def compute_sem_compl_loss_kitti360(
    sem_labels, sem_logits_at_scales, min_Cs, max_Cs, class_frequencies
):
    ce_losses = []
    lovasz_losses = []
    for scale in sem_logits_at_scales:
        sem_logits = sem_logits_at_scales[scale]
        targets = sem_labels["1_{}".format(scale)]
        bs = len(targets)

        complt_num_per_class = class_frequencies["1_{}".format(scale)]
        compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
        compl_labelweights = np.power(
            np.amax(compl_labelweights) / compl_labelweights, 1 / 1.5
        )
        compl_labelweights = torch.from_numpy(compl_labelweights).type_as(
            sem_logits[0].F
        )

        for i_subnet in range(bs):  # NOTE: assume n_subnets == bs
            target = targets[i_subnet]
            sem_logit = sem_logits[i_subnet]

            min_C = min_Cs[i_subnet]
            max_C = max_Cs[i_subnet]
            sem_logit = prune_outside_coords(sem_logit, min_C, max_C)

            coords = sem_logit.C.clone()
            nonnegative_coords = coords[:, 1:] - min_C.reshape(1, 3)
            nonnegative_coords = (nonnegative_coords // scale).long()
            target_sparse = target[
                nonnegative_coords[:, 0],
                nonnegative_coords[:, 1],
                nonnegative_coords[:, 2],
            ]

            crit = nn.CrossEntropyLoss(
                weight=compl_labelweights, ignore_index=255, reduction="mean"
            )

            ce_loss = crit(sem_logit.F, target_sparse.long())

            lovasz_loss = lovasz_softmax_flat(
                sem_logit.F, target_sparse, ignores=[255], classes="present"
            )

            ce_losses.append(ce_loss)
            lovasz_losses.append(lovasz_loss)
    ce_losses = torch.stack(ce_losses).mean()
    lovasz_losses = torch.stack(lovasz_losses).mean()
    return ce_losses, lovasz_losses


def compute_sem_compl_loss(
    sem_labels, sem_logits_at_scales, min_Cs, max_Cs, class_frequencies
):
    ce_losses = []
    lovasz_losses = []
    for scale in sem_logits_at_scales:
        sem_logits = sem_logits_at_scales[scale]
        targets = sem_labels["1_{}".format(scale)]
        bs = len(targets)

        complt_num_per_class = class_frequencies["1_{}".format(scale)]
        compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
        compl_labelweights = np.power(
            np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0
        )
        compl_labelweights = torch.from_numpy(compl_labelweights).type_as(
            sem_logits[0].F
        )

        for i_subnet in range(bs):  # NOTE: assume n_subnets == bs
            target = targets[i_subnet]
            sem_logit = sem_logits[i_subnet]

            min_C = min_Cs[i_subnet]
            max_C = max_Cs[i_subnet]
            sem_logit = prune_outside_coords(sem_logit, min_C, max_C)
            if sem_logit.F.shape[0] == 0:

                continue

            coords = sem_logit.C.clone()
            nonnegative_coords = coords[:, 1:] - min_C.reshape(1, 3)
            nonnegative_coords = (nonnegative_coords // scale).long()
            target_sparse = target[
                nonnegative_coords[:, 0],
                nonnegative_coords[:, 1],
                nonnegative_coords[:, 2],
            ]

            crit = nn.CrossEntropyLoss(
                weight=compl_labelweights, ignore_index=255, reduction="mean"
            )

            ce_loss = crit(sem_logit.F, target_sparse.long())

            lovasz_loss = lovasz_softmax_flat(
                sem_logit.F, target_sparse, ignores=[255], classes="present"
            )

            ce_losses.append(ce_loss)
            lovasz_losses.append(lovasz_loss)
    if len(ce_losses) == 0:
        return 0.0, 0.0
    ce_losses = torch.stack(ce_losses).mean()
    lovasz_losses = torch.stack(lovasz_losses).mean()
    return ce_losses, lovasz_losses
