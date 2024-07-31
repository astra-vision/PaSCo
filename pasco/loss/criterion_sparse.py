# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import pdb

import torch
import torch.nn.functional as F
from torch import nn
import MinkowskiEngine as ME
from pasco.models.helper import semantic_inference_v2


from pasco.loss.losses import dice_loss, sigmoid_focal_loss, CE_ssc_loss
from pasco.loss.lovasz import lovasz_softmax_flat


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        class_weights,
        compl_labelweights,
        alpha=0.1,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.alpha = alpha
        self.class_weights = class_weights
        empty_weight = torch.ones(self.num_classes + 1)
        self.register_buffer("empty_weight", empty_weight)
        self.sigmoid = ME.MinkowskiSigmoid()
        self.register_buffer("compl_labelweights", compl_labelweights)

    def loss_labels(self, outputs, targets, indice, class_weight):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "query_logits" in outputs

        src_logits = outputs["query_logits"]  # [100, 21]

        src_idx, target_idx = indice
        target_classes_o = targets["labels"][target_idx]

        target_classes = torch.full(
            src_logits.shape[:1],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        target_classes[src_idx] = target_classes_o.long()

        loss_ce = F.cross_entropy(
            src_logits, target_classes, class_weight, reduction="none"
        )

        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indice, class_weight, unknown_mask_dense):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "voxel_logits" in outputs

        src_idx, tgt_idx = indice

        src_masks = outputs["voxel_logits"]  # [39166, 100]
        tgt_masks = targets["masks"]  # [39166, 24]

        src_mask = src_masks.F[:, src_idx]
        tgt_mask = tgt_masks.F[:, tgt_idx].type_as(src_mask)
        tgt_mask_label = targets["labels"][tgt_idx]
        tgt_weights = class_weight[tgt_mask_label.long()]

        coords = tgt_masks.C.long()
        unknown_mask = unknown_mask_dense[
            coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        ]
        valid_mask = ~unknown_mask
        src_mask = src_mask[valid_mask]
        tgt_mask = tgt_mask[valid_mask]
        loss_mask = sigmoid_focal_loss(src_mask, tgt_mask) * tgt_weights.unsqueeze(0)
        loss_dice = dice_loss(src_mask, tgt_mask) * tgt_weights

        losses = {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }

        return losses

    def loss_completion(
        self,
        out_cl_F,
        out_cl_C,
        target_labels,
        tgt_mask_dense,
        indice,
        scale,
        geo_label,
    ):

        tgt_mask_dense_downscaled = F.max_pool3d(
            tgt_mask_dense.float().unsqueeze(0), kernel_size=scale, stride=scale
        )

        tgt_mask_sparse_downscaled = (
            tgt_mask_dense_downscaled[
                :,
                :,
                out_cl_C[:, 1].long() // scale,
                out_cl_C[:, 2].long() // scale,
                out_cl_C[:, 3].long() // scale,
            ]
            .squeeze()
            .T
        )
        geo_label_sparse = geo_label[
            out_cl_C[:, 1].long() // scale,
            out_cl_C[:, 2].long() // scale,
            out_cl_C[:, 3].long() // scale,
        ]

        valid_mask = tgt_mask_sparse_downscaled.sum(1) > 0
        assert valid_mask.sum() == (geo_label_sparse != 255).sum()

        src_idx, target_idx = indice

        src_mask = out_cl_F[:, src_idx]
        tgt_mask_sparse_downscaled = tgt_mask_sparse_downscaled[:, target_idx]

        src_mask = src_mask[valid_mask, :]
        tgt_mask_sparse_downscaled = tgt_mask_sparse_downscaled[valid_mask, :]

        losses = {
            "completion_focal": sigmoid_focal_loss(
                src_mask, tgt_mask_sparse_downscaled
            ),
            "completion_dice": dice_loss(src_mask, tgt_mask_sparse_downscaled),
        }
        return losses

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def compute_ssc_sparse_loss(
        self, voxel_logits, query_logits, min_C, semantic_label
    ):
        voxel_probs = self.sigmoid(voxel_logits)
        ssc_logit_sparse = semantic_inference_v2(voxel_probs, query_logits)
        ssc_ce_loss = 0
        ssc_lovasz_loss = 0
        if ssc_logit_sparse is not None:
            coords = ssc_logit_sparse.C.clone()

            coords[:, 1:] -= min_C.reshape(1, 3)
            coords = coords.long()
            semantic_label_sparse = semantic_label[
                0, coords[:, 1], coords[:, 2], coords[:, 3]
            ]
            ssc_logit_sparse_F = ssc_logit_sparse.F[semantic_label_sparse != 255]
            semantic_label_sparse = semantic_label_sparse[semantic_label_sparse != 255]

            ssc_ce_loss += CE_ssc_loss(
                ssc_logit_sparse_F, semantic_label_sparse, self.compl_labelweights
            )

            ssc_lovasz_loss += lovasz_softmax_flat(
                ssc_logit_sparse_F,
                semantic_label_sparse,
                ignores=[0],  # ignore empty space
                classes="present",
            )

        return ssc_ce_loss, ssc_lovasz_loss

    @staticmethod
    def JSD(P_logit, Q_logit):
        if len(P_logit) == 0 or len(Q_logit) == 0:
            return 0
        P = F.softmax(P_logit, dim=1) + 1e-8
        Q = F.softmax(Q_logit, dim=1) + 1e-8

        m = 0.5 * (P + Q)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(P_logit, dim=1), m, reduction="mean")
        loss += F.kl_div(F.log_softmax(Q_logit, dim=1), m, reduction="mean")

        return 0.5 * loss

    def compute_query_avg_ssc_loss(
        self, indice, pred_logit, voxel_logits, sem_logits_pruned_1_1, min_C
    ):
        voxel_probs = self.sigmoid(voxel_logits)
        if voxel_probs.F.shape[0] != sem_logits_pruned_1_1.F.shape[0]:
            print(voxel_probs.shape, sem_logits_pruned_1_1.shape)
            return 0
        avg_ssc_query_logit = (voxel_probs.F.T @ sem_logits_pruned_1_1.F) / (
            voxel_probs.F.sum(0).unsqueeze(-1) + 1e-8
        )
        query_class = pred_logit.argmax(-1)
        keep = query_class != self.num_classes
        query_logit = pred_logit[keep, :-1]
        avg_ssc_query_logit = avg_ssc_query_logit[keep]
        return self.JSD(query_logit, avg_ssc_query_logit)

    def compute_losses(
        self,
        sem_logits_pruned_1_1,
        outputs,
        targets,
        semantic_label,
        unknown_mask_dense,
        i_infer,
        indices=[],
        min_C=None,
    ):
        bs = len(targets)
        voxel_logits = outputs["voxel_logits"]  # SparseTensor(bs*N, num_queries)
        query_logits = outputs["query_logits"]  # [bs, num_queries, num_classes]

        loss_label_agg = {
            "loss_ce": 0,
        }
        loss_mask_agg = {
            "loss_mask": 0,
            "loss_dice": 0,
            "ssc_ce_loss": 0,
            "ssc_lovasz_loss": 0,
        }

        for i in range(bs):

            class_weight = self.class_weights[i_infer].to(voxel_logits.device)
            target = targets[i]
            target_labels = target["labels"]  # [num_target_boxes]
            target_masks = target["masks"]  # [num_target_boxes, 256, 256, 32]

            pred_mask_F = voxel_logits.features_at(i)  # [N, num_queries]
            pred_mask_C = voxel_logits.coordinates_at(i)  # [N, 4]
            pred_mask_C = ME.utils.batched_coordinates([pred_mask_C]).to(
                pred_mask_F.device
            )
            voxel_logit = ME.SparseTensor(features=pred_mask_F, coordinates=pred_mask_C)

            if min_C is not None:
                pred_mask_C[:, 1:] -= min_C.reshape(1, 3)

            pred_mask_sparse = ME.SparseTensor(
                features=pred_mask_F, coordinates=pred_mask_C
            )
            pred_logit = query_logits[i]  # [num_queries, num_classes]

            target_masks_sparse_F = target_masks[
                :,
                pred_mask_C[:, 1].long(),
                pred_mask_C[:, 2].long(),
                pred_mask_C[:, 3].long(),
            ].T  # [num_target_boxes, num_queries]

            target_masks_sparse = ME.SparseTensor(
                features=target_masks_sparse_F, coordinates=pred_mask_C
            )

            out_sparse_dict = {
                "query_logits": pred_logit,  # [num_queries, num_classes]
                "voxel_logits": pred_mask_sparse,  # SparseTensor[N, num_queries]
            }
            target_sparse_dict = {
                "labels": target_labels,  # [num_target_boxes]
                "masks": target_masks_sparse,  # SparseTensor[N, num_queries]
            }

            indice = self.matcher(
                out_sparse_dict, target_sparse_dict, class_weight, unknown_mask_dense
            )

            loss_label = self.loss_labels(
                out_sparse_dict, target_sparse_dict, indice, class_weight
            )
            loss_mask = self.loss_masks(
                out_sparse_dict,
                target_sparse_dict,
                indice,
                class_weight,
                unknown_mask_dense,
            )
            loss_mask["loss_mask"] = loss_mask["loss_mask"]
            loss_mask["loss_dice"] = loss_mask["loss_dice"]

            loss_mask_ins = (
                loss_mask["loss_mask"].mean(0) * self.weight_dict["loss_mask"]
            )
            loss_dice_ins = loss_mask["loss_dice"] * self.weight_dict["loss_dice"]
            loss_ce_ins = loss_label["loss_ce"] * self.weight_dict["loss_ce"]

            loss_label_agg["loss_ce"] += loss_ce_ins.mean()

            loss_mask_agg["loss_mask"] += loss_mask_ins.mean()
            loss_mask_agg["loss_dice"] += loss_dice_ins.mean()

            ssc_ce_loss, ssc_lovasz_loss = self.compute_ssc_sparse_loss(
                voxel_logit, pred_logit, min_C, semantic_label
            )
            loss_mask_agg["ssc_ce_loss"] += ssc_ce_loss * self.weight_dict["ssc_ce"]
            loss_mask_agg["ssc_lovasz_loss"] += (
                ssc_lovasz_loss * self.weight_dict["ssc_lovasz"]
            )

        loss_label_agg["loss_ce"] /= bs

        loss_mask_agg["loss_mask"] /= bs
        loss_mask_agg["loss_dice"] /= bs
        loss_mask_agg["ssc_ce_loss"] /= bs
        loss_mask_agg["ssc_lovasz_loss"] /= bs

        return loss_label_agg, loss_mask_agg, indices

    def forward(
        self,
        sem_logits_pruned_1_1,
        outputs,
        targets,
        semantic_label,
        unknown_mask_dense,
        i_infer,
        indices,
        min_C=None,
    ):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        losses = {}
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        loss_label_agg, loss_mask_agg, indices = self.compute_losses(
            sem_logits_pruned_1_1,
            outputs_without_aux,
            targets,
            semantic_label,
            unknown_mask_dense,
            i_infer,
            indices,
            min_C,
        )
        losses.update(loss_label_agg)
        losses.update(loss_mask_agg)

        loss_aux = {}
        if "aux_outputs" in outputs:

            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                loss_label_agg, loss_mask_agg, _ = self.compute_losses(
                    sem_logits_pruned_1_1,
                    aux_outputs,
                    targets,
                    semantic_label,
                    unknown_mask_dense,
                    i_infer,
                    indices,
                    min_C,
                )

                for k in loss_label_agg.keys():
                    loss_aux[k + "_level{}".format(i)] = loss_label_agg[k]
                for k in loss_mask_agg.keys():
                    loss_aux[k + "_level{}".format(i)] = loss_mask_agg[k]

        losses["indices"] = indices

        losses["loss_aux"] = loss_aux

        return losses
