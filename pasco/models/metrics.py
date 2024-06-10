"""
Part of the code is taken from https://github.com/waterljwant/SSC/blob/master/sscMetrics.py
"""

import numpy as np
import torch
from torchmetrics.functional.classification import binary_calibration_error
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from typing import List
import torch.nn.functional as F
from typing import Literal, Optional

import time
from numpy.typing import ArrayLike


def stable_cumsum(arr: ArrayLike, rtol: float = 1e-05, atol: float = 1e-08):
    """
    From https://github.com/hendrycks/anomaly-seg
    Uses high precision for cumsum and checks that the final value matches
    the sum.
    Args:
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):  # coverage: ignore
        raise RuntimeError(
            "cumsum was found to be unstable: "
            "its last element does not correspond to sum"
        )
    return out


def get_iou(iou_sum, cnt_class):
    _C = iou_sum.shape[0]  # 12
    iou = np.zeros(_C, dtype=float)  # iou for each class
    for idx in range(_C):
        iou[idx] = iou_sum[idx] / cnt_class[idx] if cnt_class[idx] else 0

    mean_iou = np.sum(iou[1:]) / np.count_nonzero(cnt_class[1:])
    return iou, mean_iou


def get_accuracy(predict, target, weight=None):  # 0.05s
    _bs = predict.shape[0]  # batch size
    _C = predict.shape[1]  # _C = 12
    target = np.int32(target)
    target = target.reshape(_bs, -1)  # (_bs, 60*36*60) 129600
    predict = predict.reshape(_bs, _C, -1)  # (_bs, _C, 60*36*60)
    predict = np.argmax(
        predict, axis=1
    )  # one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.

    correct = predict == target  # (_bs, 129600)
    if weight:  # 0.04s, add class weights
        weight_k = np.ones(target.shape)
        for i in range(_bs):
            for n in range(target.shape[1]):
                idx = 0 if target[i, n] == 255 else target[i, n]
                weight_k[i, n] = weight[idx]
        correct = correct * weight_k
    acc = correct.sum() / correct.size
    return acc


class UncertaintyMetrics:
    def __init__(self):
        self.reset()

    def add_batch_variation(self, variation):
        for k in self.variation:
            self.variation[k] += variation[k]
        self.variation_count += 1

    def compute_ece_dense(self, prob_mask, gt_panoptic_seg_query_id, query_id_map):
        preds = prob_mask.F
        coords = prob_mask.C.long().cpu().numpy()
        target = gt_panoptic_seg_query_id[coords[:, 1], coords[:, 2], coords[:, 3]]
        target = torch.from_numpy(target).long().to(preds.device)
        mask_ids = query_id_map[coords[:, 1], coords[:, 2], coords[:, 3]]

        mask_ids = torch.from_numpy(mask_ids).long().to(preds.device)
        known_mask = (target != 255) & (
            mask_ids != 255
        )  # NOTE: in our current prediction, the output doesn't contain the map to the empty voxels
        mask_ids = mask_ids[known_mask]
        preds = preds[known_mask]
        target = target[known_mask]
        probs = preds.squeeze()
        import pdb

        pdb.set_trace()
        correct_voxels = target == mask_ids
        np.save("probs.npy", coords[correct_voxels])

        return binary_calibration_error(probs, target == mask_ids, n_bins=15, norm="l1")

    def compute_ece_empty(self, occ_logits, occ_targets):
        ce_error = 0
        for occ_logit, occ_target in zip(occ_logits, occ_targets):
            occ_logit = occ_logit.F.squeeze()
            occ_target = occ_target.squeeze()
            known_mask = occ_target != 255
            occ_logit = occ_logit[known_mask]
            occ_target = occ_target[known_mask]
            occ_prob = torch.sigmoid(occ_logit)
            ce_error += binary_calibration_error(
                1 - occ_prob, 1 - occ_target, n_bins=15, norm="l1"
            ).item()
        self.empty_ece += ce_error / len(occ_targets)
        self.empty_ece_count += 1

    def compute_ece_panop(
        self,
        pred_panoptic_seg,
        pred_segments_info,
        vox_confidence_denses,
        vox_all_mask_probs_dense,
        pred_gt_matched_segms,
        gt_panoptic_seg,
        gt_segments_info,
        n_classes,
    ):
        pred2gt = {t[1]: t[0] for t in pred_gt_matched_segms}
        gt2pred = {t[0]: t[1] for t in pred_gt_matched_segms}
        pred_segments_dict = {t["id"]: t for t in pred_segments_info}
        gt_segments_dict = {t["id"]: t for t in gt_segments_info}

        nll_mask_labels = []
        idx = 0

        for segment_info in pred_segments_info:
            prob = segment_info["all_class_probs"]

            self.ins_confs.append(segment_info["confidence"])
            pred_id = segment_info["id"]
            if pred_id not in pred2gt:
                is_correct = False
                gt_class = n_classes
            else:
                gt_id = pred2gt[pred_id]
                gt_segment_info = gt_segments_dict[gt_id]
                is_correct = (
                    gt_segment_info["category_id"] == segment_info["category_id"]
                )
                gt_class = gt_segment_info["category_id"]

            self.ins_all_class_probs.append(prob)
            self.ins_sem_labels.append(gt_class)
            self.ins_correct.append(is_correct)

        gt_panoptic_seg = torch.from_numpy(gt_panoptic_seg).to(
            vox_confidence_denses.device
        )
        pred_panoptic_seg_mapped = torch.zeros_like(gt_panoptic_seg)

        for gt_id, pred_id in pred_gt_matched_segms:
            pred_panoptic_seg_mapped[pred_panoptic_seg == pred_id] = gt_id

        nonempty_mask = (gt_panoptic_seg != 0) & (vox_confidence_denses != 0)
        gt_panoptic_seg = gt_panoptic_seg[nonempty_mask]
        pred_panoptic_seg_mapped = pred_panoptic_seg_mapped[nonempty_mask]
        vox_confidence_denses = vox_confidence_denses[nonempty_mask]

        correct_voxels = pred_panoptic_seg_mapped == gt_panoptic_seg

        mask_ece, mask_auprc, mask_auroc, mask_fpr95 = (
            self.compute_all_uncertainty_metrics(vox_confidence_denses, correct_voxels)
        )
        self.mask_ece += mask_ece if isinstance(mask_ece, int) else mask_ece.item()
        self.mask_auprc += (
            mask_auprc if isinstance(mask_auprc, int) else mask_auprc.item()
        )
        self.mask_auroc += (
            mask_auroc if isinstance(mask_auroc, int) else mask_auroc.item()
        )
        self.mask_fpr95 += (
            mask_fpr95 if isinstance(mask_fpr95, int) else mask_fpr95.item()
        )
        self.count += 1

    @staticmethod
    def compute_all_uncertainty_metrics(confidences, labels):
        if len(labels) == 0:
            return 0, 0, 0, 0

        ece = binary_calibration_error(confidences, labels)

        fpr95 = 0
        auprc = 0
        auroc = 0
        return ece, auprc, auroc, fpr95

    def get_stats(self):
        if self.count != 0:
            mask_ece = self.mask_ece / self.count
            mask_nll = self.mask_nll / self.count
            mask_auprc = self.mask_auprc / self.count
            mask_auroc = self.mask_auroc / self.count
            mask_fpr95 = self.mask_fpr95 / self.count
        else:
            mask_ece = 0
            mask_nll = 0
            mask_auprc = 0
            mask_auroc = 0
            mask_fpr95 = 0

        ins_confs = torch.tensor(self.ins_confs).cuda()
        ins_correct = torch.tensor(self.ins_correct).cuda()

        if len(self.ins_all_class_probs) > 0:
            ins_all_class_probs = torch.stack(self.ins_all_class_probs, dim=0)
            ins_sem_labels = torch.tensor(self.ins_sem_labels).to(
                ins_all_class_probs.device
            )

            nll = F.nll_loss(
                torch.log(ins_all_class_probs + 1e-8), ins_sem_labels, reduction="mean"
            )
            nll = nll.item()
        else:
            nll = 0.0

        if ins_confs.numel() != 0:
            ins_ece, ins_auprc, ins_auroc, ins_fpr95 = (
                self.compute_all_uncertainty_metrics(ins_confs, ins_correct)
            )
        else:
            ins_ece = 0
            ins_auprc = 0
            ins_auroc = 0
            ins_fpr95 = 0
        ret = {
            "mask_ece": mask_ece,
            "mask_nll": mask_nll,
            "mask_auprc": mask_auprc,
            "mask_auroc": mask_auroc,
            "mask_fpr95": mask_fpr95,
            "ins_ece": ins_ece,
            "ins_nll": nll,
            "ins_brier": 0.0,
            "ins_auprc": ins_auprc,
            "ins_auroc": ins_auroc,
            "ins_fpr95": ins_fpr95,
            "count": len(ins_confs),
        }
        return ret

    def reset(self):
        self.ins_confs = []
        self.ins_correct = []
        self.ins_all_class_probs = []
        self.ins_sem_labels = []

        self.mask_ece = 0.0
        self.mask_nll = 0.0
        self.mask_auprc = 0.0
        self.mask_auroc = 0.0
        self.mask_fpr95 = 0.0

        self.count = 0.0
        self.empty_ece_count = 0.0
        self.variation = {
            "voxel_disagree": 0.0,
            "query_disagree": 0.0,
            "completion_disagree": 0.0,
            "voxel_kl": 0.0,
            "completion_kl": 0.0,
            "query_kl": 0.0,
        }
        self.variation_count = 0.0


class BrierScore(Metric):
    r"""The Brier Score Metric.

    Args:
        reduction (str, optional): Determines how to reduce over the
            :math:`B`/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

        kwargs: Additional keyword arguments, see `Advanced metric settings
            <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

    Inputs:
        - :attr:`probs`: :math:`(B, C)` or :math:`(B, N, C)`
        - :attr:`target`: :math:`(B)` or :math:`(B, C)`

        where :math:`B` is the batch size, :math:`C` is the number of classes
        and :math:`N` is the number of estimators.

    Note:
        If :attr:`probs` is a 3D tensor, then the metric computes the mean of
        the Brier score over the estimators ie. :math:`t = \frac{1}{N}
        \sum_{i=0}^{N-1} BrierScore(probs[:,i,:], target)`.

    Warning:
        Make sure that the probabilities in :attr:`probs` are normalized to sum
        to one.

    Raises:
        ValueError:
            If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
            ``'none'`` or ``None``.
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        reduction: Literal["mean", "sum", "none", None] = "mean",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )

        self.num_classes = num_classes
        self.reduction = reduction
        self.num_estimators = 1

        if self.reduction in ["mean", "sum"]:
            self.add_state("values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the current Brier score with a new tensor of probabilities.

        Args:
            probs (torch.Tensor): A probability tensor of shape
                (batch, num_estimators, num_classes) or
                (batch, num_classes)
        """
        if target is None:
            target = torch.zeros(len(target, self.num_classes)).to(probs.device)
        elif target.ndim == 1:
            target = F.one_hot(target, self.num_classes)

        if self.num_classes == 1:
            probs = probs.unsqueeze(-1)

        if probs.ndim == 2:
            batch_size = probs.size(0)

        else:
            raise ValueError(
                f"Expected `probs` to be of shape (batch, num_classes) or "
                f"but got {probs.shape}"
            )

        brier_score = F.mse_loss(probs, target, reduction="none").sum(dim=-1)

        if self.reduction is None or self.reduction == "none":
            self.values.append(brier_score)
        else:
            self.values += brier_score.sum()
            self.total += batch_size

    def compute(self) -> torch.Tensor:
        """
        Compute the final Brier score based on inputs passed to ``update``.

        Returns:
            torch.Tensor: The final value(s) for the Brier score
        """
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum(dim=-1) / self.num_estimators
        elif self.reduction == "mean":
            return values.sum(dim=-1) / self.total / self.num_estimators
        else:  # reduction is None
            return values


class FPR95:
    """Class which computes the False Positive Rate at 95% Recall."""

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    conf: List[Tensor]
    targets: List[Tensor]

    def __init__(self, pos_label: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.pos_label = pos_label

    def compute(self, conf, targets) -> Tensor:
        r"""From https://github.com/hendrycks/anomaly-seg
        Compute the actual False Positive Rate at 95% Recall.
        Returns:
            Tensor: The value of the FPR95.
        """
        # import pdb;pdb.set_trace()
        # conf = dim_zero_cat(self.conf).cpu().numpy()
        # targets = dim_zero_cat(self.targets).cpu().numpy()
        conf = conf.cpu().numpy()
        targets = targets.cpu().numpy()

        # out_labels is an array of 0s and 1s - 0 if IOD 1 if OOD
        out_labels = targets == self.pos_label

        in_scores = conf[np.logical_not(out_labels)]
        out_scores = conf[out_labels]

        # pos = OOD
        neg = np.array(in_scores[:]).reshape((-1, 1))
        pos = np.array(out_scores[:]).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[: len(pos)] += 1

        # make labels a boolean vector, True if OOD
        labels = labels == self.pos_label

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(examples, kind="mergesort")[::-1]
        examples = examples[desc_score_indices]
        labels = labels[desc_score_indices]

        # examples typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(examples))[0]
        threshold_idxs = np.r_[distinct_value_indices, labels.shape[0] - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(labels)[threshold_idxs]
        fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

        thresholds = examples[threshold_idxs]

        recall = tps / tps[-1]

        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)  # [last_ind::-1]
        recall, fps, tps, thresholds = (
            np.r_[recall[sl], 1],
            np.r_[fps[sl], 0],
            np.r_[tps[sl], 0],
            thresholds[sl],
        )

        cutoff = np.argmin(np.abs(recall - 0.95))

        return fps[cutoff] / (np.sum(np.logical_not(labels)))


class SSCMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def hist_info(self, n_cl, pred, gt):
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl**2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    @staticmethod
    def compute_score(hist, correct, labeled):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        mean_IU_no_back = np.nanmean(iu[1:])
        freq = hist.sum(1) / hist.sum()
        freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
        mean_pixel_acc = correct / labeled if labeled != 0 else 0

        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    def add_batch_ece(self, ssc_confidence, ssc_pred, sem_prob, target, inference_time):
        with torch.no_grad():
            # fpr95_metric = FPR95(pos_label=0)

            ssc_confidence = ssc_confidence.reshape(-1)
            ssc_pred = ssc_pred.reshape(-1)
            target = target.reshape(-1)
            sem_prob = sem_prob.reshape(sem_prob.shape[0], -1).T
            keep = target != 255
            ssc_confidence = ssc_confidence[keep]
            target = target[keep]
            ssc_pred = ssc_pred[keep]
            sem_prob = sem_prob[keep]
            accuracies = target == ssc_pred

            empty_mask = ssc_pred == 0

            empty_ece = binary_calibration_error(
                ssc_confidence[empty_mask], accuracies[empty_mask]
            )
            nonempty_ece = binary_calibration_error(
                ssc_confidence[~empty_mask], accuracies[~empty_mask]
            )

            empty_nll = F.nll_loss(
                torch.log(sem_prob[empty_mask] + 1e-12),
                target[empty_mask],
                reduction="sum",
            )
            nonempty_nll = F.nll_loss(
                torch.log(sem_prob[~empty_mask] + 1e-12),
                target[~empty_mask],
                reduction="sum",
            )
            self.empty_nll += empty_nll.item()
            self.nonempty_nll += nonempty_nll.item()
            self.n_empty_voxels += empty_mask.sum().item()
            self.n_nonempty_voxels += (~empty_mask).sum().item()

            self.empty_ece += empty_ece.item()
            self.nonempty_ece += nonempty_ece.item()
            self.ece_count += 1

            self.inference_time += inference_time

    def add_batch(self, y_pred, y_true, nonempty=None, nonsurface=None):

        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)

        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )

        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

    def get_stats(self):
        if self.completion_tp != 0:
            precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            iou = self.completion_tp / (
                self.completion_tp + self.completion_fp + self.completion_fn
            )
        else:
            precision, recall, iou = 0, 0, 0
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        empty_ece = self.empty_ece / self.ece_count if self.ece_count != 0 else 0
        nonempty_ece = self.nonempty_ece / self.ece_count if self.ece_count != 0 else 0
        empty_nll = (
            self.empty_nll / self.n_empty_voxels if self.n_empty_voxels != 0 else 0
        )
        nonempty_nll = (
            self.nonempty_nll / self.n_nonempty_voxels
            if self.n_nonempty_voxels != 0
            else 0
        )

        inference_time = (
            self.inference_time / self.ece_count if self.ece_count != 0 else 0
        )
        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": np.mean(iou_ssc[1:]),
            "empty_ece": empty_ece,
            "nonempty_ece": nonempty_ece,
            "empty_nll": empty_nll,
            "nonempty_nll": nonempty_nll,
            "inference_time": inference_time,
        }

    def reset(self):

        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0

        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0

        self.iou_ssc = np.zeros(self.n_classes, dtype=float)
        self.cnt_class = np.zeros(self.n_classes, dtype=float)

        self.empty_ece = 0.0
        self.nonempty_ece = 0.0
        self.ece_count = 0.0

        self.nonempty_nll = 0.0
        self.empty_nll = 0.0
        self.n_empty_voxels = 0.0
        self.n_nonempty_voxels = 0.0

        self.inference_time = 0.0

    def get_score_completion(self, predict, target, nonempty=None):
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]

            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
        iou_sum = np.zeros(_C, dtype=float)  # sum of iou for each class
        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]  # GT
            y_pred = predict[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[
                    np.where(np.logical_and(nonempty_idx == 1, y_true != 255))
                ]
                y_true = y_true[
                    np.where(np.logical_and(nonempty_idx == 1, y_true != 255))
                ]
            for j in range(_C):  # for each class
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum
