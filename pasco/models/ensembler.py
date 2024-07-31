from torch import nn
import torch
import numpy as np
import MinkowskiEngine as ME
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from pasco.models.misc import to_dense_tensor
from pasco.models.transform_utils import sample_scene
from pasco.models.utils import find_matching_indices_v2
import time

OFFSET = 256 * 256 * 256


class Ensembler(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid_sparse = ME.MinkowskiSigmoid()

    def ensemble_panop(
        self,
        panop_predictions,
        ensemble_sem_prob_denses,
        scene_size,
        Ts,
        iou_threshold=0.2,
        measure_time=False,
    ):
        # print("iou_threshold", iou_threshold)
        # NOTE: currently all subnets share the same coords
        n_subnets = len(panop_predictions)
        bs, n_queries, _ = panop_predictions[0]["query_logits"].shape
        ensemble_query_probs = torch.zeros_like(panop_predictions[0]["query_logits"])
        voxel_prob_denses = []
        query_probs = []
        for i in range(n_subnets):
            T = Ts[i]
            voxel_logits = panop_predictions[i]["voxel_logits"]  # [N, n_queries]
            voxel_prob_subnet = self.sigmoid_sparse(voxel_logits)
            min_C = voxel_prob_subnet.C[:, 1:].min(0)[0]
            max_C = voxel_prob_subnet.C[:, 1:].max(0)[0]
            scene_size = max_C - min_C + 1
            voxel_prob_subnet_dense = to_dense_tensor(
                voxel_prob_subnet.F,
                voxel_prob_subnet.C,
                scene_size=scene_size,
                min_coords=min_C,
            )

            from_scene_size = (256, 256, 32)
            from_probs, from_coords = sample_scene(
                min_C, T, voxel_prob_subnet_dense, from_scene_size
            )
            from_coords = ME.utils.batched_coordinates([from_coords])
            from_voxel_prob_subnet_dense = to_dense_tensor(
                from_probs, from_coords, scene_size=from_scene_size
            )
            voxel_prob_denses.append(from_voxel_prob_subnet_dense)

            query_logits = panop_predictions[i][
                "query_logits"
            ]  # [bs, n_queries, n_classes + 1]: +1 for dustbin class
            query_prob = F.softmax(query_logits, dim=-1)
            query_probs.append(query_prob)

        anchor_query_prob = query_probs[0].clone()
        anchor_voxel_prob_dense = voxel_prob_denses[0].clone()
        ious = []
        if measure_time:
            ensemble_time = 0.0
            torch.cuda.synchronize()
            time_start = time.time()
        for i in range(1, n_subnets):
            aux_query_prob = query_probs[i]
            aux_voxel_prob_dense = voxel_prob_denses[i]
            anchor_indices, aux_indices, iou = find_matching_indices_v2(
                anchor_voxel_prob_dense,
                anchor_query_prob,
                aux_voxel_prob_dense,
                aux_query_prob,
                iou_threshold,
            )

            anchor_query_prob[:, anchor_indices, :] = (
                anchor_query_prob[:, anchor_indices, :] * i
                + aux_query_prob[:, aux_indices, :]
            ) / (i + 1)
            anchor_voxel_prob_dense[anchor_indices, :, :, :] = (
                anchor_voxel_prob_dense[anchor_indices, :, :, :] * i
                + aux_voxel_prob_dense[aux_indices, :, :, :]
            ) / (i + 1)
            ious.append(iou)

        if len(ious) > 0:
            iou = torch.stack(ious, dim=0).mean(0)

            keep = iou > iou_threshold
            anchor_voxel_prob_dense = anchor_voxel_prob_dense[keep]
            anchor_query_prob = anchor_query_prob[:, keep, :]

        ensemble_voxel_probs_dense = anchor_voxel_prob_dense
        ensemble_sem_class_dense = ensemble_sem_prob_denses[-1].argmax(0)
        ensemble_voxel_probs_dense = (
            ensemble_voxel_probs_dense * (ensemble_sem_class_dense != 0).float()
        )
        voxel_prob_denses.append(ensemble_voxel_probs_dense)

        # ensemble_query_probs = torch.stack(query_probs, dim=0).mean(0)
        ensemble_query_probs = anchor_query_prob
        query_probs.append(ensemble_query_probs)
        if measure_time:
            torch.cuda.synchronize()
            ensemble_time += time.time() - time_start

        panop_prob_predictions = []
        for i in range(len(voxel_prob_denses)):
            voxel_prob = ME.to_sparse(voxel_prob_denses[i].unsqueeze(0))
            coords = voxel_prob.C.long()
            sem_prob = ensemble_sem_prob_denses[i][
                :, coords[:, 1], coords[:, 2], coords[:, 3]
            ].T
            sem_prob = ME.SparseTensor(sem_prob, coords)
            panop_prob_prediction = {
                "sem_probs": sem_prob,
                "voxel_probs": voxel_prob,
                "query_probs": query_probs[i],
            }
            if measure_time:
                panop_prob_prediction["ensemble_time"] = ensemble_time
            panop_prob_predictions.append(panop_prob_prediction)
        return panop_prob_predictions

    def ensemble_occ(self, occ_logits_at_scales, Ts):
        bs = 1
        occ_logits_1s = occ_logits_at_scales[1]
        n_subnets = len(occ_logits_1s)
        occ_probs = []
        for i_subnet in range(n_subnets):
            occ_logits_1 = occ_logits_1s[i_subnet]
            T = Ts[i_subnet]
            occ_probs_1 = self.sigmoid_sparse(occ_logits_1)
            min_C = occ_logits_1.C[:, 1:].min(0)[0]
            max_C = occ_logits_1.C[:, 1:].max(0)[0]
            scene_size = max_C - min_C + 1
            occ_probs_1_dense = to_dense_tensor(
                occ_probs_1.F, occ_probs_1.C, scene_size=scene_size, min_coords=min_C
            )
            from_scene_size = (256, 256, 32)
            from_probs, from_coords = sample_scene(
                min_C, T, occ_probs_1_dense, from_scene_size
            )
            from_coords = ME.utils.batched_coordinates([from_coords])
            from_occ_probs_1_dense = to_dense_tensor(
                from_probs, from_coords, scene_size=from_scene_size
            )
            occ_probs.append(from_occ_probs_1_dense)
        return torch.cat(occ_probs, dim=0).mean(0, keepdim=True)

    def ensemble_sem_compl(self, sem_logits_at_scales, Ts):
        bs = 1
        sem_logits_1s = sem_logits_at_scales[1]
        n_subnets = len(sem_logits_1s)
        sem_probs = []
        for i_subnet in range(n_subnets):
            sem_logits_1 = sem_logits_1s[i_subnet]
            T = Ts[i_subnet]
            sem_probs_1 = F.softmax(sem_logits_1.F, dim=-1)
            min_C = sem_logits_1.C[:, 1:].min(0)[0]
            max_C = sem_logits_1.C[:, 1:].max(0)[0]
            scene_size = max_C - min_C + 1
            sem_probs_1_dense = to_dense_tensor(
                sem_probs_1, sem_logits_1.C, scene_size=scene_size, min_coords=min_C
            )
            from_scene_size = (256, 256, 32)
            from_probs, from_coords = sample_scene(
                min_C, T, sem_probs_1_dense, from_scene_size
            )
            from_coords = ME.utils.batched_coordinates([from_coords])
            from_sem_probs_1_dense = to_dense_tensor(
                from_probs, from_coords, scene_size=from_scene_size
            )
            empty_mask = from_sem_probs_1_dense.sum(0) == 0
            xx, yy, zz = torch.nonzero(empty_mask, as_tuple=True)
            from_sem_probs_1_dense[0, xx, yy, zz] = 1.0
            sem_probs.append(from_sem_probs_1_dense)
        sem_probs.append(torch.stack(sem_probs, dim=0).mean(0))
        return sem_probs

    def ssc_uncertainty(self, subnet_sem_prob_denses):
        subnet_sem_prob_denses = torch.stack(subnet_sem_prob_denses, dim=0)
        variance = subnet_sem_prob_denses.var(0).mean(0)
        return variance


class Merger(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, panop_outs, panop_out_probs, method="avg_matching"):
        if method == "avg_matching":
            return self._merge_avg_matching(panop_outs, panop_out_probs)
        elif method == "avg":
            return self._merge_avg(panop_outs, panop_out_probs)
        else:
            raise ValueError("Unknown merge method")

    def _merge_avg(self, panop_outs, panop_out_probs):
        merged_completion_prob_denses_at_scales = None
        merged_voxel_probs_dense = None
        merged_query_probs = None
        cnt = None
        cnt_completion = {}
        for panop_out_prob in panop_out_probs:
            if merged_completion_prob_denses_at_scales is None:
                merged_completion_prob_denses_at_scales = panop_out_prob[
                    "completion_prob_denses_at_scales"
                ]
                for k in merged_completion_prob_denses_at_scales:
                    cnt_completion[k] = (
                        panop_out_prob["completion_prob_denses_at_scales"][k] > 0.0
                    ).float()

                merged_query_probs = panop_out_prob["query_probs"]
                cnt = torch.zeros_like(merged_completion_prob_denses_at_scales[1])
                scene_size = merged_completion_prob_denses_at_scales[1].shape[2:]
                dense_shape = torch.Size(
                    (1, panop_out_prob["voxel_probs"].shape[1], *scene_size)
                )
                merged_voxel_probs_dense, _, _ = panop_out_prob["voxel_probs"].dense(
                    dense_shape
                )
            else:
                for k in merged_completion_prob_denses_at_scales.keys():
                    merged_completion_prob_denses_at_scales[k] += panop_out_prob[
                        "completion_prob_denses_at_scales"
                    ][k]
                    cnt_completion[k] += (
                        panop_out_prob["completion_prob_denses_at_scales"][k] > 0.0
                    ).float()

                merged_query_probs += panop_out_prob["query_probs"]
                coords = panop_out_prob["voxel_probs"].C.clone().long()
                merged_voxel_probs_dense[
                    coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]
                ] += panop_out_prob["voxel_probs"].F
                cnt[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] += 1.0

        for k in merged_completion_prob_denses_at_scales:
            merged_completion_prob_denses_at_scales[k] /= cnt_completion[k].clamp(min=1)

        merged_query_probs /= len(panop_out_probs)
        merged_voxel_probs_dense /= cnt.clamp(min=1)
        merged_voxel_probs_dense = (
            merged_voxel_probs_dense
            * (merged_completion_prob_denses_at_scales[1] > 0.5).float()
        )
        merged_voxel_probs = ME.to_sparse(merged_voxel_probs_dense)
        return (
            merged_completion_prob_denses_at_scales,
            merged_voxel_probs,
            merged_query_probs,
        )

    def _merge_avg_matching(self, panop_outs, panop_out_probs):
        """
        Merge two panoptic outputs by averaging the voxel probabilities and query probabilities
        """
        query_idx_0, query_idx_1 = self._match_queries(panop_outs, iou_threshold=0.5)
        num_queries = 100
        scene_size = panop_out_probs[0]["completion_prob_dense"].shape[2:]
        dense_shape = torch.Size((1, num_queries, *scene_size))

        merged_completion_prob_dense = (
            panop_out_probs[0]["completion_prob_dense"]
            + panop_out_probs[1]["completion_prob_dense"]
        ) / 2
        voxel_probs_0_dense, _, _ = panop_out_probs[0]["voxel_probs"].dense(dense_shape)
        voxel_probs_1_dense, _, _ = panop_out_probs[1]["voxel_probs"].dense(dense_shape)
        voxel_probs_0_dense_matched = voxel_probs_0_dense[:, query_idx_0, :, :, :]
        voxel_probs_1_dense_matched = voxel_probs_1_dense[:, query_idx_1, :, :, :]
        cnt = (voxel_probs_0_dense_matched.sum(1, keepdim=True) > 0).float() + (
            voxel_probs_1_dense_matched.sum(1, keepdim=True) > 0
        ).float()
        merged_voxel_probs_dense_matched = (
            voxel_probs_0_dense_matched + voxel_probs_1_dense_matched
        ) / cnt.float().clamp(min=1)

        query_probs_0 = panop_out_probs[0]["query_probs"][:, query_idx_0, :]
        query_probs_1 = panop_out_probs[1]["query_probs"][:, query_idx_1, :]
        merged_query_probs_matched = (query_probs_0 + query_probs_1) / 2

        query_idx_not_match_0 = [i for i in range(num_queries) if i not in query_idx_0]
        query_idx_not_match_1 = [i for i in range(num_queries) if i not in query_idx_1]
        query_probs_not_match_0 = panop_out_probs[0]["query_probs"][
            :, query_idx_not_match_0, :
        ]
        query_probs_not_match_1 = panop_out_probs[1]["query_probs"][
            :, query_idx_not_match_1, :
        ]
        voxel_probs_not_match_0_dense = voxel_probs_0_dense[
            :, query_idx_not_match_0, :, :, :
        ]
        voxel_probs_not_match_1_dense = voxel_probs_1_dense[
            :, query_idx_not_match_1, :, :, :
        ]

        merged_voxel_probs_dense = torch.cat(
            [
                merged_voxel_probs_dense_matched,
                voxel_probs_not_match_0_dense,
                voxel_probs_not_match_1_dense,
            ],
            dim=1,
        )
        merged_voxel_probs_dense = (
            merged_voxel_probs_dense * (merged_completion_prob_dense > 0.5).float()
        )
        merged_voxel_probs = ME.to_sparse(merged_voxel_probs_dense)

        merged_query_probs = torch.cat(
            [
                merged_query_probs_matched,
                query_probs_not_match_0,
                query_probs_not_match_1,
            ],
            dim=1,
        )
        return merged_voxel_probs, merged_query_probs

    def _match_queries(self, panop_outs, iou_threshold=0.5):
        bs = len(panop_outs[0]["segments_infos"])
        assert bs == 1, "Batch size must be 1"
        for i in range(bs):

            segment_info_0 = panop_outs[0]["segments_infos"][i]
            segment_info_1 = panop_outs[1]["segments_infos"][i]
            id2query_0 = {el["id"]: el["query_id"] for el in segment_info_0}
            id2query_1 = {el["id"]: el["query_id"] for el in segment_info_1}
            panoptic_seg_denses_0 = (
                panop_outs[0]["panoptic_seg_denses"][i].detach().cpu().numpy()
            )
            panoptic_seg_denses_1 = (
                panop_outs[1]["panoptic_seg_denses"][i].detach().cpu().numpy()
            )
            idx_0, idx_1, ious = self.match_segments(
                segment_info_0,
                segment_info_1,
                panoptic_seg_denses_0,
                panoptic_seg_denses_1,
                ignore_label=0,
            )
            mask_iou = ious > iou_threshold
            idx_0 = idx_0[mask_iou]
            idx_1 = idx_1[mask_iou]
            query_idx_0 = [id2query_0[id] for id in idx_0]
            query_idx_1 = [id2query_1[id] for id in idx_1]
        return query_idx_0, query_idx_1

    def match_segments(
        self, gt_segments_info, pred_segments_info, pan_gt, pan_pred, ignore_label=0
    ):

        gt_segms = {el["id"]: el for el in gt_segments_info}
        pred_segms = {el["id"]: el for el in pred_segments_info}
        gt_ids = list(gt_segms.keys())
        pred_ids = list(pred_segms.keys())
        ious = np.zeros((np.max(gt_ids) + 1, np.max(pred_ids) + 1))

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el["id"] for el in pred_segments_info)
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == ignore_label:
                    continue
                print("Error segment", pred_segms[label])
                raise KeyError(
                    "segment with ID {} is presented in PNG and not presented in JSON.".format(
                        label
                    )
                )
            pred_segms[label]["area"] = label_cnt
            pred_labels_set.remove(label)
        assert (
            len(pred_labels_set) == 0
        ), "Some segments from JSON are not presented in PNG."

        gt_labels_set = set(el["id"] for el in gt_segments_info)
        labels, labels_cnt = np.unique(pan_gt, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in gt_segms:
                if label == ignore_label:
                    continue
                print("Error segment", pred_segms[label])
                raise KeyError(
                    "segment with ID {} is presented in PNG and not presented in JSON.".format(
                        label
                    )
                )
            gt_segms[label]["area"] = label_cnt
            gt_labels_set.remove(label)
        assert (
            len(gt_labels_set) == 0
        ), "Some segments from JSON are not presented in PNG."

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)

        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            if gt_id == ignore_label or pred_id == ignore_label:
                continue
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue

            if (
                gt_segms[gt_label]["category_id"]
                != pred_segms[pred_label]["category_id"]
            ):
                continue

            union = (
                pred_segms[pred_label]["area"]
                + gt_segms[gt_label]["area"]
                - intersection
            )
            iou = intersection / union
            ious[int(gt_label), int(pred_label)] = iou

        gt_idx, pred_idx = linear_sum_assignment(-ious)

        ious = ious[gt_idx, pred_idx]
        return gt_idx, pred_idx, ious
