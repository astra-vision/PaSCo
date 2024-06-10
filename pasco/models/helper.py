import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from pasco.models.misc import to_dense_tensor


def semantic_inference_v2(
    voxel_prob: torch.Tensor, query_logit: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        voxel_probs: SparseTensor(N, num_queries)
        query_logits: (num_queries, n_classes)
    """
    bs = query_logit.shape[0]
    n_classes = query_logit.shape[-1] - 1  # minus dustbin class

    pred_logit = query_logit

    pred_softmax_prob = F.softmax(pred_logit, dim=-1)  # [100, 21]
    scores, labels = pred_softmax_prob.max(-1)  # 100,

    keep = labels.ne(n_classes)  # & (scores > object_mask_threshold)

    if keep.sum() > 0:

        filtered_pred_logit = pred_logit[keep, :-1]  # [num_maskes, n_classes]

        voxel_probs_i = voxel_prob.F[:, keep]  # [N, num_maskes]
        coords_i = voxel_prob.C
        # normalize
        voxel_probs_i += 1e-8
        voxel_probs_i = voxel_probs_i / voxel_probs_i.sum(1, keepdim=True)

        sparse_ssc_logit = voxel_probs_i @ filtered_pred_logit  # [N, n_classes]

        return ME.SparseTensor(features=sparse_ssc_logit, coordinates=coords_i)
    else:
        return None


def semantic_inference(
    voxel_probs: torch.Tensor, query_logits: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        voxel_probs: SparseTensor(bs * N, num_queries)
        query_logits: (bs, num_queries, n_classes)
    """
    bs = query_logits.shape[0]
    n_classes = query_logits.shape[-1] - 1  # minus dustbin class

    sparse_ssc_logits = []
    sparse_ssc_coords = []
    for i in range(bs):
        pred_logit = query_logits[i]

        pred_softmax_prob = F.softmax(pred_logit, dim=-1)  # [100, 21]
        scores, labels = pred_softmax_prob.max(-1)  # 100,

        keep = labels.ne(n_classes)

        if keep.sum() > 0:
            coord_mask_i = voxel_probs.C[:, 0] == i
            # remove dustbin class
            filtered_pred_logit = pred_logit[keep, :-1]  # [num_maskes, n_classes]

            coords_i = voxel_probs.C[coord_mask_i].clone().long()
            voxel_probs_i = voxel_probs.F[coord_mask_i][:, keep]  # [N, num_maskes]

            # normalize
            voxel_probs_i += 1e-8
            voxel_probs_i = voxel_probs_i / voxel_probs_i.sum(1, keepdim=True)

            sparse_ssc_logit = voxel_probs_i @ filtered_pred_logit  # [N, n_classes]
            sparse_ssc_logits.append(sparse_ssc_logit)
            sparse_ssc_coords.append(coords_i)

        else:
            return None

    sparse_ssc_logits = torch.cat(sparse_ssc_logits, dim=0)
    sparse_ssc_coords = torch.cat(sparse_ssc_coords, dim=0)
    ssc_logit_sparse = ME.SparseTensor(
        features=sparse_ssc_logits, coordinates=sparse_ssc_coords
    )

    return ssc_logit_sparse


def panoptic_inference(
    voxel_output: torch.Tensor,
    query_output: torch.Tensor,
    # sem_probs: torch.Tensor,
    overlap_threshold: float,
    object_mask_threshold,
    thing_ids,
    min_C,
    scene_size,
    input_query_logit=True,
    input_voxel_logit=False,
    vox_occ_threshold=0.3,
):  # TODO: tune later, could result in better performance
    """
    Args:
        voxel_probs: SparseTensor(bs * N, num_queries)
    """
    (
        semantic_seg_denses,
        panoptic_seg_denses,
        panoptic_seg_sparses,
        segments_infos,
        entropies,
    ) = ([], [], [], [], [])
    ins_uncertainty_denses, vox_uncertainty_denses = [], []
    vox_confidence_denses = []
    vox_all_mask_probs_denses = []

    bs = query_output.shape[0]
    n_classes = query_output.shape[-1] - 1  # minus dustbin class

    if input_voxel_logit:
        sigmoid = ME.MinkowskiSigmoid()
        voxel_probs = sigmoid(voxel_output)
    else:
        voxel_probs = voxel_output

    for i in range(bs):
        if input_query_logit:
            pred_logit = query_output[i]
            query_probs = F.softmax(pred_logit, dim=-1)
        else:
            query_probs = query_output[i]

        # Get query class and the corresponding probability
        probs, labels = query_probs.max(-1)  # num_queries,

        # Remove dustbin and empty classes
        # NOTE: Don't predict empty class in panoptic prediction
        keep = labels.ne(0) & labels.ne(n_classes) & (probs > object_mask_threshold)

        # Filter queries probabilities and classes
        filtered_query_probs = probs[keep]
        filtered_query_classes = labels[keep]
        filtered_query_probs_all_classes = query_probs[keep]

        # Get the filtered query ids
        filtered_query_ids = torch.arange(0, keep.shape[0], device=keep.device)[keep]

        # Get the corresponding voxel_probs for each batch item
        mask_indices = torch.nonzero(voxel_probs.C[:, 0] == i, as_tuple=True)[
            0
        ]  # [N, 4]
        voxel_probs_F = voxel_probs.F[mask_indices].clone()  # [N, num_maskes]

        # Filter the voxel_probs for each batch item
        filtered_masks_prob_F = voxel_probs_F[:, keep]  # [N, num_maskes]

        normalized_voxel_probs_F = voxel_probs_F[:, keep]
        normalized_mask_prob = normalized_voxel_probs_F / (
            normalized_voxel_probs_F.sum(1, keepdim=True) + 1e-8
        )

        # Get the corresponding coords for each batch item
        coords = voxel_probs.C[mask_indices]  # [N, 4]

        # Combine the query probs and masks probs
        combined_mask_query_probs = (
            filtered_query_probs.view(1, -1) * filtered_masks_prob_F
        )  # [N, num_maskes]

        N = combined_mask_query_probs.shape[0]
        panoptic_seg = torch.zeros(
            (N,), dtype=torch.int32, device=filtered_masks_prob_F.device
        )
        semantic_seg = torch.zeros(
            (N,), dtype=torch.int32, device=filtered_masks_prob_F.device
        )
        ins_uncertainty = torch.zeros(
            (N,), dtype=torch.float, device=filtered_masks_prob_F.device
        )
        vox_uncertainty = torch.zeros(
            (N,), dtype=torch.float, device=filtered_masks_prob_F.device
        )
        vox_confidence = torch.zeros(
            (N,), dtype=torch.float, device=filtered_masks_prob_F.device
        )
        vox_all_mask_probs = torch.zeros(
            (N, filtered_masks_prob_F.shape[1]),
            dtype=torch.float,
            device=filtered_masks_prob_F.device,
        )
        segments_info = []
        current_segment_id = 0

        # Check if we detected some masks
        if filtered_masks_prob_F.shape[1] != 0:

            # NOTE: cur_mask_ids is not query_id. It is the index of the query after filtering
            max_prob, cur_mask_ids = combined_mask_query_probs.max(1)

            # Store the segment_id of stuff classes
            stuff_memory_list = {}

            # Loop over each filtered segment
            for k in range(filtered_query_classes.shape[0]):

                pred_class = filtered_query_classes[k].item()  # class of this query
                query_max_class_prob = filtered_query_probs[
                    k
                ].item()  # softmax prob of this query

                isthing = pred_class in thing_ids  # whether this mask is thing or stuff
                mask = cur_mask_ids == k  # get the mask of this query
                mask = mask & (
                    filtered_masks_prob_F[:, k] >= vox_occ_threshold
                )  # TODO: new

                mask_area = mask.sum().item()

                original_area = (
                    (filtered_masks_prob_F[:, k] >= vox_occ_threshold).sum().item()
                )

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    if pred_class == 0:
                        panoptic_seg[mask] = 0
                        semantic_seg[mask] = 0
                    else:
                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = (
                                    current_segment_id + 1
                                )
                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id
                        semantic_seg[mask] = pred_class

                        vox_confidence[mask] = normalized_mask_prob[mask, k]
                        vox_all_mask_probs[mask] = filtered_masks_prob_F[mask]

                        ins_uncertainty[mask] = query_max_class_prob
                        vox_uncertainty[mask] = (
                            combined_mask_query_probs
                            / combined_mask_query_probs.sum(1, keepdim=True)
                        ).max(1)[0][mask]
                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                                "query_id": filtered_query_ids[k].item(),
                                "confidence": query_max_class_prob,
                                "all_class_probs": filtered_query_probs_all_classes[k],
                            }
                        )
        panoptic_seg_dense = to_dense_tensor(
            panoptic_seg.unsqueeze(-1), coords, scene_size, min_coords=min_C
        ).squeeze()
        semantic_seg_dense = to_dense_tensor(
            semantic_seg.unsqueeze(-1), coords, scene_size, min_coords=min_C
        ).squeeze()
        ins_uncertainty_dense = to_dense_tensor(
            ins_uncertainty.unsqueeze(-1), coords, scene_size, min_coords=min_C
        ).squeeze()
        vox_uncertainty_dense = to_dense_tensor(
            vox_uncertainty.unsqueeze(-1), coords, scene_size, min_coords=min_C
        ).squeeze()
        vox_confidence_dense = to_dense_tensor(
            vox_confidence.unsqueeze(-1), coords, scene_size, min_coords=min_C
        ).squeeze()
        vox_all_mask_probs_dense = to_dense_tensor(
            vox_all_mask_probs, coords, scene_size, min_coords=min_C
        ).squeeze()

        semantic_seg_denses.append(semantic_seg_dense)
        panoptic_seg_sparses.append(panoptic_seg)
        panoptic_seg_denses.append(panoptic_seg_dense)
        segments_infos.append(segments_info)
        ins_uncertainty_denses.append(ins_uncertainty_dense)
        vox_uncertainty_denses.append(vox_uncertainty_dense)
        vox_confidence_denses.append(vox_confidence_dense)
        vox_all_mask_probs_denses.append(vox_all_mask_probs_dense)

    return {
        "vox_all_mask_probs_denses": vox_all_mask_probs_denses,
        "panoptic_seg_denses": torch.stack(panoptic_seg_denses),
        "semantic_seg_denses": torch.stack(semantic_seg_denses),
        "ins_uncertainty_denses": torch.stack(ins_uncertainty_denses),
        "vox_confidence_denses": torch.stack(vox_confidence_denses),
        "vox_uncertainty_denses": torch.stack(
            vox_uncertainty_denses
        ),  # TODO: not use, remove later
        "panoptic_seg_sparses": panoptic_seg_sparses,
        "segments_infos": segments_infos,
    }


def panoptic_inference_maskpls(
    voxel_output: torch.Tensor,
    query_output: torch.Tensor,
    overlap_threshold: float,
    object_mask_threshold,
    thing_ids,
    min_C,
    scene_size,
    input_query_logit=True,
    input_voxel_logit=False,
    vox_occ_threshold=0.5,
):  # TODO: tune later, could result in better performance
    """
    Args:
        voxel_probs: SparseTensor(bs * N, num_queries)
    """
    (
        semantic_seg_denses,
        panoptic_seg_denses,
        panoptic_seg_sparses,
        segments_infos,
        entropies,
    ) = ([], [], [], [], [])
    ins_uncertainty_denses, vox_uncertainty_denses = [], []
    vox_confidence_denses = []
    vox_all_mask_probs_denses = []

    bs = query_output.shape[0]
    n_classes = query_output.shape[-1] - 1  # minus dustbin class

    if input_voxel_logit:
        sigmoid = ME.MinkowskiSigmoid()
        voxel_probs = sigmoid(voxel_output)
    else:
        voxel_probs = voxel_output

    for i in range(bs):
        if input_query_logit:
            pred_logit = query_output[i]
            query_probs = F.softmax(pred_logit, dim=-1)
        else:
            query_probs = query_output[i]

        # Get query class and the corresponding probability
        probs, labels = query_probs.max(-1)  # num_queries,

        # Remove dustbin and empty classes
        # NOTE: Don't predict empty class in panoptic prediction
        keep = labels.ne(0) & labels.ne(n_classes) & (probs > object_mask_threshold)

        # Filter queries probabilities and classes
        filtered_query_probs = probs[keep]
        filtered_query_classes = labels[keep]

        # Get the filtered query ids
        filtered_query_ids = torch.arange(0, keep.shape[0], device=keep.device)[keep]

        # Get the corresponding voxel_probs for each batch item
        mask_indices = torch.nonzero(voxel_probs.C[:, 0] == i, as_tuple=True)[
            0
        ]  # [N, 4]
        voxel_probs_F = voxel_probs.F[mask_indices].clone()  # [N, num_maskes]
        # sem_probs_F = sem_probs.F[mask_indices].clone() # [N, num_maskes]

        # Filter the voxel_probs for each batch item
        filtered_masks_prob_F = voxel_probs_F[:, keep]  # [N, num_maskes]
        # normalized_voxel_probs_F = voxel_probs_F / voxel_probs_F.sum(1, keepdim=True)
        normalized_voxel_probs_F = voxel_probs_F[:, keep]
        normalized_mask_prob = normalized_voxel_probs_F / (
            normalized_voxel_probs_F.sum(1, keepdim=True) + 1e-8
        )

        # Get the corresponding coords for each batch item
        coords = voxel_probs.C[mask_indices]  # [N, 4]

        # Combine the query probs and masks probs
        combined_mask_query_probs = (
            filtered_query_probs.view(1, -1) * filtered_masks_prob_F
        )  # [N, num_maskes]
        combined_normalized_mask_query_probs = (
            normalized_mask_prob * filtered_query_probs.view(1, -1)
        )

        N = combined_mask_query_probs.shape[0]
        panoptic_seg = torch.zeros(
            (N,), dtype=torch.int32, device=filtered_masks_prob_F.device
        )
        semantic_seg = torch.zeros(
            (N,), dtype=torch.int32, device=filtered_masks_prob_F.device
        )
        ins_uncertainty = torch.zeros(
            (N,), dtype=torch.float, device=filtered_masks_prob_F.device
        )
        vox_uncertainty = torch.zeros(
            (N,), dtype=torch.float, device=filtered_masks_prob_F.device
        )
        vox_confidence = torch.zeros(
            (N,), dtype=torch.float, device=filtered_masks_prob_F.device
        )
        vox_all_mask_probs = torch.zeros(
            (N, filtered_masks_prob_F.shape[1]),
            dtype=torch.float,
            device=filtered_masks_prob_F.device,
        )
        segments_info = []
        current_segment_id = 0

        # Check if we detected some masks
        if filtered_masks_prob_F.shape[1] != 0:

            # NOTE: cur_mask_ids is not query_id. It is the index of the query after filtering
            max_prob, cur_mask_ids = combined_mask_query_probs.max(1)

            # Store the segment_id of stuff classes
            stuff_memory_list = {}

            # Loop over each filtered segment
            for k in range(filtered_query_classes.shape[0]):

                pred_class = filtered_query_classes[k].item()  # class of this query
                query_max_class_prob = filtered_query_probs[
                    k
                ].item()  # softmax prob of this query

                isthing = pred_class in thing_ids  # whether this mask is thing or stuff
                mask = cur_mask_ids == k  # get the mask of this query
                mask = mask & (
                    filtered_masks_prob_F[:, k] >= vox_occ_threshold
                )  # TODO: remove later

                mask_area = mask.sum().item()

                original_area = (
                    (filtered_masks_prob_F[:, k] >= vox_occ_threshold).sum().item()
                )

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    if pred_class == 0:
                        panoptic_seg[mask] = 0
                        semantic_seg[mask] = 0
                    else:
                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = (
                                    current_segment_id + 1
                                )
                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id
                        semantic_seg[mask] = pred_class

                        vox_all_mask_probs[mask] = filtered_masks_prob_F[mask]
                        vox_confidence[mask] = normalized_mask_prob[mask, k]

                        ins_uncertainty[mask] = query_max_class_prob
                        vox_uncertainty[mask] = (
                            combined_mask_query_probs
                            / combined_mask_query_probs.sum(1, keepdim=True)
                        ).max(1)[0][mask]
                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                                "query_id": filtered_query_ids[k].item(),
                                "confidence": query_max_class_prob,
                            }
                        )
        panoptic_seg_dense = to_dense_tensor(
            panoptic_seg.unsqueeze(-1), coords, scene_size, min_coords=min_C
        ).squeeze()
        semantic_seg_dense = to_dense_tensor(
            semantic_seg.unsqueeze(-1), coords, scene_size, min_coords=min_C
        ).squeeze()
        ins_uncertainty_dense = to_dense_tensor(
            ins_uncertainty.unsqueeze(-1), coords, scene_size, min_coords=min_C
        ).squeeze()
        vox_uncertainty_dense = to_dense_tensor(
            vox_uncertainty.unsqueeze(-1), coords, scene_size, min_coords=min_C
        ).squeeze()
        vox_confidence_dense = to_dense_tensor(
            vox_confidence.unsqueeze(-1), coords, scene_size, min_coords=min_C
        ).squeeze()
        vox_all_mask_probs_dense = to_dense_tensor(
            vox_all_mask_probs, coords, scene_size, min_coords=min_C
        ).squeeze()

        semantic_seg_denses.append(semantic_seg_dense)
        panoptic_seg_sparses.append(panoptic_seg)
        panoptic_seg_denses.append(panoptic_seg_dense)
        segments_infos.append(segments_info)
        ins_uncertainty_denses.append(ins_uncertainty_dense)
        vox_uncertainty_denses.append(vox_uncertainty_dense)
        vox_confidence_denses.append(vox_confidence_dense)
        vox_all_mask_probs_denses.append(vox_all_mask_probs_dense)

    return {
        "vox_all_mask_probs_denses": vox_all_mask_probs_denses,
        "panoptic_seg_denses": torch.stack(panoptic_seg_denses),
        "semantic_seg_denses": torch.stack(semantic_seg_denses),
        "ins_uncertainty_denses": torch.stack(ins_uncertainty_denses),
        "vox_confidence_denses": torch.stack(vox_confidence_denses),
        "vox_uncertainty_denses": torch.stack(
            vox_uncertainty_denses
        ),  # TODO: not use, remove later
        "panoptic_seg_sparses": panoptic_seg_sparses,
        "segments_infos": segments_infos,
        # "entropies": entropies
    }
