import pickle
import os
import numpy as np
import json
import time
from collections import defaultdict
import multiprocessing
import torch


OFFSET = 256 * 256 * 256
from scipy.optimize import linear_sum_assignment


class PQStatCat:
    def __init__(self):
        self.all_iou = 0.0
        self.all_n = 0.0
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        self.all_iou += pq_stat_cat.all_iou
        self.all_n += pq_stat_cat.all_n
        return self


class PQStat:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, isthing, ignore_cat_id, thing_ids):
        pq_dagger, pq, sq, rq, n = 0, 0, 0, 0, 0
        per_class_results = {}
        for label in self.pq_per_cat.keys():

            if label == ignore_cat_id:
                continue
            if isthing is not None:
                # cat_isthing = label_info['isthing'] == 1
                cat_isthing = label in thing_ids
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            all_iou = self.pq_per_cat[label].all_iou
            all_n = self.pq_per_cat[label].all_n

            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {"pq": 0.0, "sq": 0.0, "rq": 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {"pq": pq_class, "sq": sq_class, "rq": rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

            if isthing is None:
                if label in thing_ids:
                    pq_dagger += pq_class
                else:
                    pq_dagger += all_iou / max(all_n, 1)

        n = max(n, 1)
        return {
            "pq_dagger": pq_dagger / n,
            "pq": pq / n,
            "sq": sq / n,
            "rq": rq / n,
            "n": n,
        }, per_class_results


def find_matched_segment(
    gt_segments_info,
    pred_segments_info,
    pan_gt,
    pan_pred,
    threshold=0.2,
    ignore_label=0,
):

    gt_segms = {el["id"]: el for el in gt_segments_info}
    pred_segms = {el["id"]: el for el in pred_segments_info}

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
    gt_matched = set()
    pred_matched = set()
    pred_gt_matched = set()

    max_pred_id = np.max(list(pred_segms.keys())) if list(pred_segms.keys()) else 0
    max_gt_id = np.max(list(gt_segms.keys())) if list(gt_segms.keys()) else 0

    conf_matrix = np.zeros((max_gt_id + 1, max_pred_id + 1))
    for label_tuple, intersection in gt_pred_map.items():
        gt_label, pred_label = label_tuple
        if gt_label not in gt_segms:
            continue
        if pred_label not in pred_segms:
            continue

        union = (
            pred_segms[pred_label]["area"] + gt_segms[gt_label]["area"] - intersection
        )
        iou = intersection / union

        conf_matrix[int(gt_label), int(pred_label)] = iou
        if threshold >= 0.5:
            if iou > 0.5:
                pred_gt_matched.add(label_tuple)
    if threshold >= 0.5:
        return pred_gt_matched
    else:
        gt_idx, pred_idx = linear_sum_assignment(-conf_matrix)
        ious = conf_matrix[gt_idx, pred_idx]
        pred_gt_matched = [
            (gt_idx[i], pred_idx[i]) for i in range(len(gt_idx)) if ious[i] > threshold
        ]

    return pred_gt_matched


def pq_compute_single_core(
    pq_stat,
    gt_segments_info,
    pred_segments_info,
    pan_gt,
    pan_pred,
    thing_ids,
    ignore_label=0,
):

    gt_segms = {el["id"]: el for el in gt_segments_info}
    pred_segms = {el["id"]: el for el in pred_segments_info}

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
    gt_matched = set()
    pred_matched = set()
    pred_gt_matched = set()
    # pred_gt_matched = []
    # print("=====")
    for label_tuple, intersection in gt_pred_map.items():
        gt_label, pred_label = label_tuple
        if gt_label not in gt_segms:
            continue
        if pred_label not in pred_segms:
            continue
        if gt_segms[gt_label]["category_id"] != pred_segms[pred_label]["category_id"]:
            continue
        union = (
            pred_segms[pred_label]["area"] + gt_segms[gt_label]["area"] - intersection
        )  # - gt_pred_map.get((ignore_label, pred_label), 0) ERROR before
        iou = intersection / union
        if gt_segms[gt_label]["category_id"] not in thing_ids:
            pq_stat[gt_segms[gt_label]["category_id"]].all_iou += iou
            pq_stat[gt_segms[gt_label]["category_id"]].all_n += 1
            pred_matched.add(pred_label)
            pred_gt_matched.add(label_tuple)
        if iou > 0.5:
            pq_stat[gt_segms[gt_label]["category_id"]].tp += 1
            pq_stat[gt_segms[gt_label]["category_id"]].iou += iou
            gt_matched.add(gt_label)
            pred_matched.add(pred_label)
            pred_gt_matched.add(label_tuple)

    # count false negatives
    crowd_labels_dict = {}
    for gt_label, gt_info in gt_segms.items():
        if gt_label in gt_matched:
            continue
        pq_stat[gt_info["category_id"]].fn += 1

    # count false positives
    for pred_label, pred_info in pred_segms.items():
        if pred_label in pred_matched:
            continue
        pq_stat[pred_info["category_id"]].fp += 1

    return pred_gt_matched


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print(
        "Number of cores: {}, images per core: {}".format(
            cpu_num, len(annotations_split[0])
        )
    )
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(
            pq_compute_single_core,
            (proc_id, annotation_set, gt_folder, pred_folder, categories),
        )
        processes.append(p)
    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()
    return pq_stat


def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, "r") as f:
        gt_json = json.load(f)
    with open(pred_json_file, "r") as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace(".json", "")
    if pred_folder is None:
        pred_folder = pred_json_file.replace(".json", "")
    categories = {el["id"]: el for el in gt_json["categories"]}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception(
            "Folder {} with ground truth segmentations doesn't exist".format(gt_folder)
        )
    if not os.path.isdir(pred_folder):
        raise Exception(
            "Folder {} with predicted segmentations doesn't exist".format(pred_folder)
        )

    pred_annotations = {el["image_id"]: el for el in pred_json["annotations"]}
    matched_annotations_list = []
    for gt_ann in gt_json["annotations"]:
        image_id = gt_ann["image_id"]
        if image_id not in pred_annotations:
            raise Exception("no prediction for the image with id: {}".format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    pq_stat = pq_compute_multi_core(
        matched_annotations_list, gt_folder, pred_folder, categories
    )

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(
            categories, isthing=isthing
        )
        if name == "All":
            results["per_class"] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print(
            "{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
                name,
                100 * results[name]["pq"],
                100 * results[name]["sq"],
                100 * results[name]["rq"],
                results[name]["n"],
            )
        )

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


def convert_mask_label_to_panoptic_output(labels, masks, thing_ids):
    """
    labels: [ 0.,  9., 10., 11., 13., 14., 15., 16., 17.,  1.,  1.,  1.,  3.,  5.,....]
    masks: [25, 256, 256, 32]
    """
    segments_info = []
    current_segment_id = 0
    panoptic_seg = torch.zeros(masks.shape[1:])
    stuff_memory_list = {}
    for id, cat_id in enumerate(labels):
        if cat_id == 0:  # ignore empty class
            continue
        isthing = cat_id in thing_ids
        mask = masks[id, :, :, :]  # 256, 256, 32

        # merge stuff regions
        if not isthing:
            if int(cat_id) in stuff_memory_list.keys():
                panoptic_seg[mask] = stuff_memory_list[int(cat_id)]
                continue
            else:
                stuff_memory_list[int(cat_id)] = current_segment_id + 1

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": isthing,
                "category_id": int(cat_id),
                "area": mask.sum(),
            }
        )
    return panoptic_seg, segments_info


if __name__ == "__main__":
    with open("t0.pkl", "rb") as handle:
        data = pickle.load(handle)
    mask_label = data["mask_label"]

    pred_panoptic_seg = (
        data["pred_panoptic_seg"].detach().cpu().numpy()
    )  # (256, 256, 32)
    pred_segments_info = data[
        "pred_segments_info"
    ]  # [{'id': 1, 'isthing': False, 'category_id': 9}, {'id': 2, 'isthing': True, 'category_id': 6}

    gt_panoptic_seg, gt_segments_info = convert_mask_label_to_panoptic_output(
        mask_label["labels"], mask_label["masks"], thing_ids
    )
    gt_panoptic_seg = gt_panoptic_seg.detach().cpu().numpy()

    # Do not consider empty space class
    categories = {el["category_id"]: el for el in gt_segments_info}

    pq_stat = pq_compute_single_core(
        gt_segments_info,
        pred_segments_info,
        gt_panoptic_seg,
        pred_panoptic_seg,
        categories,
    )

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(
            categories, isthing=isthing, ignore_cat_id=0
        )
        if name == "All":
            results["per_class"] = per_class_results
    print("{:15s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (15 + 7 * 4))

    for name, _isthing in metrics:
        print(
            "{:15s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
                name,
                100 * results[name]["pq"],
                100 * results[name]["sq"],
                100 * results[name]["rq"],
                results[name]["n"],
            )
        )
    print("-" * (15 + 7 * 4))

    print("Thing class")
    print("-" * (15 + 7 * 4))
    for cat_id, metric in results["per_class"].items():
        if cat_id not in thing_ids:
            continue
        print(
            "{:15s}| {:5.1f}  {:5.1f}  {:5.1f} ".format(
                class_names[cat_id],
                100 * metric["pq"],
                100 * metric["sq"],
                100 * metric["rq"],
            )
        )
    print("-" * (15 + 7 * 4))
    print("Stuff class")
    print("-" * (15 + 7 * 4))
    for cat_id, metric in results["per_class"].items():
        if cat_id in thing_ids:
            continue
        print(
            "{:15s}| {:5.1f}  {:5.1f}  {:5.1f} ".format(
                class_names[cat_id],
                100 * metric["pq"],
                100 * metric["sq"],
                100 * metric["rq"],
            )
        )
    print("-" * (15 + 7 * 4))
