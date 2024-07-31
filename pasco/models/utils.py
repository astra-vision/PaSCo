import importlib
import logging
import os
import shutil
import sys

import h5py
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def compute_entropy(probs, dim):
    normalized_probs = probs / (probs.sum(dim=dim, keepdim=True) + 1e-8)
    entropy = -torch.sum(probs * torch.log2(normalized_probs + 1e-8), dim=dim)
    normalized_entropy = entropy / np.log2(probs.shape[dim])
    return normalized_entropy


def print_metrics_table_panop_ssc(pq_stats, ssc_metrics, model):
    # Overall panoptic + semantic table
    print("=====================================")
    print(
        "method, P, R, IoU, mIoU, All PQ dagger, All PQ, All SQ, All RQ, Thing PQ, Thing SQ, Thing RQ, Stuff PQ, Stuff SQ, Stuff RQ"
    )
    for i in range(len(pq_stats)):
        ssc_stats = ssc_metrics[i].get_stats()
        panop_results = model.panoptic_metrics(pq_stats[i])
        if i == len(pq_stats) - 1:
            row_name = "ensemble"
        else:
            row_name = "subnet {}".format(i)

        print(
            "{}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}".format(
                row_name,
                ssc_stats["precision"] * 100,
                ssc_stats["recall"] * 100,
                ssc_stats["iou"] * 100,
                ssc_stats["iou_ssc_mean"] * 100,
                panop_results["All"]["pq_dagger"] * 100,
                panop_results["All"]["pq"] * 100,
                panop_results["All"]["sq"] * 100,
                panop_results["All"]["rq"] * 100,
                panop_results["Things"]["pq"] * 100,
                panop_results["Things"]["sq"] * 100,
                panop_results["Things"]["rq"] * 100,
                panop_results["Stuff"]["pq"] * 100,
                panop_results["Stuff"]["sq"] * 100,
                panop_results["Stuff"]["rq"] * 100,
            )
        )


def print_metrics_table_uncertainty(uncertainty_metrics, ssc_metrics, model):
    # Uncertainty table
    print("=====================================")
    # print("method, " + \
    #       "mask ece, mask auprc, mask auroc, mask fpr95, " + \
    #       "ins ece, ins auprc, ins auroc, ins fpr95, " + \
    #       "ssc ece, ssc auprc, ssc auroc, ssc fpr95" + ", count, inference time")
    print(
        "method, "
        + "ins ece, ins nll, "
        + "ssc nonempty ece, ssc empty ece, ssc nonempty nll, ssc empty nll, "
        + " count, inference time"
    )
    for i in range(len(uncertainty_metrics)):
        if i == len(uncertainty_metrics) - 1:
            row_name = "ensemble"
        else:
            row_name = "subnet {}".format(i)
        uncertainty_stats = uncertainty_metrics[i].get_stats()
        ssc_stats = ssc_metrics[i].get_stats()
        print(
            "{},  {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {}, {:0.2f}".format(
                row_name,
                uncertainty_stats["ins_ece"],
                uncertainty_stats["ins_nll"],
                ssc_stats["nonempty_ece"],
                ssc_stats["empty_ece"],
                ssc_stats["nonempty_nll"],
                ssc_stats["empty_nll"],
                uncertainty_stats["count"],
                ssc_stats["inference_time"],
            )
        )


def print_metrics_table_panop_per_class(pq_stats, model):
    # Overall panoptic + semantic table

    class_names = model.class_names
    print("=====================================")
    metrics_list = ["pq", "sq", "rq"]
    for metric in metrics_list:
        print("==>", metric)
        print("method" + ", " + (", ".join(class_names[1:])))
        for i in range(len(pq_stats)):
            if i == len(pq_stats) - 1:
                row_name = "ensemble"
            else:
                row_name = "subnet {}".format(i)
            panop_results = model.panoptic_metrics(pq_stats[i])
            # Per class Panop table

            ts = []
            for i in range(1, len(class_names)):
                if i in panop_results["per_class"]:
                    ts.append(panop_results["per_class"][i][metric])
                else:
                    ts.append(0)
            print(
                row_name + ", " + (", ".join(["{:0.2f}".format(t * 100) for t in ts]))
            )


# def find_matching_indices(list_voxel_probs, list_query_probs, iou_threshold=0.2):
#     n_queries = list_voxel_probs[0].shape[0]
#     aux_mask = list_voxel_probs[1].reshape(n_queries, -1)
#     anchor_mask = list_voxel_probs[0].reshape(n_queries, -1)
#     aux_query_prob = list_query_probs[1].reshape(n_queries, -1)
#     anchor_query_prob = list_query_probs[0].reshape(n_queries, -1)


#     anchor_query_prob_norm = anchor_query_prob / anchor_query_prob.norm(dim=1)[:, None]
#     aux_query_prob_norm = aux_query_prob / aux_query_prob.norm(dim=1)[:, None]
#     class_cosine_distance = 1.0 - torch.mm(anchor_query_prob_norm, aux_query_prob_norm.transpose(0,1))

#     aux_mask = (aux_mask > 0.5).float()
#     anchor_mask = (anchor_mask > 0.5).float()
#     intersection = anchor_mask @ aux_mask.T
#     union = anchor_mask.sum(dim=1, keepdim=True) + aux_mask.sum(dim=1, keepdim=True).T - intersection
#     iou = torch.zeros_like(intersection)
#     union_nonzero_mask = union != 0
#     iou[union_nonzero_mask] = intersection[union_nonzero_mask] / union[union_nonzero_mask]
#     # import pdb;pdb.set_trace()
#     iou = iou * (iou > iou_threshold)
#     mask_cost = 1.0 - iou
#     # anchor_indices, aux_indices = linear_sum_assignment(mask_cost.cpu().numpy() * class_cosine_distance.cpu().numpy())
#     anchor_indices, aux_indices = linear_sum_assignment(mask_cost.cpu().numpy())
#     keep = iou[anchor_indices, aux_indices] > iou_threshold
#     keep = keep.cpu().numpy()
#     anchor_indices = anchor_indices[keep]
#     aux_indices = aux_indices[keep]
#     # semantic_similarity = list_query_probs[0][0] @ list_query_probs[1][0].T
#     # anchor_indices, aux_indices = linear_sum_assignment(-iou.cpu().numpy() - semantic_similarity.cpu().numpy())
#     return [anchor_indices, aux_indices]


def find_matching_indices_v2(
    anchor_voxel_prob_dense,
    anchor_query_prob,
    aux_voxel_prob_dense,
    aux_query_prob,
    iou_threshold,
):
    n_queries = anchor_voxel_prob_dense.shape[0]
    aux_mask = aux_voxel_prob_dense.reshape(n_queries, -1)
    anchor_mask = anchor_voxel_prob_dense.reshape(n_queries, -1)
    aux_query_prob = aux_query_prob.reshape(n_queries, -1)
    anchor_query_prob = anchor_query_prob.reshape(n_queries, -1)

    anchor_query_prob_norm = anchor_query_prob / anchor_query_prob.norm(dim=1)[:, None]
    aux_query_prob_norm = aux_query_prob / aux_query_prob.norm(dim=1)[:, None]
    class_cosine_distance = 1.0 - torch.mm(
        anchor_query_prob_norm, aux_query_prob_norm.transpose(0, 1)
    )

    # print("hard matching")
    # aux_mask = (aux_mask > 0.5).float()
    # anchor_mask = (anchor_mask > 0.5).float()

    intersection = anchor_mask @ aux_mask.T
    union = (
        anchor_mask.sum(dim=1, keepdim=True)
        + aux_mask.sum(dim=1, keepdim=True).T
        - intersection
    )
    iou = torch.zeros_like(intersection)
    union_nonzero_mask = union != 0
    iou[union_nonzero_mask] = (
        intersection[union_nonzero_mask] / union[union_nonzero_mask]
    )
    # import pdb;pdb.set_trace()
    iou = iou * (iou > iou_threshold)
    mask_cost = 1.0 - iou
    # anchor_indices, aux_indices = linear_sum_assignment(mask_cost.cpu().numpy() * class_cosine_distance.cpu().numpy())
    anchor_indices, aux_indices = linear_sum_assignment(mask_cost.cpu().numpy())
    # keep = iou[anchor_indices, aux_indices] > iou_threshold
    # keep = keep.cpu().numpy()
    # anchor_indices = anchor_indices[keep]
    # aux_indices = aux_indices[keep]
    # semantic_similarity = list_query_probs[0][0] @ list_query_probs[1][0].T
    # anchor_indices, aux_indices = linear_sum_assignment(-iou.cpu().numpy() - semantic_similarity.cpu().numpy())
    return anchor_indices, aux_indices, iou[anchor_indices, aux_indices]


def measure_variation(prediction_a, prediction_b):
    voxel_disagree = (
        (
            prediction_a["voxel_probs_denses"].argmax(1)
            != prediction_b["voxel_probs_denses"].argmax(1)
        )
        .float()
        .mean()
    )
    query_disagree = (
        (
            prediction_a["query_probs"].argmax(-1)
            != prediction_b["query_probs"].argmax(-1)
        )
        .float()
        .mean()
    )
    completion_disagree = (
        (
            (prediction_a["completion_prob_dense"] > 0.5)
            != (prediction_b["completion_prob_dense"] > 0.5)
        )
        .float()
        .mean()
    )

    voxel_probs_denses_normed_a = prediction_a["voxel_probs_denses"] + 1e-8
    voxel_probs_denses_normed_a = (
        voxel_probs_denses_normed_a / voxel_probs_denses_normed_a.sum(1, keepdim=True)
    )
    voxel_probs_denses_normed_b = prediction_b["voxel_probs_denses"] + 1e-8
    voxel_probs_denses_normed_b = (
        voxel_probs_denses_normed_b / voxel_probs_denses_normed_b.sum(1, keepdim=True)
    )

    voxel_kl = (
        F.kl_div(
            voxel_probs_denses_normed_a.log(),
            voxel_probs_denses_normed_b,
            reduction="none",
        )
        .sum(1)
        .mean()
    )
    completion_probs_a = torch.cat(
        [
            (1.0 - prediction_a["completion_prob_dense"]),
            prediction_a["completion_prob_dense"],
        ],
        1,
    )
    completion_probs_b = torch.cat(
        [
            (1.0 - prediction_b["completion_prob_dense"]),
            prediction_b["completion_prob_dense"],
        ],
        1,
    )
    completion_kl = (
        F.kl_div(completion_probs_a.log(), completion_probs_b, reduction="none")
        .sum(1)
        .mean()
    )
    query_kl = (
        F.kl_div(
            prediction_a["query_probs"].log(),
            prediction_b["query_probs"],
            reduction="none",
        )
        .sum(-1)
        .mean()
    )
    return {
        "voxel_disagree": voxel_disagree,
        "query_disagree": query_disagree,
        "completion_disagree": completion_disagree,
        "voxel_kl": voxel_kl,
        "completion_kl": completion_kl,
        "query_kl": query_kl,
    }


def to_sparse_coo(data):
    # An intuitive way to extract coordinates and features
    coords, feats = [], []
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            for k, val in enumerate(val):
                if val != 0:
                    coords.append([i, j, k])
                    feats.append([val])
    return torch.IntTensor(coords), torch.FloatTensor(feats)


def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, "last_checkpoint.pytorch")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, "best_checkpoint.pytorch")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    model_key="model_state_dict",
    optimizer_key="optimizer_state_dict",
):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def save_network_output(output_path, output, logger=None):
    if logger is not None:
        logger.info(f"Saving network output to: {output_path}...")
    output = output.detach().cpu()[0]
    with h5py.File(output_path, "w") as f:
        f.create_dataset("predictions", data=output, compression="gzip")


loggers = {}


def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger("PatchFinder")
    in_channels = model.in_channels

    patch_shapes = [
        (64, 128, 128),
        (96, 128, 128),
        (64, 160, 160),
        (96, 160, 160),
        (64, 192, 192),
        (96, 192, 192),
    ]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype("float")

        patch = torch.from_numpy(patch).view((1, in_channels) + patch.shape).to(device)

        logger.info(f"Current patch size: {shape}")
        model(patch)


def remove_halo(patch, index, shape, patch_halo):
    """
    Remove `pad_width` voxels around the edges of a given patch.
    """
    assert len(patch_halo) == 3

    def _new_slices(slicing, max_size, pad):
        if slicing.start == 0:
            p_start = 0
            i_start = 0
        else:
            p_start = pad
            i_start = slicing.start + pad

        if slicing.stop == max_size:
            p_stop = None
            i_stop = max_size
        else:
            p_stop = -pad if pad != 0 else 1
            i_stop = slicing.stop - pad

        return slice(p_start, p_stop), slice(i_start, i_stop)

    D, H, W = shape

    i_c, i_z, i_y, i_x = index
    p_c = slice(0, patch.shape[0])

    p_z, i_z = _new_slices(i_z, D, patch_halo[0])
    p_y, i_y = _new_slices(i_y, H, patch_halo[1])
    p_x, i_x = _new_slices(i_x, W, patch_halo[2])

    patch_index = (p_c, p_z, p_y, p_x)
    index = (i_c, i_z, i_y, i_x)
    return patch[patch_index], index


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2**k for k in range(num_levels)]


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, name, batch):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert (
                img.ndim == 2 or img.ndim == 3
            ), "Only 2D (HW) and 3D (CHW) images are accepted for display"

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert (
                    C == 1 or C == 3
                ), "Only (1, H, W) or (3, H, W) images are supported"

            return tag, img

        tagged_images = self.process_batch(name, batch)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch):
        raise NotImplementedError


class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, skip_last_target=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_last_target = skip_last_target

    def process_batch(self, name, batch):
        if name == "targets" and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tag_template = "{}/batch_{}/channel_{}/slice_{}"

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))


def _find_masks(batch, min_size=10):
    """Center the z-slice in the 'middle' of a given instance, given a batch of instances

    Args:
        batch (ndarray): 5d numpy tensor (NCDHW)
    """
    result = []
    for b in batch:
        assert b.shape[0] == 1
        patch = b[0]
        z_sum = patch.sum(axis=(1, 2))
        coords = np.where(z_sum > min_size)[0]
        if len(coords) > 0:
            ind = coords[len(coords) // 2]
            result.append(b[:, ind : ind + 1, ...])
        else:
            ind = b.shape[1] // 2
            result.append(b[:, ind : ind + 1, ...])

    return np.stack(result, axis=0)


def get_tensorboard_formatter(formatter_config):
    if formatter_config is None:
        return DefaultTensorboardFormatter()

    class_name = formatter_config["name"]
    m = importlib.import_module("pytorch3dunet.unet3d.utils")
    clazz = getattr(m, class_name)
    return clazz(**formatter_config)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


def create_optimizer(optimizer_config, model):
    learning_rate = optimizer_config["learning_rate"]
    weight_decay = optimizer_config.get("weight_decay", 0)
    betas = tuple(optimizer_config.get("betas", (0.9, 0.999)))
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay
    )
    return optimizer


def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    class_name = lr_config.pop("name")
    m = importlib.import_module("torch.optim.lr_scheduler")
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config["optimizer"] = optimizer
    return clazz(**lr_config)


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported dataset class: {class_name}")


def batch_sparse_tensor(list_tensors):
    max_len = max([t.shape[0] for t in list_tensors])
    batch_F = torch.zeros(
        len(list_tensors), max_len, list_tensors[0].F.shape[1]
    ).type_as(list_tensors[0].F)
    batch_C = torch.zeros(
        len(list_tensors), max_len, list_tensors[0].C.shape[1]
    ).type_as(list_tensors[0].C)
    for i, t in enumerate(list_tensors):
        batch_F[i, : t.shape[0]] = t.F
        batch_C[i, : t.shape[0]] = t.C
    return batch_F, batch_C
