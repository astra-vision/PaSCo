# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import Tensor
import MinkowskiEngine as ME


def prune_outside_coords(x, min_coords, max_coords):
    prune = ME.MinkowskiPruning()
    mask = (
        (x.C[:, 1] <= max_coords[0])
        & (x.C[:, 2] <= max_coords[1])
        & (x.C[:, 3] <= max_coords[2])
        & (x.C[:, 1] >= min_coords[0])
        & (x.C[:, 2] >= min_coords[1])
        & (x.C[:, 3] >= min_coords[2])
    )
    x_pruned = prune(x, mask)
    return x_pruned


def compute_scene_size(min_coords, max_coords, scale=1):
    scene_size = (torch.ceil((max_coords - min_coords + 1) / scale)) * scale
    return scene_size.int()


def to_dense_tensor_batch(x, coords, bs, scene_size, dim=1):
    dim = x.shape[-1]
    x_dense = torch.zeros(bs, dim, scene_size[0], scene_size[1], scene_size[2]).type_as(
        x
    )
    coords = coords.long()

    x_dense[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] = x
    return x_dense


def to_dense_tensor(x, coords, scene_size, min_coords=None):
    assert len(x.shape) == 2
    if coords.shape[1] == 4:
        coords = coords[:, 1:].clone()
    x_dense = torch.zeros(
        x.shape[-1], scene_size[0], scene_size[1], scene_size[2]
    ).type_as(x)
    if min_coords is not None:
        coords -= min_coords.type_as(coords)
    coords = coords.long()
    x_dense[:, coords[:, 0], coords[:, 1], coords[:, 2]] = x.T
    return x_dense


def compute_entropy(probs):
    probs = probs / probs.sum(dim=1, keepdim=True)
    return -torch.sum(probs * torch.log(probs + 1e-6), dim=1)


def prune_outside_scene(x, scene_size):
    prune = ME.MinkowskiPruning()
    mask = (
        (x.C[:, 1] < scene_size[0])
        & (x.C[:, 2] < scene_size[1])
        & (x.C[:, 3] < scene_size[2])
        & (x.C[:, 1] >= 0)
        & (x.C[:, 2] >= 0)
        & (x.C[:, 3] >= 0)
    )
    x_pruned = prune(x, mask)
    return x_pruned


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):

    # TODO make it support different-sized images
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
    batch_shape = [len(tensor_list)] + max_size

    b, c, x, y, z = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, x, y, z), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2], : img.shape[3]].copy_(
            img
        )
        m[: img.shape[1], : img.shape[2], : img.shape[3]] = False

    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0])
        )
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1
        )
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
