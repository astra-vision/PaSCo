# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import numpy as np
from .lovasz import lovasz_softmax_flat
from torch.nn.functional import softmax
from torch.nn import Module, CrossEntropyLoss


def fast_hist(pred, label, n):
    assert torch.all(label > -1) & torch.all(pred > -1)
    assert torch.all(label < n) & torch.all(pred < n)
    return torch.bincount(n * label + pred, minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def overall_accuracy(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist).sum() / hist.sum()


def per_class_accuracy(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist) / hist.sum(1)


class SemSegLoss(Module):
    def __init__(self, nb_class, lovasz_weight=1.0, ignore_index=255):
        super().__init__()
        self.nb_class = nb_class
        self.ignore_index = ignore_index
        self.lovasz_weight = lovasz_weight
        self.ce = CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, pred, true):
        loss = self.ce(pred, true)

        if self.lovasz_weight > 0:
            where = true != self.ignore_index
            if where.sum() > 0:
                loss += self.lovasz_weight * lovasz_softmax_flat(
                    softmax(pred[where], dim=1),
                    true[where],
                )

        return loss
