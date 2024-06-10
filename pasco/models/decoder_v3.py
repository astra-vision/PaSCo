# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
from time import time
from uncertainty.models.metrics import SSCMetrics

# Must be imported before large libs
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
import numpy as np

import MinkowskiEngine as ME
from uncertainty.models.misc import to_dense_tensor_batch, prune_outside_coords

from uncertainty.models.misc import compute_scene_size
from uncertainty.maskpls.mink import (
    BasicGenerativeDeconvolutionBlock,
    ResidualBlock,
)
from collections import defaultdict
from uncertainty.models.utils import batch_sparse_tensor


class TransformerInFeat(nn.Module):

    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=3, bias=False, dimension=3
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=3, bias=True, dimension=3
            ),
        )
        reduction = 4
        self.fc = nn.Sequential(
            ME.MinkowskiLinear(out_channels, out_channels // reduction),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(out_channels // reduction, out_channels),
            ME.MinkowskiSigmoid(),
        )
        self.pooling = ME.MinkowskiGlobalPooling()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        y = self.broadcast_mul(x, y)
        return self.net(y)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout_layer,
        scene_size,
        voxel_size=0.2,
        drop_path_rates=[0.0, 0.0, 0.0],
        n_heads=2,
        compl_head_dim=1,
        heavy_decoder=True,
        scale=1,
        dropout=0.0,
    ):
        nn.Module.__init__(self)
        self.scene_size = scene_size
        self.voxel_size = voxel_size

        self.upsample = BasicGenerativeDeconvolutionBlock(
            in_channels, out_channels, ks=2, stride=2
        )

        self.resize = nn.Sequential(
            ME.MinkowskiBatchNorm(out_channels + 3),
            ME.MinkowskiConvolution(
                out_channels + 3, out_channels, kernel_size=1, bias=True, dimension=3
            ),
        )
        if heavy_decoder:
            self.process = nn.Sequential(
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[0]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[1]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[2]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[3]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[4]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[5]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[6]),
                dropout_layer(p=dropout),
            )
        else:
            self.process = nn.Sequential(
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[0]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[1]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[2]),
            )

        self.num_queries = 100
        self.threshold = 0
        self.pruning = ME.MinkowskiPruning()
        self.n_heads = n_heads

        self.completion_heads = nn.ModuleDict()
        for i_head in range(n_heads):
            self.completion_heads[str(i_head)] = nn.Sequential(
                ME.MinkowskiConvolution(
                    out_channels, compl_head_dim, kernel_size=1, bias=True, dimension=3
                ),
            )

    def concat_coords(self, x, voxel_size):
        coords = x.C[:, 1:].float() / x.tensor_stride[0]
        x_F = torch.cat([x.F, coords], dim=1)
        return ME.SparseTensor(
            x_F,
            x.C,
            tensor_stride=x.tensor_stride,
            coordinate_manager=x.coordinate_manager,
        )

    def forward(self, x, shortcut, global_min_coords, global_max_coords):
        dec = self.upsample(x)

        keep = (
            (dec.C[:, 1] >= global_min_coords[0])
            & (dec.C[:, 1] <= global_max_coords[0])
            & (dec.C[:, 2] >= global_min_coords[1])
            & (dec.C[:, 2] <= global_max_coords[1])
            & (dec.C[:, 3] >= global_min_coords[2])
            & (dec.C[:, 3] <= global_max_coords[2])
        )
        dec = self.pruning(dec, keep)
        dec = self.concat_coords(dec, dec.tensor_stride[0] * self.voxel_size)
        dec = self.resize(dec)

        dec = self.process(dec + shortcut)

        x = dec

        completion_logits = []
        for i_head in range(self.n_heads):
            completion_logit = self.completion_heads[str(i_head)](x)
            completion_logits.append(completion_logit)

        return x, completion_logits


class DecoderGenerativeSepConvV2(nn.Module):

    def __init__(
        self,
        f,
        n_classes,
        norm_layer,
        act_layer,
        dropout_layer,
        dropouts,
        query_dim,
        transformer_predictor,
        n_infers,
        heavy_decoder=True,
        use_se_layer=False,
        scene_size=(256, 256, 32),
        drop_path_rates=[0.0] * 9,
        num_queries=100,
    ):
        nn.Module.__init__(self)

        dec_ch = f[::-1]
        self.n_infers = n_infers
        self.num_queries = num_queries
        self.scene_size = scene_size
        self.transformer_predictor = transformer_predictor
        self.dec_blocks = nn.ModuleList()
        self.dec_panop_blocks = nn.ModuleDict()
        self.voxel_feats = nn.ModuleDict()
        self.mask_projs = nn.ModuleDict()
        self.class_projs = nn.ModuleDict()
        self.query_projs = nn.ModuleDict()
        self.completion_heads = nn.ModuleDict()
        self.injects = nn.ModuleDict()
        if n_infers <= 2:
            self.occ_thres = {
                4: 25000,
                2: 120000,
                1: 400000,
            }
        elif n_infers == 3:
            self.occ_thres = {
                4: 24000,
                2: 100000,
                1: 350000,
            }

        elif n_infers == 4:
            self.occ_thres = {
                4: 22000,
                2: 80000,
                1: 320000,
            }

        self.agg_occ_thres = {
            4: 30000,
            2: 100000,
            1: 400000,
        }

        for i in range(len(dec_ch) - 1):
            n_res_block_per_dec = 7
            scale = 2 ** (len(dec_ch) - 2 - i)

            self.dec_blocks.append(
                DecoderBlock(
                    dec_ch[i],
                    dec_ch[i + 1],
                    drop_path_rates=drop_path_rates[
                        i * n_res_block_per_dec : (i + 1) * n_res_block_per_dec
                    ],
                    n_heads=n_infers,
                    compl_head_dim=n_classes,
                    scene_size=scene_size,
                    dropout_layer=dropout_layer,
                    scale=scale,
                    heavy_decoder=heavy_decoder,
                    dropout=dropouts[i],
                ),
            )

            if scale in [1]:

                for i_infer in range(self.n_infers):
                    layer_name = "scale{}_infer{}".format(scale, i_infer)

            if scale in [1, 2, 4]:
                for i_infer in range(self.n_infers):
                    layer_name = "scale{}_infer{}".format(scale, i_infer)
                    self.voxel_feats[layer_name] = nn.Sequential(
                        ME.MinkowskiConvolution(
                            dec_ch[i + 1],
                            dec_ch[i + 1],
                            kernel_size=3,
                            bias=False,
                            dimension=3,
                        ),
                        ME.MinkowskiBatchNorm(dec_ch[i + 1]),
                        ME.MinkowskiReLU(),
                        ME.MinkowskiConvolution(
                            dec_ch[i + 1],
                            dec_ch[i + 1],
                            kernel_size=3,
                            bias=True,
                            dimension=3,
                        ),
                    )
        # pruning
        self.pruning = ME.MinkowskiPruning()
        self.sigmoid_sparse = ME.MinkowskiSigmoid()
        self.threshold = 0
        self.n_classes = 20

    def predict_completion(
        self, x, scale, occ_logits, scene_size, global_min_coords, geo_labels=None
    ):
        bs = x.C[:, 0].max() + 1
        occ_logits_F = [t.F for t in occ_logits]
        occ_logits_F = torch.cat(occ_logits_F, dim=1)
        max_occ_probs = torch.sigmoid(occ_logits_F).max(1)[0]
        keep = max_occ_probs > 0.5

        n_voxels = scene_size[0] * scene_size[1] * scene_size[2] // scale**3 * bs
        ratio = keep.sum() / n_voxels
        keep_before = keep.sum()
        if keep.sum() == 0 or ratio > self.occ_thres[scale]:
            top_voxels_indices = torch.topk(
                max_occ_probs, k=int(n_voxels * self.occ_thres[scale]), dim=0
            )[1]
            keep = torch.zeros_like(max_occ_probs).bool()
            keep[top_voxels_indices] = True

            print(
                "Scale {}".format(scale),
                "keep before is ",
                keep_before,
                "keep after is ",
                keep.sum(),
            )

        return keep

    def predict_completion_sem_logit(
        self,
        x,
        scale,
        sem_logits,
        scene_size,
        min_Cs,
        max_Cs,
        class_frequencies,
        sem_labels=None,
        test=False,
    ):
        bs = x.C[:, 0].max() + 1
        keeps = []
        for i_infer in range(self.n_infers):
            sem_logit = sem_logits[i_infer]
            min_C = min_Cs[i_infer]
            max_C = max_Cs[i_infer]
            sem_prob = F.softmax(sem_logit.F, dim=-1)
            sem_prob, sem_class = sem_prob.max(-1)
            keep = sem_class != 0
            sem_label = sem_labels["1_{}".format(scale)][
                i_infer
            ]  # not use during validation

            n_voxels = keep.sum()
            keep_before = keep.sum()

            if not test and (n_voxels > self.occ_thres[scale]):
                if (
                    (self.training and self.n_infers <= 2)
                    or self.n_infers > 3
                    or n_voxels > 500000
                ):
                    complt_num_per_class = class_frequencies["1_{}".format(scale)]
                    compl_labelweights = complt_num_per_class / np.sum(
                        complt_num_per_class
                    )
                    compl_labelweights = np.power(
                        np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0
                    )
                    compl_labelweights = torch.from_numpy(compl_labelweights).type_as(
                        sem_logits[0].F
                    )
                    occ_prob = sem_prob * compl_labelweights[sem_class]
                    top_voxels_indices = torch.multinomial(
                        occ_prob, int(self.occ_thres[scale]), replacement=False
                    )

                    keep = torch.zeros_like(keep).bool()

                    keep[top_voxels_indices] = True

                    print(
                        "Scale {}, infer {}: ".format(scale, i_infer),
                        keep_before,
                        keep.sum(),
                        ((sem_label != 0) & (sem_label != 255)).sum(),
                    )

            keeps.append(keep)
        keeps = torch.stack(keeps).sum(0)
        keep = keeps > 0
        n_keep_voxels = keep.sum()
        thres = self.agg_occ_thres[scale]
        # if self.n_infers > 3 and ((n_keep_voxels <= 2) or (n_keep_voxels > thres)):
        if self.n_infers >= 3 and ((n_keep_voxels <= 2) or (n_keep_voxels > thres)):
            if not test and self.n_infers > 2:
                top_voxels_indices = torch.topk(keeps, k=thres, dim=0)[
                    1
                ]  # avoid OOM error
                keep = torch.zeros_like(keep).bool()
                keep[top_voxels_indices] = True
                print("Overall keep", n_keep_voxels, "after", keep.sum())

        return keep

    def predict_panop(self, xs, sem_logits_at_scales, Ts, min_Cs, max_Cs, sem_labels):
        panop_predictions = []

        xs_infers = defaultdict(list)
        sem_logits_pruneds = []
        for i_infer in range(self.n_infers):
            min_C = min_Cs[i_infer]
            max_C = max_Cs[i_infer]

            for scale in xs:
                layer_name = "scale{}_infer{}".format(scale, i_infer)
                x = xs[scale]
                sem_logits = sem_logits_at_scales[scale][i_infer]
                sem_class = sem_logits.F.max(-1)[1]
                sem_label = sem_labels["1_{}".format(scale)][
                    i_infer
                ]  # not use during validation

                keep = sem_class != 0
                if keep.sum() == 0:  # keep some voxels to avoid error
                    keep = torch.zeros_like(keep).bool()
                    keep[:1000] = True
                    print("line 367: keep.sum() == 0")

                if scale == 1:
                    sem_logits_pruned = self.pruning(sem_logits, keep)
                    sem_logits_pruned = prune_outside_coords(
                        sem_logits_pruned, min_C, max_C
                    )
                    sem_logits_pruneds.append(sem_logits_pruned)
                try:
                    x_infer_pruned = self.pruning(x, keep)
                except:
                    print("Pruning error in decoder", scale, x.shape, keep.shape)
                    keep = torch.zeros_like(x.C[:, 0]).bool()
                    keep[:1000] = True
                    x_infer_pruned = self.pruning(x, keep)

                x_infer_pruned = prune_outside_coords(x_infer_pruned, min_C, max_C)
                x_infer_pruned = self.voxel_feats[layer_name](x_infer_pruned)
                xs_infers[scale].append(x_infer_pruned)

        for scale in xs_infers:
            xs_infers[scale] = batch_sparse_tensor(xs_infers[scale])

        batch_sem_logits_pruned = batch_sparse_tensor(sem_logits_pruneds)

        keep_pad = (
            (batch_sem_logits_pruned[0] != 0).sum(-1)
            + (batch_sem_logits_pruned[1] != 0).sum(-1)
        ) != 0
        panop_predictions = self.transformer_predictor(
            xs_infers, batch_sem_logits_pruned, min_Cs, max_Cs, keep_pad
        )

        return panop_predictions, sem_logits_pruneds

    def forward(
        self,
        x,
        features,
        global_min_coords,
        global_max_coords,
        min_Cs,
        max_Cs,
        class_frequencies,
        Ts,
        is_predict_panop=True,
        sem_labels=None,
        test=False,
    ):
        """
        features: [enc_s1, enc_s2, enc_s4]
        """

        features = features[::-1]
        # occ_targets_at_scales = {}
        sem_logits_at_scales = {}
        xs = {}
        for i in range(len(self.dec_blocks)):
            scale = 2 ** (len(self.dec_blocks) - 1 - i)
            x, sem_logits = self.dec_blocks[i](
                x, features[i], global_min_coords, global_max_coords
            )

            if scale in [4, 2, 1]:

                scene_size = compute_scene_size(global_min_coords, global_max_coords)
                keep = self.predict_completion_sem_logit(
                    x,
                    scale,
                    sem_logits,
                    scene_size,
                    min_Cs,
                    max_Cs,
                    class_frequencies=class_frequencies,
                    sem_labels=sem_labels,
                    test=test,
                )

                x = self.pruning(x, keep)
                sem_logits = [self.pruning(c, keep) for c in sem_logits]

                xs[scale] = x
                sem_logits_at_scales[scale] = sem_logits
        ret = {
            "sem_logits_at_scales": sem_logits_at_scales,
        }
        if is_predict_panop:
            panop_predictions, sem_logits_pruneds = self.predict_panop(
                xs, sem_logits_at_scales, Ts, min_Cs, max_Cs, sem_labels
            )
            ret["panop_predictions"] = panop_predictions
            ret["sem_logits_pruneds"] = sem_logits_pruneds

        return ret
