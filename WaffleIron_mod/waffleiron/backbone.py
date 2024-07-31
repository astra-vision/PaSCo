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
import torch.nn as nn
from torch import autocast


def build_proj_matrix(indices_non_zeros, occupied_cell, batch_size, num_2d_cells):
    num_points = indices_non_zeros.shape[1] // batch_size
    matrix_shape = (batch_size, num_2d_cells, num_points)

    # Sparse projection matrix for Inflate step
    inflate = torch.sparse_coo_tensor(
        indices_non_zeros, occupied_cell.reshape(-1), matrix_shape
    ).transpose(1, 2)

    # Count number of points in each cells (used in flatten step)
    with autocast("cuda", enabled=False):
        num_points_per_cells = torch.bmm( inflate, torch.bmm(inflate.transpose(1, 2), occupied_cell.unsqueeze(-1)))


    # Sparse projection matrix for Flatten step (projection & average in each 2d cells)
    weight_per_point = 1.0 / (num_points_per_cells.reshape(-1) + 1e-6)
    weight_per_point *= occupied_cell.reshape(-1)
    flatten = torch.sparse_coo_tensor(indices_non_zeros, weight_per_point, matrix_shape)

    return {"flatten": flatten, "inflate": inflate}


class ChannelMix(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1),
        )
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  # Implement LayerScale

    def forward(self, tokens):
        """tokens <- tokens + LayerScale( MLP( BN(tokens) ) )"""
        return tokens + self.scale(self.mlp(self.norm(tokens)))


class SpatialMix(nn.Module):
    def __init__(self, channels, grid_shape):
        super().__init__()
        self.H, self.W = grid_shape
        self.norm = nn.BatchNorm1d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        )
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  # Implement LayerScale
        self.grid_shape = grid_shape

    def extra_repr(self):
        return f"(grid): [{self.grid_shape[0]}, {self.grid_shape[1]}]"

    def forward(self, tokens, sp_mat):
        """tokens <- tokens + LayerScale( Inflate( FFN( Flatten( BN(tokens) ) ) )"""
        B, C, N = tokens.shape
        residual = self.norm(tokens)
        # Flatten
        with autocast("cuda", enabled=False):
            residual = torch.bmm(
                sp_mat["flatten"], residual.transpose(1, 2).float()
            ).transpose(1, 2)
        residual = residual.reshape(B, C, self.H, self.W)
        # FFN
        residual = self.ffn(residual)
        # Inflate
        residual = residual.reshape(B, C, self.H * self.W)
        with autocast("cuda", enabled=False):
            residual = torch.bmm(
                sp_mat["inflate"], residual.transpose(1, 2).float()
            ).transpose(1, 2)
        residual = residual.reshape(B, C, N)
        return tokens + self.scale(residual)


class WaffleIron(nn.Module):
    def __init__(self, channels, depth, grids_shape):
        super().__init__()
        self.grids_shape = grids_shape
        self.channel_mix = nn.ModuleList([ChannelMix(channels) for _ in range(depth)])
        self.spatial_mix = nn.ModuleList(
            [
                SpatialMix(channels, grids_shape[d % len(grids_shape)])
                for d in range(depth)
            ]
        )

    def forward(self, tokens, cell_ind, occupied_cell):

        # Build projection matrices
        batch_size, num_points = tokens.shape[0], tokens.shape[-1]
        point_ind = (
            torch.arange(num_points, device=tokens.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(1, -1)
        )
        batch_ind = (
            torch.arange(batch_size, device=tokens.device)
            .unsqueeze(1)
            .expand(-1, num_points)
            .reshape(1, -1)
        )
        non_zeros_ind = []
        for i in range(cell_ind.shape[1]):
            non_zeros_ind.append(
                torch.cat((batch_ind, cell_ind[:, i].reshape(1, -1), point_ind), axis=0)
            )
        sp_mat = [
            build_proj_matrix(id, occupied_cell, batch_size, np.prod(sh))
            for id, sh in zip(non_zeros_ind, self.grids_shape)
        ]

        # Actual backbone
        for d, (smix, cmix) in enumerate(zip(self.spatial_mix, self.channel_mix)):
            tokens = smix(tokens, sp_mat[d % len(sp_mat)])
            tokens = cmix(tokens)

        return tokens
