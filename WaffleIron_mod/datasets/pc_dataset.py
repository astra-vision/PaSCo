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
import utils.transforms as tr
from torch.utils.data import Dataset
from scipy.spatial import cKDTree as KDTree


class PCDataset(Dataset):
    def __init__(
        self,
        rootdir=None,
        phase="train",
        input_feat="intensity",
        voxel_size=0.1,
        train_augmentations=None,
        dim_proj=[
            0,
        ],
        grids_shape=[(256, 256)],
        fov_xyz=(
            (
                -1.0,
                -1.0,
                -1.0,
            ),
            (1.0, 1.0, 1.0),
        ),
        num_neighbors=16,
        tta=False,
        instance_cutmix=False,
    ):
        super().__init__()

        # Dataset split
        self.phase = phase
        assert self.phase in ["train", "val", "trainval", "test"]

        # Root directory of dataset
        self.rootdir = rootdir

        # Input features to compute for each point
        self.input_feat = input_feat

        # Downsample input point cloud by small voxelization
        self.downsample = tr.Voxelize(
            dims=(0, 1, 2),
            voxel_size=voxel_size,
            random=(self.phase == "train" or self.phase == "trainval"),
        )

        # Field of view
        assert len(fov_xyz[0]) == len(
            fov_xyz[1]
        ), "Min and Max FOV must have the same length."
        for i, (min, max) in enumerate(zip(*fov_xyz)):
            assert (
                min < max
            ), f"Field of view: min ({min}) < max ({max}) is expected on dimension {i}."
        self.fov_xyz = np.concatenate([np.array(f)[None] for f in fov_xyz], axis=0)
        self.crop_to_fov = tr.Crop(dims=(0, 1, 2), fov=fov_xyz)

        # Grid shape for projection in 2D
        assert len(grids_shape) == len(dim_proj)
        self.dim_proj = dim_proj
        self.grids_shape = [np.array(g) for g in grids_shape]
        self.lut_axis_plane = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

        # Number of neighbors for embedding layer
        assert num_neighbors > 0
        self.num_neighbors = num_neighbors

        # Test time augmentation
        if tta:
            assert self.phase in ["test", "val"]
            self.tta = tr.Compose(
                (
                    tr.Rotation(inplace=True, dim=2),
                    tr.RandomApply(tr.FlipXY(inplace=True), prob=2.0 / 3.0),
                    tr.Scale(inplace=True, dims=(0, 1, 2), range=0.1),
                )
            )
        else:
            self.tta = None

        # Train time augmentations
        if train_augmentations is not None:
            assert self.phase in ["train", "trainval"]
        self.train_augmentations = train_augmentations

        # Flag for instance cutmix
        self.instance_cutmix = instance_cutmix

    def get_occupied_2d_cells(self, pc):
        """Return mapping between 3D point and corresponding 2D cell"""
        cell_ind = []
        for dim, grid in zip(self.dim_proj, self.grids_shape):
            # Get plane of which to project
            dims = self.lut_axis_plane[dim]
            # Compute grid resolution
            res = (self.fov_xyz[1, dims] - self.fov_xyz[0, dims]) / grid[None]
            # Shift and quantize point cloud
            pc_quant = ((pc[:, dims] - self.fov_xyz[0, dims]) / res).astype("int")
            # Check that the point cloud fits on the grid
            min, max = pc_quant.min(0), pc_quant.max(0)
            assert min[0] >= 0 and min[1] >= 0, print(
                "Some points are outside the FOV:", pc[:, :3].min(0), self.fov_xyz
            )
            assert max[0] < grid[0] and max[1] < grid[1], print(
                "Some points are outside the FOV:", pc[:, :3].min(0), self.fov_xyz
            )
            # Transform quantized coordinates to cell indices for projection on 2D plane
            temp = pc_quant[:, 0] * grid[1] + pc_quant[:, 1]
            cell_ind.append(temp[None])
        return np.vstack(cell_ind)

    def prepare_input_features(self, pc_orig):
        # Concatenate desired input features to coordinates
        pc = [pc_orig[:, :3]]  # Initialize with coordinates
        for type in self.input_feat:
            if type == "intensity":
                pc.append(pc_orig[:, 3:])
            elif type == "height":
                pc.append(pc_orig[:, 2:3])
            elif type == "radius":
                r_xyz = np.linalg.norm(pc_orig[:, :3], axis=1, keepdims=True)
                pc.append(r_xyz)
            elif type == "xyz":
                pc.append(pc_orig[:, :3])
            else:
                raise ValueError(f"Unknown feature: {type}")
        return np.concatenate(pc, 1)

    def load_pc(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        # Load original point cloud
        pc_orig, labels_orig, filename = self.load_pc(index)

        # Prepare input feature
        pc_orig = self.prepare_input_features(pc_orig)
        pc_orig_no_tta = pc_orig.copy()

        # Test time augmentation
        if self.tta is not None:
            pc_orig, labels_orig = self.tta(pc_orig, labels_orig)

        # Voxelization
        pc, labels = self.downsample(pc_orig, labels_orig)

        # Augment data
        if self.train_augmentations is not None:
            pc, labels = self.train_augmentations(pc, labels)

        # Crop to fov
        pc, labels = self.crop_to_fov(pc, labels)

        # For each point, get index of corresponding 2D cells on projected grid
        cell_ind = self.get_occupied_2d_cells(pc)
       

        # Get neighbors for point embedding layer providing tokens to waffleiron backbone
        kdtree = KDTree(pc[:, :3])
        assert pc.shape[0] > self.num_neighbors
        _, neighbors_emb = kdtree.query(pc[:, :3], k=self.num_neighbors + 1)

        # Nearest neighbor interpolation to undo cropping & voxelisation at validation time
        if self.phase in ["train", "trainval"]:
            upsample = np.arange(pc.shape[0])
        else:
            _, upsample = kdtree.query(pc_orig[:, :3], k=1)

        # Output to return
        out = (
            pc_orig_no_tta[:, :4],
            # Point coordinates
            pc[:, :3],
            # Point features
            pc[:, 3:].T[None],
            # Point labels of original entire point cloud
            labels if self.phase in ["train", "trainval"] else labels_orig,
            # Projection 2D -> 3D: index of 2D cells for each point
            cell_ind[None],
            # Neighborhood for point embedding layer, which provides tokens to waffleiron backbone
            neighbors_emb.T[None],
            # For interpolation from voxelized & cropped point cloud to original point cloud
            upsample,
            # Filename of original point cloud
            filename,
        )

        return out


def zero_pad(feat, neighbors_emb, cell_ind, Nmax):
    N = feat.shape[-1]
    assert N <= Nmax
    occupied_cells = np.ones((1, Nmax))
    if N < Nmax:
        # Zero-pad with null features
        feat = np.concatenate((feat, np.zeros((1, feat.shape[1], Nmax - N))), axis=2)
        # For zero-padded points, associate last zero-padded points as neighbor
        neighbors_emb = np.concatenate(
            (
                neighbors_emb,
                (Nmax - 1) * np.ones((1, neighbors_emb.shape[1], Nmax - N)),
            ),
            axis=2,
        )
        # Associate zero-padded points to first 2D cell...
        cell_ind = np.concatenate(
            (cell_ind, np.zeros((1, cell_ind.shape[1], Nmax - N))), axis=2
        )
        # ... and at the same time mark zero-padded points as unoccupied
        occupied_cells[:, N:] = 0
    return feat, neighbors_emb, cell_ind, occupied_cells


class Collate:
    def __init__(self, num_points=None):
        self.num_points = num_points
        assert num_points is None or num_points > 0

    def __call__(self, list_data):

        # Extract all data
        list_of_data = (list(data) for data in zip(*list_data))
        pc_orig, coords, feat, label_orig, cell_ind, neighbors_emb, upsample, filename = list_of_data

        # Zero-pad point clouds
        Nmax = np.max([f.shape[-1] for f in feat])
        if self.num_points is not None:
            assert Nmax <= self.num_points
        occupied_cells = []
        for i in range(len(feat)):
            feat[i], neighbors_emb[i], cell_ind[i], temp = zero_pad(
                feat[i],
                neighbors_emb[i],
                cell_ind[i],
                Nmax if self.num_points is None else self.num_points,
            )
            occupied_cells.append(temp)

        # Concatenate along batch dimension
        feat = torch.from_numpy(np.vstack(feat)).float()  # B x C x Nmax
        neighbors_emb = torch.from_numpy(np.vstack(neighbors_emb)).long()  # B x Nmax
        cell_ind = torch.from_numpy(
            np.vstack(cell_ind)
        ).long()  # B x nb_2d_cells x Nmax
        occupied_cells = torch.from_numpy(np.vstack(occupied_cells)).float()  # B x Nmax
        labels_orig = torch.from_numpy(np.hstack(label_orig)).long()
        upsample = [torch.from_numpy(u) for u in upsample]

        # Prepare output variables
        out = {
            "pc_orig": pc_orig,
            "coords": coords,
            "feat": feat,
            "neighbors_emb": neighbors_emb,
            "upsample": upsample,
            "labels_orig": labels_orig,
            "cell_ind": cell_ind,
            "occupied_cells": occupied_cells,
            "filename": filename,
        }

        return out
