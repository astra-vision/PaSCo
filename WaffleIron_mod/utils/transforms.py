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


import os
import torch
import numpy as np
from glob import glob


class Compose:
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, pcloud, labels):
        for t in self.transformations:
            pcloud, labels = t(pcloud, labels)
        return pcloud, labels


class RandomApply:
    def __init__(self, transformation, prob=0.5):
        self.prob = prob
        self.transformation = transformation

    def __call__(self, pcloud, labels):
        if torch.rand(1) < self.prob:
            pcloud, labels = self.transformation(pcloud, labels)
        return pcloud, labels


class Transformation:
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, pcloud, labels):
        if labels is None:
            return pcloud if self.inplace else np.array(pcloud, copy=True)

        out = (
            (pcloud, labels)
            if self.inplace
            else (np.array(pcloud, copy=True), np.array(labels, copy=True))
        )
        return out


class Identity(Transformation):
    def __init__(self, inplace=False):
        super().__init__(inplace)

    def __call__(self, pcloud, labels):
        return super().__call__(pcloud, labels)


class Rotation(Transformation):
    def __init__(self, dim=2, range=np.pi, inplace=False):
        super().__init__(inplace)
        self.range = range
        self.inplace = inplace
        if dim == 2:
            self.dims = (0, 1)
        elif dim == 1:
            self.dims = (0, 2)
        elif dim == 0:
            self.dims = (1, 2)

    def __call__(self, pcloud, labels):
        # Build rotation matrix
        theta = (2 * torch.rand(1)[0] - 1) * self.range
        # Build rotation matrix
        rot = np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        # Apply rotation
        pcloud, labels = super().__call__(pcloud, labels)
        pcloud[:, self.dims] = pcloud[:, self.dims] @ rot
        return pcloud, labels


class Scale(Transformation):
    def __init__(self, dims=(0, 1), range=0.05, inplace=False):
        super().__init__(inplace)
        self.dims = dims
        self.range = range

    def __call__(self, pcloud, labels):
        pcloud, labels = super().__call__(pcloud, labels)
        scale = 1 + (2 * torch.rand(1).item() - 1) * self.range
        pcloud[:, self.dims] *= scale
        return pcloud, labels


class FlipXY(Transformation):
    def __init__(self, inplace=False):
        super().__init__(inplace=inplace)

    def __call__(self, pcloud, labels):
        pcloud, labels = super().__call__(pcloud, labels)
        id = torch.randint(2, (1,))[0]
        pcloud[:, id] *= -1.0
        return pcloud, labels


class LimitNumPoints(Transformation):
    def __init__(self, dims=(0, 1, 2), max_point=30000, random=False):
        super().__init__(inplace=True)
        self.dims = dims
        self.max_points = max_point
        self.random = random
        assert max_point > 0

    def __call__(self, pcloud, labels):
        pc, labels = super().__call__(pcloud, labels)
        if pc.shape[0] > self.max_points:
            if self.random:
                center = torch.randint(pc.shape[0], (1,))[0]
                center = pc[center : center + 1, self.dims]
            else:
                center = np.zeros((1, len(self.dims)))
            idx = np.argsort(np.square(pc[:, self.dims] - center).sum(axis=1))[
                : self.max_points
            ]
            pc, labels = pc[idx], labels[idx]
        return pc, labels


class Crop(Transformation):
    def __init__(self, dims=(0, 1, 2), fov=((-5, -5, -5), (5, 5, 5)), eps=1e-4):
        super().__init__(inplace=True)
        self.dims = dims
        self.fov = fov
        self.eps = eps
        assert len(fov[0]) == len(fov[1]), "Min and Max FOV must have the same length."
        for i, (min, max) in enumerate(zip(*fov)):
            assert (
                min < max
            ), f"Field of view: min ({min}) < max ({max}) is expected on dimension {i}."

    def __call__(self, pcloud, labels):
        pc, labels = super().__call__(pcloud, labels)

        where = None
        for i, d in enumerate(self.dims):  # Actually a bug below, use d in pc not i!
            temp = (pc[:, d] > self.fov[0][i] + self.eps) & (
                pc[:, d] < self.fov[1][i] - self.eps
            )
            where = temp if where is None else where & temp

        return pc[where], labels[where]


class Voxelize(Transformation):
    def __init__(self, dims=(0, 1, 2), voxel_size=0.1, random=False):
        super().__init__(inplace=True)
        self.dims = dims
        self.voxel_size = voxel_size
        self.random = random
        assert voxel_size >= 0

    def __call__(self, pcloud, labels):
        pc, labels = super().__call__(pcloud, labels)
        if self.voxel_size <= 0:
            return pc, labels

        if self.random:
            permute = torch.randperm(pc.shape[0])
            pc, labels = pc[permute], labels[permute]

        pc_shift = pc[:, self.dims] - pc[:, self.dims].min(0, keepdims=True)

        _, ind = np.unique(
            (pc_shift / self.voxel_size).astype("int"), return_index=True, axis=0
        )

        return pc[ind, :], None if labels is None else labels[ind]


class InstanceCutMix(Transformation):
    def __init__(self, phase="train"):
        """Instance cutmix coded only for SemanticKITTI"""
        super().__init__(inplace=True)

        raise ValueError("Include latest verion")

        self.phase = phase
        self.rootdir = "/root/local_storage/semantic_kitti_instance_" + self.phase
        self.bank = {1: [], 2: [], 5: [], 6: [], 7: []}
        for key in self.bank.keys():
            self.bank[key] = glob(os.path.join(self.rootdir, f"{key}", "*.bin"))
        self.loaded = self.test_loaded()
        # v2
        self.rot = Compose(
            (
                FlipXY(inplace=True),
                Rotation(inplace=True),
                Scale(dims=(0, 1, 2), range=0.1, inplace=True),
            )
        )
        self.nb_to_add = 40
        self.vox = Voxelize(dims=(0, 1, 2), voxel_size=1.0, random=True)
        """ v1
        self.rot = Rotation(inplace=False)
        self.max_size = 100 # Unused
        self.nb_to_add = 20
        self.vox = Voxelize(dims=(0, 1, 2), voxel_size=.1, random=True)
        """

    def test_loaded(self):
        if self.phase == "train":
            if len(self.bank[1]) != 5083:
                print(len(self.bank[1]), 5083)
                return False
            if len(self.bank[2]) != 3092:
                print(len(self.bank[2]), 3092)
                return False
            if len(self.bank[5]) != 8084:
                print(len(self.bank[5]), 8084)
                return False
            if len(self.bank[6]) != 1551:
                print(len(self.bank[6]), 1551)
                return False
            if len(self.bank[7]) != 560:
                print(len(self.bank[7]), 560)
                return False
        elif self.phase == "trainval":
            if len(self.bank[1]) != 8213:
                print(len(self.bank[1]), 8213)
                return False
            if len(self.bank[2]) != 4169:
                print(len(self.bank[2]), 4169)
                return False
            if len(self.bank[5]) != 12190:
                print(len(self.bank[5]), 12190)
                return False
            if len(self.bank[6]) != 2943:
                print(len(self.bank[6]), 2943)
                return False
            if len(self.bank[7]) != 701:
                print(len(self.bank[7]), 701)
                return False
        return True

    def add_in_bank(self, pc, class_label, instance_label):
        for id_class in self.bank.keys():
            where_class = class_label == id_class
            all_instances = np.unique(instance_label[where_class])
            for id_instance in all_instances:
                # Segment instance
                where_ins = instance_label == id_instance
                if where_ins.sum() <= 5:
                    continue
                pc_to_add = pc[where_ins, :]
                # Center instance
                pc_to_add[:, :2] -= pc_to_add[:, :2].mean(0, keepdims=True)
                pc_to_add[:, 2] -= pc_to_add[:, 2].min(0, keepdims=True)
                #
                pathfile = os.path.join(
                    self.rootdir, f"{id_class}", f"{len(self.bank[id_class]):07d}.bin"
                )
                os.makedirs(os.path.join(self.rootdir, f"{id_class}"), exist_ok=True)
                pc_to_add.tofile(pathfile)
                self.bank[id_class].append(pathfile)

    def add_in_pc(self, pc, class_label):
        new_pc = [pc]
        new_label = [class_label]
        # Find location where to add new object (on a surface)
        pc_vox, class_label_vox = self.vox(pc, class_label)

        # v2
        where_surface = np.where((class_label_vox >= 8) & (class_label_vox <= 10))[0]

        """ v1
        where_surface = np.where( ( (class_label_vox>=8) & (class_label_vox<=11) ) | (class_label_vox==16) )[0]
        """

        where_surface = where_surface[torch.randperm(len(where_surface))]
        id_tot = 0
        for id_class in self.bank.keys():
            which_one = torch.randint(len(self.bank[id_class]), (self.nb_to_add,))
            for ii in range(self.nb_to_add):
                p = pc_vox[where_surface[id_tot]]
                object = self.bank[id_class][which_one[ii]]
                object = np.fromfile(object, dtype=np.float32).reshape((-1, 4))
                object, _ = self.rot(object, 1)
                object[:, :3] += p[:3][None]
                new_pc.append(object)
                new_label.append(np.ones((object.shape[0],), dtype=np.int) * id_class)
                id_tot += 1
        return np.concatenate(new_pc, 0), np.concatenate(new_label, 0)

    def __call__(self, pc, class_label, instance_label):
        if not self.loaded:
            self.add_in_bank(pc, class_label, instance_label)
            return np.zeros((2, 4)), None
        return self.add_in_pc(pc, class_label)
