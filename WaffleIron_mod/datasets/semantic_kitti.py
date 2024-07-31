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
import yaml
import torch
import warnings
import numpy as np
from glob import glob
from tqdm import tqdm
import utils.transforms as tr
from .pc_dataset import PCDataset


class InstanceCutMix:
    def __init__(self, phase="train", temp_dir="/tmp/semantic_kitti_instances/"):

        # Train or Trainval
        self.phase = phase
        assert self.phase in ["train", "trainval"]

        # List of files containing instances for bicycle, motorcycle, person, bicyclist
        self.bank = {1: [], 2: [], 5: [], 6: []}

        # Directory where to store instances
        self.rootdir = os.path.join(temp_dir, self.phase)
        for id_class in self.bank.keys():
            os.makedirs(os.path.join(self.rootdir, f"{id_class}"), exist_ok=True)

        # Load instances
        for key in self.bank.keys():
            self.bank[key] = glob(os.path.join(self.rootdir, f"{key}", "*.bin"))
        self.__loaded__ = self.test_loaded()
        if not self.__loaded__:
            warnings.warn(
                "Instances must be extracted and saved on disk before training"
            )

        # Augmentations applied on Instances
        self.rot = tr.Compose(
            (
                tr.FlipXY(inplace=True),
                tr.Rotation(inplace=True),
                tr.Scale(dims=(0, 1, 2), range=0.1, inplace=True),
            )
        )

        # For each class, maximum number of instance to add
        self.num_to_add = 40

        # Voxelization of 1m to downsample point cloud to ensure that
        # center of the instances are at least 1m away
        self.vox = tr.Voxelize(dims=(0, 1, 2), voxel_size=1.0, random=True)

    def test_loaded(self):
        self.__loaded__ = False
        if self.phase == "train":
            if len(self.bank[1]) != 5083:
                print(f"Expected 5083 instances but got {len(self.bank[1])}.")
                return False
            if len(self.bank[2]) != 3092:
                print(f"Expected 3092 instances but got {len(self.bank[2])}.")
                return False
            if len(self.bank[5]) != 8084:
                print(f"Expected 8084 instances but got {len(self.bank[5])}.")
                return False
            if len(self.bank[6]) != 1551:
                print(f"Expected 1551 instances but got {len(self.bank[6])}.")
                return False
        elif self.phase == "trainval":
            if len(self.bank[1]) != 8213:
                print(f"Expected 8213 instances but got {len(self.bank[1])}.")
                return False
            if len(self.bank[2]) != 4169:
                print(f"Expected 4169 instances but got {len(self.bank[2])}.")
                return False
            if len(self.bank[5]) != 12190:
                print(f"Expected 12190 instances but got {len(self.bank[5])}.")
                return False
            if len(self.bank[6]) != 2943:
                print(f"Expected 2943 instances but got {len(self.bank[6])}.")
                return False
        self.__loaded__ = True
        return True

    def cut(self, pc, class_label, instance_label):
        for id_class in self.bank.keys():
            where_class = (class_label == id_class)
            all_instances = np.unique(instance_label[where_class])
            for id_instance in all_instances:
                # Segment instance
                where_ins = (instance_label == id_instance)
                if where_ins.sum() <= 5: continue
                instance = pc[where_ins, :]
                # Center instance
                instance[:, :2] -= instance[:, :2].mean(0, keepdims=True)
                instance[:, 2] -= instance[:, 2].min(0, keepdims=True)
                # Save instance
                pathfile = os.path.join(
                    self.rootdir, f"{id_class}", f"{len(self.bank[id_class]):07d}.bin"
                )
                instance.tofile(pathfile)
                self.bank[id_class].append(pathfile)

    def mix(self, pc, class_label):

        # Find potential location where to add new object (on a surface)
        pc_vox, class_label_vox = self.vox(pc, class_label)
        where_surface = np.where((class_label_vox >= 8) & (class_label_vox <= 10))[0]
        where_surface = where_surface[torch.randperm(len(where_surface))]

        # Add instances of each class in bank
        id_tot = 0
        new_pc, new_label  = [pc], [class_label]
        for id_class in self.bank.keys():
            nb_to_add = torch.randint(self.num_to_add, (1,))[0]
            which_one = torch.randint(len(self.bank[id_class]), (nb_to_add,))
            for ii in range(nb_to_add):
                # Point p where to add the instance
                p = pc_vox[where_surface[id_tot]]
                # Extract instance
                object = self.bank[id_class][which_one[ii]]
                object = np.fromfile(object, dtype=np.float32).reshape((-1, 4))
                # Augment instance
                label = np.ones((object.shape[0],), dtype=np.int) * id_class
                object, label = self.rot(object, label)
                # Move instance at point p
                object[:, :3] += p[:3][None]
                # Add instance in the point cloud
                new_pc.append(object)
                # Add corresponding label
                new_label.append(label)
                id_tot += 1

        return np.concatenate(new_pc, 0), np.concatenate(new_label, 0)

    def __call__(self, pc, class_label, instance_label):
        if not self.__loaded__:
            self.cut(pc, class_label, instance_label)
            return None, None

        return self.mix(pc, class_label)


class SemanticKITTI(PCDataset):

    CLASS_NAME = [
        "car",              # 0
        "bicycle",          # 1
        "motorcycle",       # 2
        "truck",            # 3
        "other-vehicle",    # 4
        "person",           # 5
        "bicyclist",        # 6
        "motorcyclist",     # 7
        "road",             # 8
        "parking",          # 9
        "sidewalk",         # 10
        "other-ground",     # 11
        "building",         # 12
        "fence",            # 13
        "vegetation",       # 14
        "trunk",            # 15
        "terrain",          # 16
        "pole",             # 17
        "traffic-sign",     # 18
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Config file and class mapping
        current_folder = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_folder, "semantic-kitti.yaml")) as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml["learning_map"]

        # Split
        if self.phase == "train":
            split = semkittiyaml["split"]["train"]
        elif self.phase == "val":
            split = semkittiyaml["split"]["valid"] + semkittiyaml["split"]["train"]
        elif self.phase == "test":
            split = semkittiyaml["split"]["test"]
        elif self.phase == "trainval":
            split = semkittiyaml["split"]["train"] 
        else:
            raise Exception(f"Unknown split {self.phase}")

        # Find all files
        self.im_idx = []
        for i_folder in np.sort(split):
            self.im_idx.extend(
                glob(
                    os.path.join(
                        self.rootdir,
                        "dataset",
                        "sequences",
                        str(i_folder).zfill(2),
                        "velodyne",
                        "*.bin",
                    )
                )
            )
        self.im_idx = np.sort(self.im_idx)
        filtered = []
        for path in self.im_idx:
            frame_id = int(path.split("/")[-1].split(".")[0])
            if frame_id % 5 == 0:
                filtered.append(path)
            # if frame_id % 1 == 0:
            #     filtered.append(path)
        self.im_idx = filtered
        # self.im_idx = filtered[:10]

        # Training with instance cutmix
        if self.instance_cutmix:
            assert (
                self.phase != "test" and self.phase != "val"
            ), "Instance cutmix should not be applied at test or val time"
            self.cutmix = InstanceCutMix(phase=self.phase)
            if not self.cutmix.test_loaded():
                print("Extracting instances before training...")
                for index in tqdm(range(len(self))):
                    self.load_pc(index)
                print("Done.")
            assert self.cutmix.test_loaded(), "Instances not extracted correctly"

    def __len__(self):
        return len(self.im_idx)

    def load_pc(self, index):
        # Load point cloud
        pc = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))

        # Extract Label
        if self.phase == "test":
            labels = np.zeros((pc.shape[0], 1), dtype=np.uint8)
        else:
            labels_inst = np.fromfile(
                self.im_idx[index].replace("velodyne", "labels")[:-3] + "label",
                dtype=np.uint32,
            ).reshape((-1, 1))
            labels = labels_inst & 0xFFFF  # delete high 16 digits binary
            labels = np.vectorize(self.learning_map.__getitem__)(labels).astype(
                np.int32
            )

        # Map ignore index (0) to 255
        labels = labels[:, 0] - 1
        labels[labels == -1] = 255

        # Instance CutMix
        if self.instance_cutmix:
            pc, labels = self.cutmix(pc, labels, labels_inst[:, 0])

        return pc, labels, self.im_idx[index]
