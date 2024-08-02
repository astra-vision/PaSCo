import pdb

import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import pickle
import torch.nn.functional as F
from pasco.models.augmenter import Augmenter
from pasco.models.misc import compute_scene_size
from pasco.models.transform_utils import transform_scene, transform, generate_random_transformation
from pasco.data.kitti360.params import thing_ids
from pasco.data.kitti360.collate import collate_fn


class Kitti360Dataset(Dataset):
    def __init__(
        self,
        split,
        kitti360_root,
        kitti360_preprocess_root,
        kitti360_label_root,
        n_subnets=2,
        data_aug=False,
        max_angle=90.0,
        translate_distance=0.2,
        scale_range=0,
        overfit=False,
        visualize=False,
        max_items=None,
        n_fuse_scans=1,
        complete_scale=8,
        frame_ids=[]
    ):
        super().__init__()
        self.kitti360_root = kitti360_root
        self.kitti360_preprocess_root = kitti360_preprocess_root
        self.kitti360_label_root = kitti360_label_root

        self.n_subnets = n_subnets
        self.complete_scale = complete_scale
        self.data_aug = data_aug
        self.max_angle = max_angle
        self.translate_distance = translate_distance
        self.scale_range = scale_range
        # self.max_translation = np.array([0.6, 0.6, 0.4])
        self.max_translation = np.array([3.0, 3.0, 2.0]) * translate_distance
        print(self.data_aug, self.max_angle, self.scale_range, self.translate_distance)

        self.instance_label_root = os.path.join(self.kitti360_preprocess_root, "instance_labels_v2")
        self.label_root = os.path.join(self.kitti360_label_root, "labels")

        self.overfit = overfit
        self.n_classes = 19
        self.max_extent = (51.2, 25.6, 4.4)
        self.min_extent = np.array([0, -25.6, -2.0])
        self.augmenter = Augmenter()
        self.n_fuse_scans = n_fuse_scans
        
        splits = {
            "train": ["2013_05_28_drive_0004_sync", "2013_05_28_drive_0000_sync", 
                      "2013_05_28_drive_0010_sync","2013_05_28_drive_0002_sync", 
                      "2013_05_28_drive_0003_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0007_sync"],
            "val": ["2013_05_28_drive_0006_sync"],
            "test": ["2013_05_28_drive_0009_sync"],
        }
            
        self.split = split
        self.sequences = splits[split]
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.voxel_size = 0.2  # 0.2m
        self.grid_size = [int(i/self.voxel_size) for i in self.scene_size]
        self.thing_ids = thing_ids
        
        id_map = self.get_match_id()
        

        self.scans = []
        for sequence in self.sequences:
            glob_path = os.path.join(self.label_root, sequence, "*_1_1.npy")
            
            
            # frame_ids = ["007960"]
            
            for label_path in glob.glob(glob_path):
                filename = os.path.basename(label_path)
                frame_id = os.path.splitext(filename)[0][:6]
                
               
                
                if len(frame_ids) > 0 and visualize and frame_id not in frame_ids:
                    continue
                
                self.scans.append(
                    {
                        "sequence": sequence,
                        "frame_id": frame_id,
                        "original_id": id_map[sequence][frame_id],
                    }
                )

        self.scans = self.scans[:max_items]
        

    def __getitem__(self, idx):
        if self.split != "train":
            idx_list = [idx] * self.n_subnets
        else:
            idx_list = np.random.choice(len(self.scans), self.n_subnets - 1, replace=False)
            idx_list = idx_list.tolist() + [idx]
        if self.overfit:
            idx_list = [0] * self.n_subnets
        items = []
        for id in idx_list:
            items.append(self.get_individual(id))
        return collate_fn(items, self.complete_scale)
    
        
    def get_individual(self, idx):
      
        scales = [1, 2, 4]

        scan = self.scans[idx]
        sequence = scan['sequence']
        frame_id = scan['frame_id']
        original_id = scan['original_id']
        
        
        # s1 = time.time()
        data = self.load_data_v3(sequence, frame_id, original_id, n_fuse_scans=self.n_fuse_scans)
        # s2 = time.time()
        # print("Load data", s2 - s1)
        
        
        semantic_label_sparse_F, semantic_label_sparse_C = data['semantic_label_sparse']
        instance_label_sparse_F, instance_label_sparse_C = data['instance_label_sparse']
        semantic_label_origin, instance_label_origin = data['semantic_label_origin'], data['instance_label_origin']
        in_feat, in_coord = data['in_feat']
        T = data['T']
        
        
        min_C_semantic = semantic_label_sparse_C.min(dim=0)[0]
        max_C_semantic = semantic_label_sparse_C.max(dim=0)[0]
        
        if instance_label_sparse_C.shape[0] > 0:
            min_C_instance = instance_label_sparse_C.min(dim=0)[0]
            max_C_instance = instance_label_sparse_C.max(dim=0)[0]
            min_C = torch.min(min_C_semantic, min_C_instance)
            max_C = torch.max(max_C_semantic, max_C_instance)
        else: 
            min_C = min_C_semantic
            max_C = max_C_semantic
        min_C = torch.floor(min_C.float() / self.complete_scale) * self.complete_scale
        min_C = min_C.int()
        max_C = torch.ceil(max_C)
        
        scene_size = compute_scene_size(min_C, max_C, scale=self.complete_scale).int()

        semantic_label = torch.full((scene_size[0], scene_size[1], scene_size[2]), 255, dtype=torch.uint8)
        nonnegative_coords = semantic_label_sparse_C - min_C
        nonnegative_coords = nonnegative_coords.long()
        semantic_label[nonnegative_coords[:, 0], nonnegative_coords[:, 1], nonnegative_coords[:, 2]] = semantic_label_sparse_F.squeeze()
        
        instance_label = torch.full((scene_size[0], scene_size[1], scene_size[2]), 0, dtype=torch.uint8)
        if instance_label_sparse_C.shape[0] > 0:
            nonnegative_coords = instance_label_sparse_C - min_C
            nonnegative_coords = nonnegative_coords.long()
            instance_label[nonnegative_coords[:, 0], nonnegative_coords[:, 1], nonnegative_coords[:, 2]] = instance_label_sparse_F.squeeze()

        
        mask_label = self.prepare_mask_label(semantic_label, instance_label)
        mask_label_origin = self.prepare_mask_label(semantic_label_origin, instance_label_origin)
        
        # complete_voxel = np.copy(semantic_label)
        complete_voxel = semantic_label.clone().float()
        complete_voxel[(semantic_label > 0) & (semantic_label != 255)] = 1
        complete_voxel_remove_255 = complete_voxel.clone()
        complete_voxel_remove_255[semantic_label == 255] = 0 # ignore 255
        
        
        scales = [1, 2, 4]
        geo_labels = {}
        sem_labels = {}
        
        temp = semantic_label.clone().long()
        temp[temp == 255] = self.n_classes
        sem_label_oh = F.one_hot(temp, num_classes=self.n_classes + 1).permute(3, 0, 1, 2).float()
        for scale in scales:
            if scale == 1:
                downscaled_label = complete_voxel
                downscaled_sem_label = semantic_label
            else:
                downscaled_label = F.max_pool3d(complete_voxel_remove_255.unsqueeze(0).unsqueeze(0), kernel_size=scale, stride=scale).squeeze(0).squeeze(0)
                downscaled_mask_255 = F.avg_pool3d(complete_voxel.unsqueeze(0).unsqueeze(0), kernel_size=scale, stride=scale).squeeze(0).squeeze(0)
                downscaled_label[downscaled_mask_255 == 255] = 255
                
                sem_label_oh_occ = sem_label_oh.clone()
                sem_label_oh_occ[0, :, :, :] = 0
                sem_label_oh_occ[self.n_classes, :, :, :] = 0
                downscaled_sem_label = F.avg_pool3d(sem_label_oh_occ.unsqueeze(0), kernel_size=scale, stride=scale).squeeze(0)
                downscaled_sem_label = torch.argmax(downscaled_sem_label, dim=0)
                
                sem_label_oh_0_255 = sem_label_oh.clone()
                sem_label_oh_0_255[1:self.n_classes, :, :, :] = 0
                downscaled_sem_label_0_255_oh = F.avg_pool3d(sem_label_oh_0_255.unsqueeze(0), kernel_size=scale, stride=scale).squeeze(0)
                downscaled_sem_label_0_255 = torch.full_like(downscaled_sem_label, 255)
                downscaled_sem_label_0_255[downscaled_sem_label_0_255_oh[self.n_classes, :, :, :] == 1] = 0
                
                empty_mask = downscaled_sem_label == 0
                downscaled_sem_label[empty_mask] = downscaled_sem_label_0_255[empty_mask]
                
            
            sem_labels['1_{}'.format(scale)] = downscaled_sem_label.type(torch.uint8)
            geo_labels['1_{}'.format(scale)] = downscaled_label.type(torch.uint8)
        
        
            # items.append(item)
        
        ret_data = {
            "xyz": data['xyz'],
            "frame_id": frame_id,
            "sequence": sequence,
            
            "in_feat": in_feat.float(),
            "in_coord": in_coord,
            
            "T": T,
            
            "min_C": min_C,
            "max_C": max_C,
            
            "semantic_label": semantic_label,
            "instance_label": instance_label,
            "mask_label": mask_label,
            "geo_labels": geo_labels,
            "sem_labels": sem_labels,
            
            "semantic_label_origin": semantic_label_origin,
            "instance_label_origin": instance_label_origin,
            "mask_label_origin": mask_label_origin,

            
            
        }
        
        # np.save("label.npy", semantic_label.numpy())
        
        return ret_data
    
    def load_file(self, path):
        # data_seg_feats = os.path.join("/lustre/fsn1/projects/rech/kvd/uyl37fq/monoscene_preprocess/kitti/waffleiron/sequences", sequence, "seg_feats_tta", "{}.pkl".format(frame_id))
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
            # import pdb; pdb.set_trace()
            
            embedding = data['embedding']
            random_embedding_idx = np.random.randint(0, embedding.shape[0])
            embedding = embedding[random_embedding_idx].T
            
            xyz_and_density = data['coords']
            xyz = xyz_and_density[:, :3]
            vote = data['vote']
            intensity = xyz_and_density[:, 3:]
            
        return xyz, vote, intensity, embedding
    
    def voxelize(self, xyz, vote):
        vox_origin = self.vox_origin.reshape(1, 3)
        coords = (xyz - vox_origin.reshape(1, 3)) // self.voxel_size

        voxel_centers = (coords.astype(float) + 0.5) * self.voxel_size + vox_origin.reshape(1, 3)
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)
        return return_xyz, vote, coords
    

    def load_input_pcd_instance_label(self, sequence, frame_id):
        path = os.path.join(self.root, "dataset", "sequences", sequence, "labels", "{}.label".format(frame_id))
        instance_label = np.fromfile(path, dtype=np.int32).reshape((-1, 1))
        instance_label = instance_label & 0xFFFF  # delete high 16 digits binary
        return instance_label


    def load_data_v3(self, sequence, frame_id, original_id, downsample=1, n_fuse_scans=1):
        data_path = os.path.join(self.instance_label_root, sequence, "{}_1_{}.pkl".format(frame_id, downsample))

        with open(data_path, 'rb') as handle:
            data = pickle.load(handle)
            semantic_label = data['semantic_labels'].astype(np.uint8)
            instance_label = data['instance_labels'].astype(np.uint8)
           
        
        pc_path = os.path.join(self.kitti360_root, "data_3d_raw",  sequence, "velodyne_points/data", "{:010d}.bin".format(int(original_id)))
        pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))
        xyz = pc[:, :3]
        intensity = pc[:, 3:]
        keep = (xyz[:, 0] < self.max_extent[0]) & (xyz[:, 0] >= self.min_extent[0]) & \
               (xyz[:, 1] < self.max_extent[1]) & (xyz[:, 1] >= self.min_extent[1]) & \
               (xyz[:, 2] < self.max_extent[2]) & (xyz[:, 2] >= self.min_extent[2])
        # keep = keep & (semantic_label != 255)
        xyz = xyz[keep]
        intensity = intensity[keep]

 


        if self.data_aug:
            T = generate_random_transformation(max_angle=self.max_angle,
                                               flip=True,
                                               scale_range=self.scale_range,
                                               max_translation=self.max_translation)
                                               
        else:
            T = torch.eye(4)
        


        
        
        semantic_label = torch.from_numpy(semantic_label)
        semantic_label_origin = semantic_label.clone()
        semantic_coords = torch.nonzero(semantic_label != 255)
        semantic_label_sparse, semantic_coords, to_coords_bnd = transform_scene(semantic_coords, T, semantic_label.unsqueeze(0) + 1)
        non_zero = semantic_label_sparse.sum(dim=1) != 0
        semantic_label_sparse = semantic_label_sparse[non_zero]
        semantic_label_sparse -= 1
        semantic_coords = semantic_coords[non_zero]
    
        
        instance_label = torch.from_numpy(instance_label)
        instance_label_origin = instance_label.clone()
        instance_coords = torch.nonzero(instance_label)
        if instance_coords.shape[0] > 0:
            instance_label_sparse, instance_coords, _ = transform_scene(instance_coords, T, instance_label.unsqueeze(0) + 1, to_coords_bnd=to_coords_bnd)
        else:
            instance_label_sparse = torch.zeros((0, 1), dtype=torch.uint8)
            instance_coords = torch.zeros((0, 3)).long()
        non_zero = instance_label_sparse.sum(dim=1) != 0
        instance_label_sparse = instance_label_sparse[non_zero]
        instance_coords = instance_coords[non_zero]
        instance_label_sparse -= 1


        # xyz = transform_xyz(torch.from_numpy(xyz), T).numpy()
        radius = np.linalg.norm(xyz, axis=1)[..., np.newaxis]
        feat = np.concatenate((intensity, radius), axis=1)

        return_xyz, feat, coords = self.voxelize(xyz, feat)
        in_feat = np.concatenate([feat, return_xyz], axis=1)
        in_coords = torch.from_numpy(coords)
        in_coords = transform(in_coords, T)
        in_coords = in_coords.long()
        in_feat = torch.from_numpy(in_feat)
        
        # min_coords = semantic_coords.min(dim=0)[0]
        # max_coords = semantic_coords.max(dim=0)[0]
        # semantic_coords -= min_coords
        # instance_coords -= min_coords
        # in_coords -= min_coords
        xyz = xyz - self.vox_origin.reshape(1, 3)

        return {
            "xyz": xyz,
            "in_feat": (in_feat, in_coords),
            "semantic_label_sparse": (semantic_label_sparse.type(torch.uint8), semantic_coords),
            "instance_label_sparse": (instance_label_sparse.type(torch.uint8), instance_coords),
            "semantic_label_origin": semantic_label_origin.type(torch.uint8),
            "instance_label_origin": instance_label_origin.type(torch.uint8),
            "T": T,
        }
    
    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        # for seq in range(0, 22):
        for seq in range(0, 11):
            seq_folder = os.path.join(self.root, "dataset", "sequences", str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(os.path.join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(os.path.join(seq_folder, 'times.txt'), dtype=float))

            # Read poses
            poses_f64 = self.parse_poses(os.path.join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(float) for pose in poses_f64])


    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    @staticmethod
    def prepare_target(target: torch.Tensor, ignore_labels: list[int]) -> dict:
        # z, y, x = target.shape
        unique_ids = torch.unique(target)
        unique_ids = torch.tensor([unique_id for unique_id in unique_ids if unique_id not in ignore_labels])
        masks = []
        
        for id in unique_ids:
            masks.append(target == id)
        
        masks = torch.stack(masks)
        
        return {
            "labels": unique_ids,
            "masks": masks
        }
    
    def prepare_mask_label(self, semantic_label, instance_label):
        mask_semantic_label = self.prepare_target(semantic_label, ignore_labels=[0, 255]) # NOTE: remove empty class
    
        stuff_filtered_mask = [t not in self.thing_ids for t in mask_semantic_label['labels']]
        stuff_semantic_labels = mask_semantic_label['labels'][stuff_filtered_mask]
        stuff_semantic_masks = mask_semantic_label['masks'][stuff_filtered_mask]
        labels = [stuff_semantic_labels]
        masks = [stuff_semantic_masks]

        mask_instance_label = self.prepare_instance_target(
            semantic_target=semantic_label, 
            instance_target=instance_label, 
            ignore_label=0) # The empty class is already included in the stuff_semantic_labels

        
        if mask_instance_label is not None: # there are thing objects
            labels.append(mask_instance_label['labels'])
            masks.append(mask_instance_label['masks'])
        
        mask_label = {
            "labels": torch.cat(labels, dim=0),
            "masks": torch.cat(masks, dim=0)
        }
        
        return mask_label
    
    
    @staticmethod
    def prepare_instance_target(semantic_target: torch.Tensor, instance_target: torch.Tensor, ignore_label: int) -> dict:
        # z, y, x = target.shape
        unique_instance_ids = torch.unique(instance_target)
        
        unique_instance_ids = unique_instance_ids[unique_instance_ids != ignore_label]
        masks = []
        semantic_labels = []

        
        for id in unique_instance_ids:
            masks.append(instance_target == id)
            semantic_labels.append(semantic_target[instance_target == id][0])

        
        if len(masks) == 0:
            return None

        masks = torch.stack(masks)
        semantic_labels = torch.tensor(semantic_labels)

        return {
            # "instance_ids": unique_instance_ids,
            "labels": semantic_labels,
            "masks": masks
        }


        # for targets_per_scene in targets:
        #     # pad gt
        #     gt_masks = targets_per_image.gt_masks
        #     padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
        #     padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
        #     new_targets.append(
        #         {
        #             "labels": targets_per_image.gt_classes,
        #             "masks": padded_masks,
        #         }
        #     )
        # return new_targets



  

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out


    def get_match_id(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''
        # Define a dictionary to store the data
        data_dict = {}
        # file_path = "./uncertainty/data/kitti360/kitti_360_match.txt"
        file_path = "/gpfswork/rech/kvd/uyl37fq/code/uncertainty/uncertainty/data/kitti360/kitti_360_match.txt"
        # Open the file for reading
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()  # Split the line into parts based on spaces
                if len(parts) == 3:
                    sequence, id1, id2 = parts  # Assign the values to variables
                    # Check if the sequence already exists in the dictionary
                    id1 = id1.rsplit(".", 1)[0]
                    id2 = id2.rsplit(".", 1)[0]
                    if sequence in data_dict:
                        data_dict[sequence][id2] = id1
                    else:
                        data_dict[sequence] = {id2: id1}

        return data_dict

def count_label(sem_label):
    unique, counts = np.unique(sem_label[sem_label != 255], return_counts=True)
    return unique.astype(int), counts
if __name__ == "__main__":
    kitti360_root="/gpfsdswork/dataset/KITTI-360"
    kitti360_preprocess_root="/lustre/fsn1/projects/rech/kvd/uyl37fq/monoscene_preprocess/kitti360"
    kitti360_label_root="/gpfsdswork/dataset/SSCBench-KITTI-360"

    dataset = Kitti360Dataset("train", 
        kitti360_root=kitti360_root,
        kitti360_preprocess_root=kitti360_preprocess_root,
        kitti360_label_root=kitti360_label_root,
        overfit=False,
        n_subnets=1,
        data_aug=False)
    
    
    # item = dataset[2200]
    # import pdb; pdb.set_trace()
    cnts_1_1 = np.zeros(19).astype(np.int64)
    cnts_1_2 = np.zeros(19).astype(np.int64)
    cnts_1_4 = np.zeros(19).astype(np.int64)
    iters = 0
    for item in tqdm(dataset):
        sem_labels_1_1 = item['sem_labels']["1_1"][0].cpu().numpy().astype(np.uint8)
        sem_labels_1_2 = item['sem_labels']["1_2"][0].cpu().numpy().astype(np.uint8)
        sem_labels_1_4 = item['sem_labels']["1_4"][0].cpu().numpy().astype(np.uint8)
        unique_1_1, counts_1_1 = count_label(sem_labels_1_1)
        unique_1_2, counts_1_2 = count_label(sem_labels_1_2)
        unique_1_4, counts_1_4 = count_label(sem_labels_1_4)
        cnts_1_1[unique_1_1] += counts_1_1
        cnts_1_2[unique_1_2] += counts_1_2
        cnts_1_4[unique_1_4] += counts_1_4
        iters += 1
        # if iters == 10:
        #     break
    print("1_1", ', '.join(str(x) for x in cnts_1_1))
    print("1_2", ', '.join(str(x) for x in cnts_1_2))
    print("1_4", ', '.join(str(x) for x in cnts_1_4))
    # print("1_1:", cnts_1_1)
    # print("1_2:", cnts_1_2)
    # print("1_4:", cnts_1_4)
    # filepath = "t_aug.pkl"
    # out_dict = {
    #     # "instance_label": item['instance_label'][0].cpu().numpy().astype(np.uint16),
    #     # "semantic_label": item['semantic_label'][0].cpu().numpy().astype(np.uint8),
    #     "sem_labels_1_1": item['sem_labels']["1_1"][0].cpu().numpy().astype(np.uint8),
    #     "sem_labels_1_2": item['sem_labels']["1_2"][0].cpu().numpy().astype(np.uint8),
    #     "sem_labels_1_4": item['sem_labels']["1_4"][0].cpu().numpy().astype(np.uint8),
    #     "in_coords": item['in_coords'][0].cpu().numpy().astype(np.int16),
    #     "min_C": item['min_Cs'][0].cpu().numpy().astype(np.int16),
    # }
    # with open(filepath, "wb") as handle:
    #     pickle.dump(out_dict, handle)
    #     print("wrote to", filepath)