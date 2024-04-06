import numpy as np
from tqdm import tqdm
import os
import glob
import pasco.data.semantic_kitti.io_data as SemanticKittiIO
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset    
import click    
import pickle
from pasco.data.semantic_kitti.params import thing_ids


class DummyDataset(Dataset):
    """Use torch dataloader for multiprocessing"""
    def __init__(self, kitti_config, kitti_root, kitti_preprocess_root, frame_interval=5):
        self.preprocess_root = kitti_preprocess_root
        sequences = ["08", "00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
        
        self.label_paths = []
        self.scale = 1
        self.scene_size = (256//self.scale, 256//self.scale, 32//self.scale)
        
        self.thing_ids = thing_ids
        self.remap_lut = SemanticKittiIO.get_remap_lut(kitti_config)
        self.scans = []
        for sequence in sequences:
            os.path.join(kitti_preprocess_root, "instance_labels", sequence)
            sequence_path = os.path.join(kitti_root, "dataset", "sequences", sequence)
            label_paths = sorted(
                glob.glob(os.path.join(sequence_path, "voxels", "*.label"))
            )
            for label_path in label_paths:
                frame_id, extension = os.path.splitext(os.path.basename(label_path))
                if float(frame_id) % frame_interval != 0:
                    continue
                invalid_path = os.path.join(sequence_path, "voxels", frame_id + ".invalid")
                out_dir = os.path.join(kitti_preprocess_root, "instance_labels_v2", sequence)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "{}_1_{}.pkl".format(frame_id, self.scale))

                self.scans.append((frame_id, sequence, label_path, invalid_path, out_path))


    def floodfill(self, volume, mask, x, y, z, instance_id):  
        stack = [(x, y, z)]
        while len(stack) > 0:
            x, y, z = stack.pop()
            mask[x, y, z] = instance_id
            volume[x, y, z] = 0
            for x_offset in [-1, 0, 1]:
                for y_offset in [-1, 0, 1]:
                    for z_offset in [-1, 0, 1]:
                        if x_offset == 0 and y_offset == 0 and z_offset == 0:
                            continue
                        x_next = x + x_offset
                        y_next = y + y_offset
                        z_next = z + z_offset
                        if x_next < 0 or x_next > (self.scene_size[0] - 1):
                            continue
                        if y_next < 0 or y_next > (self.scene_size[1] - 1):
                            continue
                        if z_next < 0 or z_next > (self.scene_size[2] - 1):
                            continue
                        if mask[x_next, y_next, z_next] == 0 and volume[x_next, y_next, z_next] != 0:
                            stack.append((x_next, y_next, z_next))
    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        frame_id, sequence, label_path, invalid_path, out_path = self.scans[idx]
        print("Process", label_path)

        if os.path.exists(out_path):
            return np.zeros(self.scene_size)
        # frame_id, extension = os.path.splitext(os.path.basename(label_path))
        if self.scale == 1:
            LABEL = SemanticKittiIO._read_label_SemKITTI(label_path)
            
            INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_path)
            LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(
                np.float32
            )  # Remap 20 classes semanticKITTI SSC
            LABEL[
                np.isclose(INVALID, 1)
            ] = 255  # Setting to unknown all voxels marked on invalid mask...
            LABEL = LABEL.reshape(self.scene_size)
        else:
            label_dir = os.path.join(self.preprocess_root, "label", sequence)
            filename = "{}_1_{}.npy".format(frame_id, self.scale)
            label_filename = os.path.join(label_dir, filename)
            LABEL = np.load(label_filename)

        filter_sem_label = np.copy(LABEL)
        instance_id = 1
        mask = np.zeros(self.scene_size)
        for thing_id in self.thing_ids:
            # thing_id = 1
            target = np.copy(LABEL)
            target[LABEL != thing_id] = 0
            
            for x in range(target.shape[0]):
                for y in range(target.shape[1]):
                    for z in range(target.shape[2]):
                        if target[x, y, z] == 0 or mask[x, y, z] != 0:
                            continue
                        self.floodfill(target, mask, x, y, z, instance_id)
                        instance_id += 1
        # Filter too small instance
        instance_ids = np.unique(mask)
        for instance_id in instance_ids:
            count = (mask == instance_id).sum()
            if count < 8:
                filter_sem_label[mask == instance_id] = 255
                mask[mask == instance_id] = 0
                

        instance_ids = np.unique(mask)
        old_mask = np.copy(mask)
        for i, instance_id in enumerate(instance_ids):
            if instance_id == 0:
                continue
            mask[old_mask == instance_id] = i
        print("#instances", i)
        
        out_dict = {
            "instance_labels": mask,
            "semantic_labels": filter_sem_label,
        }
        with open(out_path, "wb") as handle:
            pickle.dump(out_dict, handle)
            print("wrote to", out_path)
        return mask


@click.command()
@click.option('--kitti_config', default="/gpfswork/rech/kvd/uyl37fq/code/uncertainty/uncertainty/data/semantic_kitti/semantic-kitti.yaml")
@click.option('--kitti_root', default="/gpfsdswork/dataset/SemanticKITTI", help='Semantic kitti root')
@click.option('--kitti_preprocess_root', default="/gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/kitti", help='Semantic kitti root')
@click.option('--n_process', default=10, help='number of parallel processes')
def main(kitti_config, kitti_root, kitti_preprocess_root, n_process):
    dataset = DummyDataset(
        kitti_config=kitti_config,
        kitti_root=kitti_root,
        kitti_preprocess_root=kitti_preprocess_root,
        frame_interval=5)
    dataloader = DataLoader(dataset, shuffle=True, num_workers=n_process)

    for data_dict in tqdm(dataloader):
        pass

 





if __name__ == "__main__":
    main()
