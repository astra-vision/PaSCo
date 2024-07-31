from torch.utils.data.dataloader import DataLoader
from pasco.data.kitti360.kitti360_dataset import Kitti360Dataset
import pytorch_lightning as pl
from pasco.data.semantic_kitti.collate import collate_fn_simple
from pasco.utils.torch_util import worker_init_fn


class Kitti360DataModule(pl.LightningDataModule):
    def __init__(
        self,
        kitti360_root,
        kitti360_preprocess_root,
        kitti360_label_root,
        batch_size=4,
        num_workers=6,
        overfit=False,
        data_aug=True,
        val_aug=True,
        max_angle=5.0,
        scale_range=0,
        translate_distance=0.2,
        max_val_items=None,
        n_fuse_scans=1,
        n_subnets=1,
        complete_scale=8,
    ):
        super().__init__()
        self.max_val_items = max_val_items
        self.kitti360_root = kitti360_root
        self.kitti360_preprocess_root = kitti360_preprocess_root
        self.kitti360_label_root = kitti360_label_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.overfit = overfit
        self.data_aug = data_aug
        self.max_angle = max_angle
        self.scale_range = scale_range
        self.translate_distance = translate_distance
        self.n_fuse_scans = n_fuse_scans
        self.n_subnets = n_subnets
        self.complete_scale = complete_scale
        self.val_aug = val_aug


    def setup_val_loader_visualization(self, frame_ids, max_items=None, data_aug=True):

        self.val_ds = Kitti360Dataset(
            split="val",
            kitti360_root=self.kitti360_root,
            kitti360_preprocess_root=self.kitti360_preprocess_root,
            kitti360_label_root=self.kitti360_label_root,
            overfit=self.overfit,
            data_aug=data_aug,
            visualize=True,
            max_items=max_items,
            max_angle=self.max_angle,
            frame_ids=frame_ids, 
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans = self.n_fuse_scans,
            n_subnets = self.n_subnets,
            complete_scale=self.complete_scale
        )


    def setup_val_loader(self, visualize=False, max_items=None, data_aug=True):

        self.val_ds = Kitti360Dataset(
            split="val",
            kitti360_root=self.kitti360_root,
            kitti360_preprocess_root=self.kitti360_preprocess_root,
            kitti360_label_root=self.kitti360_label_root,
            overfit=self.overfit,
            data_aug=data_aug,
            visualize=visualize,
            max_items=max_items,
            max_angle=self.max_angle,
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans = self.n_fuse_scans,
            n_subnets = self.n_subnets,
            complete_scale=self.complete_scale
        )

    def setup_test_loader(self, visualize=False, max_items=None, data_aug=True):

        self.test_ds = Kitti360Dataset(
            split="test",
            kitti360_root=self.kitti360_root,
            kitti360_preprocess_root=self.kitti360_preprocess_root,
            kitti360_label_root=self.kitti360_label_root,
            overfit=self.overfit,
            data_aug=data_aug,
            visualize=visualize,
            max_items=max_items,
            max_angle=self.max_angle,
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans = self.n_fuse_scans,
            n_subnets = self.n_subnets,
            complete_scale=self.complete_scale
        )
        
    def setup_train_loader(self, visualize=False, max_items=None, data_aug=True):

        self.train_ds = Kitti360Dataset(
            split="train",
            kitti360_root=self.kitti360_root,
            kitti360_preprocess_root=self.kitti360_preprocess_root,
            kitti360_label_root=self.kitti360_label_root,
            overfit=self.overfit,
            data_aug=data_aug,
            visualize=visualize,
            max_items=max_items,
            max_angle=self.max_angle,
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans = self.n_fuse_scans,
            n_subnets = self.n_subnets,
            complete_scale=self.complete_scale
        )

   

    def setup(self, stage=None):
        self.train_ds = Kitti360Dataset(
            split="train",
            kitti360_root=self.kitti360_root,
            kitti360_preprocess_root=self.kitti360_preprocess_root,
            kitti360_label_root=self.kitti360_label_root,
            overfit=self.overfit,
            data_aug=self.data_aug,
            max_angle=self.max_angle,
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans = self.n_fuse_scans,
            n_subnets = self.n_subnets,
            complete_scale=self.complete_scale
        )

        self.val_ds = Kitti360Dataset(
            split="val",
            kitti360_root=self.kitti360_root,
            kitti360_preprocess_root=self.kitti360_preprocess_root,
            kitti360_label_root=self.kitti360_label_root,
            overfit=self.overfit,
            data_aug=self.val_aug,
            max_angle=self.max_angle,
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans = self.n_fuse_scans,
            n_subnets = self.n_subnets,
            complete_scale=self.complete_scale
        )

        self.test_ds = Kitti360Dataset(
            split="test",
            kitti360_root=self.kitti360_root,
            kitti360_preprocess_root=self.kitti360_preprocess_root,
            kitti360_label_root=self.kitti360_label_root,
            overfit=self.overfit,
            data_aug=self.val_aug,
            max_angle=self.max_angle,
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans = self.n_fuse_scans,
            n_subnets = self.n_subnets,
            complete_scale=self.complete_scale
        )

      

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            # drop_last=True,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            # collate_fn=collate_fn,
            collate_fn=collate_fn_simple
        )

    def val_dataloader(self):
        
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            # collate_fn=collate_fn,
            collate_fn=collate_fn_simple
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            # collate_fn=collate_fn,
            collate_fn=collate_fn_simple
        )