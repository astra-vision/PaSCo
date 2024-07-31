from torch.utils.data.dataloader import DataLoader
from pasco.data.semantic_kitti.kitti_dataset import KittiDataset
import pytorch_lightning as pl
from pasco.data.semantic_kitti.collate import collate_fn_simple
from pasco.utils.torch_util import worker_init_fn


class KittiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        preprocess_root,
        config_path,
        batch_size=4,
        num_workers=6,
        data_aug=True,
        val_aug=True,
        max_angle=5.0,
        scale_range=0,
        translate_distance=0.2,
        max_val_items=None,
        n_fuse_scans=1,
        n_subnets=1,
        complete_scale=8,
        frame_interval=5,
    ):
        super().__init__()
        self.root = root
        self.max_val_items = max_val_items
        self.preprocess_root = preprocess_root
        self.config_path = config_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_aug = data_aug
        self.max_angle = max_angle
        self.scale_range = scale_range
        self.translate_distance = translate_distance
        self.n_fuse_scans = n_fuse_scans
        self.n_subnets = n_subnets
        self.complete_scale = complete_scale
        self.val_aug = val_aug
        self.frame_interval = frame_interval

    def setup_val_loader_visualization(self, frame_ids, data_aug=True, max_items=None):

        self.val_ds = KittiDataset(
            split="val",
            root=self.root,
            config_path=self.config_path,
            preprocess_root=self.preprocess_root,
            data_aug=data_aug,
            visualize=True,
            max_items=max_items,
            max_angle=self.max_angle,
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans=self.n_fuse_scans,
            n_subnets=self.n_subnets,
            complete_scale=self.complete_scale,
            frame_ids=frame_ids,
            frame_interval=self.frame_interval,
        )

    def setup_val_loader(self, visualize=False, max_items=None, data_aug=True):

        self.val_ds = KittiDataset(
            split="val",
            root=self.root,
            config_path=self.config_path,
            preprocess_root=self.preprocess_root,
            data_aug=data_aug,
            visualize=visualize,
            max_items=max_items,
            max_angle=self.max_angle,
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans=self.n_fuse_scans,
            n_subnets=self.n_subnets,
            complete_scale=self.complete_scale,
        )

    def setup_train_loader(self, visualize=False, max_items=None, data_aug=True):

        self.train_ds = KittiDataset(
            split="train",
            root=self.root,
            config_path=self.config_path,
            preprocess_root=self.preprocess_root,
            data_aug=data_aug,
            visualize=visualize,
            max_items=max_items,
            max_angle=self.max_angle,
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans=self.n_fuse_scans,
            n_subnets=self.n_subnets,
            complete_scale=self.complete_scale,
        )

    def setup(self, stage=None):
        self.train_ds = KittiDataset(
            split="train",
            root=self.root,
            config_path=self.config_path,
            preprocess_root=self.preprocess_root,
            data_aug=self.data_aug,
            max_angle=self.max_angle,
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans=self.n_fuse_scans,
            n_subnets=self.n_subnets,
            complete_scale=self.complete_scale,
        )

        self.val_ds = KittiDataset(
            split="val",
            root=self.root,
            config_path=self.config_path,
            preprocess_root=self.preprocess_root,
            data_aug=self.val_aug,
            max_angle=self.max_angle,
            scale_range=self.scale_range,
            translate_distance=self.translate_distance,
            n_fuse_scans=self.n_fuse_scans,
            n_subnets=self.n_subnets,
            complete_scale=self.complete_scale,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn_simple,
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
            collate_fn=collate_fn_simple,
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
            collate_fn=collate_fn_simple,
        )
