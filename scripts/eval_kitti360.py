import os
from pasco.data.kitti360.kitti360_dm import Kitti360DataModule
from pytorch_lightning import Trainer
import click
from pasco.data.kitti360.params import  kitti_360_class_frequencies as class_frequencies
from pasco.models.net_panoptic_sparse_kitti360 import Net_kitti360

from pasco.utils.torch_util import set_random_seed




@click.command()
@click.option('--kitti360_root', default="/gpfsdswork/dataset/KITTI-360")
@click.option('--kitti360_preprocess_root', default="/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti360")
@click.option('--kitti360_label_root', default="/gpfsdswork/dataset/SSCBench-KITTI-360")
@click.option('--model_path')
@click.option('--n_infers', default=2, help="#subnets")
@click.option('--n_gpus', default=1, help="number of GPUs")
@click.option('--n_workers_per_gpu', default=3, help="Number of workers per GPU")
@click.option('--max_angle', default=10.0, help="")
@click.option('--translate_distance', default=0.2, help="")
@click.option('--split', default="test", help="val/test")
@click.option('--seed', default=1024, help="val/test")
def main(
    n_workers_per_gpu, n_gpus,
    n_infers, split, model_path,
    max_angle, translate_distance, seed,
    kitti360_root, kitti360_preprocess_root, kitti360_label_root):
    
    set_random_seed(seed)
   
     
    print("n_infers", n_infers)
    # Setup dataloaders
    data_module = Kitti360DataModule(
        kitti360_root=kitti360_root,
        kitti360_preprocess_root=kitti360_preprocess_root,
        kitti360_label_root=kitti360_label_root,
        batch_size=1,   
        num_workers=int(n_workers_per_gpu),
        overfit=False,
        n_fuse_scans=1,
        n_subnets=n_infers,
        
        translate_distance=translate_distance,
        max_angle=max_angle
    )
    data_module.setup_val_loader(visualize=False, max_items=None)
    data_module.setup_test_loader(visualize=False, max_items=None)
    

    print(model_path)

    trainer = Trainer(
            deterministic=False,
            devices=n_gpus,
            logger=False,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            accelerator="gpu",
        )

    model = Net_kitti360.load_from_checkpoint(model_path, class_frequencies=class_frequencies)
    model.cuda()
    model.eval()
    
    if split == "test":
        print("test")
        trainer.test(model=model, dataloaders=data_module.test_dataloader())
    elif split == "val":
        print("validation")
        trainer.test(model=model, dataloaders=data_module.val_dataloader())
    else:
        raise NotImplementedError("split {} not implemented".format(split))


if __name__ == "__main__":
    main()
