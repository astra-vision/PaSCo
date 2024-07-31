import os
from pasco.data.semantic_kitti.kitti_dm import KittiDataModule
from pytorch_lightning import Trainer
import click
from pasco.data.semantic_kitti.params import class_frequencies
from pasco.models.net_panoptic_sparse import Net
from pasco.utils.torch_util import set_random_seed


set_random_seed(42)


@click.command()
@click.option("--dataset_root", default="/gpfsdswork/dataset/SemanticKITTI")
@click.option(
    "--config_path",
    default="semantic-kitti.yaml",
)
@click.option(
    "--dataset_preprocess_root",
    default="/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti",
)
@click.option("--model_path", default="")

@click.option("--n_infers", default=1, help="batch size")
@click.option("--n_gpus", default=1, help="number of GPUs")
@click.option("--iou_threshold", default=0.2, help="number of GPUs")
@click.option("--max_angle", default=30.0, help="")
@click.option("--translate_distance", default=0.2, help="")
@click.option("--n_workers_per_gpu", default=3, help="Number of workers per GPU")
def main(
    n_workers_per_gpu,
    n_gpus,
    max_angle,
    translate_distance,
    n_infers,
    model_path,
    iou_threshold,
    dataset_root,
    dataset_preprocess_root,
    config_path,
):

    print("n_infers", n_infers)
    # Setup dataloaders
    data_module = KittiDataModule(
        root=dataset_root,
        config_path=config_path,
        preprocess_root=dataset_preprocess_root,
        batch_size=1,
        num_workers=int(n_workers_per_gpu),
        n_subnets=n_infers,
        translate_distance=translate_distance,
        max_angle=max_angle,
    )
    data_module.setup_val_loader(visualize=False, max_items=None, data_aug=True)

    print(model_path)

    trainer = Trainer(
        deterministic=False,
        devices=n_gpus,
        logger=False,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        accelerator="gpu",
    )

    model = Net.load_from_checkpoint(
        model_path, class_frequencies=class_frequencies, iou_threshold=iou_threshold
    )
    model.cuda()
    model.eval()

    # enable_dropout(model)
    trainer.test(model=model, dataloaders=data_module.val_dataloader())
    # trainer.test(model=model, dataloaders=data_module.val_dataloader())


if __name__ == "__main__":
    main()
