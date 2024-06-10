import os
from pasco.data.semantic_kitti.kitti_dm import KittiDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import click
import numpy as np
import torch
from pasco.data.semantic_kitti.params import class_names, class_frequencies
from pasco.models.net_panoptic_sparse import Net
from pytorch_lightning.strategies import DDPStrategy
from pasco.utils.torch_util import set_random_seed
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal

import os



@click.command()
@click.option('--log_dir', default="logs", help='logging directory')
@click.option('--dataset_root', default="/gpfsdswork/dataset/SemanticKITTI")
@click.option('--config_path', default="semantic-kitti.yaml")
@click.option('--dataset_preprocess_root', default="/gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/kitti")
@click.option('--n_infers', default=1, help='number of subnets')

@click.option('--lr', default=3e-4, help='learning rate')
@click.option('--wd', default=0.0, help='weight decay')
@click.option('--bs', default=1, help="batch size")
@click.option('--scale', default=1, help="Scale")
@click.option('--n_gpus', default=2, help="number of GPUs")
@click.option('--n_workers_per_gpu', default=3, help="Number of workers per GPU")
@click.option('--exp_prefix', default="exp", help='prefix of logging directory')
@click.option('--enable_log', default=True, help='Enable logging')

@click.option('--transformer_dropout', default=0.2)
@click.option('--net_3d_dropout', default=0.0)
@click.option('--n_dropout_levels', default=3)

@click.option('--max_angle', default=5.0, help='random augmentation angle from -max_angle to max_angle')
@click.option('--translate_distance', default=0.2, help='randomly translate 3D scene in 3 dimensions: np.array([3.0, 3.0, 2.0]) * translate_distance')
@click.option('--point_dropout_ratio', default=0.05, help='randomly drop from 0 to 5% points in 3D input')
@click.option('--data_aug', default=True, help='Use data augmentation if True')
@click.option('--scale_range', default=0.0, help='random scaling the scene')

@click.option('--alpha', default=0.0, help='uncertainty weight')


@click.option('--transformer_enc_layers', default=0, help='Transformer encoder layer')
@click.option('--transformer_dec_layers', default=1, help='Transformer decoder layer')

@click.option('--num_queries', default=100, help='Number of queries')
@click.option('--mask_weight', default=40.0, help='mask weight')
@click.option('--occ_weight', default=1.0, help='mask weight')


@click.option('--use_se_layer', default=False, help='mask weight')
@click.option('--sample_query_class', default=False, help='Sample query class')
@click.option('--heavy_decoder', default=True, help='mask weight')

@click.option('--use_voxel_query_loss', default=True, help='uncertainty weight')

@click.option('--accum_batch', default=1, help='') # Quite slow to train, test later
@click.option('--n_fuse_scans', default=1, help='#scans to fuse')

@click.option('--pretrained_model', default="")
@click.option('--f', default=64)
@click.option('--seed', default=42)
def main(
    lr, wd, 
    bs, scale, alpha,
    n_workers_per_gpu, n_gpus,
    exp_prefix, log_dir, enable_log,
    data_aug, mask_weight, heavy_decoder,
    transformer_dropout, net_3d_dropout, n_dropout_levels,
    point_dropout_ratio, use_voxel_query_loss,
    max_angle, translate_distance, scale_range, seed,
    transformer_dec_layers, transformer_enc_layers, n_infers, occ_weight,
    num_queries, use_se_layer, sample_query_class, accum_batch, pretrained_model,
    dataset_root, dataset_preprocess_root, config_path, n_fuse_scans, f):
    
    set_random_seed(seed)
    
    encoder_dropouts = [point_dropout_ratio, 0.0, 0.0, 0.0, 0.0, 0.0]
    decoder_dropouts = [0.0, 0.0, 0.0, 0.0, 0.0]
    for l in range(n_dropout_levels):
        encoder_dropouts[len(encoder_dropouts) - l - 1] = net_3d_dropout
        decoder_dropouts[l] = net_3d_dropout

   
    print("log_dir", log_dir)
    exp_name = exp_prefix
    
    exp_name += "bs{}_Fuse{}".format(bs, n_fuse_scans)
    exp_name += "_alpha{}_wd{}_lr{}_Aug{}R{}T{}S{}_DropoutPoints{}Trans{}net3d{}nLevels{}".format(
        alpha, wd, lr, 
        data_aug, max_angle, translate_distance, scale_range,
        point_dropout_ratio, transformer_dropout, net_3d_dropout, n_dropout_levels)
    exp_name += "_TransLay{}Enc{}Dec_queries{}".format(transformer_enc_layers, transformer_dec_layers, num_queries)
    exp_name += "_maskWeight{}".format(mask_weight)
    
    if occ_weight != 1.0:
        exp_name += "_occWeight{}".format(occ_weight) 
    
    if sample_query_class:
        exp_name += "_sampleQueryClass"
        
    exp_name += "_nInfers{}".format(n_infers)
    
    if not use_voxel_query_loss:
        exp_name += "_noVoxelQueryLoss"
    if not heavy_decoder:
        exp_name += "_noHeavyDecoder"
    
    query_sample_ratio = 1.0

    print(exp_name)
    max_epochs = 60
    n_classes = 20
    
    class_weights = []

    for _ in range(n_infers):
        class_weight = torch.ones(n_classes + 1)
        class_weight[0] = 0.1
        class_weight[-1] = 0.1 # dustbin class
        class_weights.append(class_weight)

    complt_num_per_class = class_frequencies["1_1"]
    compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
    compl_labelweights = np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)
    compl_labelweights = torch.from_numpy(compl_labelweights).float()
    

    data_module = KittiDataModule(
        root=dataset_root,
        config_path=config_path,
        preprocess_root=dataset_preprocess_root,
        batch_size=int(bs / n_gpus),
        num_workers=int(n_workers_per_gpu),
        data_aug=data_aug,
        max_angle=max_angle,
        translate_distance=translate_distance,
        scale_range=scale_range,
        max_val_items=None,
        n_fuse_scans=n_fuse_scans,
        n_subnets=n_infers,
    )

    model = Net(
        heavy_decoder=heavy_decoder,
        class_frequencies=class_frequencies,
        n_classes=n_classes,
        occ_weight=occ_weight,
        class_names=class_names,
        lr=lr,
        weight_decay=wd,
        class_weights=class_weights,
        transformer_dropout=transformer_dropout,
        encoder_dropouts=encoder_dropouts,
        decoder_dropouts=decoder_dropouts,
        dense3d_dropout=net_3d_dropout,
        scale=scale, # not use
        enc_layers=transformer_enc_layers,
        dec_layers=transformer_dec_layers,
        aux_loss=False,
        num_queries=num_queries,
        mask_weight=mask_weight,
        use_se_layer=use_se_layer,
        alpha=alpha,
        sample_query_class=sample_query_class,
        query_sample_ratio=query_sample_ratio,
        n_infers=n_infers,
        f=f,
        compl_labelweights=compl_labelweights,
        use_voxel_query_loss=use_voxel_query_loss
    )
    
    

    if enable_log:
        logger = TensorBoardLogger(save_dir=log_dir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="val_subnet" + str(n_infers) + "/pq_dagger_all",
                save_top_k=50,
                mode="max",
                filename="{epoch:03d}-{val_subnet" + str(n_infers) + "/pq_dagger_all:.5f}",
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False

    model_path = os.path.join(log_dir, exp_name, "checkpoints/last.ckpt")
    if not os.path.isfile(model_path) and pretrained_model != "":
        assert os.path.isfile(pretrained_model), "Pretrained model not found"
        model = Net.load_from_checkpoint(checkpoint_path=pretrained_model)
        print("Load pretrained model from {}".format(model_path))

    if os.path.isfile(model_path):
              
        trainer = Trainer(
            accumulate_grad_batches=accum_batch,
            limit_val_batches=0.25 * accum_batch * n_gpus,
            limit_train_batches=0.25 * accum_batch * n_gpus,
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,            
            max_epochs=max_epochs,
            gradient_clip_val=0.5,
            logger=logger,
            check_val_every_n_epoch=1,
            accelerator="gpu",
            strategy=DDPStrategy(find_unused_parameters=True),
            num_nodes=1,
            devices=n_gpus,
            sync_batchnorm=True,
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        )
    else:
        # Train from scratch
        
        trainer = Trainer(
            accumulate_grad_batches=accum_batch,
            limit_val_batches=0.25 * accum_batch * n_gpus,
            limit_train_batches=0.25 * accum_batch * n_gpus,
            callbacks=checkpoint_callbacks,            
            max_epochs=max_epochs,
            gradient_clip_val=0.5,
            logger=logger,
            strategy=DDPStrategy(find_unused_parameters=True),
            check_val_every_n_epoch=1,
            accelerator="gpu",
            devices=n_gpus,
            num_nodes=1,
            sync_batchnorm=True,
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
        )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
