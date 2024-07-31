import os
from pasco.data.kitti360.kitti360_dm import Kitti360DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import click
import numpy as np
import torch
from pasco.data.kitti360.params import kitti_360_class_names as class_names, kitti_360_class_frequencies as class_frequencies
from pasco.models.net_panoptic_sparse_kitti360 import Net_kitti360
from pytorch_lightning.strategies import DDPStrategy
from pasco.utils.torch_util import set_random_seed
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal

import os



@click.command()
@click.option('--lr', default=3e-4, help='learning rate')
@click.option('--wd', default=0.0, help='weight decay')
@click.option('--bs', default=1, help="batch size")
@click.option('--scale', default=1, help="Scale")
@click.option('--n_gpus', default=2, help="number of GPUs")
@click.option('--n_workers_per_gpu', default=3, help="Number of workers per GPU")
@click.option('--exp_prefix', default="exp", help='prefix of logging directory')
@click.option('--log_dir', default="", help='logging directory')
@click.option('--enable_log', default=True, help='Enable logging')

@click.option('--overfit', default=False, help='If true, overfit on 1 sample')

@click.option('--transformer_dropout', default=0.2, help='dropout')
@click.option('--net_3d_dropout', default=0.0, help='net 3d dropout')
@click.option('--n_dropout_levels', default=3, help='net 3d dropout')

@click.option('--max_angle', default=5.0, help='net 3d dropout')
@click.option('--translate_distance', default=0.2, help='net 3d dropout')
@click.option('--point_dropout_ratio', default=0.05, help='net 3d dropout')
@click.option('--data_aug', default=True, help='Data Aug')
@click.option('--scale_range', default=0.0, help='random scaling')

@click.option('--alpha', default=0.0, help='uncertainty weight')

@click.option('--kitti360_root', default="/gpfsdswork/dataset/KITTI-360")
@click.option('--kitti360_preprocess_root', default="/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti360")
@click.option('--kitti360_label_root', default="/gpfsdswork/dataset/SSCBench-KITTI-360")

@click.option('--transformer_enc_layers', default=0, help='Transformer encoder layer')
@click.option('--transformer_dec_layers', default=1, help='Transformer decoder layer')

@click.option('--num_queries', default=100, help='Number of queries')
@click.option('--mask_weight', default=40.0, help='mask weight')

@click.option('--use_se_layer', default=False, help='mask weight')

@click.option('--use_voxel_query_loss', default=True, help='uncertainty weight')
@click.option('--n_infers', default=3, help='uncertainty weight')
@click.option('--accum_batch', default=1, help='') # Quite slow to train, test later
@click.option('--n_fuse_scans', default=1, help='#scans to fuse')

@click.option('--pretrained_model', default="")
@click.option('--f', default=64)
@click.option('--seed', default=42)
@click.option('--occ_weight', default=1.0, help='net 3d dropout')
@click.option('--heavy_decoder', default=False)
def main(
    lr, wd, 
    bs, scale, alpha,
    n_workers_per_gpu, n_gpus,
    exp_prefix, log_dir, enable_log,
    data_aug, mask_weight, use_voxel_query_loss,
    transformer_dropout, net_3d_dropout, n_dropout_levels,
    overfit, point_dropout_ratio,
    max_angle, translate_distance, scale_range, occ_weight,
    transformer_dec_layers, transformer_enc_layers, n_infers, seed,
    num_queries, use_se_layer, accum_batch, pretrained_model,
    kitti360_root, kitti360_preprocess_root, kitti360_label_root, n_fuse_scans, f, heavy_decoder):
    
    set_random_seed(seed)
    
    encoder_dropouts = [point_dropout_ratio, 0.0, 0.0, 0.0, 0.0, 0.0]
    decoder_dropouts = [0.0, 0.0, 0.0, 0.0, 0.0]
    for l in range(n_dropout_levels):
        encoder_dropouts[len(encoder_dropouts) - l - 1] = net_3d_dropout
        decoder_dropouts[l] = net_3d_dropout

    exp_name = exp_prefix
    if overfit:
        exp_name = "overfit_" + exp_name
    exp_name += "bs{}_Fuse{}".format(bs, n_fuse_scans)
    exp_name += "_alpha{}_wd{}_lr{}_Aug{}R{}T{}S{}_DropoutPoints{}Trans{}net3d{}nLevels{}".format(
        alpha, wd, lr, 
        data_aug, max_angle, translate_distance, scale_range,
        point_dropout_ratio, transformer_dropout, net_3d_dropout, n_dropout_levels)
    exp_name += "_TransLay{}Enc{}Dec_queries{}".format(transformer_enc_layers, transformer_dec_layers, num_queries)
    exp_name += "_maskWeight{}".format(mask_weight)
    if occ_weight != 1.0:
        exp_name += "_occWeight{}".format(occ_weight)
    exp_name += "_nInfers{}".format(n_infers)
    
    if not use_voxel_query_loss:
        exp_name += "_noVoxelQueryLoss"
    if not heavy_decoder:
        exp_name += "_noHeavyDecoder"
    
    query_sample_ratio = 1.0 # not use

    print(exp_name)
    # Setup dataloaders
    max_epochs = 80
    if overfit:
        max_epochs = 1000
        
    n_classes = 19
    
    class_weights = []
  
    for i_infer in range(n_infers):
        class_weight = torch.ones(n_classes + 1)
        class_weight[0] = 0.1
        class_weight[-1] = 0.1 # dustbin class
        class_weights.append(class_weight)

    complt_num_per_class = class_frequencies["1_1"]
    compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
    compl_labelweights = np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)
    compl_labelweights = torch.from_numpy(compl_labelweights).float()
    
    data_module = Kitti360DataModule(
        kitti360_root=kitti360_root,
        kitti360_preprocess_root=kitti360_preprocess_root,
        kitti360_label_root=kitti360_label_root,
        
        batch_size=int(bs / n_gpus),
        num_workers=int(n_workers_per_gpu),
        data_aug=data_aug,
        
        max_angle=max_angle,
        translate_distance=translate_distance,
        scale_range=scale_range,
        
        overfit=overfit,
        max_val_items=None,

        n_fuse_scans=n_fuse_scans,
        n_subnets=n_infers,
    )

    model = Net_kitti360(
            heavy_decoder=heavy_decoder,
            in_channels=8,
            class_frequencies=class_frequencies,
            n_classes=n_classes,
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
            query_sample_ratio=query_sample_ratio,
            n_infers=n_infers,
            f=f,
            occ_weight=occ_weight,
            compl_labelweights=compl_labelweights,
            use_voxel_query_loss=use_voxel_query_loss
        )
    
    

    if enable_log:
        logger = TensorBoardLogger(save_dir=log_dir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                # monitor="val/mIoU",
                monitor="val_subnet" + str(n_infers) + "/pq_dagger_all",
                save_top_k=60,
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
        # model_path = pretrained_model
        assert os.path.isfile(pretrained_model), "Pretrained model not found"
        model = Net_kitti360.load_from_checkpoint(checkpoint_path=pretrained_model)
        print("Load pretrained model from {}".format(model_path))

    if os.path.isfile(model_path):
        # model = Net.load_from_checkpoint(model_path)
    #     # Continue training from last.ckpt
              
        trainer = Trainer(
            accumulate_grad_batches=accum_batch,
            # limit_val_batches=0.25 * accum_batch * n_gpus,
            # limit_train_batches=0.25 * accum_batch * n_gpus,
            limit_val_batches=0.1 * accum_batch * n_gpus,
            limit_train_batches=0.125 * accum_batch * n_gpus,
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,            
            max_epochs=max_epochs,
            # gradient_clip_val=0.01,
            gradient_clip_val=0.5,
            logger=logger,
            check_val_every_n_epoch=1,
            # val_check_interval=0.25,
            # log_every_n_steps=10,
            accelerator="gpu",
            strategy=DDPStrategy(find_unused_parameters=True),
            # strategy=DDPStrategy(find_unused_parameters=False),
            num_nodes=1,
            devices=n_gpus,
            sync_batchnorm=True,
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        )
    else:
        # Train from scratch
        
        trainer = Trainer(
            accumulate_grad_batches=accum_batch,
            limit_val_batches=0.1 * accum_batch * n_gpus,
            limit_train_batches=0.125 * accum_batch * n_gpus,
            # profiler="advanced",
            callbacks=checkpoint_callbacks,            
            max_epochs=max_epochs,
            # gradient_clip_val=0.01,
            gradient_clip_val=0.5,
            logger=logger,
            strategy=DDPStrategy(find_unused_parameters=True),
            # strategy=DDPStrategy(find_unused_parameters=False),
            check_val_every_n_epoch=1,
            # val_check_interval=0.25,
            # log_every_n_steps=10,
            accelerator="gpu",
            devices=n_gpus,
            num_nodes=1,
            sync_batchnorm=True,
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
        )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
