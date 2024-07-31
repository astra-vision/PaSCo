#!/bin/bash
#SBATCH --job-name=waffleiron_robo3d
#SBATCH --nodes=1
#SBATCH -A trg@v100
#SBATCH --ntasks-per-node=1

#SBATCH --time=20:00:00

##SBATCH --time=01:59:00
##SBATCH --qos=qos_gpu-dev

#SBATCH --output=waffleiron_robo3d_v100_%j.out
#SBATCH --error=waffleiron_robo3d_%j.err
#SBATCH --hint=nomultithread

#SBATCH --gres=gpu:1

#SBATCH -C v100-32g
#SBATCH --cpus-per-task=10

##SBATCH -C a100
##SBATCH --cpus-per-task=6
##SBATCH -A trg@a100

##SBATCH --cpus-per-task=3
##SBATCH --partition=gpu_p2

##SBATCH --cpus-per-task=6
##SBATCH --partition=gpu_p4

##SBATCH --signal=SIGUSR1@90

# module load cpuarch/amd

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default

conda activate uncertainty

set -x


    


python eval_kitti_robo3d.py \
--path_dataset /gpfsdswork/dataset/SemanticKITTI \
--ckpt pretrained_models/WaffleIron-48-256__kitti/ckpt_last.pth \
--config ./configs/WaffleIron-48-256__kitti.yaml \
--result_folder /gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/kitti/waffleiron_v2 \
--phase val \
--num_workers 10 \
--num_votes 10 \
--batch_size 5


