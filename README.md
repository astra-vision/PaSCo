<div align='center'>
 
# PaSCo: Urban 3D Panoptic Scene Completion with Uncertainty Awareness

CVPR 2024 Oral, Best Paper Award Candidate


[Anh-Quan Cao](https://anhquancao.github.io)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Angela Dai](https://www.3dunderstanding.org/)<sup>2</sup>&nbsp;&nbsp;&nbsp;
[Raoul de Charette](https://team.inria.fr/rits/membres/raoul-de-charette/)<sup>1</sup>&nbsp;&nbsp;&nbsp;

<div>
<sup>1</sup> Inria
<sup>2</sup> Technical University of Munich
</div>

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2312.02158-darkred)](https://arxiv.org/abs/2312.02158) 
[![Project page](https://img.shields.io/badge/Project%20Page-PaSCo-darkgreen)](https://astra-vision.github.io/PaSCo/)

</div>

If you find this work or code useful, please cite our [paper](https://arxiv.org/abs/2212.02501) and [give this repo a star](https://github.com/astra-vision/SceneRF/stargazers):
```
@InProceedings{cao2024pasco,
      title={PaSCo: Urban 3D Panoptic Scene Completion with Uncertainty Awareness}, 
      author={Anh-Quan Cao and Angela Dai and Raoul de Charette},
      year={2024},
      booktitle = {CVPR}
}
```

# Teaser
![](./teaser/psc.GIF)

# Table of Contents
- [News](#news)
- [1. Installation](#1-installation)
- [2. Data](#2-data)
  - [2.1. Semantic KITTI](#21-semantic-kitti)
  - [2.2. SSCBench-KITTI360](#22-sscbench-kitti360)
- [3. Panoptic labels generation](#3-panoptic-labels-generation)
  - [3.1. Semantic KITTI](#31-semantic-kitti)
  - [3.2. SSCBench-KITTI360](#32-sscbench-kitti360)
- [4. Training and evaluation](#4-training-and-evaluation)
  - [4.1. Semantic KITTI](#41-semantic-kitti)
    - [4.1.1 Extract point features](#411-extract-point-features)
    - [4.1.2 Training](#412-training)
    - [4.1.3 Evaluation](#413-evaluation)
  - [4.2. SSCBench-KITTI360](#42-sscbench-kitti360)
    - [4.2.1 Training](#421-training)
    - [4.2.2 Evaluation](#422-evaluation)
- [5. Visualization](#5-visualization)
- [Acknowledgment](#acknowledgment)


# News
- 30/07/2024: We added the training, evaluation, and checkpoints for PaSCo on SSCBench-KITTI360.
- 23/07/2024: The training/evaluation code and the checkpoint for PaSCo on SemanticKITTI has been released.
- 25/06/2024: Added visualization code.
- 10/06/2024: Training and evaluation code for PaSCo w/o MIMO has been released.
- 06/04/2024: Dataset download instructions and label generation code for SemanticKITTI are now available.
- 04/04/2024: PaSCo has been accepted as Oral paper at [CVPR 2024](https://cvpr.thecvf.com/) (0.8% = 90/11,532).
- 05/12/2023: Paper released on arXiv! Code will be released soon! Please [watch this repo](https://github.com/astra-vision/PaSCo/watchers) for updates.


# 1. Installation
1. Download the source code with git
      ```
      git clone https://github.com/astra-vision/PaSCo.git
      ```
2. Create conda environment:
      ```
      conda create -y -n pasco python=3.9
      conda activate pasco
      ```
3. Install pytorch 1.13.0
      ```
      pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
      ```
4. Install [Minkowski Engine v0.5.4](https://github.com/NVIDIA/MinkowskiEngine?tab=readme-ov-file#pip)

5. Install pytorch_lightning 1.9.0 with torchmetrics 1.4.0.post0
   
6. Install the additional dependencies:
      ```
      cd PaSCo/
      pip install -r requirements.txt
      ```

7. Install PaSCo
      ```
      pip install -e ./
      ```
# 2. Data

## 2.1. Semantic KITTI
Please download the following data into a folder e.g. **/gpfsdswork/dataset/SemanticKITTI** and unzip:

- The **Semantic Scene Completion dataset v1.1** (SemanticKITTI voxel data (700 MB)) from [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download)

-  The [KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) calibration data (Download odometry data set **(calibration files, 1 MB)**). 

- The[ KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) Velodyne data (Download odometry data set  **(velodyne laser data, 80 GB)**).

- The dataset folder at **/gpfsdswork/dataset/SemanticKITTI** should have the following structure:
    ```
    └── /gpfsdswork/dataset/SemanticKITTI
      └── dataset
        └── sequences
    ```

## 2.2. SSCBench-KITTI360
Please download the following data:
- The **SSCBench-KITTI360** `preprocess.sqf`  from [SSCBench-KITTI360 official github](https://github.com/ai4ce/SSCBench/tree/main/dataset/KITTI-360) and unsquash it into a folder e.g. **/gpfsdswork/dataset/SSCBench-KITTI-360**. I also uploaded the file `preprocess.sqf` [here as backup](https://huggingface.co/datasets/anhquancao/SSCBench-KITTI360/blob/main/preprocess.sqf).
- The **KITTI-360** `Raw Velodyne Scans (119G)` from [KITTI-360 download page](https://www.cvlibs.net/datasets/kitti-360/download.php) and put into folder e.g. **/gpfsdswork/dataset/KITTI-360**.

- The folder  **/gpfsdswork/dataset/KITTI-360** should have the following structure:
   ```
   /gpfsdswork/dataset/KITTI-360
   └── data_3d_raw
       ├── 2013_05_28_drive_0000_sync
       ├── 2013_05_28_drive_0002_sync
       ├── 2013_05_28_drive_0003_sync
       ├── 2013_05_28_drive_0004_sync
       ├── 2013_05_28_drive_0005_sync
       ├── 2013_05_28_drive_0006_sync
       ├── 2013_05_28_drive_0007_sync
       └── 2013_05_28_drive_0009_sync
   ```

- The folder **/gpfsdswork/dataset/SSCBench-KITTI-360** should have the following structure:
   ```
   /gpfsdswork/dataset/SSCBench-KITTI-360/
   ├── labels                        
   │   ├── 2013_05_28_drive_0000_sync
   │   ├── 2013_05_28_drive_0002_sync
   │   ├── 2013_05_28_drive_0003_sync
   │   ├── 2013_05_28_drive_0004_sync
   │   ├── 2013_05_28_drive_0005_sync
   │   ├── 2013_05_28_drive_0006_sync
   │   ├── 2013_05_28_drive_0007_sync
   │   ├── 2013_05_28_drive_0009_sync
   │   └── 2013_05_28_drive_0010_sync
   ├── labels_half                   
   │   ├── 2013_05_28_drive_0000_sync
   │   ├── 2013_05_28_drive_0002_sync
   │   ├── 2013_05_28_drive_0003_sync
   │   ├── 2013_05_28_drive_0004_sync
   │   ├── 2013_05_28_drive_0005_sync
   │   ├── 2013_05_28_drive_0006_sync
   │   ├── 2013_05_28_drive_0007_sync  
   │   ├── 2013_05_28_drive_0009_sync
   │   └── 2013_05_28_drive_0010_sync
   ├── README            
   └── unified                         
   └── labels
   ```
 

# 3. Panoptic labels generation
## 3.1. Semantic KITTI
1. Create a folder to store preprocess data for Semantic KITTI dataset e.g. **/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti** .
2. Execute the command below to generate panoptic labels, or **move to the next step** to directly download the **pre-generated labels**:
      ```
      cd PaSCo/
      python label_gen/gen_instance_labels.py \
          --kitti_config=pasco/data/semantic_kitti/semantic-kitti.yaml \
          --kitti_root=/gpfsdswork/dataset/SemanticKITTI \
          --kitti_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti \
          --n_process=10
      ```

> [!NOTE]
> This command processes 4649 files on CPU took approximately 10 hours using 10 processes. The number of processes can be adjusted by modifying the `n_process` parameter.


3. You can download the generated panoptic labels for Semantic KITTI:
   1. Go to the preprocess folder for KITTI:
      ```
      cd /gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti
      ```
   2. Download the compressed file:
      ```
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.0.1/kitti_instance_label_v2.tar.gz
      ```
   3. Extract the file:
      ```
      tar xvf kitti_instance_label_v2.tar.gz
      ```
4. Your folder structure should look as follows:
      ```
      /gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti
      └── instance_labels_v2
          ├── 00
          ├── 01
          ├── 02
          ├── 03
          ├── 04
          ├── 05
          ├── 06
          ├── 07
          ├── 08
          ├── 09
          └── 10
      ```
   

## 3.2. SSCBench-KITTI360
1. Create a folder to store preprocess data for SSCBench-KITTI360 dataset e.g. **/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti360** .
2. Execute the command below to generate panoptic labels (took approximately 8.5 hours to process 12464 scans using 10 processes), or **move to the next step** to directly download the **pre-generated labels**:
      ```
      cd PaSCo/
      python label_gen/gen_instance_labels_kitti360.py \
          --kitti360_label_root=/gpfsdswork/dataset/SSCBench-KITTI-360 \
          --kitti360_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti360 \
          --n_process=10
      ```
3. You can download the generated panoptic labels for SSCBench-KITTI360:
   1. Go to the preprocess folder for KITTI360:
      ```
      cd /gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti360
      ```
   2. Create a folder to store the instance label and cd into it:
      ```
      mkdir instance_labels_v2
      cd instance_labels_v2
      ```
   3. Run the following commands to download the compressed files into it:
      ```
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0000_sync.tar.gz
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0002_sync.tar.gz
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0003_sync.tar.gz
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0004_sync.tar.gz
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0005_sync.tar.gz
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0006_sync.tar.gz
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0007_sync.tar.gz
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0009_sync.tar.gz
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0010_sync.tar.gz
      ```
   3. Extract all the downloaded files:
      ```
      tar xvf *.tar.gz
      ```
4. Your folder structure with the instance labels should look as follows:
      ```
      /gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti360
      └── instance_labels_v2
          ├── 2013_05_28_drive_0000_sync
          ├── 2013_05_28_drive_0002_sync
          ├── 2013_05_28_drive_0003_sync
          ├── 2013_05_28_drive_0004_sync
          ├── 2013_05_28_drive_0005_sync
          ├── 2013_05_28_drive_0006_sync
          ├── 2013_05_28_drive_0007_sync
          ├── 2013_05_28_drive_0009_sync
          └── 2013_05_28_drive_0010_sync
      ```

# 4. Training and evaluation
> [!IMPORTANT]
> During training, the reported metric is lower than the final metrics because we limit the number of generated voxels to prevent running out of memory. The training metrics are used solely to assess the progress of the training. The final metrics are determined during evaluation.

> [!NOTE]
> The architecture has been slightly modified from the paper to improve training stability. The paper uses 7 residual blocks after each upsampling layer and none in the corresponding encoder block. The new design uses 3 residual blocks after each upsampling layer and 3 residual blocks in the corresponding encoder block. The original design occasionally caused GPU memory errors due to the large number of generated voxels processed by 7 residual blocks. The new design is more stable and lighter, albeit with slightly lower performance.

## 4.1. Semantic KITTI
### 4.1.1 Extract point features

> [!NOTE]
> This step is only necessary when training on SemanticKITTI because of the availability of the WaffleIron pretrained model.

> [!TIP]
> A better approach could be to explore the features of pretrained models available at [https://github.com/valeoai/ScaLR](https://github.com/valeoai/ScaLR).


1. Install WaffleIron in a separate conda environment:
      ```
      conda create -y -n waffleiron 
      pip install pyYAML==6.0 tqdm==4.63.0 scipy==1.8.0 torch==1.11.0 tensorboard==2.8.0
      cd PaSCo/WaffleIron_mod
      pip install -e ./
      ```

> [!CAUTION]
> I used the older version of WaffleIron which requires pytorch 1.11.0.


2. Run the following command to extract point features from the pretrained WaffleIron model (require 10883Mb GPU memory) pretrained on SemanticKITTI. The extracted features will be stored in the `result_folder`:
      ```
      cd PaSCo/WaffleIron_mod
      python extract_point_features.py \
      --path_dataset /gpfsdswork/dataset/SemanticKITTI \
      --ckpt pretrained_models/WaffleIron-48-256__kitti/ckpt_last.pth \
      --config configs/WaffleIron-48-256__kitti.yaml \
      --result_folder /gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti/waffleiron_v2 \
      --phase val \
      --num_workers 3 \
      --num_votes 10 \
      --batch_size 2
      ```
### 4.1.2 Training
> [!NOTE]
> The generated instance label is supposed to be stored in os.path.join(dataset_preprocess_root, "instance_labels_v2")

1. Change the `dataset_preprocess_root` and `dataset_root` of the training command below to the preprocess and raw data folder respectively.
2. The `log_dir` is the folder to store the training logs and checkpoints.
3. **Train PaSCo with MIMO** (i.e. 1 subnet) using the following command with a batchsize of 2 on 2 V100-32G GPUs (1 item per GPU):

      ```
      cd PaSCo/
      python scripts/train.py --bs=2 --n_gpus=2 \
            --dataset_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti \
            --dataset_root=/gpfsdswork/dataset/SemanticKITTI \
            --log_dir=logs \
            --exp_prefix=pasco_single --lr=1e-4 --seed=0 \
            --data_aug=True --max_angle=30.0 --translate_distance=0.2 \
            --enable_log=True \
            --n_infers=1
      ```

4. **Train PaSCo (3 subnets)**  by setting **--n_infers=3 (number of subnets = 3)** with batchsize of 2 on 2 A100-80G GPUs (1 items per GPU):
      ```
      cd PaSCo/
      python scripts/train.py --bs=2 --n_gpus=2 \
            --dataset_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti \
            --dataset_root=/gpfsdswork/dataset/SemanticKITTI \
            --log_dir=logs \
            --exp_prefix=pasco_single --lr=1e-4 --seed=0 \
            --data_aug=True --max_angle=30.0 --translate_distance=0.2 \
            --enable_log=True \
            --n_infers=3
      ```


## 4.1.3 Evaluation
1. Download the **pretrained checkpoint of [PaSCO](https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/pasco.ckpt) or [PaSCO without MIMO](https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/pasco_single.ckpt)** and put it into `ckpt` folder or use your trained checkpoint.
      ```
      cd ckpt
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/pasco.ckpt
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/pasco_single.ckpt
      ```
2. Evaluate **PaSCo without MIMO** on 1 V100-32G GPUs (1 item per GPU). `ckpt/pasco_single.ckpt` is the path to the downloaded checkpoint:
      ```
      python scripts/eval.py --n_infers=1 --model_path=ckpt/pasco_single.ckpt \
            --dataset_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti \
            --dataset_root=/gpfsdswork/dataset/SemanticKITTI 
      ```
3. Evaluate **PaSCo** on 1 V100-32G GPUs (1 item per GPU). `ckpt/pasco.ckpt` is the path to the downloaded checkpoint:
      ```
      python scripts/eval.py --n_infers=3 --model_path=ckpt/pasco.ckpt \
            --dataset_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti \
            --dataset_root=/gpfsdswork/dataset/SemanticKITTI 
      ```

   

4. Output looks like following: 
      - **PaSCo without MIMO**:
      ```
      =====================================
      method, P, R, IoU, mIoU, All PQ dagger, All PQ, All SQ, All RQ, Thing PQ, Thing SQ, Thing RQ, Stuff PQ, Stuff SQ, Stuff RQ
      subnet 0, 86.41, 57.98, 53.13, 29.15, 26.33, 15.71, 53.82, 24.27, 12.27, 47.18, 18.86, 18.21, 58.65, 28.20
      ensemble, 86.41, 57.98, 53.13, 29.15, 26.33, 15.71, 53.82, 24.27, 12.27, 47.18, 18.86, 18.21, 58.65, 28.20
      =====================================
      ==> pq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, road, parking, sidewalk, other-ground, building, fence, vegetation, trunk, terrain, pole, traffic-sign
      subnet 0, 27.53, 6.21, 16.86, 34.27, 9.77, 3.53, 0.00, 0.00, 74.51, 26.63, 39.70, 0.54, 4.10, 4.64, 6.87, 3.80, 29.58, 7.68, 2.28
      ensemble, 27.53, 6.21, 16.86, 34.27, 9.77, 3.53, 0.00, 0.00, 74.51, 26.63, 39.70, 0.54, 4.10, 4.64, 6.87, 3.80, 29.58, 7.68, 2.28
      ==> sq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, road, parking, sidewalk, other-ground, building, fence, vegetation, trunk, terrain, pole, traffic-sign
      subnet 0, 69.83, 57.87, 64.56, 65.42, 59.95, 59.80, 0.00, 0.00, 75.74, 63.11, 58.65, 52.57, 56.08, 55.89, 52.51, 58.07, 62.12, 55.01, 55.42
      ensemble, 69.83, 57.87, 64.56, 65.42, 59.95, 59.80, 0.00, 0.00, 75.74, 63.11, 58.65, 52.57, 56.08, 55.89, 52.51, 58.07, 62.12, 55.01, 55.42
      ==> rq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, road, parking, sidewalk, other-ground, building, fence, vegetation, trunk, terrain, pole, traffic-sign
      subnet 0, 39.43, 10.73, 26.11, 52.38, 16.29, 5.91, 0.00, 0.00, 98.38, 42.19, 67.69, 1.04, 7.31, 8.31, 13.07, 6.54, 47.61, 13.96, 4.11
      ensemble, 39.43, 10.73, 26.11, 52.38, 16.29, 5.91, 0.00, 0.00, 98.38, 42.19, 67.69, 1.04, 7.31, 8.31, 13.07, 6.54, 47.61, 13.96, 4.11
      [2.621915578842163, 0.8142204284667969, 0.8685343265533447, 0.7775185108184814, 0.9801337718963623, 0.6943247318267822]
      inference time:  0.7034459143364459
      [0.004038333892822266, 0.003854036331176758, 0.005398988723754883, 0.003660440444946289, 0.004451274871826172, 0.003663778305053711]
      ensemble time:  0.004062994399293342
      Uncertainty threshold:  0.5
      =====================================
      method, ins ece, ins nll, ssc nonempty ece, ssc empty ece, ssc nonempty nll, ssc empty nll,  count, inference time
      subnet 0,  0.6235, 4.6463, 0.0911, 0.0357, 0.7075, 0.9657, 11702, 0.00
      ensemble,  0.6235, 4.6463, 0.0911, 0.0357, 0.7075, 0.9657, 11702, 0.00
      allocated 8895.119325153375
      ```
      - **PaSCo**:
      ```
      =====================================
      method, P, R, IoU, mIoU, All PQ dagger, All PQ, All SQ, All RQ, Thing PQ, Thing SQ, Thing RQ, Stuff PQ, Stuff SQ, Stuff RQ
      subnet 0, 77.41, 65.46, 54.96, 27.55, 25.17, 14.67, 56.88, 23.18, 10.80, 52.60, 17.42, 17.49, 59.99, 27.38
      subnet 1, 79.38, 63.54, 54.54, 27.36, 25.56, 14.51, 53.75, 22.78, 10.80, 46.09, 17.11, 17.20, 59.33, 26.90
      subnet 2, 75.10, 67.29, 55.01, 27.79, 25.54, 14.98, 53.23, 23.51, 11.74, 53.30, 18.49, 17.33, 53.18, 27.17
      ensemble, 83.59, 62.10, 55.35, 29.54, 30.61, 16.38, 55.41, 25.24, 13.49, 47.20, 20.93, 18.49, 61.38, 28.37
      =====================================
      ==> pq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, road, parking, sidewalk, other-ground, building, fence, vegetation, trunk, terrain, pole, traffic-sign
      subnet 0, 25.31, 7.34, 14.46, 28.97, 7.78, 2.27, 0.24, 0.00, 73.71, 21.03, 35.21, 0.78, 6.46, 4.35, 10.70, 2.65, 29.88, 6.09, 1.55
      subnet 1, 23.68, 4.74, 9.70, 37.66, 7.98, 2.62, 0.00, 0.00, 73.47, 18.59, 35.12, 0.68, 6.72, 3.65, 10.17, 3.25, 29.53, 5.67, 2.38
      subnet 2, 24.04, 7.17, 16.35, 36.08, 8.14, 1.92, 0.24, 0.00, 73.99, 22.03, 34.39, 0.00, 6.30, 2.59, 10.16, 3.24, 29.58, 5.95, 2.39
      ensemble, 27.44, 8.70, 16.79, 42.71, 9.55, 2.70, 0.00, 0.00, 75.65, 25.51, 36.83, 0.90, 6.27, 0.32, 11.13, 4.31, 31.61, 8.15, 2.70
      ==> sq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, road, parking, sidewalk, other-ground, building, fence, vegetation, trunk, terrain, pole, traffic-sign
      subnet 0, 66.64, 58.42, 62.35, 60.01, 59.58, 59.09, 54.75, 0.00, 75.02, 62.31, 57.57, 69.60, 55.13, 55.76, 52.73, 59.12, 61.43, 54.35, 56.92
      subnet 1, 66.31, 58.21, 63.06, 63.01, 59.65, 58.46, 0.00, 0.00, 74.97, 62.37, 57.99, 66.12, 55.11, 55.67, 52.16, 57.16, 61.72, 54.83, 54.52
      subnet 2, 66.35, 58.97, 65.16, 63.13, 60.61, 56.54, 55.67, 0.00, 75.26, 61.39, 57.35, 0.00, 54.78, 55.43, 52.69, 57.71, 61.29, 54.21, 54.87
      ensemble, 69.20, 59.77, 65.81, 63.30, 61.22, 58.33, 0.00, 0.00, 76.99, 63.37, 59.20, 72.50, 56.39, 62.66, 52.83, 57.40, 62.94, 54.98, 55.88
      ==> rq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, road, parking, sidewalk, other-ground, building, fence, vegetation, trunk, terrain, pole, traffic-sign
      subnet 0, 37.97, 12.56, 23.19, 48.28, 13.06, 3.85, 0.44, 0.00, 98.25, 33.74, 61.16, 1.12, 11.72, 7.81, 20.29, 4.48, 48.63, 11.21, 2.73
      subnet 1, 35.71, 8.14, 15.38, 59.77, 13.38, 4.49, 0.00, 0.00, 98.00, 29.81, 60.56, 1.03, 12.20, 6.55, 19.49, 5.69, 47.85, 10.33, 4.36
      subnet 2, 36.23, 12.16, 25.10, 57.14, 13.43, 3.40, 0.44, 0.00, 98.32, 35.88, 59.97, 0.00, 11.49, 4.67, 19.29, 5.61, 48.26, 10.97, 4.35
      ensemble, 39.65, 14.55, 25.51, 67.47, 15.60, 4.63, 0.00, 0.00, 98.25, 40.25, 62.21, 1.23, 11.11, 0.52, 21.08, 7.52, 50.23, 14.83, 4.83
      [2.5787551403045654, 1.415881872177124, 1.4355676174163818, 1.3622872829437256, 1.3630497455596924, 1.491441011428833]
      inference time:  1.1929051064741991
      [0.03452730178833008, 0.03274846076965332, 0.03339242935180664, 0.03439188003540039, 0.03410673141479492, 0.03477001190185547]
      ensemble time:  0.03270599180123144
      Uncertainty threshold:  0.5
      =====================================
      method, ins ece, ins nll, ssc nonempty ece, ssc empty ece, ssc nonempty nll, ssc empty nll,  count, inference time
      subnet 0,  0.6447, 5.5524, 0.1090, 0.0315, 0.8212, 0.6874, 12014, 0.00
      subnet 1,  0.6582, 5.5356, 0.0870, 0.0323, 0.7848, 0.7034, 12034, 0.00
      subnet 2,  0.6499, 5.5946, 0.1191, 0.0312, 0.8515, 0.6640, 11867, 0.00
      ensemble,  0.5239, 4.5508, 0.0570, 0.0216, 0.6968, 0.4914, 8982, 0.00
      allocated 24330.26740797546
      ```

> [!IMPORTANT]
> Note that **voxel ece = (ssc empty ece + ssc nonempty ece)/2** and **voxel nll = (ssc empty nll + ssc nonempty nll)/2**.
> 
> The inference time reported in the paper was measured on an A100 GPU, making it faster than on a V100. For SemanticKITTI, the time also includes the WaffleIron feature extraction duration.


## 4.2. SSCBench-KITTI360
### 4.2.1 Training
> [!NOTE]
> The generated instance label is supposed to be stored in os.path.join(dataset_preprocess_root, "instance_labels_v2")

1. Change the `kitti360_root`, `kitti360_label_root` and `kitti360_preprocess_root` of the training command below to your data folders respectively.
2. The `log_dir` is the folder to store the training logs and checkpoints.
3. **Train PaSCo with MIMO** (i.e. 1 subnet) using the following command with a batchsize of 2 on 2 V100-32G GPUs (1 item per GPU):

      ```
      cd PaSCo/
      python scripts/train_kitti360.py --bs=2 --n_gpus=2 \
            --exp_prefix=pasco_single_kitti360 --lr=1e-4 \
            --kitti360_root=/gpfsdswork/dataset/KITTI-360 \
            --kitti360_label_root=/gpfsdswork/dataset/SSCBench-KITTI-360 \
            --kitti360_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti360 \
            --log_dir=logs \
            --transformer_dropout=0.2 --n_dropout_levels=3 \
            --data_aug=True --max_angle=10.0 --translate_distance=0.2 --scale_range=0.0 \
            --enable_log=True \
            --alpha=0.0 --n_infers=1

      ```

4. **Train PaSCo (2 subnets)**  by setting **--n_infers=2 (number of subnets = 2)** with batchsize of 2 on 2 A100-80G GPUs (1 items per GPU):
      ```
      cd PaSCo/
      python scripts/train_kitti360.py --bs=2 --n_gpus=2 \
            --exp_prefix=pasco_kitti360 --lr=1e-4 \
            --kitti360_root=/gpfsdswork/dataset/KITTI-360 \
            --kitti360_label_root=/gpfsdswork/dataset/SSCBench-KITTI-360 \
            --kitti360_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti360 \
            --log_dir=logs \
            --transformer_dropout=0.2 --n_dropout_levels=3 \
            --data_aug=True --max_angle=10.0 --translate_distance=0.2 --scale_range=0.0 \
            --enable_log=True \
            --alpha=0.0 --n_infers=2
      ```
## 4.2.2 Evaluation
1. Download the **pretrained checkpoint of [PaSCO](https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/pasco_kitti360.ckpt) or [PaSCO without MIMO](https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/pasco_single_kitti360.ckpt)** and put it into `ckpt` folder or use your trained checkpoint.
      ```
      cd ckpt
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/pasco_single_kitti360.ckpt
      wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/pasco_kitti360.ckpt
      ```
2. Evaluate **PaSCo without MIMO** on 1 V100-32G GPUs (1 item per GPU). `ckpt/pasco_single_kitti360.ckpt` is the path to the downloaded checkpoint:
      ```
      python scripts/eval_kitti360.py --n_infers=1 --model_path=ckpt/pasco_single_kitti360.ckpt \
            --kitti360_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti360 \
            --kitti360_root=/gpfsdswork/dataset/KITTI-360 \
            --kitti360_label_root=/gpfsdswork/dataset/SSCBench-KITTI-360
      ```


3. Evaluate **PaSCo** on 1 V100-32G GPUs (1 item per GPU). `ckpt/pasco_kitti360.ckpt` is the path to the downloaded checkpoint:
      ```
      python scripts/eval_kitti360.py --n_infers=2 --model_path=ckpt/pasco_kitti360.ckpt \
            --kitti360_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti360 \
            --kitti360_root=/gpfsdswork/dataset/KITTI-360 \
            --kitti360_label_root=/gpfsdswork/dataset/SSCBench-KITTI-360
      ```

4. Output looks like following: 
      - **PaSCo without MIMO**:
      ```
      method, P, R, IoU, mIoU, All PQ dagger, All PQ, All SQ, All RQ, Thing PQ, Thing SQ, Thing RQ, Stuff PQ, Stuff SQ, Stuff RQ
      subnet 0, 65.07, 58.86, 44.73, 21.18, 19.50, 10.09, 56.86, 15.67, 3.71, 49.16, 6.39, 13.29, 60.71, 20.31
      ensemble, 65.07, 58.86, 44.73, 21.18, 19.50, 10.09, 56.86, 15.67, 3.71, 49.16, 6.39, 13.29, 60.71, 20.31
      =====================================
      ==> pq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, road, parking, sidewalk, other-ground, building, fence, vegetation, terrain, pole, traffic-sign, other-structure, other-object
      subnet 0, 13.74, 0.00, 1.51, 5.07, 1.00, 0.91, 72.51, 6.78, 37.23, 0.53, 15.95, 0.70, 5.94, 7.28, 1.46, 9.89, 0.22, 0.97
      ensemble, 13.74, 0.00, 1.51, 5.07, 1.00, 0.91, 72.51, 6.78, 37.23, 0.53, 15.95, 0.70, 5.94, 7.28, 1.46, 9.89, 0.22, 0.97
      ==> sq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, road, parking, sidewalk, other-ground, building, fence, vegetation, terrain, pole, traffic-sign, other-structure, other-object
      subnet 0, 58.36, 0.00, 55.35, 56.86, 55.39, 69.01, 76.30, 60.57, 58.73, 62.33, 55.46, 57.64, 53.99, 56.83, 56.50, 65.51, 55.38, 69.22
      ensemble, 58.36, 0.00, 55.35, 56.86, 55.39, 69.01, 76.30, 60.57, 58.73, 62.33, 55.46, 57.64, 53.99, 56.83, 56.50, 65.51, 55.38, 69.22
      ==> rq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, road, parking, sidewalk, other-ground, building, fence, vegetation, terrain, pole, traffic-sign, other-structure, other-object
      subnet 0, 23.55, 0.00, 2.73, 8.91, 1.80, 1.32, 95.03, 11.19, 63.39, 0.85, 28.75, 1.21, 11.00, 12.81, 2.59, 15.10, 0.41, 1.39
      ensemble, 23.55, 0.00, 2.73, 8.91, 1.80, 1.32, 95.03, 11.19, 63.39, 0.85, 28.75, 1.21, 11.00, 12.81, 2.59, 15.10, 0.41, 1.39
      [1.7900667190551758, 0.6379802227020264, 0.8541789054870605, 0.5246858596801758, 0.539797306060791, 0.570587158203125]
      inference time:  0.564805249520018
      [0.004273653030395508, 0.0037992000579833984, 0.0037772655487060547, 0.0036439895629882812, 0.0037114620208740234, 0.0037202835083007812]
      ensemble time:  0.003746812533098316
      Uncertainty threshold:  0.5
      =====================================
      method, ins ece, ins nll, ssc nonempty ece, ssc empty ece, ssc nonempty nll, ssc empty nll,  count, inference time
      subnet 0,  0.7872, 5.3554, 0.2259, 0.1306, 1.1598, 3.5766, 36668, 0.00
      ensemble,  0.7872, 5.3554, 0.2259, 0.1306, 1.1598, 3.5766, 36668, 0.00
      allocated 9058.6876443418
      ```
      - **PaSCo**:
      ```
      method, P, R, IoU, mIoU, All PQ dagger, All PQ, All SQ, All RQ, Thing PQ, Thing SQ, Thing RQ, Stuff PQ, Stuff SQ, Stuff RQ
      subnet 0, 59.27, 68.86, 46.74, 20.39, 20.78, 10.96, 55.53, 17.42, 4.47, 46.48, 7.89, 14.20, 60.05, 22.19
      subnet 1, 57.90, 69.95, 46.37, 20.14, 20.35, 10.57, 58.42, 16.77, 4.02, 56.09, 7.03, 13.84, 59.58, 21.65
      ensemble, 62.66, 65.70, 47.22, 22.07, 28.43, 11.04, 52.86, 17.34, 5.09, 47.10, 8.93, 14.02, 55.74, 21.55
      =====================================
      ==> pq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, road, parking, sidewalk, other-ground, building, fence, vegetation, terrain, pole, traffic-sign, other-structure, other-object
      subnet 0, 14.98, 0.00, 1.20, 7.34, 1.53, 1.79, 70.77, 4.22, 34.66, 0.25, 24.68, 0.62, 10.31, 6.42, 3.31, 12.67, 0.32, 2.19
      subnet 1, 14.12, 0.18, 0.92, 5.76, 1.57, 1.55, 70.25, 4.13, 33.80, 0.29, 24.80, 0.73, 9.59, 6.28, 2.50, 12.09, 0.13, 1.55
      ensemble, 16.57, 0.00, 1.59, 8.89, 1.92, 1.58, 71.71, 4.08, 36.25, 0.00, 23.52, 0.61, 8.91, 4.66, 3.27, 13.23, 0.11, 1.82
      ==> sq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, road, parking, sidewalk, other-ground, building, fence, vegetation, terrain, pole, traffic-sign, other-structure, other-object
      subnet 0, 58.22, 0.00, 52.01, 54.43, 54.62, 59.60, 74.87, 58.31, 57.89, 62.00, 55.82, 57.20, 54.99, 56.37, 56.55, 66.91, 54.67, 65.05
      subnet 1, 57.65, 52.41, 52.77, 56.32, 54.96, 62.41, 74.45, 58.30, 57.92, 58.69, 55.82, 60.48, 54.95, 56.47, 55.98, 67.00, 53.16, 61.78
      ensemble, 58.39, 0.00, 54.00, 55.12, 54.42, 60.70, 75.73, 57.44, 59.22, 0.00, 55.90, 63.81, 55.49, 58.33, 55.86, 67.74, 55.85, 63.47
      ==> rq
      method, car, bicycle, motorcycle, truck, other-vehicle, person, road, parking, sidewalk, other-ground, building, fence, vegetation, terrain, pole, traffic-sign, other-structure, other-object
      subnet 0, 25.72, 0.00, 2.31, 13.49, 2.81, 3.01, 94.52, 7.24, 59.87, 0.41, 44.21, 1.09, 18.74, 11.39, 5.86, 18.94, 0.59, 3.37
      subnet 1, 24.49, 0.35, 1.74, 10.23, 2.85, 2.49, 94.36, 7.08, 58.36, 0.49, 44.43, 1.20, 17.45, 11.12, 4.46, 18.04, 0.25, 2.51
      ensemble, 28.38, 0.00, 2.95, 16.13, 3.53, 2.61, 94.70, 7.11, 61.22, 0.00, 42.09, 0.96, 16.06, 7.99, 5.85, 19.53, 0.19, 2.87
      [2.1090309619903564, 1.5366406440734863, 1.534111738204956, 1.6147339344024658, 1.208867073059082, 1.52060866355896]
      inference time:  1.3322471255736077
      [0.02621603012084961, 0.02443861961364746, 0.027606964111328125, 0.02429676055908203, 0.024660348892211914, 0.0240786075592041]
      ensemble time:  0.024390966416286672
      Uncertainty threshold:  0.5
      =====================================
      method, ins ece, ins nll, ssc nonempty ece, ssc empty ece, ssc nonempty nll, ssc empty nll,  count, inference time
      subnet 0,  0.7668, 4.9833, 0.1879, 0.1252, 1.1738, 2.6776, 35104, 0.00
      subnet 1,  0.7735, 5.1097, 0.1990, 0.1225, 1.1935, 2.6386, 35466, 0.00
      ensemble,  0.5899, 3.8083, 0.1616, 0.1068, 1.1075, 2.1397, 18990, 0.00
      allocated 17902.39387990762
      ```


# 5. Visualization
1. Install Mayavi following [the official instructions](https://docs.enthought.com/mayavi/mayavi/installation.html). 
2. Run the following command to generate the prediction using the downloaded checkpoint `ckpt/pasco_single.ckpt`:
      ```
      cd PaSCo/
      python scripts/save_outputs_panoptic.py --model_path=ckpt/pasco_single.ckpt \
            --dataset_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti \
            --dataset_root=/gpfsdswork/dataset/SemanticKITTI
      ```
3. Draw the generated output:
      ```
      cd PaSCo/
      python scripts/visualize.py
      ```

# Acknowledgment
We thank the authors of the following repositories for making their code and models publicly available:
- https://github.com/valeoai/WaffleIron
- https://github.com/PRBonn/MaskPLS
- https://github.com/SCPNet/Codes-for-SCPNet
- https://github.com/astra-vision/LMSCNet
- https://github.com/yanx27/JS3C-Net
- https://github.com/facebookresearch/Mask2Former
- https://github.com/ENSTA-U2IS-AI/torch-uncertainty


The research was supported by the French project SIGHT (ANR-20-CE23-0016), the ERC Starting Grant SpatialSem (101076253), and the SAMBA collaborative project co-funded by
BpiFrance in the Investissement d’Avenir Program. Computation was performed using HPC resources from GENCI–IDRIS (2023-AD011014102,
AD011012808R2). We thank all Astra-Vision members for their valuable feedbacks, including Andrei Bursuc and Gilles Puy for excellent suggestions and
Tetiana Martyniuk for her kind proofreading.
