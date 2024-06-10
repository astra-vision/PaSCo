<div align='center'>
 
# PaSCo: Urban 3D Panoptic Scene Completion with Uncertainty Awareness

CVPR 2024 Oral - Best paper award candidate


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

# Table of Content
- [News](#news) 
- [1. Installation](#1-installation)
- [2. Data](#2-data)
- [3. Panoptic labels generation](#3-panoptic-label-generation)
- [4. Training and evaluation](#4-training-and-evaluation)


# News
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

5. Install the additional dependencies:
      ```
      cd PaSCo/
      pip install -r requirements.txt
      ```

6. Install PaSCo
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
        ├── poses
        └── sequences
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
> This command doesn't need GPU. Processing 4649 files took approximately 10 hours using 10 processes. The number of processes can be adjusted by modifying the `n_process` parameter.
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
   


## 3.2. KITTI-360
WORK IN PROGRESS

# 4. Training and evaluation
## 4.1. PaSCo w/o MIMO
## 4.1.1 Training
> [!NOTE]
> The generated instance label is supposed to be stored in os.path.join(dataset_preprocess_root, "instance_labels_v2")
1. Change the `dataset_preprocess_root` and `dataset_root` to the preprocess and raw data folder respectively.
2. The `log_dir` is the folder to store the training logs and checkpoints.
3. Run the following command to train PaSCo w/o MIMO with batchsize of 2 on 2 V100-32G GPUs (1 item per GPU):

      ```
      cd PaSCo/
      python scripts/train.py --bs=2 --n_gpus=2 \
            --dataset_preprocess_root=/gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/kitti \
            --dataset_root=/gpfsdswork/dataset/SemanticKITTI \
            --log_dir=logs \
            --exp_prefix=pasco_single --lr=1e-4 --seed=0 \
            --data_aug=True --max_angle=30.0 --translate_distance=0.2 \
            --enable_log=True \
            --sample_query_class=True --n_infers=1
    
      ```
> [!NOTE]
> During training, the reported metric is lower than the final metrics because we limit the number of generated voxels to prevent running out of memory. The final metric is determined during evaluation and is used solely to assess if the training is progressing well.

## 4.1.2 Evaluation
1. Download the pretrained checkpoint at [here]() and put it into `ckpt` folder or use your trained checkpoint.
2. Run the following command to evaluate PaSCo w/o MIMO on 1 V100-32G GPUs (1 item per GPU). CHECKPOINT_PATH is the path to the downloaded checkpoint:
      ```
      python scripts/eval.py --n_infers=1 --model_path=ckpt/pasco_single.ckpt
      ```

      

3. Output looks like following:
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
> [!NOTE]
> Note that **voxel ece = (ssc empty ece + ssc nonempty ece)/2** and **voxel nll = (ssc empty nll + ssc nonempty nll)/2**.
> The inference time reported in the paper was measured on an A100 GPU. So it will be faster than on v100.

## 4.2. PaSCo w/ MIMO
WORK IN PROGRESS

# Acknowledgment
The research was supported by the French project SIGHT (ANR-20-CE23-0016), the ERC Starting Grant SpatialSem (101076253), and the SAMBA collaborative project co-funded by
BpiFrance in the Investissement d’Avenir Program. Computation was performed using HPC resources from GENCI–IDRIS (2023-AD011014102,
AD011012808R2). We thank all Astra-Vision members for their valuable feedbacks, including Andrei Bursuc and Gilles Puy for excellent suggestions and
Tetiana Martyniuk for her kind proofreading.
