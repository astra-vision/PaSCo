<div align='center'>

# PaSCo: Urban 3D Panoptic Scene Completion with Uncertainty Awareness

CVPR 2024 Oral


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
- [3. Panoptic labels generation](#3-panoptic-labels-generation)


# News
- 06/04/2024: Dataset download instructions and label generation code for SemanticKITTI are now available. KITTI-360 will be released next weekend 🚨.
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
   
5. The **partial dataloader** for the KITTI dataset is available [here](https://github.com/astra-vision/PaSCo/blob/main/pasco/data/semantic_kitti/kitti_dataset.py). The full version will be released later.


## 2.2. KITTI-360



# Acknowledgment
The research was supported by the French project SIGHT (ANR-20-CE23-0016), the ERC Starting Grant SpatialSem (101076253), and the SAMBA collaborative project co-funded by
BpiFrance in the Investissement d’Avenir Program. Computation was performed using HPC resources from GENCI–IDRIS (2023-AD011014102,
AD011012808R2). We thank all Astra-Vision members for their valuable feedbacks, including Andrei Bursuc and Gilles Puy for excellent suggestions and
Tetiana Martyniuk for her kind proofreading.
