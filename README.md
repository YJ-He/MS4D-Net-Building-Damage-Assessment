# **MS4D-Net: Multitask-Based Semi-Supervised Semantic Segmentation Framework with Perturbed Dual Mean Teachers for Building Damage Assessment from High-Resolution Remote Sensing Imagery**

This is a PyTorch implementation of a semi-supervised learning framework for building damage assessment. The manuscript can be visited via https://www.mdpi.com/2072-4292/15/2/478.

## 1. Directory Structure    
You need to first generate lists of pre- and post-image/label files and place as the structure shown below. Every txt file contains the full absolute path of the files, each file per line. To ensure you can understand what are the files, we give three tif files as examples. The image file is the RGB-band file and label file is the 256-bit image with value from 0 to 255.
```
/root
    /train_image_pre.txt
    /train_image_post.txt
    /train_label.txt
    /test_image_pre.txt
    /test_image_post.txt
    /test_label.txt
    /val_image_pre.txt
    /val_image_post.txt
    /val_label.txt
    /train_unsup_image_pre.txt
    /train_unsup_image_post.txt
```
## 2. Code
### Installation
The code is developed using Python 3.7 with PyTorch 1.9.1. The code is developed and tested using singel RTX 2080 Ti GPU.

**(1) Clone this repo.**
```
git clone https://github.com/YJ-He/MS4D-Net-Building-Damage-Assessment.git
```

**(2) Create a conda environment.**  
```
conda env create -f environment.yaml
conda activate building_damage_assessment
```

### Training
1. set `root_dir` and hyper-parameters configuration in `./configs/config.cfg`.
2. run `python train.py`.

### Evaludation
1. set `root_dir` and hyper-parameters configuration in `./configs/config.cfg`.
2. set `pathCkpt` in `test.py` to indicate the model checkpoint file.
3. run `python test.py`.


## 3.Citation
If this repo is useful in your research, please kindly consider citing our paper as follow.
```
@article{he2023ms4d,
  title={MS4D-Net: Multitask-Based Semi-Supervised Semantic Segmentation Framework with Perturbed Dual Mean Teachers for Building Damage Assessment from High-Resolution Remote Sensing Imagery},
  author={He, Yongjun and Wang, Jinfei and Liao, Chunhua and Zhou, Xin and Shan, Bo},
  journal={Remote Sensing},
  volume={15},
  number={2},
  pages={478},
  year={2023},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

**If our work give you some insights and hints, star me please! Thank you~**


