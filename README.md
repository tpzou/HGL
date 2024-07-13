# **HGL: Hierarchical Geometry Learning for Test-time Adaptation in 3D Point Cloud [ECCV2024]**

The official implementation of our work "HGL: Hierarchical Geometry Learning for Test-time Adaptation in 3D Point Cloud".

## Introduction
3D point cloud segmentation has received significant interest for its growing applications. However, the generalization ability of models suffers in dynamic scenarios due to the distribution shift between test and training data. To promote robustness and adaptability across diverse scenarios, test-time adaptation (TTA) has recently been introduced. Nevertheless, most existing TTA methods are developed for images, and limited approaches applicable to point clouds ignore the inherent hierarchical geometric structures in point cloud streams, i.e., local (point-level), global (object-level), and temporal (frame-level) structures. In this paper, we delve into TTA in 3D point cloud segmentation and propose a novel Hierarchical Geometry Learning (HGL) framework. HGL comprises three complementary modules from local, global to temporal learning in a bottom-up manner. Technically, we first construct a local geometry learning module for pseudo-label generation. Next, we build prototypes from the global geometry perspective for pseudo-label fine-tuning. Furthermore, we introduce a temporal consistency regularization module to mitigate negative transfer. Extensive experiments on four datasets demonstrate the effectiveness and superiority of our HGL. Remarkably, on the SynLiDAR to SemanticKITTI task, HGL achieves an overall mIoU of 46.91\%, improving GIPSO by 3.0\% and significantly reducing the required adaptation time by 80\%.

### Environment
- [MinkowskiEnginge](https://github.com/NVIDIA/MinkowskiEngine).
- [open3d 0.13.0](http://www.open3d.org)
- [KNN-CUDA](https://github.com/unlimblue/KNN_CUDA)
- [pytorch-lighting 1.4.1](https://www.pytorchlightning.ai)
- [wandb](https://docs.wandb.ai/quickstart)
- [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
- tqdm
- pickle



## Source training

To train the source model on SynLiDAR
```
python train_lighting.py --config_file configs/source/synlidar_source.yaml
```
For Synth4D   ``--config_file configs/source/synth4dkitti_source.yaml``.

For nuScenes ``--config_file configs/source/synth4dnusc_source.yaml``


## Adaptation to target

To adapt the source model SynLiDAR to the target domain SemanticKITTI

```
sh train.sh
``` 
If you want to save point cloud for future visualization you will need to add ``--save_predictions`` and they will be saved in ```pipeline.save_dir```. 






