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






