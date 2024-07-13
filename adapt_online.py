import os
import time
import argparse
import numpy as np
import random
import torch

import models
from models import MinkUNet18_HEADS, MinkUNet18_MCMC
from utils.config import get_config
from utils.collation import CollateSeparated, CollateFN
from utils.dataset_online import get_online_dataset
from utils.online_logger import OnlineWandbLogger, OnlineCSVLogger
from pipelines import OnlineTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/deva/nuscenes_sequence.yaml",
                    type=str,
                    help="Path to config file")
parser.add_argument("--split_size",
                    default=4071,
                    type=int,
                    help="Num frames per sub sequence (SemanticKITTI only)")
parser.add_argument("--drop_prob",
                    default=None,
                    type=float,
                    help="Dropout prob MCMC")
parser.add_argument("--save_predictions",
                    default=False,
                    action='store_true')
parser.add_argument("--note",
                    default=None,
                    type=str)


parser.add_argument("--use_pseudo_new",
                    default=False,
                    action='store_true')
parser.add_argument("--use_prototype",
                    default=False,
                    action='store_true')
parser.add_argument("--use_all_pseudo",
                    default=False,
                    action='store_true')
parser.add_argument("--score_weight",
                    default=False,
                    action='store_true')
parser.add_argument("--loss_use_score_weight",
                    default=False,
                    action='store_true')


parser.add_argument("--without_pre_eval_synlidar2kitti",
                    default=False,
                    action='store_true')
parser.add_argument("--without_pre_eval_synth4d2kitti",
                    default=False,
                    action='store_true')
parser.add_argument("--without_pre_eval_synth4dnusc",
                    default=False,
                    action='store_true')


parser.add_argument("--kitti_sim",
                    default=False,
                    action='store_true')
parser.add_argument("--only_certainty",
                    default=False,
                    action='store_true')
parser.add_argument("--only_purity",
                    default=False,
                    action='store_true')
parser.add_argument("--without_reload",
                    default=False,
                    action='store_true')
parser.add_argument("--save_gem_predictions",
                    default=False,
                    action='store_true')
parser.add_argument("--sample_pos",
                    default=False,
                    action='store_true')
parser.add_argument("--coord_weight",
                    default=False,
                    action='store_true')
parser.add_argument("--use_hard_label",
                    default=False,
                    action='store_true')
parser.add_argument("--BMD_prototype",
                    default=False,
                    action='store_true')
parser.add_argument("--only_use_BMD_prototype",
                    default=False,
                    action='store_true')
parser.add_argument("--score_weight_new",
                    default=False,
                    action='store_true')
parser.add_argument("--use_ema",
                    default=False,
                    action='store_true')
parser.add_argument("--use_pre_label",
                    default=False,
                    action='store_true')
parser.add_argument("--without_ssl_loss",
                    default=False,
                    action='store_true')
parser.add_argument("--only_use_prototype",
                    default=False,
                    action='store_true')


parser.add_argument("--lr",
                    default=0.0,
                    type=float)
parser.add_argument("--ssl_beta",
                    default=1.0,
                    type=float)
parser.add_argument("--pseudo_th",
                    default=0.5,
                    type=float)
parser.add_argument("--loss_eps",
                    default=0.25,
                    type=float)
parser.add_argument("--segmentation_beta",
                    default=1.0,
                    type=float)
parser.add_argument("--max_time_window",
                    default=0,
                    type=int)
parser.add_argument("--loss_method_num",
                    default=0,
                    type=int)
parser.add_argument("--pre_label_num",
                    default=2,
                    type=int)
parser.add_argument("--pre_label_knn",
                    default=1,
                    type=int)
parser.add_argument("--pseudo_knn",
                    default=5,
                    type=int)
parser.add_argument("--seed",
                    default=1234,
                    type=int)

AUG_DICT = None


def get_mini_config(main_c):
    return dict(time_window=main_c.dataset.max_time_window,
                mcmc_it=main_c.pipeline.num_mc_iterations,
                metric=main_c.pipeline.metric,
                cbst_p=main_c.pipeline.top_p,
                th_pseudo=main_c.pipeline.th_pseudo,
                top_class=main_c.pipeline.top_class,
                propagation_size=main_c.pipeline.propagation_size,
                drop_prob=main_c.model.drop_prob)


def train(config, split_size=4071, save_preds=False, args=None):

    mapping_path = config.dataset.mapping_path


    if args.max_time_window != 0:
        config.dataset.max_time_window = args.max_time_window
    
    eval_dataset = get_online_dataset(dataset_name=config.dataset.name,
                                      dataset_path=config.dataset.dataset_path,
                                      voxel_size=config.dataset.voxel_size,
                                      augment_data=config.dataset.augment_data,
                                      max_time_wdw=config.dataset.max_time_window,
                                      version=config.dataset.version,
                                      sub_num=config.dataset.num_pts,
                                      ignore_label=config.dataset.ignore_label,
                                      split_size=split_size,
                                      mapping_path=mapping_path,
                                      num_classes=config.model.out_classes,
                                      args=args)

    adapt_dataset = get_online_dataset(dataset_name=config.dataset.name,
                                       dataset_path=config.dataset.dataset_path,
                                       voxel_size=config.dataset.voxel_size,
                                       augment_data=config.dataset.augment_data,
                                       max_time_wdw=config.dataset.max_time_window,
                                       version=config.dataset.version,
                                       sub_num=config.dataset.num_pts,
                                       ignore_label=config.dataset.ignore_label,
                                       split_size=split_size,
                                       mapping_path=mapping_path,
                                       num_classes=config.model.out_classes,
                                       args=args)

    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)

    if config.model.name == 'MinkUNet18':
        model = MinkUNet18_HEADS(model)

    if config.pipeline.is_double:
        source_model = Model(config.model.in_feat_size, config.model.out_classes)
        if config.pipeline.use_mcmc:
            if args.drop_prob is not None:
                config.model.drop_prob = args.drop_prob

            source_model = MinkUNet18_MCMC(source_model, p_drop=config.model.drop_prob)
    else:
        source_model = None

    if config.pipeline.delayed_freeze_list is not None:
        delayed_list = dict(zip(config.pipeline.delayed_freeze_list, config.pipeline.delayed_freeze_frames))
    else:
        delayed_list = None


    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    if config.pipeline.wandb.run_name is not None:
        run_name = run_time + '_' + config.pipeline.wandb.run_name
    else:
        run_name = run_time

    mini_configs = get_mini_config(config)

    if args.note is not None:
        run_name += f'_{args.note}'
    else:
        for k, v in mini_configs.items():
            run_name += f'_{str(k)}:{str(v)}'

    save_dir = os.path.join(config.pipeline.save_dir, run_name)
    args.save_dir = save_dir
    # save_dir += "_normal_test"
    os.makedirs(save_dir, exist_ok=True)

    wandb_logger = OnlineWandbLogger(project=config.pipeline.wandb.project_name,
                                     entity=config.pipeline.wandb.entity_name,
                                     name=run_name,
                                     offline=config.pipeline.wandb.offline,
                                     config=mini_configs)

    csv_logger = OnlineCSVLogger(save_dir=save_dir,
                                 version='logs')

    loggers = [wandb_logger, csv_logger]

    if args.lr != 0.0:
        config.pipeline.optimizer.lr = args.lr
    trainer = OnlineTrainer(
                            eval_dataset=eval_dataset,
                            adapt_dataset=adapt_dataset,
                            model=model,
                            num_classes=config.model.out_classes,
                            source_model=source_model,
                            criterion=config.pipeline.loss,
                            epsilon=config.pipeline.eps,
                            ssl_criterion=config.pipeline.ssl_loss,
                            ssl_beta=config.pipeline.ssl_beta,
                            seg_beta=config.pipeline.segmentation_beta,
                            optimizer_name=config.pipeline.optimizer.name,
                            adaptation_batch_size=config.pipeline.dataloader.adaptation_batch_size,
                            stream_batch_size=config.pipeline.dataloader.stream_batch_size,
                            lr=config.pipeline.optimizer.lr,
                            clear_cache_int=config.pipeline.trainer.clear_cache_int,
                            scheduler_name=config.pipeline.scheduler.scheduler_name,
                            train_num_workers=config.pipeline.dataloader.num_workers,
                            val_num_workers=config.pipeline.dataloader.num_workers,
                            use_random_wdw=config.pipeline.random_time_window,
                            freeze_list=config.pipeline.freeze_list,
                            delayed_freeze_list=delayed_list,
                            num_mc_iterations=config.pipeline.num_mc_iterations,

                            collate_fn_eval=CollateFN(),
                            collate_fn_adapt=CollateSeparated(),
                            device=config.pipeline.gpu,
                            default_root_dir=config.pipeline.save_dir,
                            weights_save_path=os.path.join(save_dir, 'checkpoints'),
                            loggers=loggers,
                            save_checkpoint_every=config.pipeline.trainer.save_checkpoint_every,
                            source_checkpoint=config.pipeline.source_model,
                            student_checkpoint=config.pipeline.student_model,
                            is_double=config.pipeline.is_double,
                            is_pseudo=config.pipeline.is_pseudo,
                            use_mcmc=config.pipeline.use_mcmc,
                            sub_epochs=config.pipeline.sub_epoch,
                            save_predictions=save_preds,
                            args=args,)

    trainer.adapt_double()

def set_random_seed(seed=0):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    set_random_seed(args.seed)
    train(config, split_size=args.split_size, save_preds=args.save_predictions, args=args)
