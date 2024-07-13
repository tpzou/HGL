import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import csv
import pickle
import open3d as o3d
from knn_cuda import KNN
from sklearn.metrics import davies_bouldin_score

from utils.losses import CELoss, SoftCELoss, DICELoss, SoftDICELoss, HLoss, SCELoss
from utils.collation import CollateSeparated, CollateStream
from utils.sampler import SequentialSampler
from utils.dataset_online import PairedOnlineDataset, FrameOnlineDataset
from models import MinkUNet18_HEADS, MinkUNet18_SSL, MinkUNet18_MCMC
from knn_cuda import KNN
import csv
import random
import copy
from pytorch3d.ops import knn_points, knn_gather
import math
import time

MemoryBank_Data = []
MemoryBank_Label = []

Label_Bank = []
Coordinate_Bank = []
prototypes = torch.zeros(7, 96).cuda()
score_list = []
score_list_new = []


time_lists = []


# freeze BN
def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def get_cbst_th(preds, vals):
    pc = torch.unique(preds)
    c_th = torch.zeros(pc.max()+1)
    for c in pc:
        c_idx = preds == c
        vals_c, _ = torch.sort(vals[c_idx], descending=False)
        p = 0.8
        c_th[c] = vals_c[torch.floor(torch.tensor((vals_c.shape[0]-1)*p)).long()]
    return c_th


def get_cbst_th_2(preds, vals, p=0.5):
    pc = torch.unique(preds)
    c_th = torch.zeros(pc.max()+1)
    for c in pc:
        c_idx = preds == c
        vals_c, _ = torch.sort(vals[c_idx], descending=False)
        c_th[c] = vals_c[torch.floor(torch.tensor((vals_c.shape[0]-1)*p)).long()]
    return c_th


class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = model
        self.alpha = alpha

    def update(self, model):
        # decay = min(1 - 1 / (self.step + 1), self.alpha)
        decay = self.alpha
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data

        # if self.step > 1000:
        #     for ema_param, param in zip(self.model.parameters(), model.parameters()):
        #         ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        # else:
        #     pass
        self.step += 1

def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[: j]),
                                                high=sum(seg_num_list[: j+1]),
                                                size=int(samp_num[i, j])).tolist()
    
    return negative_index


def label_onehot(inputs, num_class):
    '''
    inputs is class label
    return one_hot label 
    dim will be increasee
    '''
    inputs = torch.relu(inputs)
    outputs = torch.zeros([inputs.shape[0], num_class]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)





def BMD_prototype_update(prototypes, all_cls_out, all_emd_feat):
    class_num = 7
    topk_seg = 3
    alpha = 0.99

    topk_num = max(all_emd_feat.shape[0] // (class_num * topk_seg), 1)
    _, all_psd_label = torch.max(all_cls_out, dim=1)
    for cls_idx in range(class_num):
        with torch.no_grad():
            feat_samp_idx = torch.topk(all_cls_out[:, cls_idx], topk_num)[1]
            feat_cls_sample = all_emd_feat[feat_samp_idx, :]
            proto_rep_ = torch.mean(feat_cls_sample, dim=0, keepdim=True)
            if (prototypes[cls_idx].sum() == torch.tensor(0.0)):
                prototypes[cls_idx] = proto_rep_
            else:
                # Update gloal prototype
                prototypes[cls_idx] = alpha * prototypes[cls_idx] + (1 - alpha) * proto_rep_
    



def prototype_update(rep, label, mask, prototypes):
    num_segments = 7
    topk_seg = 3
    alpha = 0.99

    valid_pixel_all_prt = label * mask.unsqueeze(-1).repeat(1, num_segments)
    for i in range(num_segments): #7
        valid_pixel_gather = valid_pixel_all_prt[:, i]
        if valid_pixel_gather.sum() == 0:
            continue

        with torch.no_grad():
            proto_rep_ = torch.mean((rep[valid_pixel_gather.bool()]), dim=0, keepdim=True)
            if (prototypes[i].sum() == torch.tensor(0.0)):
                prototypes[i] = proto_rep_
            else:
                # Update gloal prototype
                prototypes[i] = alpha * prototypes[i] + (1 - alpha) * proto_rep_



class Contrast_Loss(nn.Module):
    def __init__(self, num_queries=256, num_negatives=512, temp=0.5, mean=False, strong_threshold=0.9, alpha=0.99):
        super(Contrast_Loss, self).__init__()
        self.temp = temp
        self.mean = mean
        self.num_queries = num_queries
        self.num_negatives = num_negatives
        self.strong_threshold = strong_threshold
        self.alpha = alpha

    def forward(self, rep, label, mask, prob, prototypes):
        # we gather all representations (mu and sigma) cross mutiple GPUs during this progress
        with torch.no_grad():
            rep_prt = rep # For protoype computing on all cards (w/o gradients)
        size, num_feat = rep.shape
        num_segments = label.shape[1] #7
        valid_pixel_all = label * mask.unsqueeze(-1).repeat(1, num_segments)
        with torch.no_grad():
            valid_pixel_all_prt = (valid_pixel_all) # For protoype computing on all cards 

        rep_all_list = []
        rep_hard_list = []
        num_list = []
        proto_rep_list = []

        for i in range(num_segments): #7
            valid_pixel = valid_pixel_all[:, i]
            valid_pixel_gather = valid_pixel_all_prt[:, i]
            if valid_pixel.sum() == 0:
                continue
            prob_seg = prob[:, i]
            rep_mask_hard = (prob_seg < self.strong_threshold) * valid_pixel.bool() # Only on single card

            with torch.no_grad():
                proto_rep_ = torch.mean((rep_prt[valid_pixel_gather.bool()]), dim=0, keepdim=True)
                if (prototypes[i].sum() == torch.tensor(0.0)):
                    proto_rep_list.append(proto_rep_)
                    prototypes[i] = proto_rep_
                else:
                    # Update gloal prototype
                    prototypes[i] = self.alpha * prototypes[i] + (1 - self.alpha) * proto_rep_
                    proto_rep_list.append(prototypes[i].unsqueeze(0))

            rep_all_list.append(rep[valid_pixel.bool()])
            rep_hard_list.append(rep[rep_mask_hard])
            num_list.append(int(valid_pixel.sum().item()))

        # Compute Probabilistic Representation Contrastive Loss
        if (len(num_list) <= 1) : # in some rare cases, a small mini-batch only contain 1 or no semantic class
            return torch.tensor(0.0) + 0 * rep.sum() # A trick for avoiding data leakage in DDP training
        else:
            contrast_loss = torch.tensor(0.0)
            proto_rep = torch.cat(proto_rep_list) # [c]
            valid_num = len(num_list)
            seg_len = torch.arange(valid_num)

            for i in range(valid_num):
                if len(rep_hard_list[i]) > 0:
                    # Random Sampling anchor representations
                    sample_idx = torch.randint(len(rep_hard_list[i]), size=(self.num_queries, ))
                    anchor_rep = rep_hard_list[i][sample_idx]
                else:
                    continue
                with torch.no_grad():
                    # Select negatives
                    id_mask = torch.cat(([seg_len[i: ], seg_len[: i]]))

                    proto_sim = torch.cosine_similarity(proto_rep[id_mask[0]].unsqueeze(0), proto_rep[id_mask[1:]], dim=1)
                    proto_prob = torch.softmax(proto_sim / self.temp, dim=0)
                    negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                    samp_class = negative_dist.sample(sample_shape=[self.num_queries, self.num_negatives])
                    samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)
                    negative_num_list = num_list[i+1: ] + num_list[: i]
                    negative_index = negative_index_sampler(samp_num, negative_num_list)
                    negative_rep_all = torch.cat(rep_all_list[i+1: ] + rep_all_list[: i])
                    negative_rep = negative_rep_all[negative_index].reshape(self.num_queries, self.num_negatives, num_feat)
                    positive_rep = proto_rep[i].unsqueeze(0).unsqueeze(0).repeat(self.num_queries, 1, 1)
                    all_rep = torch.cat((positive_rep, negative_rep), dim=1)
                
                logits = torch.cosine_similarity(anchor_rep.unsqueeze(1), all_rep, dim=2)
                contrast_loss = contrast_loss + F.cross_entropy(logits / self.temp, torch.zeros(self.num_queries).long().cuda())

            return contrast_loss / valid_num


class OnlineTrainer(object):

    def __init__(self,
                 model,
                 eval_dataset,
                 adapt_dataset,
                 source_model=None,
                 optimizer_name='SGD',
                 criterion='CELoss',
                 epsilon=0.,
                 ssl_criterion='Cosine',
                 ssl_beta=0.5,
                 seg_beta=1.0,
                 temperature=0.5,
                 lr=1e-3,
                 stream_batch_size=1,
                 adaptation_batch_size=2,
                 weight_decay=1e-5,
                 momentum=0.8,
                 val_batch_size=6,
                 train_num_workers=10,
                 val_num_workers=10,
                 num_classes=7,
                 clear_cache_int=2,
                 scheduler_name='ExponentialLR',
                 pseudor=None,
                 use_random_wdw=False,
                 freeze_list=None,
                 delayed_freeze_list=None,
                 num_mc_iterations=10,
                 use_global=False,

                 collate_fn_eval=None,
                 collate_fn_adapt=None,
                 device='cpu',
                 default_root_dir=None,
                 weights_save_path=None,
                 loggers=None,
                 save_checkpoint_every=2,
                 source_checkpoint=None,
                 student_checkpoint=None,
                 boost=True,
                 save_predictions=False,
                 is_double=True,
                 is_pseudo=True,
                 use_mcmc=True,
                 sub_epochs=0,
                 args=None):

        super().__init__()


        for name, value in list(vars().items()):
            if name != "self":
                setattr(self, name, value)

        # loss
        if criterion == 'CELoss':
            self.criterion = CELoss(ignore_label=self.adapt_dataset.ignore_label,
                                    weight=None)

        elif criterion == 'WCELoss':
            self.criterion = CELoss(ignore_label=self.adapt_dataset.ignore_label,
                                    weight=self.adapt_dataset.weights)

        elif criterion == 'SoftCELoss':
            self.criterion = SoftCELoss(ignore_label=self.adapt_dataset.ignore_label)

        elif criterion == 'DICELoss':
            self.criterion = DICELoss(ignore_label=self.adapt_dataset.ignore_label)
        elif criterion == 'SoftDICELoss':
            self.criterion = SoftDICELoss(ignore_label=self.adapt_dataset.ignore_label,
                                          neg_range=True, eps=self.args.loss_eps)

        elif criterion == 'SCELoss':
            self.criterion = SCELoss(alpha=1, beta=0.1, num_classes=self.num_classes, ignore_label=self.adapt_dataset.ignore_label)
        else:
            raise NotImplementedError

        if self.ssl_criterion == 'CosineSimilarity':
            self.ssl_criterion = nn.CosineSimilarity(dim=-1)
        else:
            raise NotImplementedError

        self.ignore_label = self.eval_dataset.ignore_label
        self.global_step = 0
        self.max_time_wdw = self.eval_dataset.max_time_wdw
        self.delayed_freeze_list = delayed_freeze_list
        self.topk_matches = 0
        self.dataset_name = self.adapt_dataset.name
        self.configure_optimizers()
    
        ########################################
        if device is not None:
            self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cpu')

        self.default_root_dir = default_root_dir
        self.weights_save_path = weights_save_path
        self.loggers = loggers
        self.save_checkpoint_every = save_checkpoint_every
        self.source_checkpoint = source_checkpoint
        self.student_checkpoint = student_checkpoint


        self.is_double = is_double
        self.use_mcmc = use_mcmc
        self.model = model

        if self.is_double:
            self.source_model = source_model

        self.eval_dataset = eval_dataset
        self.adapt_dataset = adapt_dataset

        self.max_time_wdw = self.eval_dataset.max_time_wdw

        self.eval_dataset.eval()
        self.adapt_dataset.train()

        self.online_sequences = np.arange(self.adapt_dataset.num_sequences())
        self.num_frames = len(self.eval_dataset)

        self.collate_fn_eval = collate_fn_eval
        self.collate_fn_adapt = collate_fn_adapt
        self.collate_fn_eval.device = self.device
        self.collate_fn_adapt.device = self.device

        self.sequence = -1

        self.adaptation_results_dict = {s: [] for s in self.online_sequences}
        self.source_results_dict = {s: [] for s in self.online_sequences}

        # for speed up
        self.eval_dataloader = None
        self.adapt_dataloader = None

        self.boost = boost

        self.save_predictions = save_predictions

        self.is_pseudo = is_pseudo
        self.sub_epochs = sub_epochs
        self.num_classes = num_classes

        self.args = args


    def freeze(self):
        # here we freeze parts that have to be frozen forever
        if self.freeze_list is not None:
            for name, p in self.model.named_parameters():
                for pf in self.freeze_list:
                    if pf in name:
                        p.requires_grad = False

    def delayed_freeze(self, frame):
        # here we freeze parts that have to be frozen only for a certain period
        if self.delayed_freeze_list is not None:
            for name, p in self.model.named_parameters():
                for pf, frame_act in self.delayed_freeze_list.items():
                    if pf in name and frame <= frame_act:
                        p.requires_grad = False

    def entropy_loss(self, p):
        p = F.softmax(p, dim=1)
        log_p = F.log_softmax(p, dim=1)
        loss = -torch.sum(p * log_p, dim=1)
        return loss


    def configure_optimizers(self):

        parameters = self.model.parameters()

        if self.scheduler_name is None:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(parameters,
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(parameters,
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError
            self.optimizer = optimizer
            self.scheduler = None

        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(parameters,
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay)
            elif self.optimizer_name == 'Adam' or self.optimizer_name == 'ADAM':
                optimizer = torch.optim.Adam(parameters,
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError
            if self.scheduler_name == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            elif self.scheduler_name == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            elif self.scheduler_name == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr/10000, max_lr=self.lr,
                                                              step_size_up=5, mode="triangular2")
            else:
                raise NotImplementedError
            self.optimizer = optimizer
            self.scheduler = scheduler


    def get_online_dataloader(self, dataset, is_adapt=False):
        if is_adapt:
            collate = CollateSeparated(torch.device('cpu'))
            sampler = SequentialSampler(dataset, is_adapt=True, adapt_batchsize=self.adaptation_batch_size,
                                        max_time_wdw=self.max_time_wdw)
            dataloader = DataLoader(dataset,
                                    collate_fn=collate,
                                    sampler=sampler,
                                    pin_memory=False,
                                    num_workers=self.train_num_workers)
        else:
            # collate = CollateFN(torch.device('cpu'))
            collate = CollateStream(torch.device('cpu'))
            sampler = SequentialSampler(dataset, is_adapt=False, adapt_batchsize=self.stream_batch_size)
            dataloader = DataLoader(dataset,
                                    collate_fn=collate,
                                    sampler=sampler,
                                    pin_memory=False,
                                    num_workers=self.train_num_workers)
        return dataloader

    def save_pcd(self, batch, preds, labels, save_path, frame, is_global=False):
        pcd = o3d.geometry.PointCloud()

        if not is_global:
            pts = batch['coordinates']
            pcd.points = o3d.utility.Vector3dVector(pts[:, 1:])
        else:
            pts = batch['global_points'][0]
            pcd.points = o3d.utility.Vector3dVector(pts)
        if self.num_classes == 7 or self.num_classes == 2:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[labels+1])
        else:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[labels])

        # os.makedirs(os.path.join(save_path, 'gt'), exist_ok=True)
        # o3d.io.write_point_cloud(os.path.join(save_path, 'gt', str(frame)+'.ply'), pcd)
        if self.num_classes == 7 or self.num_classes == 2:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[preds+1])
        else:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[preds])

        os.makedirs(os.path.join(save_path, 'preds'), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_path, 'preds', str(frame)+'.ply'), pcd)


    def save_pcd_wogt_1(self, batch, preds, labels, save_path, frame, is_global=False):
        pcd = o3d.geometry.PointCloud()

        if not is_global:
            pts = batch['coordinates_all'][0]
            pcd.points = o3d.utility.Vector3dVector(pts[:, :])
        else:
            pts = batch['global_pts'][0]
            pcd.points = o3d.utility.Vector3dVector(pts)
        if self.num_classes == 7 or self.num_classes == 2:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[preds+1])
        else:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[preds])

        os.makedirs(os.path.join(save_path, 'preds'), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_path, 'preds', str(frame)+'.ply'), pcd)

    def save_pcd_wogt_2(self, batch, preds, labels, save_path, frame, is_global=False):
        pcd = o3d.geometry.PointCloud()

        if not is_global:
            pts = batch['coordinates0']
            pcd.points = o3d.utility.Vector3dVector(pts[:, 1:])
        else:
            pts = batch['global_pts'][0]
            pcd.points = o3d.utility.Vector3dVector(pts)
        if self.num_classes == 7 or self.num_classes == 2:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[preds+1])
        else:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[preds])

        os.makedirs(os.path.join(save_path, 'preds'), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_path, 'preds', str(frame)+'.ply'), pcd)


    def adaptation_double_pseudo_step(self, batch, frame):

        self.model.train()
        self.freeze()
        self.source_model.eval()

        coords = batch["coordinates_all"][0]

        batch_all = torch.zeros([coords.shape[0], 1])
        coords_all = torch.cat([batch_all, coords], dim=-1)
        feats_all = torch.ones([coords_all.shape[0], 1]).float()

        # we assume that data the loader gives frames in pairs
        stensor_all = ME.SparseTensor(coordinates=coords_all.int().to(self.device),
                                     features=feats_all.to(self.device),
                                     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        # pseudo label generation
        with torch.no_grad():
            global score_list
            if self.args.use_ema:
                self.ema.model.eval()
                source_tmp, source_feats, source_bottle = self.ema.model(stensor_all, is_train=False)
            else:
                self.source_model.eval()
                source_tmp, source_feats, source_bottle = self.source_model(stensor_all, is_train=False)
            source_tmp = F.softmax(source_tmp, dim=-1).unsqueeze(0) # [1, N, C]


            if self.args.use_pseudo_new:
                # local geometric label aggregation
                K = self.args.pseudo_knn
                class_num = source_tmp.shape[-1]

                global_pts = batch["global_pts0"].unsqueeze(0).float().cuda() # [1,N,3]
                dists, idx, nn = knn_points(global_pts, global_pts, K=K, return_nn=True) # [1, N, K], [1, N, K], [1, N, K, 3]
                knn_tmp = knn_gather(source_tmp, idx) #[1, N, K, 7]

                if self.args.use_hard_label:
                    knn_predict = torch.argmax(knn_tmp.squeeze().reshape(-1, class_num), dim=-1)  # [N*K]
                    knn_one_hot = F.one_hot(knn_predict, num_classes=class_num).float() # [N*K, 7]
                    knn_tmp = knn_one_hot.reshape(1, -1, K, class_num) # [1, N, K, 7]

                knn_dist = torch.exp(-dists) # [1,N,K]
                knn_dist = knn_dist / knn_dist.sum(dim=-1, keepdim=True)
                knn_dist = knn_dist.unsqueeze(-1).repeat(1,1,1,class_num) # [1,N,K,7]
                source_label = torch.sum(knn_tmp * knn_dist, dim=2) / K # [1,N,7]

                # local purity
                p = source_label
                point_certainty = 1.0 - torch.sum(-p * torch.log(p + 1e-6), dim=-1) / math.log(class_num)  #[1, N]

                predict = torch.argmax(p.squeeze(), dim=-1)  # [N]
                one_hot = F.one_hot(predict, num_classes=class_num).float() # [N, 7]
                knn_label = knn_gather(one_hot.unsqueeze(0), idx) #[1, N, K, 7] 
                knn_label = torch.mean(knn_label, dim=2)
                region_purity = 1.0 - torch.sum( - knn_label * torch.log(knn_label + 1e-6), dim=-1) / math.log(class_num)  # [1, N]

                score = point_certainty * region_purity  # [1, N]

                if self.args.only_certainty:
                    score = point_certainty
                if self.args.only_purity:
                    score = region_purity

            else:
                # local purity
                p = source_tmp # [1, N, C]
                class_num = source_tmp.shape[-1]
                source_label = source_tmp
                point_certainty = 1.0 - torch.sum(-p * torch.log(p + 1e-6), dim=-1) / math.log(class_num)  #[1, N]
                score = point_certainty
            
            score_list.append(score.detach())
            # compute pseudo labels
            pseudo_logits_rep, pseudo_labels_rep = torch.max(source_label.squeeze(), dim=1)
            pseudo = pseudo_labels_rep
            pseudo_logits_rep = score.squeeze()

            class_th = get_cbst_th_2(pseudo, pseudo_logits_rep, p=self.args.pseudo_th)
            present_classes = torch.unique(pseudo)
            new_pseudo = -torch.ones(pseudo.shape[0]).long().cuda()
            main_idx = torch.arange(pseudo.shape[0])
            valid_pseudo = []
            for c in present_classes:
                c_idx = main_idx[pseudo == c]
                pseudo_logits_rep_c = pseudo_logits_rep[c_idx]
                valid_unc = pseudo_logits_rep_c > class_th[c]
                c_idx = c_idx[valid_unc]
                new_pseudo[c_idx] = c
                valid_pseudo.append(c_idx)

            valid_pseudo = torch.cat(valid_pseudo)
            pseudo0 = new_pseudo.detach()
            pseudo_all = pseudo_labels_rep.clone().detach()


            if self.args.save_predictions or self.args.save_gem_predictions:
                save_path = os.path.join(self.weights_save_path, 'pcd')
                phase = 'Gem_pseudo'
                save_path_tmp = os.path.join(save_path, phase)
                preds = pseudo0
                labels = batch['labels0'].long().cuda()
                self.save_pcd_wogt_1(batch, preds.cpu().numpy(),
                            labels.cpu().numpy(), save_path_tmp, frame,
                            is_global=False)

        # knn-pseudo
        if self.args.use_pre_label:

            if len(Label_Bank) == 0:
                Label_Bank.append(pseudo0.clone())
                Coordinate_Bank.append(batch["global_pts0"].unsqueeze(0).float().cuda().clone())  #[1,N,3]
                previous_label = pseudo0.clone()
            else:
                Label_Bank.append(pseudo0.clone())
                Coordinate_Bank.append(batch["global_pts0"].unsqueeze(0).float().cuda().clone())  #[1,N,3]
                if len(Label_Bank) > self.args.pre_label_num:
                    Label_Bank.pop(0)
                    Coordinate_Bank.pop(0)
                
                previous_label_list = []
                for i in range(len(Label_Bank)-1):
                    global_pts0 = Coordinate_Bank[-1].clone()
                    global_pts1 = Coordinate_Bank[i].clone()
                    K = self.args.pre_label_knn
                    dists, idx, _ = knn_points(global_pts0, global_pts1, K=K, return_nn=True) # [1, N, K], [1, N, K], [1, N, K, 3]
                    previous_label = knn_gather(Label_Bank[i].unsqueeze(0).unsqueeze(-1).clone(), idx) #[1, N, K, C] 
                    previous_label = previous_label.squeeze(-1) #[N, K]
                    previous_label_list.append(previous_label)
                previous_label_list = torch.cat(previous_label_list, dim=0) #[pre_label_num, N, K]
                previous_label_list = previous_label_list.permute(1, 0, 2) #[N, pre_label_num, K]
                N = previous_label_list.shape[0]
                previous_label_list = previous_label_list.reshape(N, -1) #[N, pre_label_num*K]
                # 取出现次数最多的label
                # torch one-hot
                previous_label_list = previous_label_list.reshape(-1)
                previous_label_list = previous_label_list + 1
                previous_label_list = label_onehot(previous_label_list, 8) #[N*pre_label_num*K, 8]
                previous_label_list = previous_label_list.reshape(N, -1, 8) #[N, pre_label_num*K, 8]
                previous_label_list = previous_label_list.sum(dim=1) #[N, 8]
                previous_label_list = previous_label_list.argmax(dim=-1) #[N]
                previous_label_list = previous_label_list - 1
                previous_label = previous_label_list

            mask = (pseudo0 == -1)
            pseudo0[mask] = previous_label[mask]


        if (pseudo0 != -1).sum() > 0:
            # we assume that data the loader gives frames in pairs
            stensor0 = ME.SparseTensor(coordinates=batch["coordinates0"].int().to(self.device),
                                       features=batch["features0"].to(self.device),
                                       quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

            stensor1 = ME.SparseTensor(coordinates=batch["coordinates1"].int().to(self.device),
                                       features=batch["features1"].to(self.device),
                                       quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

            # Must clear cache at regular interval
            if self.global_step % self.clear_cache_int == 0:
                torch.cuda.empty_cache()

            self.optimizer.zero_grad()

            # forward in mink
            out_seg0, out_en0, out_pred0, out_bck0, _, out_seg1, out_en1, out_pred1, out_bck1, _ = self.model((stensor0, stensor1))
            
            # segmentation loss for t0
            labels0 = batch['labels0'].long()

            # prototype generation
            if self.args.use_prototype:
                global prototypes
                Contrast_loss = Contrast_Loss()
                rep_all = out_bck0
                pseudo0_2 = pseudo0.clone()
                pseudo0_2[pseudo0_2 == -1] = 0
                label_all = label_onehot(pseudo0_2, 7)
                mask_all = pseudo0 != -1
                pred_all = out_seg0

                if self.args.BMD_prototype:
                    source_label = source_label * score.unsqueeze(-1).repeat(1,1,7) # [1, N, 7]
                    BMD_prototype_update(prototypes.cuda(), source_label.squeeze(), out_bck0)
                elif self.args.use_prototype:
                    prototype_update(rep_all, label_all, mask_all, prototypes.cuda())
                else:
                    contrast_loss = Contrast_loss(rep_all.cuda(), label_all.cuda(), mask_all.cuda(), pred_all.cuda(), prototypes.cuda())


                norm_rep_u = F.normalize(out_bck0, dim=-1) # [N, C]
                norm_proto = F.normalize(prototypes, dim=-1).permute(1, 0) # [C, C]
                sim_mat = torch.mm(norm_rep_u, norm_proto) # [N, C] * [C, C] = [N, C]
                temp = 0.25
                # temp = 1.0
                num_classes = 7
                pseudo_logits_rep, pseudo_labels_rep = torch.max(F.softmax(sim_mat / temp, dim=1), dim=1)

                label_mask = pseudo0.eq(pseudo_labels_rep)
                label_mask = (~label_mask).float()
                pseudo_labels = pseudo0 - label_mask * num_classes
                pseudo_labels[pseudo_labels < 0] = -1
                pseudo_labels = pseudo_labels

                if self.args.use_all_pseudo:
                    label_mask = pseudo_all.eq(pseudo_labels_rep)
                    label_mask = (~label_mask).float()
                    pseudo_labels = pseudo_all - label_mask * num_classes
                    pseudo_labels[pseudo_labels < 0] = -1
                    pseudo_labels = pseudo_labels

                # pseudo_labels = pseudo_labels.long()
                # pseudo_labels[pseudo_labels == -1] = pseudo_labels_rep[pseudo_labels == -1]

                # labels0 = batch['labels0'].long().cuda()
                # valid_idx_pseudo = torch.logical_and(pseudo_labels != -1, labels0 != -1)
                # # valid_idx_pseudo = labels0 != -1
                # pseudo_acc = (pseudo_labels[valid_idx_pseudo] == labels0[valid_idx_pseudo]).sum() / labels0[valid_idx_pseudo].shape[0]
                # print((pseudo_labels[valid_idx_pseudo] == labels0[valid_idx_pseudo]).sum())
                # print(labels0[valid_idx_pseudo].shape[0])
                # print(pseudo_acc)

                if self.args.only_use_prototype:
                    pseudo = pseudo_labels_rep
                    class_th = get_cbst_th(pseudo, pseudo_logits_rep)
                    present_classes = torch.unique(pseudo)
                    new_pseudo = -torch.ones(pseudo.shape[0]).long().cuda()
                    main_idx = torch.arange(pseudo.shape[0])
                    valid_pseudo = []
                    for c in present_classes:
                        c_idx = main_idx[pseudo == c]
                        pseudo_logits_rep_c = pseudo_logits_rep[c_idx]
                        valid_unc = pseudo_logits_rep_c > class_th[c]
                        # valid_unc = pseudo_logits_rep_c < class_th[c]
                        c_idx = c_idx[valid_unc]
                        new_pseudo[c_idx] = c
                        valid_pseudo.append(c_idx)

                    pseudo_labels = -torch.ones(pseudo_labels_rep.shape[0]).long().cuda()
                    valid_pseudo = torch.cat(valid_pseudo)

                    pseudo_labels = new_pseudo.detach()

                if self.args.only_use_BMD_prototype:
                    pseudo = pseudo_labels_rep
                    pseudo_labels = new_pseudo.detach()

                if self.args.use_prototype:
                    pseudo0 = pseudo_labels.long()

                if self.args.save_predictions:
                    save_path = os.path.join(self.weights_save_path, 'pcd')
                    phase = 'Sem_pseudo'
                    save_path_tmp = os.path.join(save_path, phase)
                    preds = pseudo0
                    labels = batch['labels0'].long().cuda()
                    self.save_pcd_wogt_2(batch, preds.cpu().numpy(),
                                labels.cpu().numpy(), save_path_tmp, frame,
                                is_global=False)

            if self.args.loss_use_score_weight:
                loss_seg_head = self.criterion(out_seg0, pseudo0, score=score, loss_method_num=self.args.loss_method_num)
            else:
                loss_seg_head = self.criterion(out_seg0, pseudo0)


            pseudo0 = pseudo0
            labels0 = labels0

            # get matches in t0 and t1 (used for selection)
            matches0 = batch['matches0'].to(self.device)
            matches1 = batch['matches1'].to(self.device)


            # 2FUTURE CONTRASTIVE
            # forward preds (t0 -> t1)
            future_preds = torch.index_select(out_pred0, 0, matches0)
            # forward gt feats and stop grad
            future_gt = torch.index_select(out_en1.detach(), 0, matches1)
            future_neg_cos_sim = -self.ssl_criterion(future_preds, future_gt)


            if self.args.sample_pos or self.args.coord_weight or self.args.score_weight:
                if self.topk_matches > 0:
                    # select top-k worst performing matches
                    future_neg_cos_sim = future_neg_cos_sim.topk(self.topk_matches, dim=0).values.mean()
                else:
                    if self.args.sample_pos:
                        future_seg = torch.index_select(out_seg1, 0, matches1)
                        future_seg = F.softmax(future_seg, dim=1)
                        future_seg, _ = torch.max(future_seg, dim=1)
                        future_neg_cos_sim = (future_neg_cos_sim * future_seg)
                    if self.args.coord_weight:
                        global_pts0 = batch["global_pts0"].unsqueeze(0).float().cuda()
                        global_pts1 = batch["global_pts1"].unsqueeze(0).float().cuda()
                        K = 1
                        dists, idx, _ = knn_points(global_pts1, global_pts0, K=K, return_nn=True) # [1, N, K], [1, N, K], [1, N, K, 3]
                        dists = dists.squeeze(0)
                        dists_weight = torch.index_select(dists, 0, matches1)
                        dists_weight = (dists_weight.sigmoid() - 0.5) * 2
                        dists_weight = (1 - dists_weight).squeeze()
                        future_neg_cos_sim = (future_neg_cos_sim * dists_weight)
                    if self.args.score_weight:
                        if frame >= 2 * self.max_time_wdw:
                            if self.args.score_weight_new:
                                score1 = out_seg1.detach().clone().softmax(dim=-1)
                                score1 = torch.max(score1, dim=-1)[0]
                                score_weight = torch.index_select(score1.cuda().squeeze(0).unsqueeze(-1), 0, matches1).squeeze()
                            else:
                                score_weight = score_list[-1-self.max_time_wdw].cuda()
                                score_weight = torch.index_select(score_weight.squeeze(0).unsqueeze(-1), 0, matches1).squeeze()
                        else:
                            score_weight = torch.ones(future_neg_cos_sim.shape[0]).cuda()
                        future_neg_cos_sim = (future_neg_cos_sim * score_weight)

                    future_neg_cos_sim = future_neg_cos_sim.mean(dim=0)
            else:
                if self.topk_matches > 0:
                    # select top-k worst performing matches
                    future_neg_cos_sim = future_neg_cos_sim.topk(self.topk_matches, dim=0).values.mean()
                else:
                    future_neg_cos_sim = future_neg_cos_sim.mean(dim=0)

            # 2PAST CONTRASTIVE
            # backward preds (t1 -> t0)
            past_preds = torch.index_select(out_pred1, 0, matches1)
            # backward gt feats and stop grad
            past_gt = torch.index_select(out_en0.detach(), 0, matches0)
            past_neg_cos_sim = -self.ssl_criterion(past_preds, past_gt)

            if self.args.sample_pos or self.args.coord_weight or self.args.score_weight:
                if self.topk_matches > 0:
                    # select top-k worst performing matches
                    past_neg_cos_sim = past_neg_cos_sim.topk(self.topk_matches, dim=0).values.mean()
                else:
                    if self.args.sample_pos:
                        past_seg = torch.index_select(out_seg0, 0, matches0)
                        past_seg = F.softmax(past_seg, dim=1)
                        past_seg, _ = torch.max(past_seg, dim=1)
                        past_neg_cos_sim = (past_neg_cos_sim * past_seg)
                    if self.args.coord_weight:
                        global_pts0 = batch["global_pts0"].unsqueeze(0).float().cuda()
                        global_pts1 = batch["global_pts1"].unsqueeze(0).float().cuda()
                        K = 1
                        dists, idx, _ = knn_points(global_pts0, global_pts1, K=K, return_nn=True) # [1, N, K], [1, N, K], [1, N, K, 3]
                        dists = dists.squeeze(0)
                        dists_weight = torch.index_select(dists, 0, matches0)
                        dists_weight = (dists_weight.sigmoid() - 0.5) * 2
                        dists_weight = (1 - dists_weight).squeeze()
                        past_neg_cos_sim = (past_neg_cos_sim * dists_weight)
                    if self.args.score_weight:
                        if frame >= 2 * self.max_time_wdw:
                            if self.args.score_weight_new:
                                score0 = out_seg0.detach().clone().softmax(dim=-1)
                                score0 = torch.max(score0, dim=-1)[0]
                                score_weight = torch.index_select(score0.cuda().squeeze(0).unsqueeze(-1), 0, matches0).squeeze()
                            else:
                                score_weight = score_list[-1].cuda()
                                score_weight = torch.index_select(score_weight.squeeze(0).unsqueeze(-1), 0, matches0).squeeze()
                        else:
                            score_weight = torch.ones(past_neg_cos_sim.shape[0]).cuda()
                        past_neg_cos_sim = (past_neg_cos_sim * score_weight)

                    past_neg_cos_sim = past_neg_cos_sim.mean(dim=0)
            else:
                if self.topk_matches > 0:
                    # select top-k worst performing matches
                    past_neg_cos_sim = past_neg_cos_sim.topk(self.topk_matches, dim=0).values.mean()
                else:
                    past_neg_cos_sim = past_neg_cos_sim.mean(dim=0)

            # sum up to total
            ssl_loss = (future_neg_cos_sim + past_neg_cos_sim) * self.ssl_beta
            total_loss = self.args.segmentation_beta * loss_seg_head + self.args.ssl_beta * ssl_loss 

            if self.args.without_ssl_loss:
                total_loss = self.args.segmentation_beta * loss_seg_head

            # backward and optimize
            total_loss.backward()
            self.optimizer.step()

            if self.args.use_ema:
                self.ema.update(self.model)

        else:
            # if no pseudo we skip the frame (happens never basically)
            # we assume that data the loader gives frames in pairs
            stensor0 = ME.SparseTensor(coordinates=batch["coordinates0"].int().to(self.device),
                                       features=batch["features0"].to(self.device),
                                       quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

            stensor1 = ME.SparseTensor(coordinates=batch["coordinates1"].int().to(self.device),
                                       features=batch["features1"].to(self.device),
                                       quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

            # Must clear cache at regular interval
            if self.global_step % self.clear_cache_int == 0:
                torch.cuda.empty_cache()

            self.model.eval()
            with torch.no_grad():
                # forward in mink
                out_seg0, out_en0, out_pred0, out_bck0, _, out_seg1, out_en1, out_pred1, out_bck1, _ = self.model((stensor0, stensor1))

            # segmentation loss for t0
            labels0 = batch['labels0'].long()

            loss_seg_head = self.criterion(out_seg0, pseudo0)

            pseudo0 = pseudo0
            labels0 = labels0

            # get matches in t0 and t1 (used for selection)
            matches0 = batch['matches0'].to(self.device)
            matches1 = batch['matches1'].to(self.device)

            # 2FUTURE CONTRASTIVE
            # forward preds (t0 -> t1)
            future_preds = torch.index_select(out_pred0, 0, matches0)
            # forward gt feats and stop grad
            future_gt = torch.index_select(out_en1.detach(), 0, matches1)
            future_neg_cos_sim = -self.ssl_criterion(future_preds, future_gt)

            if self.topk_matches > 0:
                # select top-k worst performing matches
                future_neg_cos_sim = future_neg_cos_sim.topk(self.topk_matches, dim=0).values.mean()
            else:
                future_neg_cos_sim = future_neg_cos_sim.mean(dim=0)

            # 2PAST CONTRASTIVE
            # backward preds (t1 -> t0)
            past_preds = torch.index_select(out_pred1, 0, matches1)
            # backward gt feats and stop grad
            past_gt = torch.index_select(out_en0.detach(), 0, matches0)
            past_neg_cos_sim = -self.ssl_criterion(past_preds, past_gt)
            if self.topk_matches > 0:
                # select top-k worst performing matches
                past_neg_cos_sim = past_neg_cos_sim.topk(self.topk_matches, dim=0).values.mean()
            else:
                past_neg_cos_sim = past_neg_cos_sim.mean(dim=0)

            # sum up to total
            ssl_loss = (future_neg_cos_sim + past_neg_cos_sim) * self.ssl_beta

        # print((pseudo0 != -1).sum())
        # print((pseudo0 != -1).sum())
        # increase step
        self.global_step += self.stream_batch_size
        labels0 = batch['labels0'].long()
        pseudo0 = pseudo0.cpu()
        labels0 = labels0.cpu()

        # additional metrics
        _, pred_seg0 = out_seg0.detach().max(1)
        # iou
        iou_tmp = jaccard_score(pred_seg0.cpu().numpy(), labels0.cpu().numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0.)

        # forward preds (t0 -> t1)
        future_preds = torch.index_select(out_bck0.detach(), 0, matches0)
        # forward gt feats and stop grad
        future_gt = torch.index_select(out_bck1.detach(), 0, matches1)
        frame_match_sim = -self.ssl_criterion(future_preds, future_gt).mean()

        # we check pseudo labelling accuracy, not IoU as union of points changes
        valid_idx_pseudo = torch.logical_and(pseudo0 != -1, labels0 != -1)
        pseudo_acc = (pseudo0[valid_idx_pseudo] == labels0[valid_idx_pseudo]).sum() / labels0[valid_idx_pseudo].shape[0]

        present_labels, class_occurs = np.unique(labels0.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.adapt_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join('training', p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        # check pseudo nums
        valid_pseudo = (pseudo0 != -1)
        pseudo_classes, pseudo_num = torch.unique(pseudo0[valid_pseudo], return_counts=True)
        pseudo_names = self.adapt_dataset.class2names[pseudo_classes].tolist()
        classes_count = dict(zip(pseudo_names, pseudo_num.int().tolist()))
        classes_print = dict()
        for c in self.adapt_dataset.class2names[pseudo_classes]:
            if c in classes_count.keys():
                classes_print[f'training/pseudo_number/{c}'] = classes_count[c]
            else:
                classes_print[f'training/pseudo_number/{c}'] = -1

        results_dict.update(classes_print)
        # degeneration check
        out_en0_dg = out_en0.detach().clone()
        out_en1_dg = out_en1.detach().clone()

        max_val = 1/np.sqrt(out_en0_dg.shape[-1])

        out_en0_dg = F.normalize(out_en0_dg, p=2, dim=-1).std(dim=-1).mean()
        out_en1_dg = F.normalize(out_en1_dg, p=2, dim=-1).std(dim=-1).mean()

        results_dict['training/seg_loss'] = loss_seg_head
        results_dict['training/future_ssl'] = future_neg_cos_sim
        results_dict['training/past_ssl'] = past_neg_cos_sim
        results_dict['training/frame_similarity'] = frame_match_sim
        results_dict['training/future_degeneration'] = out_en0_dg
        results_dict['training/past_degeneration'] = out_en1_dg
        results_dict['training/max_degeneration'] = max_val
        results_dict['training/iou'] = np.mean(iou_tmp[present_labels])
        results_dict['training/lr'] = self.optimizer.param_groups[0]["lr"]
        results_dict['training/pseudo_accuracy'] = pseudo_acc
        results_dict['training/pseudo_number'] = torch.sum(pseudo_num)
        # results_dict['training/source_similarity'] = source_sim

        return results_dict

    def validation_step(self, batch, is_source=False, save_path=None, frame=None):
        self.model.eval()
        # for multiple dataloaders
        phase = 'validation' if not is_source else 'source'
        coords_name = 'coordinates'
        feats_name = 'features'
        label_name = 'labels'

        if save_path is not None:
            save_path_tmp = os.path.join(save_path, phase)

        # clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        # sparsify
        stensor = ME.SparseTensor(coordinates=batch[coords_name].int().to(self.device),
                                  features=batch[feats_name].to(self.device))

        # get output
        out, out_bck, out_bottle = self.model(stensor, is_train=False)

        labels = batch[label_name].long().cuda()
        present_lbl = torch.unique(labels)


        loss = self.criterion(out, labels)
        _, preds = out.max(1)

        preds = preds.cpu()
        labels = labels.cpu()
        self.global_step += self.stream_batch_size

        # eval iou and log
        iou_tmp = jaccard_score(preds.numpy(), labels.numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0.)

        valid_feats_idx = torch.where(labels != -1)[0].view(-1).long()
        db_index = davies_bouldin_score(out_bck.cpu()[valid_feats_idx].numpy(), labels.cpu()[valid_feats_idx].numpy())

        present_labels, class_occurs = np.unique(labels.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.adapt_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join(phase, p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        results_dict[f'{phase}/loss'] = loss.cpu().item()
        results_dict[f'{phase}/iou'] = np.mean(iou_tmp[present_labels])
        results_dict[f'{phase}/db_index'] = db_index

        if save_path is not None:

            self.save_pcd(batch, preds.cpu().numpy(),
                          labels.cpu().numpy(), save_path_tmp, frame,
                          is_global=False)

        return results_dict

    def adapt_double(self):

        self.load_source_model()
        if self.args.use_ema:
            self.ema = EMA(self.source_model,0.999)

        if self.args.without_pre_eval_synth4dnusc:
            self.online_sequences = self.online_sequences[:5]

        self.model_init_dict = self.model.state_dict()

        # first we eval getting performance of source model
        self.eval(is_adapt=True)

        # adapt
        for sequence in tqdm(self.online_sequences, desc='Online Adaptation'):
            # load source model
            if self.args.without_reload:
                pass
            else:
                self.reload_model()
            # self.model = MinkUNet18_HEADS(self.model.seg_model)
            # self.reload_model_from_scratch()
            # set sequence in dataset, in weight path and loggers
            self.set_sequence(sequence)
            # adapt on sequence
            sequence_dict = self.online_adaptation_routine()
            self.adaptation_results_dict[sequence] = sequence_dict

        print(np.array(time_lists).mean())
        print(np.array(time_lists).mean())

        self.save_final_results()

    def eval(self, is_adapt=False):
        # load model only once
        self.reload_model(is_adapt=False)
        
        if self.args.without_pre_eval_synth4dnusc:
            if "synth" in self.args.note:
                sequence_dict = np.load("/home/XXXX/A0_TTA-Point/eval_result_64to32_sy.npy", allow_pickle=True).item()
            elif "syn4d" in self.args.note:
                sequence_dict = np.load("/home/XXXX/A0_TTA-Point/eval_result_64to32_4d.npy", allow_pickle=True).item()
            else:
                sequence_dict = np.load("/home/XXXX/A0_TTA-Point/TTA-Pointcloud/experiments/synth4dnusc/eval_result.npy", allow_pickle=True).item()
            for sequence in tqdm(self.online_sequences, desc='Online Evaluation', leave=True):
                self.source_results_dict[sequence] = sequence_dict[sequence]

        elif "32to64" in self.args.note:
            sequence_dict = np.load("/home/XXXX/A0_TTA-Point/eval_result_32to64.npy", allow_pickle=True).item()
            for sequence in tqdm(self.online_sequences, desc='Online Evaluation', leave=True):
                self.source_results_dict[sequence] = sequence_dict[sequence]

        elif "kitti_sim" in self.args.note:
            sequence_dict = np.load("/home/XXXX/A0_TTA-Point/eval_result_kitti_sim_3.npy", allow_pickle=True).item()
            for sequence in tqdm(self.online_sequences, desc='Online Evaluation', leave=True):
                self.source_results_dict[sequence] = sequence_dict[sequence]

        else:
            for sequence in tqdm(self.online_sequences, desc='Online Evaluation', leave=True):
                # set sequence
                self.set_sequence(sequence)
                # evaluate
                sequence_dict = self.online_evaluation_routine()
                # store dict
                self.source_results_dict[sequence] = sequence_dict

        if not is_adapt:
            self.save_eval_results()

    def check_frame(self, fr):
        return (fr+1) >= self.adaptation_batch_size and fr >= self.max_time_wdw

    def online_adaptation_routine(self):
        # move to device
        self.model.to(self.device)

        if self.is_double:
            self.source_model.to(self.device)

        # for storing
        adaptation_results = []

        if self.save_predictions:
            save_path = os.path.join(self.weights_save_path, 'pcd')
        else:
            save_path = None

        for f in tqdm(range(len(self.eval_dataset)), desc=f'Seq: {self.sequence}', leave=True):

            # get eval batch (1 frame at a time)
            val_batch = self.get_evaluation_batch(f)
            # eval
            with torch.no_grad():
                val_dict = self.validation_step(val_batch, save_path=save_path, frame=f)
            val_dict['validation/frame'] = f
            # log
            self.log(val_dict)

            # if enough frames
            if self.check_frame(f):
                train_dict = {}
                # get adaptation batch (t-b, t)
                # print('FRAME', f)
                batch = self.get_adaptation_batch(f)

                for _ in range(self.sub_epochs):

                    if self.is_pseudo:
                        t0 = time.perf_counter()
                        train_dict = self.adaptation_double_pseudo_step(batch, f)
                        t1 = time.perf_counter()
                        # print(f'Adaptation time: {t1-t0}')
                        time_lists.append(t1-t0)
                    else:
                        raise NotImplementedError

                    if train_dict is not None:
                        train_dict.update(train_dict)
                    # log
                self.log(train_dict)

            ###########################################################
            ###########################################################
            ###########################################################
            # if self.args.without_pre_eval_synlidar2kitti:
            #     self.save_checkpoint_every = 400
            #     if (f+1) % self.save_checkpoint_every == 0:
            #         # save weights
            #         self.save_state_dict(f)
            ###########################################################
            ###########################################################
            ###########################################################

            # append dict
            adaptation_results.append(val_dict)

        return adaptation_results

    def online_evaluation_routine(self):
        # move model to device
        self.model.to(self.device)
        # for store
        source_results = []

        if self.save_predictions:
            save_path = os.path.join(self.weights_save_path, 'pcd')
        else:
            save_path = None

        if self.args.without_pre_eval_synlidar2kitti:
            file_text = []
            if self.args.split_size == 100:
                path_name = "/home/XXXX/A0_TTA-Point/TTA-Pointcloud/experiments/synlidar2kitti/logs_100/0.csv"
            elif self.args.split_size == 150:
                path_name = "/home/XXXX/A0_TTA-Point/TTA-Pointcloud/experiments/synlidar2kitti/logs_150/0.csv"
            else:
                path_name = "/home/XXXX/A0_TTA-Point/TTA-Pointcloud/experiments/synlidar2kitti/logs_all/0.csv"
            with open(path_name, encoding='utf-8-sig') as f:
                for row in csv.reader(f, skipinitialspace=True):
                    file_text.append(row)
            f.close()

            for i in range(len(self.eval_dataset)):
                val_dict = {}
                for j in range(len(file_text[0])):
                    if "source" in file_text[0][j]:
                        #str to num
                        if len(file_text[1+i][j]) != 0:
                            val_dict[file_text[0][j]] = float(file_text[1+i][j])
                        else:
                            # val_dict[file_text[0][j]] = 0
                            pass
                source_results.append(val_dict)

            return source_results

        if self.args.without_pre_eval_synth4d2kitti:
            file_text = []
            path_name = "/home/XXXX/A0_TTA-Point/TTA-Pointcloud/experiments/synth4d2kitti/logs_all/0.csv"
            with open(path_name, encoding='utf-8-sig') as f:
                for row in csv.reader(f, skipinitialspace=True):
                    file_text.append(row)
            f.close()

            for i in range(len(self.eval_dataset)):
                val_dict = {}
                for j in range(len(file_text[0])):
                    if "source" in file_text[0][j]:
                        #str to num
                        if len(file_text[1+i][j]) != 0:
                            val_dict[file_text[0][j]] = float(file_text[1+i][j])
                        else:
                            # val_dict[file_text[0][j]] = 0
                            pass
                source_results.append(val_dict)

            return source_results
        
        with torch.no_grad():
            for f in tqdm(range(len(self.eval_dataset)), desc=f'Seq: {self.sequence}', leave=True):
                # get eval batch
                val_batch = self.get_evaluation_batch(f)
                # eval
                val_dict = self.validation_step(val_batch, is_source=True, save_path=save_path, frame=f)
                val_dict['source/frame'] = f
                # store results
                self.log(val_dict)
                source_results.append(val_dict)

        return source_results

    def set_loggers(self, sequence):
        # set current sequence in loggers, for logging purposes
        for logger in self.loggers:
            logger.set_sequence(sequence)

    def set_sequence(self, sequence):
        # update current weight saving path
        self.sequence = str(sequence)
        path, _ = os.path.split(self.weights_save_path)
        self.weights_save_path = os.path.join(path, self.sequence)

        ###########################################################
        ###########################################################
        ###########################################################
        if "NUSCENES" not in self.weights_save_path:
            os.makedirs(self.weights_save_path, exist_ok=True)
        ###########################################################
        ###########################################################
        ###########################################################

        self.eval_dataset.set_sequence(sequence)
        self.adapt_dataset.set_sequence(sequence)

        if self.boost:
            self.eval_dataloader = iter(self.get_online_dataloader(FrameOnlineDataset(self.eval_dataset),
                                                                            is_adapt=False))
            self.adapt_dataloader = iter(self.get_online_dataloader(PairedOnlineDataset(self.adapt_dataset,
                                                                                                 use_random=self.use_random_wdw),
                                                                                 is_adapt=True))

        # set sequence in path of loggers
        self.set_loggers(sequence)

    def log(self, results_dict):
        # log in ach logger
        for logger in self.loggers:
            logger.log(results_dict)

    def save_state_dict(self, frame):
        # save stat dict of the model
        save_dict = {'frame': frame,
                     'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(save_dict, os.path.join(self.weights_save_path, f'checkpoint-frame{frame}.pth'))

    def reload_model(self, is_adapt=True):
        # reloads model
        def clean_state_dict(state):
            # clean state dict from names of PL
            for k in list(ckpt.keys()):
                if "model" in k:
                    ckpt[k.replace("model.", "")] = ckpt[k]
                del ckpt[k]
            return state

        if self.student_checkpoint is not None and is_adapt:
            checkpoint_path = self.student_checkpoint
            print(f'--> Loading student checkpoint {checkpoint_path}')
        else:
            checkpoint_path = self.source_checkpoint
            print(f'--> Loading source checkpoint {checkpoint_path}')

        # in case of SSL pretraining
        if isinstance(self.model, MinkUNet18_SSL):
            if checkpoint_path.endswith('.pth'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(ckpt)

            elif checkpoint_path.endswith('.ckpt'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
                ckpt = clean_state_dict(ckpt)
                self.model.load_state_dict(ckpt, strict=True)

            else:
                raise NotImplementedError('Invalid source model extension (allowed .pth and .ckpt)')

        # in case of segmentation pretraining
        elif isinstance(self.model, MinkUNet18_HEADS):
            def clean_student_state_dict(ckpt):
                # clean state dict from names of PL
                for k in list(ckpt.keys()):
                    if "seg_model" in k:
                        ckpt[k.replace("seg_model.", "")] = ckpt[k]
                    del ckpt[k]
                return ckpt
            if checkpoint_path.endswith('.pth'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                ckpt = clean_student_state_dict(ckpt['model_state_dict'])
                self.model.seg_model.load_state_dict(ckpt)

            elif checkpoint_path.endswith('.ckpt'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
                ckpt = clean_state_dict(ckpt)
                self.model.seg_model.load_state_dict(ckpt, strict=True)

            # init_dict_predictor = {}
            # init_dict_encoder = {}
            # for k, v in self.model_init_dict.items():
            #     if "predictor" in k :
            #         init_dict_predictor[k.replace("predictor.", "")] = v
            #     elif "encoder" in k:
            #         init_dict_encoder[k.replace("encoder.", "")] = v
            # self.model.predictor.load_state_dict(init_dict_predictor)
            # self.model.encoder.load_state_dict(init_dict_encoder)


    def reload_model_from_scratch(self):

        # in case of SSL pretraining
        if isinstance(self.model, MinkUNet18_SSL):
            self.model.weight_initialization()

        # in case of segmentation pretraining
        elif isinstance(self.model, MinkUNet18_HEADS):
            seg_model = self.model.seg_model
            seg_model.weight_initialization()
            self.model = MinkUNet18_HEADS(seg_model=seg_model)

    def load_source_model(self):
        # reloads model
        def clean_state_dict(state):
            # clean state dict from names of PL
            for k in list(ckpt.keys()):
                if "model" in k:
                    ckpt[k.replace("model.", "")] = ckpt[k]
                del ckpt[k]
            return state

        print(f'--> Loading source checkpoint {self.source_checkpoint}')

        if self.source_checkpoint.endswith('.pth'):
            ckpt = torch.load(self.source_checkpoint, map_location=torch.device('cpu'))
            if isinstance(self.source_model, MinkUNet18_MCMC):
                self.source_model.seg_model.load_state_dict(ckpt)
            else:
                self.source_model.load_state_dict(ckpt)

        elif self.source_checkpoint.endswith('.ckpt'):
            ckpt = torch.load(self.source_checkpoint, map_location=torch.device('cpu'))["state_dict"]
            ckpt = clean_state_dict(ckpt)
            if isinstance(self.source_model, MinkUNet18_MCMC):
                self.source_model.seg_model.load_state_dict(ckpt, strict=True)
            else:
                self.source_model.load_state_dict(ckpt, strict=True)

        else:
            raise NotImplementedError('Invalid source model extension (allowed .pth and .ckpt)')

    def get_adaptation_batch(self, frame_idx):
        if self.adapt_dataloader is None:
            frame_idx += 1
            batch_idx = np.arange(frame_idx - self.adaptation_batch_size, frame_idx)

            batch_data = [self.adapt_dataset.__getitem__(b) for b in batch_idx]
            batch_data = [self.adapt_dataset.get_double_data(batch_data[b-1], batch_data[b]) for b in range(1, len(batch_data))]
            batch = self.collate_fn_adapt(batch_data)
        else:
            batch = next(self.adapt_dataloader)

        return batch

    def get_evaluation_batch(self, frame_idx):
        if self.eval_dataloader is None:
            data = self.eval_dataset.__getitem__(frame_idx)
            data = self.eval_dataset.get_single_data(data)

            batch = self.collate_fn_eval([data])
        else:
            batch = next(self.eval_dataloader)

        return batch

    def save_final_results(self):
        # stores final results in a final dict
        # finally saves results in a csv file

        final_dict = {}

        for seq in self.online_sequences:
            source_results = self.source_results_dict[seq]
            adaptation_results = self.adaptation_results_dict[seq]

            assert len(source_results) == len(adaptation_results)
            num_frames = len(source_results)

            source_results = self.format_val_dict(source_results)
            adaptation_results = self.format_val_dict(adaptation_results)

            final_dict[seq] = {}

            for k in adaptation_results.keys():
                relative_tmp = adaptation_results[k] - source_results[k]
                final_dict[seq][f'relative_{k}'] = relative_tmp
                final_dict[seq][f'source_{k}'] = source_results[k]
                final_dict[seq][f'adapted_{k}'] = adaptation_results[k]

        self.write_csv(final_dict, phase='final')
        self.write_csv(final_dict, phase='source')
        self.save_pickle(final_dict)

    def save_eval_results(self):
        # stores final results in a final dict
        # finally saves results in a csv file

        final_dict = {}

        for seq in self.online_sequences:
            eval_results = self.source_results_dict[seq]

            eval_results = self.format_val_dict(eval_results)

            final_dict[seq] = {}

            for k in eval_results.keys():
                final_dict[seq][f'eval_{k}'] = eval_results[k]

        self.write_csv(final_dict, phase='eval')
        self.save_pickle(final_dict)

    def format_val_dict(self, list_dict):
        # input is a list of dicts for each frame
        # returns a dict with [miou, iou_per_frame, per_class_miou, per_class_iou_frame]

        def change_names(in_dict):
            for k in list(in_dict.keys()):
                if "validation/" in k:
                    in_dict[k.replace("validation/", "")] = in_dict[k]
                    del in_dict[k]
                elif "source/" in k:
                    in_dict[k.replace("source/", "")] = in_dict[k]
                    del in_dict[k]

            return in_dict

        list_dict = [change_names(list_dict[f]) for f in range(len(list_dict))]

        if self.num_classes == 7:
            classes = {'vehicle_iou': [],
                       'pedestrian_iou': [],
                       'road_iou': [],
                       'sidewalk_iou': [],
                       'terrain_iou': [],
                       'manmade_iou': [],
                       'vegetation_iou': []}
        elif self.num_classes == 3:
            classes = {'background_iou': [],
                       'vehicle_iou': [],
                       'pedestrian_iou': []}
        else:
            classes = {'vehicle_iou': [],
                       'pedestrian_iou': []}

        for f in range(len(list_dict)):
            val_tmp = list_dict[f]
            for key in classes.keys():
                if key in val_tmp:
                    classes[key].append(val_tmp[key])
                else:
                    classes[key].append(np.nan)

        all_iou = np.concatenate([np.asarray(v)[np.newaxis, ...] for k, v in classes.items()], axis=0).T

        per_class_iou = np.nanmean(all_iou, axis=0)
        miou = np.nanmean(per_class_iou)

        per_frame_miou = np.nanmean(all_iou, axis=-1)

        return {'miou': miou,
                'per_frame_miou': per_frame_miou,
                'per_class_iou': per_class_iou,
                'per_class_frame_iou': all_iou}

    def write_csv(self, results_dict, phase='final'):
        if self.num_classes == 7:
            if phase == 'final':
                headers = ['sequence', 'relative_miou', 'relative_vehicle_iou',
                           'relative_pedestrian_iou', 'relative_road_iou',
                           'relative_sidewalk_iou', 'relative_terrain_iou',
                           'relative_manmade_iou', 'relative_vegetation_iou']
                file_name = 'final_main.csv'
            elif phase == 'source':
                headers = ['sequence', 'miou', 'source_vehicle_iou',
                           'source_pedestrian_iou', 'source_road_iou',
                           'source_sidewalk_iou', 'source_terrain_iou',
                           'source_manmade_iou', 'source_vegetation_iou']
                file_name = 'source_main.csv'
            elif phase == 'eval':
                headers = ['sequence', 'miou', 'eval_vehicle_iou',
                           'eval_pedestrian_iou', 'eval_road_iou',
                           'eval_sidewalk_iou', 'eval_terrain_iou',
                           'eval_manmade_iou', 'eval_vegetation_iou']
                file_name = 'evaluation_main.csv'
            else:
                raise NotImplementedError
        elif self.num_classes == 3:
            if phase == 'final':
                headers = ['sequence', 'relative_miou',
                           'relative_background_iou',
                           'relative_vehicle_iou',
                           'relative_pedestrian_iou']
                file_name = 'final_main.csv'
            elif phase == 'source':
                headers = ['sequence', 'miou',
                           'source_background_iou',
                           'source_vehicle_iou',
                           'source_pedestrian_iou']
                file_name = 'source_main.csv'
            elif phase == 'eval':
                headers = ['sequence','miou',
                           'source_backround_iou',
                           'eval_vehicle_iou',
                           'eval_pedestrian_iou']
                file_name = 'evaluation_main.csv'
            else:
                raise NotImplementedError
        elif self.num_classes == 2:
            if phase == 'final':
                headers = ['sequence', 'relative_miou',
                           'relative_vehicle_iou',
                           'relative_pedestrian_iou']
                file_name = 'final_main.csv'
            elif phase == 'source':
                headers = ['sequence', 'miou',
                           'source_vehicle_iou',
                           'source_pedestrian_iou']
                file_name = 'source_main.csv'
            elif phase == 'eval':
                headers = ['sequence','miou',
                           'eval_vehicle_iou',
                           'eval_pedestrian_iou']
                file_name = 'evaluation_main.csv'
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if self.dataset_name == 'nuScenes':
            cumul = []

        results_dir = os.path.join(os.path.split(self.weights_save_path)[0], 'final_results')
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, file_name), 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(headers)

            for seq in results_dict.keys():
                dict_tmp = results_dict[seq]
                if phase == 'final':
                    per_class = dict_tmp['relative_per_class_iou']
                    if self.num_classes == 7:
                        data = [seq,
                                dict_tmp['relative_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100,
                                per_class[3]*100,
                                per_class[4]*100,
                                per_class[5]*100,
                                per_class[6]*100]
                    elif self.num_classes == 3:
                        data = [seq,
                                dict_tmp['relative_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100]
                    elif self.num_classes == 2:
                        data = [seq,
                                dict_tmp['relative_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100]

                elif phase == 'source':
                    per_class = dict_tmp['source_per_class_iou']
                    if self.num_classes == 7:
                        data = [seq,
                                dict_tmp['source_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100,
                                per_class[3]*100,
                                per_class[4]*100,
                                per_class[5]*100,
                                per_class[6]*100]
                    elif self.num_classes == 3:
                        data = [seq,
                                dict_tmp['source_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100]

                    elif self.num_classes == 2:
                        data = [seq,
                                dict_tmp['source_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100]

                elif phase == 'eval':
                    per_class = dict_tmp['eval_per_class_iou']
                    if self.num_classes == 7:
                        data = [seq,
                                dict_tmp['eval_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100,
                                per_class[3]*100,
                                per_class[4]*100,
                                per_class[5]*100,
                                per_class[6]*100]

                    elif self.num_classes == 3:
                        data = [seq,
                                dict_tmp['eval_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100]
                    
                    elif self.num_classes == 2:
                        data = [seq,
                                dict_tmp['eval_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100]

                # write the data
                writer.writerow(data)

                if self.dataset_name == 'nuScenes':
                    if phase == 'final':
                        first_iou = dict_tmp['relative_miou']
                    elif phase == 'source':
                        first_iou = dict_tmp['source_miou']
                    elif phase == 'eval':
                        first_iou = dict_tmp['eval_miou']

                    if self.num_classes == 7:
                        cumul.append([first_iou*100,
                                      per_class[0]*100,
                                      per_class[1]*100,
                                      per_class[2]*100,
                                      per_class[3]*100,
                                      per_class[4]*100,
                                      per_class[5]*100,
                                      per_class[6]*100])
                    elif self.num_classes == 3:
                        cumul.append([first_iou*100,
                                      per_class[0]*100,
                                      per_class[1]*100,
                                      per_class[2]*100])
                    elif self.num_classes == 2:
                        cumul.append([first_iou*100,
                                      per_class[0]*100,
                                      per_class[1]*100])

            if self.dataset_name == 'nuScenes':
                avg_cumul = np.array(cumul)
                avg_cumul_tmp = np.nanmean(avg_cumul, axis=0)
                if self.num_classes == 7:
                    data = ['Average',
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2],
                            avg_cumul_tmp[3],
                            avg_cumul_tmp[4],
                            avg_cumul_tmp[5],
                            avg_cumul_tmp[6],
                            avg_cumul_tmp[7]]
                elif self.num_classes == 3:
                    data = ['Average',
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2]]

                elif self.num_classes == 2:
                    data = ['Average',
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1]]

                # write cumulative results
                writer.writerow(data)
                seq_locs = np.array([self.adapt_dataset.names2locations[self.adapt_dataset.online_keys[s]] for s in results_dict.keys()])

                for location in ['singapore-queenstown', 'boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth']:
                    valid_sequences = seq_locs == location
                    avg_cumul_tmp = np.nanmean(avg_cumul[valid_sequences, :], axis=0)
                    if self.num_classes == 7:
                        data = [location,
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2],
                            avg_cumul_tmp[3],
                            avg_cumul_tmp[4],
                            avg_cumul_tmp[5],
                            avg_cumul_tmp[6],
                            avg_cumul_tmp[7]]

                    elif self.num_classes == 3:
                        data = [location,
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2]]

                    elif self.num_classes == 2:
                        data = [location,
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1]]

                    # write cumulative results
                    writer.writerow(data)

    def save_pickle(self, results_dict):
        results_dir = os.path.join(os.path.split(self.weights_save_path)[0], 'final_results')
        with open(os.path.join(results_dir, 'final_all.pkl'), 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
