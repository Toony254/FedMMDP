import copy
import gc
import torch

torch.backends.cudnn.enabled = True

import numpy as np
import os
import random
import torch.multiprocessing
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import math

import torch.nn as nn

from src.algorithms.fedmmdp_base import EngineBase
from tqdm import tqdm
import torch

try:
    from apex import amp
except ImportError:
    print('failed to import apex')

from src.utils.serialize_utils import flatten_dict


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


##################################################
# step -1: Predefined function
##################################################
import torch.utils.data.sampler as sampler


class SubsetSampler(sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def cur_step(cur_epoch, idx, N, fmt=None):
    _cur_step = cur_epoch + idx / N
    if fmt:
        return fmt.format(_cur_step)
    else:
        return _cur_step

def compute_rmg(image_features, text_features):
    image_to_text_map = torch.arange(image_features.shape[0]).reshape(image_features.shape[0], 1) # [batch_size, 1]
    text_to_image_map = torch.arange(image_features.shape[0]) # [batch_size * 1]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features_original = image_features.clone()
    image_features = torch.stack([image_features[l] for l in text_to_image_map], dim=0)
    text_feature_per_image = text_features
    labels_to_idx_map = torch.ones((text_features.size(0),text_features.size(0))).bool()
    for txt_idcs in image_to_text_map:
        for i,j in [(x.item(), y.item()) for x in txt_idcs for y in txt_idcs]:
            labels_to_idx_map[i,j] = False
    
    image_features_matching = torch.sum(image_features*text_feature_per_image, dim=1).mean()
    image_features_matching = 1-(image_features_matching+1)/2 # [0, 1] & flip
    image_features_matching = torch.where(image_features_matching > 0, image_features_matching, torch.ones_like(image_features_matching)*1e-3) # [1e-3, 1]

    i_x_i = image_features_original @ image_features_original.T

    i_x_i.fill_diagonal_(0)
    mean_img_similarity = i_x_i.sum() / (math.prod(i_x_i.shape)-i_x_i.shape[0]) # [-1, 1]
    mean_img_similarity = 1-(mean_img_similarity+1)/2 # [0,1] & flip

    t_x_t = text_features @ text_features.T
    t_x_t.fill_diagonal_(0)
    mean_txt_similarity = t_x_t.sum() / (math.prod(t_x_t.shape)-t_x_t.shape[0]) # [-1, 1]
    mean_txt_similarity = 1-(mean_txt_similarity+1)/2 # [0,1] & flip

    normalizer = image_features_matching.mean() + (mean_img_similarity.mean() + mean_txt_similarity.mean()) / 2
    dist = image_features_matching.mean() / normalizer.clamp_min(1e-8)

    return dist

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


gpuid = 'cuda:0' if torch.cuda.is_available() else 'cpu'

is_test = False


class MMClientTrainer(EngineBase):
    def _get_global_client_key(self):
        if hasattr(self, 'client_idx') and self.client_idx is not None:
            return self.client_idx
        local_client_id = getattr(self, 'client', getattr(self, 'client_id', -1))
        return local_client_id + self.args.num_img_clients + self.args.num_txt_clients

    def _set_local_head_state(self, enabled):
        if hasattr(self.model, "img_enc") and hasattr(self.model.img_enc, "mlp_local"):
            self.model.img_enc.mlp_local = enabled
        if hasattr(self.model, "txt_enc") and hasattr(self.model.txt_enc, "mlp_local"):
            self.model.txt_enc.mlp_local = enabled

    def run(self, cluster_centers, client_cluster_list):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval().cuda()
        self.model.cuda()
        self._set_local_head_state(self.args.mlp_local)
        if self.local_epoch == 0:
            _, self.optimizer = amp.initialize([], self.optimizer,
                                                        opt_level='O2')
        self.model.train()

        for i in range(self.local_epochs):
            self.local_epoch += 1
            if self.logger is not None:
                self.logger.log(f"Epoch {self.local_epoch}")
            self.train_epoch(cluster_centers, client_cluster_list)

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/mm/Client{self.client}-model_{self.local_epoch}.pth')

        self.old_model.cpu()
        self.model.cpu()

        del self.old_model
        import gc
        gc.collect()

    def train_epoch(self, cluster_centers, client_cluster_list):
        client_key = self._get_global_client_key()
        cluster_list = client_cluster_list.get(client_key)
        if cluster_list is None:
            if self.logger is not None:
                self.logger.log(f"Skip clustered local epoch for client {client_key}: no assigned cluster labels")
            return
        self.train_loader.generator.manual_seed(self.cur_epoch)
        loss_dict = {}
        last_idx = -1
        for idx, data in enumerate(self.train_loader):
            last_idx = idx
            images = data["processed_img"].to(self.device)
            captions = data["cap_tokens"].to(self.device)
            output = self.model(images, captions)
                
            loss, loss_dict = self.criterion(**output)
            
            image_features = output['image_features']
            caption_features = output['caption_features']
            
            cluster_list_batch = cluster_list[idx*self.config.dataloader["batch_size"]: (idx+1)*self.config.dataloader["batch_size"]]
            if not cluster_list_batch:
                continue
            cluster_features = torch.stack([
                torch.tensor(cluster_centers[i],dtype=image_features.dtype,device=self.device) 
                for i in cluster_list_batch
            ])
            image_features_norm = F.normalize(image_features, p=2, dim=1)
            caption_features_norm = F.normalize(caption_features, p=2, dim=1)
            cluster_features_norm = F.normalize(cluster_features, p=2, dim=1)
            # logits_inter_image = torch.div(torch.matmul(image_features, cluster_features.T), 0.5)
            # logits_inter_text = torch.div(torch.matmul(caption_features, cluster_features.T), 0.5)
            # labels_inter = torch.tensor(cluster_list_batch, dtype=torch.long).cuda()
            # loss_cluster_image = criterion(logits_inter_image, labels_inter)
            # loss_cluster_caption = criterion(logits_inter_text, labels_inter)
            # Align each sample with its assigned cluster center via cosine distance.
            loss_cluster_image = (1.0 - (image_features_norm * cluster_features_norm).sum(dim=1)).mean()
            loss_cluster_caption = (1.0 - (caption_features_norm * cluster_features_norm).sum(dim=1)).mean()
            loss_rmg = compute_rmg(image_features, caption_features)
            total_loss = loss/10 + self.args.cluster_weight * (loss_cluster_image + loss_cluster_caption)/2 + loss_rmg * self.args.rmg_weight
            print(f"total_loss: {total_loss}, loss: {loss}, loss_cluster_image: {loss_cluster_image}, loss_cluster_caption: {loss_cluster_caption}")
            self.optimizer.zero_grad()
            if self.config.train.get('use_fp16'):
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()

        if last_idx < 0:
            if self.logger is not None:
                self.logger.log(f"Skip clustered local epoch for client {client_key}: train loader has no full batch")
            return

        loss_dict = {'{}'.format(key): val
                     for key, val in loss_dict.items()}
        loss_dict['step'] = cur_step(self.cur_epoch, last_idx, len(self.train_loader))

    def generate_logits(self, key):
        self.model.cuda()
        was_training = self.model.training
        old_img_mlp_local = getattr(self.model.img_enc, "mlp_local", None) if hasattr(self.model, "img_enc") else None
        old_txt_mlp_local = getattr(self.model.txt_enc, "mlp_local", None) if hasattr(self.model, "txt_enc") else None
        self.model.eval()
        self._set_local_head_state(False)
        img_vec = []
        txt_vec = []
        self.train_loader.generator.manual_seed(self.cur_epoch)
        for idx, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader),disable=True):
            with torch.no_grad():
                images = data["processed_img"].to(self.device)
                captions = data["cap_tokens"].to(self.device)
                output = self.model(images, captions)

                out_img = output['image_features'].cpu().numpy()
                out_txt = output['caption_features'].cpu().numpy()
                img_vec.append(out_img)
                txt_vec.append(out_txt)

        if not img_vec or not txt_vec:
            self.model.cpu()
            if old_img_mlp_local is not None:
                self.model.img_enc.mlp_local = old_img_mlp_local
            if old_txt_mlp_local is not None:
                self.model.txt_enc.mlp_local = old_txt_mlp_local
            if was_training:
                self.model.train()
            torch.cuda.empty_cache()
            gc.collect()
            return np.empty((0, key.shape[1]), dtype=np.float32), self.dset_name

        img_vec = np.concatenate(img_vec, axis=0)
        txt_vec = np.concatenate(txt_vec, axis=0)
        img_vec = np.nan_to_num(img_vec, nan=0.0, posinf=0.0, neginf=0.0)
        txt_vec = np.nan_to_num(txt_vec, nan=0.0, posinf=0.0, neginf=0.0)
        img_norm = np.linalg.norm(img_vec, axis=1, keepdims=True)
        txt_norm = np.linalg.norm(txt_vec, axis=1, keepdims=True)
        img_norm = np.clip(img_norm, a_min=1e-12, a_max=None)
        txt_norm = np.clip(txt_norm, a_min=1e-12, a_max=None)
        img_vec = img_vec / img_norm
        txt_vec = txt_vec / txt_norm
        
        self.model.cpu()
        if old_img_mlp_local is not None:
            self.model.img_enc.mlp_local = old_img_mlp_local
        if old_txt_mlp_local is not None:
            self.model.txt_enc.mlp_local = old_txt_mlp_local
        if was_training:
            self.model.train()
        
        encrypted_img = np.dot(img_vec, key)
        encrypted_txt = np.dot(txt_vec, key)
        encrypted_img = np.nan_to_num(encrypted_img, nan=0.0, posinf=0.0, neginf=0.0)
        encrypted_txt = np.nan_to_num(encrypted_txt, nan=0.0, posinf=0.0, neginf=0.0)
        
        # use mean representation as features
        encrypted_feature = (encrypted_img + encrypted_txt)/2
        torch.cuda.empty_cache()
        del encrypted_img, encrypted_txt, img_vec, txt_vec
        gc.collect()
        return encrypted_feature, self.dset_name

    def report_scores(self, step, scores, metadata, prefix=''):
        report_dict = {data_key: flatten_dict(_scores, sep='_')
                       for data_key, _scores in scores.items()}
        report_dict = flatten_dict(report_dict, sep='__')
        tracker_data = report_dict.copy()

        report_dict = {'{}{}'.format(prefix, key): val for key, val in report_dict.items()}
        report_dict['step'] = step
        if self.logger is not None:
            self.logger.report(report_dict,
                               prefix='[Eval] Report @step: ',
                               pretty=True)

        tracker_data['metadata'] = metadata
        tracker_data['scores'] = scores
        if self.logger is not None:
            self.logger.update_tracker(tracker_data)

