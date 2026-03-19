import copy
import gc
import math
import os
import random

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.algorithms.fedmmdp_base import EngineBase
from src.utils.serialize_utils import flatten_dict

try:
    from apex import amp
except ImportError:
    print('failed to import apex')


torch.backends.cudnn.enabled = True
torch.multiprocessing.set_sharing_strategy('file_system')


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cur_step(cur_epoch, idx, n_batches, fmt=None):
    current_step = cur_epoch + idx / n_batches
    if fmt:
        return fmt.format(current_step)
    return current_step


def compute_rmg(image_features, text_features):
    image_to_text_map = torch.arange(image_features.shape[0]).reshape(image_features.shape[0], 1)
    text_to_image_map = torch.arange(image_features.shape[0])
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features_original = image_features.clone()
    image_features = torch.stack([image_features[l] for l in text_to_image_map], dim=0)
    text_feature_per_image = text_features
    labels_to_idx_map = torch.ones((text_features.size(0), text_features.size(0))).bool()
    for txt_idcs in image_to_text_map:
        for i, j in [(x.item(), y.item()) for x in txt_idcs for y in txt_idcs]:
            labels_to_idx_map[i, j] = False

    image_features_matching = torch.sum(image_features * text_feature_per_image, dim=1).mean()
    image_features_matching = 1 - (image_features_matching + 1) / 2
    image_features_matching = torch.where(
        image_features_matching > 0,
        image_features_matching,
        torch.ones_like(image_features_matching) * 1e-3,
    )

    i_x_i = image_features_original @ image_features_original.T
    i_x_i.fill_diagonal_(0)
    mean_img_similarity = i_x_i.sum() / (math.prod(i_x_i.shape) - i_x_i.shape[0])
    mean_img_similarity = 1 - (mean_img_similarity + 1) / 2

    t_x_t = text_features @ text_features.T
    t_x_t.fill_diagonal_(0)
    mean_txt_similarity = t_x_t.sum() / (math.prod(t_x_t.shape) - t_x_t.shape[0])
    mean_txt_similarity = 1 - (mean_txt_similarity + 1) / 2

    normalizer = image_features_matching.mean() + (mean_img_similarity.mean() + mean_txt_similarity.mean()) / 2
    dist = image_features_matching.mean() / normalizer.clamp_min(1e-8)
    return dist


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

    def _normalize_sample_ids(self, sample_ids):
        return [str(sample_id) for sample_id in sample_ids]

    def _cached_targets_from_batch(self, data):
        sample_ids = self._normalize_sample_ids(data["id"])
        if any(sample_id not in self.cached_domain_labels for sample_id in sample_ids):
            return None
        targets = [self.cached_domain_labels[sample_id] for sample_id in sample_ids]
        return torch.tensor(targets, dtype=torch.long, device=self.device)

    def run(self, global_centroids):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval().cuda()
        self.model.cuda()
        self._set_local_head_state(self.args.mlp_local)
        if self.local_epoch == 0:
            _, self.optimizer = amp.initialize([], self.optimizer, opt_level='O2')
        self.model.train()

        for _ in range(self.local_epochs):
            self.local_epoch += 1
            if self.logger is not None:
                self.logger.log(f"Epoch {self.local_epoch}")
            self.train_epoch(global_centroids)

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/mm/Client{self.client}-model_{self.local_epoch}.pth')

        self.old_model.cpu()
        self.model.cpu()
        del self.old_model
        gc.collect()

    def compute_local_cluster_statistics(self, global_centroids):
        self.model.cuda()
        was_training = self.model.training
        old_img_mlp_local = getattr(self.model.img_enc, "mlp_local", None) if hasattr(self.model, "img_enc") else None
        old_txt_mlp_local = getattr(self.model.txt_enc, "mlp_local", None) if hasattr(self.model, "txt_enc") else None
        self.model.eval()
        self._set_local_head_state(False)

        centroids = F.normalize(global_centroids.to(self.device, dtype=torch.float32), p=2, dim=1)
        local_sum = torch.zeros(self.args.n_clusters, self.args.feature_dim, dtype=torch.float32)
        local_count = torch.zeros(self.args.n_clusters, dtype=torch.float32)
        self.cached_domain_labels = {}

        self.train_loader.generator.manual_seed(self.cur_epoch)
        for _, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader), disable=True):
            with torch.no_grad():
                images = data["processed_img"].to(self.device)
                captions = data["cap_tokens"].to(self.device)
                output = self.model(images, captions)
                image_features = F.normalize(
                    torch.nan_to_num(output['image_features'].float(), nan=0.0, posinf=0.0, neginf=0.0),
                    p=2,
                    dim=1,
                )
                caption_features = F.normalize(
                    torch.nan_to_num(output['caption_features'].float(), nan=0.0, posinf=0.0, neginf=0.0),
                    p=2,
                    dim=1,
                )
                fused_features = F.normalize((image_features + caption_features) / 2.0, p=2, dim=1)
                logits = torch.matmul(fused_features, centroids.t())
                domain_labels = torch.argmax(logits, dim=1)

                sample_ids = self._normalize_sample_ids(data["id"])
                for sample_id, label in zip(sample_ids, domain_labels.tolist()):
                    self.cached_domain_labels[sample_id] = int(label)

                cpu_features = fused_features.cpu()
                cpu_labels = domain_labels.cpu()
                local_sum.index_add_(0, cpu_labels, cpu_features)
                local_count += torch.bincount(cpu_labels, minlength=self.args.n_clusters).float()

        self.model.cpu()
        if old_img_mlp_local is not None:
            self.model.img_enc.mlp_local = old_img_mlp_local
        if old_txt_mlp_local is not None:
            self.model.txt_enc.mlp_local = old_txt_mlp_local
        if was_training:
            self.model.train()
        torch.cuda.empty_cache()
        gc.collect()
        return {'sum': local_sum, 'count': local_count}, self.dset_name

    def train_epoch(self, global_centroids):
        centroids = F.normalize(global_centroids.to(self.device, dtype=torch.float32), p=2, dim=1)
        self.train_loader.generator.manual_seed(self.cur_epoch)
        loss_dict = {}
        last_idx = -1

        for idx, data in enumerate(self.train_loader):
            last_idx = idx
            images = data["processed_img"].to(self.device)
            captions = data["cap_tokens"].to(self.device)
            output = self.model(images, captions)
            loss, loss_dict = self.criterion(**output)

            image_features = F.normalize(
                torch.nan_to_num(output['image_features'].float(), nan=0.0, posinf=0.0, neginf=0.0),
                p=2,
                dim=1,
            )
            caption_features = F.normalize(
                torch.nan_to_num(output['caption_features'].float(), nan=0.0, posinf=0.0, neginf=0.0),
                p=2,
                dim=1,
            )

            domain_targets = self._cached_targets_from_batch(data)
            if domain_targets is None:
                continue

            image_logits = torch.matmul(image_features, centroids.t()) / self.args.tau
            caption_logits = torch.matmul(caption_features, centroids.t()) / self.args.tau
            loss_cluster_image = F.cross_entropy(image_logits, domain_targets)
            loss_cluster_caption = F.cross_entropy(caption_logits, domain_targets)
            loss_rmg = compute_rmg(image_features, caption_features)
            total_loss = (
                loss / 10 +
                self.args.cluster_weight * (loss_cluster_image + loss_cluster_caption) / 2 +
                loss_rmg * self.args.rmg_weight
            )

            self.optimizer.zero_grad()
            if self.config.train.get('use_fp16'):
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
            self.optimizer.step()

        if last_idx < 0:
            if self.logger is not None:
                self.logger.log(f"Skip clustered local epoch for client {self._get_global_client_key()}: train loader has no full batch")
            return

        loss_dict = {'{}'.format(key): val for key, val in loss_dict.items()}
        loss_dict['step'] = cur_step(self.cur_epoch, last_idx, len(self.train_loader))

    def report_scores(self, step, scores, metadata, prefix=''):
        report_dict = {data_key: flatten_dict(_scores, sep='_') for data_key, _scores in scores.items()}
        report_dict = flatten_dict(report_dict, sep='__')
        tracker_data = report_dict.copy()

        report_dict = {'{}{}'.format(prefix, key): val for key, val in report_dict.items()}
        report_dict['step'] = step
        if self.logger is not None:
            self.logger.report(report_dict, prefix='[Eval] Report @step: ', pretty=True)

        tracker_data['metadata'] = metadata
        tracker_data['scores'] = scores
        if self.logger is not None:
            self.logger.update_tracker(tracker_data)
