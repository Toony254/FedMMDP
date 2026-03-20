import copy
import gc
import torch
import operator

torch.backends.cudnn.enabled = True

import numpy as np
import os
import random
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from src.algorithms.base import EngineBase
from tqdm import tqdm
import torch

try:
    from apex import amp
except ImportError:
    print('failed to import apex')

from src.utils.serialize_utils import flatten_dict
from src.algorithms.distill_utils import compute_distill_loss


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
    def run_with_prox(self, prefix='FedProx_'):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval().cuda()
        self.model.cuda()
        if self.local_epoch == 0:
            _, self.optimizer = amp.initialize([], self.optimizer, opt_level='O2')
        self.model.train()

        mu = getattr(self.args, 'mu', 0.01)
        global_params = {k: v.clone().detach() for k, v in self.global_model.state_dict().items()}

        for i in range(self.local_epochs):
            self.local_epoch += 1
            for idx, data in enumerate(self.train_loader):
                images = data["processed_img"].to(self.device)
                captions = data["cap_tokens"].to(self.device)
                output = self.model(images, captions)
                loss, loss_dict = self.criterion(**output)
                # === Proximal term ===
                prox_loss = 0.0
                for name, param in self.model.named_parameters():
                    prox_loss += ((param - global_params[name].to(param.device)) ** 2).sum()
                total_loss = loss + 0.5 * mu * prox_loss
                # print(f"Client {self.client} - Epoch {self.local_epoch}, Step {idx}, Loss: {loss:.4f}, Prox Loss: {prox_loss:.4f}")
                self.optimizer.zero_grad()
                if self.config.train.get('use_fp16'):
                    with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()
                if self.config.train.grad_clip > 0:
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                self.optimizer.step()
                if is_test:
                    break

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/mm/Client{self.client}-model_{self.local_epoch}.pth')
        self.old_model.cpu()
        self.model.cpu()
        del self.old_model
        import gc
        gc.collect()
    
    def run(self, prefix=''):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval().cuda()
        self.model.cuda()
        if getattr(self.args, 'FL_algorithm', '') == 'MASA':
            self._ensure_masa_modules()
            self._move_masa_modules(self.device)
        if self.local_epoch == 0:
            _, self.optimizer = amp.initialize([], self.optimizer,
                                                        opt_level='O2')
        self.model.train()

        for i in range(self.local_epochs):
            self.local_epoch += 1
            if self.logger is not None:
                self.logger.log(f"Epoch {self.local_epoch}")
            if self.args.FL_algorithm == 'FedAvg':
                self.train_epoch(prefix='FedAvg_')
            elif self.args.FL_algorithm == 'Harmony':
                self.train_epoch(prefix='Harmony_')
            elif self.args.FL_algorithm == 'MASA':
                self.train_gcmd_epoch(prefix='MASA_')

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/mm/Client{self.client}-model_{self.local_epoch}.pth')

        self.old_model.cpu()
        self.model.cpu()
        self._move_masa_modules('cpu')

        del self.old_model
        import gc
        gc.collect()
    def set_global_anchor(self, anchor):
        self.global_anchor = anchor
    def train_with_anchor(self, prefix='FedMEMA_'):
        self.model.cuda()
        if self.local_epoch == 0 and self.config.train.get('use_fp16'):
            _, self.optimizer = amp.initialize([], self.optimizer, opt_level='O2')

        self.model.train()
        mse_loss = nn.MSELoss()
        anchor_accum = []

        for _ in range(self.local_epochs):
            self.local_epoch += 1
            for idx, data in enumerate(self.train_loader):
                images = data["processed_img"].to(self.device)
                captions = data["cap_tokens"].to(self.device)

                output = self.model(images, captions)
                loss, _ = self.criterion(**output)

                img_features = output['image_features']
                txt_features = output['caption_features']

                if getattr(self, 'global_anchor', None) is not None:
                    batch_anchor = torch.cat([img_features, txt_features], dim=0).mean(dim=0)
                    anchor_target = torch.tensor(self.global_anchor, device=batch_anchor.device,
                                                 dtype=batch_anchor.dtype)
                    loss = loss + 0.1 * mse_loss(batch_anchor, anchor_target)
                else:
                    batch_anchor = torch.cat([img_features, txt_features], dim=0).mean(dim=0)

                self.optimizer.zero_grad()
                if self.config.train.get('use_fp16'):
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if self.config.train.grad_clip > 0:
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                self.optimizer.step()

                anchor_accum.append(batch_anchor.detach().cpu().numpy())
                if is_test:
                    break

        if self.args.save_client:
            torch.save(self.model.state_dict(),
                       f'./saved_clients/mm/Client{self.client}-model_{self.local_epoch}.pth')

        if anchor_accum:
            local_anchor = np.mean(np.stack(anchor_accum), axis=0)
        else:
            local_anchor = np.zeros(self.args.feature_dim, dtype=np.float32)

        model_copy = copy.deepcopy(self.model).cpu()
        self.model.cpu()
        gc.collect()
        return model_copy, local_anchor
    
    def run_with_moon(self, global_model, prev_models=None, temperature=0.5, mu=1.0):
        self.model.cuda()
        self.model.train()
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval().cuda()
        global_model.eval()
        global_model.cuda()
        if prev_models is not None:
            prev_models.eval()
            prev_models.cuda()
        for i in range(self.local_epochs):
            for idx, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                images = data["processed_img"].to(self.device)
                captions = data["cap_tokens"].to(self.device)
                output = self.model(images, captions)
                img_fvec = output['image_features']
                txt_fvec = output['caption_features']
                with torch.no_grad():
                    output_global = global_model(images, captions)
                    img_fvec_global = output_global['image_features']
                    txt_fvec_global = output_global['caption_features']
                    img_fvec_prev = prev_models(images, captions)['image_features'] if prev_models else None
                    txt_fvec_prev = prev_models(images, captions)['caption_features'] if prev_models else None
                # 多模态损失
                loss_cls, _ = self.criterion(**output)
                # MOON对比损失
                cos = nn.CosineSimilarity(dim=-1)
                posi_img = cos(img_fvec, img_fvec_global)
                posi_txt = cos(txt_fvec, txt_fvec_global)
                img_logits = posi_img.reshape(-1, 1)
                txt_logits = posi_txt.reshape(-1, 1)
                if prev_models:
                    nega = cos(img_fvec, img_fvec_prev)
                    img_logits = torch.cat((img_logits, nega.reshape(-1, 1)), dim=1)
                    nega = cos(txt_fvec, txt_fvec_prev)
                    txt_logits = torch.cat((txt_logits, nega.reshape(-1, 1)), dim=1)
                img_logits /= temperature
                txt_logits /= temperature
                contrastive_labels = torch.zeros(images.size(0)).long().to(self.device)
                loss_con_img = mu * nn.CrossEntropyLoss()(img_logits, contrastive_labels)
                loss_con_txt = mu * nn.CrossEntropyLoss()(txt_logits, contrastive_labels)
                loss = loss_cls + loss_con_img + loss_con_txt
                loss.backward()
                self.optimizer.step()
    def _ensure_masa_modules(self):
        if getattr(self, '_masa_modules_ready', False):
            return
        feature_dim = self.args.feature_dim
        self.masa_img_decoder = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True), nn.Linear(feature_dim, feature_dim))
        self.masa_txt_decoder = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True), nn.Linear(feature_dim, feature_dim))
        self.masa_img_classifier = nn.Linear(feature_dim, self.class_size)
        self.masa_txt_classifier = nn.Linear(feature_dim, self.class_size)
        self.optimizer.add_param_group({'params': self.masa_img_decoder.parameters()})
        self.optimizer.add_param_group({'params': self.masa_txt_decoder.parameters()})
        self.optimizer.add_param_group({'params': self.masa_img_classifier.parameters()})
        self.optimizer.add_param_group({'params': self.masa_txt_classifier.parameters()})
        default_cluster = getattr(self, 'client_idx', self.client) % max(int(getattr(self.args, 'num_clusters', 1)), 1)
        if not hasattr(self, 'selected_clusters'):
            self.selected_clusters = {'img': int(default_cluster), 'txt': int(default_cluster)}
        self._masa_modules_ready = True

    def _move_masa_modules(self, device):
        for name in ['masa_img_decoder', 'masa_txt_decoder', 'masa_img_classifier', 'masa_txt_classifier']:
            if hasattr(self, name):
                getattr(self, name).to(device)

    def _compute_masa_mis(self, features, labels):
        labels = labels.tolist() if hasattr(labels, 'tolist') else labels
        if len(features) == 0 or len(set(labels)) <= 1:
            return 0.0
        n_clusters = min(len(set(labels)), len(features))
        if n_clusters <= 1:
            return 0.0
        return float(normalized_mutual_info_score(labels, KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(features)))

    def _compute_scm(self, features):
        return torch.matmul(F.normalize(features, dim=-1), F.normalize(features, dim=-1).t())

    def _collect_masa_support(self, max_batches=8):
        self.model.eval(); img_features = []; txt_features = []; labels_all = []
        with torch.no_grad():
            for idx, data in enumerate(self.train_loader):
                images = data['processed_img'].to(self.device).float(); captions = data['cap_tokens'].to(self.device).float(); labels = data['class_id']
                if isinstance(labels, list): labels = torch.tensor(labels, dtype=torch.long)
                labels = labels.to(self.device); output = self.model(images, captions)
                img_features.append(output['image_features'].detach().cpu()); txt_features.append(output['caption_features'].detach().cpu()); labels_all.append(labels.detach().cpu())
                if idx + 1 >= max_batches or is_test: break
        self.model.train()
        if not img_features or not txt_features: return None
        return {'img': torch.cat(img_features, dim=0).numpy(), 'txt': torch.cat(txt_features, dim=0).numpy(), 'labels': torch.cat(labels_all, dim=0).numpy()}

    def get_masa_encoder_state(self, modality):
        module = self.model.img_enc if modality == 'img' else self.model.txt_enc
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

    def load_masa_encoder_state(self, modality, state_dict):
        (self.model.img_enc if modality == 'img' else self.model.txt_enc).load_state_dict(state_dict, strict=False)

    def _masa_temp_encoder(self, modality, state_dict):
        module = copy.deepcopy(self.model.img_enc if modality == 'img' else self.model.txt_enc)
        module.load_state_dict(state_dict, strict=False); module.to(self.device); module.eval()
        for param in module.parameters(): param.requires_grad_(False)
        return module

    def _masa_forward_modality(self, module, modality, inputs):
        with torch.no_grad(): output = module(inputs)
        return output['embedding'] if modality == 'img' else output

    def _masa_query_batches(self, modality, ratio=None):
        ratio = float(getattr(self.args, 'ascc_query_ratio', 0.2) if ratio is None else ratio)
        query_batches = []
        for batch in self.train_loader:
            labels = batch['class_id']
            if isinstance(labels, list): labels = torch.tensor(labels, dtype=torch.long)
            inputs = batch['processed_img'] if modality == 'img' else batch['cap_tokens']
            n_query = max(1, int(len(inputs) * ratio)); idx = torch.randperm(len(inputs))[:n_query]
            query_batches.append((inputs[idx].to(self.device).float(), labels[idx].to(self.device)))
            if is_test: break
        return query_batches

    def masa_personalize_from_clusters(self, modality, cluster_states, cluster_labels=None):
        if not cluster_states: return
        self.model.cuda(); self._ensure_masa_modules(); self._move_masa_modules(self.device)
        cluster_labels = list(range(len(cluster_states))) if cluster_labels is None else cluster_labels
        query_batches = self._masa_query_batches(modality)
        if not query_batches:
            self.load_masa_encoder_state(modality, cluster_states[0]); self.selected_clusters[modality] = int(cluster_labels[0]); return
        temp_encoders = [self._masa_temp_encoder(modality, state) for state in cluster_states]
        classifier = self.masa_img_classifier if modality == 'img' else self.masa_txt_classifier
        attn_logits = nn.Parameter(torch.zeros(len(cluster_states), device=self.device))
        optimizer = torch.optim.Adam([attn_logits], lr=float(getattr(self.args, 'ascc_lr', 5e-2)))
        for _ in range(int(getattr(self.args, 'ascc_attn_epoch', 3))):
            optimizer.zero_grad(); weights = torch.softmax(attn_logits, dim=0); total_loss = 0.0
            for inputs, labels in query_batches:
                feats = [self._masa_forward_modality(enc, modality, inputs) for enc in temp_encoders]
                mixed_feature = sum(weights[idx] * feat for idx, feat in enumerate(feats))
                total_loss = total_loss + F.cross_entropy(classifier(mixed_feature), labels)
            total_loss.backward(); optimizer.step()
        final_weights = torch.softmax(attn_logits.detach(), dim=0).cpu(); personalized_state = {}
        for key in cluster_states[0].keys():
            personalized_state[key] = sum(state[key].detach().cpu() * float(final_weights[idx].item()) for idx, state in enumerate(cluster_states))
        self.load_masa_encoder_state(modality, personalized_state); self.selected_clusters[modality] = int(cluster_labels[int(torch.argmax(final_weights).item())])
        for enc in temp_encoders: enc.cpu()
        del temp_encoders; gc.collect()

    def masa_finetune_local(self, ft_epochs=1):
        self.model.cuda(); self._ensure_masa_modules(); self._move_masa_modules(self.device)
        for _ in range(int(ft_epochs)):
            self.train_gcmd_epoch(log_metrics=False, recompute_superior=False)
        self.model.cpu(); self._move_masa_modules('cpu')

    def train_gcmd_epoch(self, prefix='', log_metrics=True, recompute_superior=True):
        self._ensure_masa_modules(); self.model.train(); self._move_masa_modules(self.device)
        if recompute_superior or not hasattr(self, 'masa_superior_modality'):
            support = self._collect_masa_support()
            if support is None:
                if self.logger is not None: self.logger.log(f'Skip MASA local epoch for client {self.client}: train loader has no full batch')
                return
            mis_img = self._compute_masa_mis(support['img'], support['labels']); mis_txt = self._compute_masa_mis(support['txt'], support['labels'])
            self.masa_superior_modality = 'img' if mis_img >= mis_txt else 'txt'
            if self.logger is not None: self.logger.log(f'MASA client {self.client}: superior modality={self.masa_superior_modality}, MIS(img)={mis_img:.4f}, MIS(txt)={mis_txt:.4f}')
        cls_weight = float(getattr(self.args, 'masa_cls_weight', 0.2)); rec_weight = float(getattr(self.args, 'masa_rec_weight', 0.1)); distill_weight = float(getattr(self.args, 'masa_distill_weight', 0.1)); last_loss = None
        for _, data in enumerate(self.train_loader):
            images = data['processed_img'].to(self.device).float(); captions = data['cap_tokens'].to(self.device).float(); labels = data['class_id']
            if isinstance(labels, list): labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(self.device); output = self.model(images, captions); retrieval_loss, _ = self.criterion(**output)
            img_feat = output['image_features']; txt_feat = output['caption_features']
            img_cls_loss = F.cross_entropy(self.masa_img_classifier(img_feat), labels); txt_cls_loss = F.cross_entropy(self.masa_txt_classifier(txt_feat), labels)
            img_rec_loss = F.mse_loss(self.masa_img_decoder(img_feat), images); txt_rec_loss = F.mse_loss(self.masa_txt_decoder(txt_feat), captions)
            scm_img = self._compute_scm(img_feat); scm_txt = self._compute_scm(txt_feat)
            distill_loss = F.mse_loss(scm_txt, scm_img.detach()) if self.masa_superior_modality == 'img' else F.mse_loss(scm_img, scm_txt.detach())
            loss = retrieval_loss + cls_weight * (img_cls_loss + txt_cls_loss) + rec_weight * (img_rec_loss + txt_rec_loss) + distill_weight * distill_loss
            self.optimizer.zero_grad()
            if self.config.train.get('use_fp16'):
                with amp.scale_loss(loss, self.optimizer) as scaled_loss: scaled_loss.backward()
            else:
                loss.backward()
            if self.config.train.grad_clip > 0: nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
            self.optimizer.step(); last_loss = float(loss.item())
            if is_test: break
        if log_metrics and self.logger is not None and last_loss is not None: self.logger.log(f'{prefix}client {self.client} loss={last_loss:.6f}')

    def train_epoch(self, prefix=''):
        loss_dict = {}
        last_idx = -1
        for idx, data in enumerate(self.train_loader):
            last_idx = idx
            images = data["processed_img"].to(self.device)
            captions = data["cap_tokens"].to(self.device)
            output = self.model(images, captions)
            
            loss, loss_dict = self.criterion(**output)
            
            self.optimizer.zero_grad()
            if self.config.train.get('use_fp16'):
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()
            
            if is_test:
                break

        if last_idx < 0:
            if self.logger is not None:
                self.logger.log(f"Skip local epoch for client {self.client}: train loader has no full batch")
            return

        loss_dict = {'{}'.format(key): val
                     for key, val in loss_dict.items()}
        loss_dict['step'] = cur_step(self.cur_epoch, last_idx, len(self.train_loader))
        
        loss_dict = {'{}{}'.format(prefix, key): val
                     for key, val in loss_dict.items()}
        loss_dict['step'] = cur_step(self.cur_epoch, last_idx, len(self.train_loader))
        
    def predict_logits(self, dataloader):
        self.model.cuda()
        self.model.eval()
        img_logits_list = []
        txt_logits_list = []
        with torch.no_grad():
            for i, (images, captions, _, _, a_, b_, index) in enumerate(dataloader):
                images = images.to(self.device)
                captions = captions.to(self.device)
                output = self.model(images, captions)
                img_logits = output['image_features']
                txt_logits = output['caption_features']
                img_logits_list.append(img_logits.float().cpu().numpy().astype(np.float32))
                txt_logits_list.append(txt_logits.float().cpu().numpy().astype(np.float32))
        if not img_logits_list or not txt_logits_list:
            return np.empty((0, self.args.feature_dim), dtype=np.float32), np.empty((0, self.args.feature_dim), dtype=np.float32)
        return np.concatenate(img_logits_list, axis=0).astype(np.float32, copy=False), np.concatenate(txt_logits_list, axis=0).astype(np.float32, copy=False)

    def distill_with_logits(self, dataloader, avg_img_logits, avg_txt_logits):
        self.model.cuda()
        self.model.train()
        idx = 0
        for i, (images, captions, _, _, a_, b_, index) in enumerate(dataloader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            batch_size = images.size(0)
            img_soft_label = torch.as_tensor(avg_img_logits[idx:idx+batch_size], dtype=torch.float32, device=self.device)
            txt_soft_label = torch.as_tensor(avg_txt_logits[idx:idx+batch_size], dtype=torch.float32, device=self.device)
            if torch.isnan(img_soft_label).any() or torch.isnan(txt_soft_label).any():
                print("Found nan in soft labels!")
            idx += batch_size
            self.optimizer.zero_grad()
            output = self.model(images, captions)
            loss = compute_distill_loss(output['image_features'], img_soft_label)
            loss += compute_distill_loss(output['caption_features'], txt_soft_label)
            if torch.isnan(loss) or torch.isinf(loss):
                print("Found invalid loss during distillation!")
                continue
            if i == 0 or (i + 1) % 20 == 0:
                client_key = getattr(self, 'client_idx', self.client)
                self.logger.log(f"Distill client {client_key} ({self.dset_name}) step {i}: loss={loss.item():.6f}")
            loss.backward()
            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()
        
    def generate_logits(self, dataloader):
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            img_vec = []
            txt_vec = []
            distill_index = []
            for idx, (images, captions, _, _, _, _, index) in enumerate(dataloader):
                images = images.to(self.device)
                captions = captions.to(self.device)

                output = self.model(images, captions)

                out_img = output['image_features'].sum(axis=1) if len(output['image_features'].shape) == 3 else output[
                    'image_features']
                out_txt = output['caption_features'].sum(axis=1) if len(output['caption_features'].shape) == 3 else \
                    output['caption_features']
                img_vec.extend(out_img)
                txt_vec.extend(out_txt)
                distill_index.extend(index)

                if is_test and idx == 1:
                    break

        img_vec = torch.cat(img_vec, dim=0).view(-1, self.args.feature_dim)
        txt_vec = torch.cat(txt_vec, dim=0).view(-1, self.args.feature_dim)

        img_vec = img_vec.cpu()
        txt_vec = txt_vec.cpu()
        self.model.cpu()

        return {'img': img_vec, 'txt': txt_vec}, distill_index

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
