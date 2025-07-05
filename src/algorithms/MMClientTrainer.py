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

from src.algorithms.base import EngineBase
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
                if idx > 10:
                    break
                images = data["processed_img"].to(self.device)
                captions = data["cap_tokens"].to(self.device)
                output = self.model(images, captions)
                loss, loss_dict = self.criterion(**output)
                # === Proximal term ===
                prox_loss = 0.0
                for name, param in self.model.named_parameters():
                    prox_loss += ((param - global_params[name].to(param.device)) ** 2).sum()
                total_loss = loss + 0.5 * mu * prox_loss
                print(f"Client {self.client} - Epoch {self.local_epoch}, Step {idx}, Loss: {loss:.4f}, Prox Loss: {prox_loss:.4f}")
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
            elif self.args.FL_algorithm == 'MASA':
                self.train_gcmd_epoch(prefix='MASA_')

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/mm/Client{self.client}-model_{self.local_epoch}.pth')

        self.old_model.cpu()
        self.model.cpu()

        del self.old_model
        import gc
        gc.collect()
        
    def run_with_moon(self, global_model, prev_models=None, temperature=0.5, mu=1.0):
        self.model.cuda()
        self.model.train()
        global_model.eval()
        global_model.cuda()
        if prev_models is not None:
            for m in prev_models:
                m.eval()
                m.cuda()
        for i in range(self.local_epochs):
            for idx, data in enumerate(self.train_loader):
                if idx > 10:
                    break
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
                    img_fvec_prev = [m(images, captions)['image_features'] for m in prev_models] if prev_models else []
                    txt_fvec_prev = [m(images, captions)['caption_features'] for m in prev_models] if prev_models else []
                # 多模态损失
                loss_cls, _ = self.criterion(**output)
                # MOON对比损失
                cos = nn.CosineSimilarity(dim=-1)
                posi_img = cos(img_fvec, img_fvec_global)
                posi_txt = cos(txt_fvec, txt_fvec_global)
                img_logits = posi_img.reshape(-1, 1)
                txt_logits = posi_txt.reshape(-1, 1)
                if prev_models:
                    for img_fvec_p in img_fvec_prev:
                        nega = cos(img_fvec, img_fvec_p)
                        img_logits = torch.cat((img_logits, nega.reshape(-1, 1)), dim=1)
                    for txt_fvec_p in txt_fvec_prev:
                        nega = cos(txt_fvec, txt_fvec_p)
                        txt_logits = torch.cat((txt_logits, nega.reshape(-1, 1)), dim=1)
                img_logits /= temperature
                txt_logits /= temperature
                contrastive_labels = torch.zeros(images.size(0)).long().to(self.device)
                loss_con_img = mu * nn.CrossEntropyLoss()(img_logits, contrastive_labels)
                loss_con_txt = mu * nn.CrossEntropyLoss()(txt_logits, contrastive_labels)
                loss = loss_cls + loss_con_img + loss_con_txt
                loss.backward()
                self.optimizer.step()
    def train_gcmd_epoch(self, prefix=''):
        # 1. 收集有标签样本特征
        self.model.eval()
        img_features, img_labels = [], []
        txt_features, txt_labels = [], []
        for idx, data in enumerate(self.train_loader):
            if idx > 10: break
            images = data["processed_img"].to(self.device)
            captions = data["cap_tokens"].to(self.device)
            labels = data["class_id"]
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
            # 提取特征
            img_feat = self.model.img_enc(images)["embedding"].detach().cpu().numpy()
            txt_feat = self.model.txt_enc(captions).detach().cpu().numpy()
            img_features.append(img_feat)
            txt_features.append(txt_feat)
            img_labels.append(labels.numpy())
            txt_labels.append(labels.numpy())
        img_features = np.concatenate(img_features, axis=0)
        txt_features = np.concatenate(txt_features, axis=0)
        img_labels = np.concatenate(img_labels, axis=0)
        txt_labels = np.concatenate(txt_labels, axis=0)

        # 2. 计算互信息分数MIS
        def compute_mis(features, labels):
            unique_labels, labels_mapped = np.unique(labels, return_inverse=True)
            kmeans = KMeans(n_clusters=len(unique_labels))
            cluster_labels = kmeans.fit_predict(features)
            # 互信息分数
            from scipy.stats import entropy
            C = len(unique_labels)
            joint_pmf = np.zeros((C, C))
            for y_true, y_pred in zip(labels_mapped, cluster_labels):
                joint_pmf[y_true, y_pred] += 1
            joint_pmf /= len(labels_mapped)
            marginal_pmf_true = np.sum(joint_pmf, axis=1)
            marginal_pmf_pred = np.sum(joint_pmf, axis=0)
            mi = 0.0
            for c in range(C):
                for c_tilde in range(C):
                    if joint_pmf[c, c_tilde] > 0:
                        mi += joint_pmf[c, c_tilde] * np.log2(
                            joint_pmf[c, c_tilde] / (marginal_pmf_true[c] * marginal_pmf_pred[c_tilde])
                        )
            H_true = entropy(marginal_pmf_true, base=2)
            H_pred = entropy(marginal_pmf_pred, base=2)
            mis = mi / (H_true + H_pred + 1e-8)
            return mis

        mis_img = compute_mis(img_features, img_labels)
        mis_txt = compute_mis(txt_features, txt_labels)
        superior = 'img' if mis_img >= mis_txt else 'txt'
        inferior = 'txt' if superior == 'img' else 'img'

        # 3. 采样无标签数据，计算SCM
        def compute_scm(features):
            return torch.matmul(features, features.T)
        
        self.model.train()
        for idx, data in enumerate(self.train_loader):
            if idx <= 10: continue
            images = data["processed_img"].to(self.device)
            captions = data["cap_tokens"].to(self.device)
            # 提取特征
            img_feat = self.model.img_enc(images)["embedding"]
            txt_feat = self.model.txt_enc(captions)
            scm_img = compute_scm(img_feat)
            scm_txt = compute_scm(txt_feat)

            # 4. 损失计算
            loss = 0
            if superior == 'img':
                rec_loss_img = F.mse_loss(img_feat, img_feat)
                rec_loss_txt = F.mse_loss(txt_feat, txt_feat)
                distill_loss = F.mse_loss(scm_img, scm_txt)
                loss = rec_loss_img + rec_loss_txt + distill_loss
            else:
                rec_loss_txt = F.mse_loss(txt_feat, txt_feat)
                rec_loss_img = F.mse_loss(img_feat, img_feat)
                distill_loss = F.mse_loss(scm_txt, scm_img)
                loss = rec_loss_txt + rec_loss_img + distill_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_epoch(self, prefix=''):
        for idx, data in enumerate(self.train_loader):
            if idx > 10:
                break
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

        loss_dict = {'{}'.format(key): val
                     for key, val in loss_dict.items()}
        loss_dict['step'] = cur_step(self.cur_epoch, idx, len(self.train_loader))
        
        loss_dict = {'{}{}'.format(prefix, key): val
                     for key, val in loss_dict.items()}
        loss_dict['step'] = cur_step(self.cur_epoch, idx, len(self.train_loader))
        
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
                img_logits_list.append(img_logits.cpu().numpy())
                txt_logits_list.append(txt_logits.cpu().numpy())
        return np.concatenate(img_logits_list, axis=0), np.concatenate(txt_logits_list, axis=0)

    def distill_with_logits(self, dataloader, avg_img_logits, avg_txt_logits):
        self.model.cuda()
        self.model.train()
        idx = 0
        for i, (images, captions, _, _, a_, b_, index) in enumerate(dataloader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            batch_size = images.size(0)
            img_soft_label = torch.tensor(avg_img_logits[idx:idx+batch_size]).to(self.device)
            txt_soft_label = torch.tensor(avg_txt_logits[idx:idx+batch_size]).to(self.device)
            idx += batch_size
            self.optimizer.zero_grad()
            output = self.model(images, captions)
            image_logits = output['image_features']
            text_logits = output['caption_features']
            loss = nn.MSELoss()(image_logits, img_soft_label)
            loss += nn.MSELoss()(text_logits, txt_soft_label)
            loss.backward()
            self.optimizer.step()

    def train_on_private_data(self):
        self.model.cuda()
        self.model.train()
        for i in range(self.local_epochs):
            for idx, data in enumerate(self.train_loader):
                if idx > 10:
                    break
                self.optimizer.zero_grad()
                images = data["processed_img"].to(self.device)
                captions = data["cap_tokens"].to(self.device)
                output = self.model(images, captions)
                image_features = output['image_features']
                text_features = output['caption_features']
                loss = self.criterion(image_features, text_features)
                loss.backward()
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
