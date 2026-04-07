import copy
import gc
import operator
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from apex import amp
from sklearn.metrics import pairwise_distances

from algorithms.mm_eval import batch
from src import losses
from src.networks.clip_model import ClientTextEncoder
from src.networks.clip_model import ClientImageEncoder
from src.networks.language_model import EncoderText
from src.networks.resnet_client import resnet18_client
from src.utils.Utils import to_one_hot
from src.utils.model_utils import is_embedding_model
from src.algorithms.distill_utils import compute_distill_loss

torch.backends.cudnn.enabled = True

from tqdm import tqdm

import numpy as np
import os
import random
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def seed_torch(seed=42):
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


def get_result_list(query_sorted_idx, gt_list, ignore_list, top_k):
    return_retrieval_list = []
    count = 0
    while len(return_retrieval_list) < top_k:
        query_idx = query_sorted_idx[count]
        if query_idx in ignore_list:
            pass
        else:
            if query_idx in gt_list:
                return_retrieval_list.append(1)
            else:
                return_retrieval_list.append(0)
        count += 1
    return return_retrieval_list


def recall_at_k(feature, query_id, retrieval_list, top_k):
    distance = pairwise_distances(feature, feature)
    result = 0
    for i in range(len(query_id)):
        query_distance = distance[query_id[i], :]
        gt_list = retrieval_list[i][0]
        ignore_list = retrieval_list[i][1]
        query_sorted_idx = np.argsort(query_distance)
        query_sorted_idx = query_sorted_idx.tolist()
        result_list = get_result_list(query_sorted_idx, gt_list, ignore_list, top_k)
        result += 1. if sum(result_list) > 0 else 0
    result = result / float(len(query_id))
    return result


gpuid = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    if isinstance(target, list):
        target = torch.tensor(target, dtype=torch.long)
    batch_size = target.size(0)
    
    n_classes = output.size(1)
    if target.min() < 0 or target.max() >= n_classes:
        raise ValueError(f"Target contains invalid class index. Expected 0-{n_classes-1}, got {target.min()}-{target.max()}")
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    device = pred.device
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.to(device)
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


is_test = False

class ClientTrainer:
    def __init__(self, args, dataset, class_size, logger, inter_distance=4, loss='softmax',
                 gpuid='cuda:0', num_epochs=30, init_lr=0.0001, decay=0.1,
                 num_workers=4, print_freq=10, save_step=10, scale=128, pool_type='max_avg', client_id=-1, wandb=None):
        seed_torch()
        self.args = args
        self.client_id = client_id
        self.dset_name = dataset
        self.local_feature = None
        self.classSize = class_size
        self.selected_cluster = client_id % 5
        self.global_model = None

        self.gpuid = gpuid if torch.cuda.is_available() else 'cpu'

        self.batch_size = args.batch_size
        self.num_workers = num_workers

        self.decay_time = [False, False]
        self.init_lr = init_lr
        self.decay_rate = decay
        self.num_epochs = num_epochs
        self.cur_epoch = -1

        self.record = []
        self.epoch = 0
        self.print_freq = print_freq
        self.save_step = save_step
        self.loss = loss
        self.losses = AverageMeter()
        self.top1, self.test_top1 = AverageMeter(), AverageMeter()
        self.top5, self.test_top5 = AverageMeter(), AverageMeter()

        # model parameter
        self.scale = scale
        self.pool_type = pool_type
        self.inter_distance = inter_distance
        if not self.setsys(): print('system error'); return

        self.logger = logger
        self.wandb = wandb

        self.loadData()
        self.setModel()

        self.old_model = None
        self.global_anchor = None

        self.local_epochs = args.local_epochs
        self.local_epoch = 0

    def run_with_prox(self):
        self.model.to(self.gpuid)
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.old_model.cuda()

        self.lr_scheduler(self.cur_epoch)

        mu = getattr(self.args, 'mu', 0.01)
        if self.dset_name == 'image':
            global_params = {k: v.clone().detach() for k, v in self.global_model.img_enc.state_dict().items()}
        if self.dset_name == 'text':
            global_params = {k: v.clone().detach() for k, v in self.global_model.txt_enc.state_dict().items()}

        for i in range(self.local_epochs):
            self.local_epoch += 1
            self.model.train()
            for idx, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                if self.dset_name == 'image':
                    inputs_bt = data["processed_img"]
                    labels_bt = data["class_id"]
                    if isinstance(labels_bt, list):
                        labels_bt = torch.tensor(labels_bt, dtype=torch.long)
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    labels_var = torch.autograd.Variable(labels_bt).to(self.gpuid)
                    if is_embedding_model(self.args.model):
                        fvec, _, _ = self.model(inputs_var)
                    elif self.args.model == 'resnet':
                        fvec, _, _, _ = self.model(inputs_var)
                elif self.dset_name == 'text':
                    inputs_bt = data["cap_tokens"]
                    labels_bt = data["class_id"]
                    if isinstance(labels_bt, list):
                        labels_bt = torch.tensor(labels_bt, dtype=torch.long)
                    inputs_bt, labels_bt = map(lambda t: torch.cat(t) if type(t) != torch.Tensor else t,
                                               (inputs_bt, labels_bt))
                    inputs_bt, labels_var = map(lambda t: t.to(self.gpuid).contiguous(), (inputs_bt, labels_bt))
                    if is_embedding_model(self.args.model):
                        fvec, _, _ = self.model(inputs_bt)
                    elif self.args.model == 'resnet':
                        fvec, _, _, _ = self.model(inputs_bt)
                loss = self.criterion(fvec, labels_var)
                # === Proximal term ===
                common_keys = set(dict(self.model.named_parameters()).keys()) & set(global_params.keys())
                prox_loss = 0.0
                for name in common_keys:
                    param = dict(self.model.named_parameters())[name]
                    prox_loss += ((param - global_params[name].to(param.device)) ** 2).sum()
                total_loss = loss + 0.5 * mu * prox_loss
                # print(f'Client {self.client_id} - Epoch {self.local_epoch}, Step {idx}, Loss: {total_loss:.4f}, Prox Loss: {prox_loss:.4f}')
                total_loss.backward()
                self.optimizer.step()
                if is_test:
                    break

        self.test()
        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/{self.dset_name}/Client{self.client_id}-model_{self.local_epoch}.pth')
        self.model.cpu()
        self.old_model.cpu()
        del self.old_model
        import gc
        gc.collect()
    def run(self):
        self.model.to(self.gpuid)
        if getattr(self.args, 'FL_algorithm', '') == 'MASA':
            self._ensure_masa_modules()
            self._move_masa_modules(self.gpuid)
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.old_model.cuda()

        self.lr_scheduler(self.cur_epoch)

        for i in range(self.local_epochs):
            self.local_epoch += 1
            if getattr(self.args, 'FL_algorithm', '') == 'MASA':
                self.train_masa_epoch()
            else:
                self.tra()

        self.test()

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/{self.dset_name}/Client{self.client_id}-model_{self.local_epoch}.pth')

        self.model.cpu()
        self.old_model.cpu()
        if getattr(self.args, 'FL_algorithm', '') == 'MASA':
            self._move_masa_modules('cpu')

        del self.old_model
        import gc
        gc.collect()

    def set_global_anchor(self, anchor):
        self.global_anchor = anchor

    def _ensure_masa_modules(self):
        if getattr(self, '_masa_modules_ready', False):
            return
        feature_dim = self.args.feature_dim
        self.masa_decoder = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True), nn.Linear(feature_dim, feature_dim))
        self.optimizer.add_param_group({'params': self.masa_decoder.parameters()})
        self._masa_modules_ready = True

    def _move_masa_modules(self, device):
        masa_decoder = self.__dict__.get('masa_decoder', None)
        if masa_decoder is not None:
            masa_decoder.to(device)

    def _masa_prepare_batch(self, data):
        if self.dset_name == 'image':
            inputs_bt = data['processed_img']
            labels_bt = data['class_id']
            if isinstance(labels_bt, list):
                labels_bt = torch.tensor(labels_bt, dtype=torch.long)
            inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
            labels_var = torch.autograd.Variable(labels_bt).to(self.gpuid)
        else:
            inputs_bt = data['cap_tokens']
            labels_bt = data['class_id']
            if isinstance(labels_bt, list):
                labels_bt = torch.tensor(labels_bt, dtype=torch.long)
            inputs_bt, labels_bt = map(lambda t: torch.cat(t) if type(t) != torch.Tensor else t, (inputs_bt, labels_bt))
            inputs_var, labels_var = map(lambda t: t.to(self.gpuid).contiguous(), (inputs_bt, labels_bt))
        if is_embedding_model(self.args.model):
            logits, _, features = self.model(inputs_var)
        elif self.args.model == 'resnet':
            logits, features, _, _ = self.model(inputs_var)
        else:
            raise ValueError(f'Unsupported model for MASA single-modal training: {self.args.model}')
        return inputs_var.float(), labels_var, logits, features

    def _masa_reconstruction_target(self, inputs, features):
        target = inputs.float().view(inputs.size(0), -1)
        feat = features.view(features.size(0), -1)
        if target.size(-1) != feat.size(-1):
            common_dim = min(target.size(-1), feat.size(-1))
            target = target[..., :common_dim]
            feat = feat[..., :common_dim]
        return target, feat

    def train_masa_epoch(self, prefix='MASA_', log_metrics=True):
        self._ensure_masa_modules()
        self.model.train()
        self.masa_decoder.train()
        rec_weight = float(getattr(self.args, 'masa_rec_weight', 0.1))
        for _, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            inputs_var, labels_var, logits, features = self._masa_prepare_batch(data)
            recon_target, feature_vec = self._masa_reconstruction_target(inputs_var, features)
            total_loss = self.criterion(logits, labels_var) + rec_weight * F.mse_loss(self.masa_decoder(feature_vec), recon_target)
            prec1, prec5 = accuracy(logits.data, labels_var, topk=(1, 5))
            self.top1.update(prec1[0], inputs_var.size(0))
            self.top5.update(prec5[0], inputs_var.size(0))
            self.losses.update(total_loss.item(), inputs_var.size(0))
            total_loss.backward()
            self.optimizer.step()
            if is_test:
                break
        if log_metrics and self.logger is not None:
            self.logger.log('Epoch: [{0}] {1}	Loss {loss.val:.4f} ({loss.avg:.4f})	Prec@1 {top1.val:.3f} ({top1.avg:.3f})	Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(self.local_epoch, f'{prefix}{self.dset_name}', loss=self.losses, top1=self.top1, top5=self.top5))
            self.losses = AverageMeter(); self.top1 = AverageMeter(); self.top5 = AverageMeter()

    def get_masa_encoder_state(self):
        return {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items() if not key.startswith('class_fc')}

    def load_masa_encoder_state(self, state_dict):
        self.model.load_state_dict(state_dict, strict=False)

    def _masa_temp_model(self, state_dict):
        temp_model = copy.deepcopy(self.model)
        temp_model.load_state_dict(state_dict, strict=False)
        temp_model.to(self.gpuid)
        temp_model.eval()
        for param in temp_model.parameters():
            param.requires_grad_(False)
        return temp_model

    def _masa_extract_features(self, model, inputs):
        with torch.no_grad():
            if is_embedding_model(self.args.model):
                _, _, features = model(inputs)
            elif self.args.model == 'resnet':
                _, features, _, _ = model(inputs)
            else:
                raise ValueError(f'Unsupported model for MASA personalization: {self.args.model}')
        return features

    def _masa_classify_features(self, features):
        if hasattr(self.model, 'class_fc_2'):
            weight = self.model.relu(self.model.class_fc_2.weight) if hasattr(self.model, 'relu') else self.model.class_fc_2.weight
            return F.linear(features, weight, self.model.class_fc_2.bias)
        if hasattr(self.model, 'class_fc'):
            return self.model.class_fc(features)
        raise AttributeError('No classifier head found for MASA single-modal personalization')

    def _masa_query_batches(self, ratio=None):
        ratio = float(getattr(self.args, 'ascc_query_ratio', 0.2) if ratio is None else ratio)
        query_batches = []
        for batch in self.train_loader:
            inputs = batch['processed_img'] if self.dset_name == 'image' else batch['cap_tokens']
            labels = batch['class_id']
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
            if type(inputs) != torch.Tensor:
                inputs = torch.cat(inputs)
            n_query = max(1, int(len(inputs) * ratio))
            idx = torch.randperm(len(inputs))[:n_query]
            query_batches.append((inputs[idx].to(self.gpuid).float(), labels[idx].to(self.gpuid)))
            if is_test:
                break
        return query_batches

    def masa_personalize_from_clusters(self, cluster_states, cluster_labels=None):
        if not cluster_states:
            return
        self.model.to(self.gpuid)
        self._ensure_masa_modules(); self._move_masa_modules(self.gpuid)
        cluster_labels = list(range(len(cluster_states))) if cluster_labels is None else cluster_labels
        query_batches = self._masa_query_batches()
        if not query_batches:
            self.load_masa_encoder_state(cluster_states[0]); self.selected_cluster = int(cluster_labels[0]); return
        temp_models = [self._masa_temp_model(state) for state in cluster_states]
        attn_logits = nn.Parameter(torch.zeros(len(cluster_states), device=self.gpuid))
        optimizer = optim.Adam([attn_logits], lr=float(getattr(self.args, 'ascc_lr', 5e-2)))
        for _ in range(int(getattr(self.args, 'ascc_attn_epoch', 3))):
            optimizer.zero_grad(); weights = torch.softmax(attn_logits, dim=0); total_loss = 0.0
            for inputs, labels in query_batches:
                features = [self._masa_extract_features(model, inputs) for model in temp_models]
                mixed_feature = sum(weights[idx] * feat for idx, feat in enumerate(features))
                total_loss = total_loss + F.cross_entropy(self._masa_classify_features(mixed_feature), labels)
            total_loss.backward(); optimizer.step()
        final_weights = torch.softmax(attn_logits.detach(), dim=0).cpu(); personalized_state = {}
        for key in cluster_states[0].keys():
            personalized_state[key] = sum(state[key].detach().cpu() * float(final_weights[idx].item()) for idx, state in enumerate(cluster_states))
        self.load_masa_encoder_state(personalized_state)
        self.selected_cluster = int(cluster_labels[int(torch.argmax(final_weights).item())])
        for model in temp_models:
            model.cpu()
        del temp_models; gc.collect()

    def masa_finetune_local(self, ft_epochs=1):
        self.model.to(self.gpuid); self._ensure_masa_modules(); self._move_masa_modules(self.gpuid)
        for _ in range(int(ft_epochs)):
            self.train_masa_epoch(log_metrics=False)
        self.model.cpu(); self._move_masa_modules('cpu')

    def train_with_anchor(self):
        self.model.to(self.gpuid)
        self.lr_scheduler(self.cur_epoch)

        mse_loss = nn.MSELoss()
        anchor_accum = []

        for _ in range(self.local_epochs):
            self.local_epoch += 1
            self.model.train()
            for idx, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                if self.dset_name == 'image':
                    inputs_bt = data["processed_img"]
                    labels_bt = data["class_id"]
                    if isinstance(labels_bt, list):
                        labels_bt = torch.tensor(labels_bt, dtype=torch.long)
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    labels_var = torch.autograd.Variable(labels_bt).to(self.gpuid)

                    if is_embedding_model(self.args.model):
                        logits, _, embedding = self.model(inputs_var)
                    elif self.args.model == 'resnet':
                        logits, embedding, _, _ = self.model(inputs_var)

                elif self.dset_name == 'text':
                    inputs_bt = data["cap_tokens"]
                    labels_bt = data["class_id"]
                    if isinstance(labels_bt, list):
                        labels_bt = torch.tensor(labels_bt, dtype=torch.long)

                    inputs_bt, labels_bt = map(
                        lambda t: torch.cat(t) if not isinstance(t, torch.Tensor) else t,
                        (inputs_bt, labels_bt)
                    )
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    labels_var = torch.autograd.Variable(labels_bt).to(self.gpuid)

                    if is_embedding_model(self.args.model):
                        logits, _, embedding = self.model(inputs_var)
                    elif self.args.model == 'resnet':
                        logits, embedding, _, _ = self.model(inputs_var)

                if embedding.dim() > 2:
                    embedding = embedding.view(embedding.size(0), -1)

                loss_cls = self.criterion(logits, labels_var)
                if getattr(self, 'global_anchor', None) is not None:
                    anchor_target = torch.tensor(self.global_anchor, device=embedding.device, dtype=embedding.dtype)
                    anchor_loss = mse_loss(embedding.mean(dim=0), anchor_target)
                    total_loss = loss_cls + 0.1 * anchor_loss
                else:
                    total_loss = loss_cls

                total_loss.backward()
                self.optimizer.step()

                anchor_accum.append(embedding.detach().mean(dim=0).cpu().numpy())
                if is_test:
                    break

        if self.args.save_client:
            torch.save(self.model.state_dict(),
                       f'./saved_clients/{self.dset_name}/Client{self.client_id}-model_{self.local_epoch}.pth')

        if anchor_accum:
            local_anchor = np.mean(np.stack(anchor_accum), axis=0)
        else:
            local_anchor = np.zeros(self.args.feature_dim, dtype=np.float32)

        model_copy = copy.deepcopy(self.model).cpu()
        self.model.cpu()
        gc.collect()
        return model_copy, local_anchor
    def run_with_moon(self, global_model, prev_models=None, temperature=0.5, mu=1.0):
        def printnreset(name):
            self.logger.log('Epoch: [{0}] {1}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                self.local_epoch, name, loss=self.losses, top1=self.top1, top5=self.top5))

            self.losses = AverageMeter()
            self.top1 = AverageMeter()
            self.top5 = AverageMeter()
            
        self.model.train()
        self.model.cuda()
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
                if self.dset_name == 'image':
                    inputs = data["processed_img"].to(self.gpuid)
                    labels = data["class_id"]
                    if isinstance(labels, list):
                        labels = torch.tensor(labels,dtype=torch.long)
                    labels = labels.to(self.gpuid)
                    if is_embedding_model(self.args.model):
                        fvec, _, local_logits = self.model(inputs)
                    elif self.args.model == 'resnet':
                        fvec, _, _, _ = self.model(inputs)
                        self.model.phase = "extract_conv_feature"
                        self.model.is_train = False
                        local_logits = self.model(inputs)
                        self.model.phase = "None"
                        self.model.is_train = True
                    with torch.no_grad():
                        fvec_global = global_model.img_enc(inputs)["embedding"]
                        fvec_prev = prev_models(inputs)[0] if prev_models else None
                elif self.dset_name == 'text':
                    inputs = data["cap_tokens"].to(self.gpuid)
                    labels = data["class_id"]
                    if isinstance(labels, list):
                        labels = torch.tensor(labels,dtype=torch.long)
                    labels = labels.to(self.gpuid)
                    if is_embedding_model(self.args.model):
                        fvec, _, local_logits = self.model(inputs)
                    elif self.args.model == 'resnet':
                        fvec, _, _, _ = self.model(inputs)
                        self.model.phase = "extract_conv_feature"
                        self.model.is_train = False
                        local_logits = self.model(inputs).squeeze()
                        self.model.phase = "None"
                        self.model.is_train = True
                    with torch.no_grad():
                        fvec_global = global_model.txt_enc(inputs)
                        fvec_prev = prev_models(inputs)[0] if prev_models else None
                # print(f'fvec: {fvec}, local_logits: {local_logits}, fvec_global: {fvec_global}, fvec_prev: {fvec_prev}')
                # classification loss
                loss_cls = self.criterion(fvec, labels)
                # MOON contrastive loss
                cos = nn.CosineSimilarity(dim=-1)
                posi = cos(local_logits, fvec_global)
                logits = posi.reshape(-1, 1)
                if prev_models:
                    nega = cos(fvec, fvec_prev)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                logits /= temperature
                contrastive_labels = torch.zeros(inputs.size(0)).long().to(self.gpuid)
                loss_con = mu * nn.CrossEntropyLoss()(logits, contrastive_labels)
                loss = loss_cls + loss_con
                # print(f'loss: {loss:.3f}, cls: {loss_cls:.3f}, con: {loss_con:.3f}')
                loss.backward()
                self.optimizer.step()
                
                prec1, prec5 = accuracy(fvec.data, labels, topk=(1, 5))
                self.top1.update(prec1[0], inputs.size(0))
                self.top5.update(prec5[0], inputs.size(0))

                self.losses.update(loss.item(), inputs.size(0))

        printnreset(self.dset_name)

    ##################################################
    # step 0: System check and predefine function
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        return True

    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):
        self.class_label = torch.Tensor(np.array(range(self.classSize)))
        print('output size: {}'.format(self.classSize))

        return

    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self):
        if self.logger is not None:
            self.logger.log(f'Setting model {self.client_id}')
        if self.dset_name == 'image' and is_embedding_model(self.args.model):
            self.model = ClientImageEncoder(num_class=self.classSize, embed_dim=self.args.feature_dim, 
                                        mlp_local=self.args.mlp_local, is_train=True,
                                        use_pretrained_proj=bool(self.args.use_pretrained_proj), model_name=self.args.model,
                                        pretrained_proj_variant=getattr(self.args, 'pretrained_proj_variant', ''),
                                        pretrained_proj_path=getattr(self.args, 'pretrained_proj_path', ''))
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
        elif self.dset_name == 'image' and self.args.model == 'resnet':
            self.model = resnet18_client(pretrained=True, num_class=self.classSize, pool_type=self.pool_type,
                                         is_train=True, scale=self.scale, mlp_local=self.args.mlp_local, embed_dim=self.args.feature_dim)
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
            # params = [p for n, p in self.model.named_parameters() if "lora" in n and p.requires_grad]
        elif self.dset_name == 'text' and is_embedding_model(self.args.model):
            self.model = ClientTextEncoder(num_class=self.classSize, embed_dim=self.args.feature_dim,
                                        mlp_local=self.args.mlp_local, use_pretrained_proj=bool(self.args.use_pretrained_proj), model_name=self.args.model,
                                        pretrained_proj_variant=getattr(self.args, 'pretrained_proj_variant', ''),
                                        pretrained_proj_path=getattr(self.args, 'pretrained_proj_path', ''))
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
        elif self.dset_name == 'text' and self.args.model == 'resnet':
            self.model = EncoderText(embed_dim=self.args.feature_dim, num_class=self.classSize, scale=self.scale, mlp_local=self.args.mlp_local)
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
            # params = [p for n, p in self.model.named_parameters() if "lora" in n and p.requires_grad]
        self.center_criterion = nn.MSELoss()
        self.optimizer = optim.SGD(params, lr=self.init_lr,
                                momentum=0.9, weight_decay=0.00005)
        return

    def lr_scheduler(self, epoch):
        if epoch >= 0.5 * self.num_epochs and not self.decay_time[0]:
            self.decay_time[0] = True
            lr = self.init_lr * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch >= 0.8 * self.num_epochs and not self.decay_time[1]:
            self.decay_time[1] = True
            lr = self.init_lr * self.decay_rate * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return

    ##################################################
    # step 3: Learning
    ##################################################
    def tra(self):
        def printnreset(name):
            self.logger.log('Epoch: [{0}] {1}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                self.local_epoch, name, loss=self.losses, top1=self.top1, top5=self.top5))

            self.losses = AverageMeter()
            self.top1 = AverageMeter()
            self.top5 = AverageMeter()

        # Set model to training mode
        self.model.train()
        
        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                center_labels_var = torch.autograd.Variable(self.class_label.to(torch.long)).to(self.gpuid)
                if self.dset_name == 'image':
                    inputs_bt = data["processed_img"]
                    labels_bt = data["class_id"]
                    if isinstance(labels_bt, list):
                        labels_bt = torch.tensor(labels_bt,dtype=torch.long)
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    labels_var = torch.autograd.Variable(labels_bt).to(self.gpuid)

                    if self.args.model == 'resnet':
                        fvec, _, _, _ = self.model(inputs_var)
                        
                    elif is_embedding_model(self.args.model):
                        fvec, _, _ = self.model(inputs_var)

                elif self.dset_name == 'text':
                    inputs_bt = data["cap_tokens"]
                    labels_bt = data["class_id"]
                    if isinstance(labels_bt, list):
                        labels_bt = torch.tensor(labels_bt,dtype=torch.long)
                    
                    inputs_bt, labels_bt = map(lambda t: torch.cat(t) if type(t) != torch.Tensor else t,
                                               (inputs_bt, labels_bt))
                    inputs_bt, labels_var = map(lambda t: t.to(self.gpuid).contiguous(), (inputs_bt, labels_bt))
                    
                    if self.args.model == 'resnet':
                        fvec, _, _, _ = self.model(inputs_bt)
                        
                    elif is_embedding_model(self.args.model):
                        fvec, _, _ = self.model(inputs_bt)

                # intra_class_distance
                loss = self.criterion(fvec, labels_var)
                total_loss = loss
                # print("Prediction_label: ", fvec.data)
                # print("Ground_truth: ", labels_bt)
                prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 5))
                self.top1.update(prec1[0], inputs_bt.size(0))
                self.top5.update(prec5[0], inputs_bt.size(0))

                self.losses.update(total_loss.item(), inputs_bt.size(0))
                total_loss.backward()
                self.optimizer.step()

            if is_test:
                break

        printnreset(self.dset_name)

    def test(self):
        def printnreset(name):
            self.logger.log('TEST:  Epoch: [{0}] {1}\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                self.local_epoch, name, top1=self.test_top1, top5=self.test_top5))

            self.losses = AverageMeter()
            self.test_top1 = AverageMeter()
            self.test_top5 = AverageMeter()

        self.model.eval()
        self.model.cuda()

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                if self.dset_name == 'image' and is_embedding_model(self.args.model):
                    inputs_bt = data["processed_img"]
                    labels_bt = data["class_id"]
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    fvec, _, _ = self.model(inputs_var)
                    
                elif self.dset_name == 'image' and self.args.model == 'resnet':
                    inputs_bt = data["processed_img"]
                    labels_bt = data["class_id"]
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    fvec, _, _, _ = self.model(inputs_var)
                    
                elif self.dset_name == 'text' and is_embedding_model(self.args.model):
                    inputs_bt = data["cap_tokens"]
                    labels_bt = data["class_id"]
                    inputs_bt = inputs_bt.to(self.gpuid)
                    fvec, _, _ = self.model(inputs_bt)
                    
                elif self.dset_name == 'text' and self.args.model == 'resnet':
                    inputs_bt = data["cap_tokens"]
                    labels_bt = data["class_id"]
                    inputs_bt = inputs_bt.to(self.gpuid)
                    fvec, _, _, _ = self.model(inputs_bt)

                prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 5))
                self.test_top1.update(prec1[0], inputs_bt.size(0))
                self.test_top5.update(prec5[0], inputs_bt.size(0))

        if self.wandb is not None:
            self.wandb.log({f"client_{self.client_id}_val_top1": self.test_top1.avg,
                            f"client_{self.client_id}_val_top5": self.test_top5.avg}, step=self.local_epoch)

        printnreset(self.dset_name)
        self.model.train()
        return self.losses.avg, self.test_top1.avg, self.test_top5.avg
        
    def predict_logits(self, dataloader):
        """Generate logits on the public alignment dataset."""
        self.model.cuda()
        self.model.eval()
        self.model.is_train = False
        logits_list = []
        with torch.no_grad():
            for i, (images, captions, _, _, a_, b_, index) in enumerate(dataloader):
                if self.dset_name == 'image' and is_embedding_model(self.args.model):
                    inputs = images.to(self.gpuid)
                    output = self.model(inputs)
                elif self.dset_name == 'text' and is_embedding_model(self.args.model):
                    inputs = captions.to(self.gpuid)
                    output = self.model(inputs)
                elif self.dset_name == 'image' and self.args.model == 'resnet':
                    inputs = images.to(self.gpuid)
                    self.model.phase = "extract_conv_feature"
                    output = self.model(inputs)
                    self.model.phase = "None"
                elif self.dset_name == 'text' and self.args.model == 'resnet':
                    inputs = captions.to(self.gpuid)
                    self.model.phase = "extract_conv_feature"
                    output = self.model(inputs)
                    self.model.phase = "None"
                logits_list.append(output.float().cpu().numpy().astype(np.float32))
        self.model.is_train = True
        
        if not logits_list:
            return np.empty((0, self.args.feature_dim), dtype=np.float32)
        return np.concatenate(logits_list, axis=0).astype(np.float32, copy=False)

    def distill_with_logits(self, dataloader, avg_img_logits, avg_txt_logits):
        def printnreset(name):
            self.logger.log('Epoch: [{0}] {1}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                self.local_epoch, name, loss=self.losses))

            self.losses = AverageMeter()
            
        """Train with aggregated soft labels for public-data distillation."""
        self.model.cuda()
        self.model.train()
        idx = 0
        for i, (images, captions, _, _, a_, b_, index) in enumerate(dataloader):
            if self.dset_name == 'image':
                inputs = images.to(self.gpuid)
                batch_size = inputs.size(0)
                img_soft_label = torch.as_tensor(avg_img_logits[idx:idx+batch_size], dtype=torch.float32, device=self.gpuid)
                idx += batch_size
                self.optimizer.zero_grad()
                self.model.is_train = False
                if is_embedding_model(self.args.model):
                    output = self.model(inputs).float()
                elif self.args.model == 'resnet':
                    self.model.phase = "extract_conv_feature"
                    output = self.model(inputs).float()
                    self.model.phase = "None"
                self.model.is_train = True
                loss = compute_distill_loss(output, img_soft_label)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                self.optimizer.step()
                if i == 0 or (i + 1) % 20 == 0:
                    client_key = getattr(self, 'client_idx', self.client_id)
                    self.logger.log(f"Distill client {client_key} ({self.dset_name}) step {i}: loss={loss.item():.6f}")
            elif self.dset_name == 'text':
                inputs = captions.to(self.gpuid)
                batch_size = inputs.size(0)
                txt_soft_label = torch.as_tensor(avg_txt_logits[idx:idx+batch_size], dtype=torch.float32, device=self.gpuid)
                idx += batch_size
                self.optimizer.zero_grad()
                self.model.is_train = False
                if is_embedding_model(self.args.model):
                    output = self.model(inputs).float()
                elif self.args.model == 'resnet':
                    self.model.phase = "extract_conv_feature"
                    output = self.model(inputs).float()
                    self.model.phase = "None"
                self.model.is_train = True
                loss = compute_distill_loss(output, txt_soft_label)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                self.optimizer.step()
                if i == 0 or (i + 1) % 20 == 0:
                    client_key = getattr(self, 'client_idx', self.client_id)
                    self.logger.log(f"Distill client {client_key} ({self.dset_name}) step {i}: loss={loss.item():.6f}")

            self.losses.update(loss.item(), inputs.size(0))

            if is_test:
                break

        printnreset(self.dset_name)
    def to_half(self):
        # Mixed precision
        # https://nvidia.github.io/apex/amp.html
        _, self.optimizer = amp.initialize(models=[], optimizers=self.optimizer,
                                                    opt_level='O2')

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError
