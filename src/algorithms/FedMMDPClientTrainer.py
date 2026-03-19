import copy
import gc
import os
import random

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from apex import amp
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from src import losses
from src.networks.clip_model import ClientImageEncoder
from src.networks.clip_model import ClientTextEncoder
from src.networks.language_model import EncoderText
from src.networks.resnet_client import resnet18_client
from src.utils.model_utils import is_embedding_model


torch.backends.cudnn.enabled = True


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
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
        if query_idx not in ignore_list:
            return_retrieval_list.append(1 if query_idx in gt_list else 0)
        count += 1
    return return_retrieval_list


def recall_at_k(feature, query_id, retrieval_list, top_k):
    distance = pairwise_distances(feature, feature)
    result = 0
    for i in range(len(query_id)):
        query_distance = distance[query_id[i], :]
        gt_list = retrieval_list[i][0]
        ignore_list = retrieval_list[i][1]
        query_sorted_idx = np.argsort(query_distance).tolist()
        result_list = get_result_list(query_sorted_idx, gt_list, ignore_list, top_k)
        result += 1.0 if sum(result_list) > 0 else 0
    return result / float(len(query_id))


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    if isinstance(target, list):
        target = torch.tensor(target, dtype=torch.long)
    batch_size = target.size(0)

    n_classes = output.size(1)
    if target.min() < 0 or target.max() >= n_classes:
        raise ValueError(f"Target contains invalid class index. Expected 0-{n_classes-1}, got {target.min()}-{target.max()}")

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred).to(pred.device)
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
                 num_workers=0, print_freq=10, save_step=10, scale=128, pool_type='max_avg', client_id=-1, wandb=None):
        seed_torch()
        self.args = args
        self.client_id = client_id
        self.dset_name = dataset
        self.local_feature = None
        self.classSize = class_size
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
        self.scale = scale
        self.pool_type = pool_type
        self.inter_distance = inter_distance
        if not self.setsys():
            print('system error')
            return

        self.logger = logger
        self.wandb = wandb
        self.loadData()
        self.setModel()
        self.old_model = None
        self.local_epochs = args.local_epochs
        self.local_epoch = 0
        self.cached_domain_labels = {}

    def _get_global_client_key(self):
        if hasattr(self, 'client_idx') and self.client_idx is not None:
            return self.client_idx
        if self.dset_name == 'image':
            return self.client_id
        if self.dset_name == 'text':
            return self.client_id + self.args.num_img_clients
        return self.client_id + self.args.num_img_clients + self.args.num_txt_clients

    def _normalize_sample_ids(self, sample_ids):
        return [str(sample_id) for sample_id in sample_ids]

    def _cached_targets_from_batch(self, data):
        sample_ids = self._normalize_sample_ids(data["id"])
        if any(sample_id not in self.cached_domain_labels for sample_id in sample_ids):
            return None
        targets = [self.cached_domain_labels[sample_id] for sample_id in sample_ids]
        return torch.tensor(targets, dtype=torch.long, device=self.gpuid)

    def _extract_assignment_features(self, data):
        if self.dset_name == 'image' and is_embedding_model(self.args.model):
            images = data["processed_img"].to(self.gpuid)
            return self.model(images)
        if self.dset_name == 'image' and self.args.model == 'resnet':
            images = data["processed_img"].to(self.gpuid)
            self.model.phase = "extract_conv_feature"
            self.model.is_train = False
            features = self.model(images)
            self.model.phase = "None"
            self.model.is_train = True
            return features
        if self.dset_name == 'text' and is_embedding_model(self.args.model):
            captions = data["cap_tokens"].to(self.gpuid)
            return self.model(captions)
        captions = data["cap_tokens"].to(self.gpuid)
        self.model.phase = "extract_conv_feature"
        self.model.is_train = False
        features = self.model(captions).squeeze()
        self.model.phase = "None"
        self.model.is_train = True
        return features

    def run(self, global_centroids):
        self.model.to(self.gpuid)
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval().cuda()
        self.lr_scheduler(self.cur_epoch)

        for _ in range(self.local_epochs):
            self.local_epoch += 1
            self.tra(global_centroids)

        self.test()

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/{self.dset_name}/Client{self.client_id}-model_{self.local_epoch}.pth')

        self.model.cpu()
        self.old_model.cpu()
        del self.old_model
        gc.collect()

    def setsys(self):
        if not torch.cuda.is_available():
            print('No GPU detected')
            return False
        return True

    def loadData(self):
        self.class_label = torch.Tensor(np.array(range(self.classSize)))
        print('output size: {}'.format(self.classSize))

    def setModel(self):
        if self.logger is not None:
            self.logger.log(f'Setting model {self.client_id}')
        if self.dset_name == 'image' and is_embedding_model(self.args.model):
            self.model = ClientImageEncoder(num_class=self.classSize, embed_dim=self.args.feature_dim, mlp_local=self.args.mlp_local, is_train=True,
                                        use_pretrained_proj=bool(self.args.use_pretrained_proj))
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
        elif self.dset_name == 'image' and self.args.model == 'resnet':
            self.model = resnet18_client(
                pretrained=True, num_class=self.classSize, pool_type=self.pool_type,
                is_train=True, scale=self.scale, mlp_local=self.args.mlp_local, embed_dim=self.args.feature_dim,
            )
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
        elif self.dset_name == 'text' and is_embedding_model(self.args.model):
            self.model = ClientTextEncoder(num_class=self.classSize, embed_dim=self.args.feature_dim, mlp_local=self.args.mlp_local, use_pretrained_proj=bool(self.args.use_pretrained_proj))
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
        else:
            self.model = EncoderText(embed_dim=self.args.feature_dim, num_class=self.classSize, scale=self.scale, mlp_local=self.args.mlp_local)
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
        self.center_criterion = nn.MSELoss()
        self.optimizer = optim.SGD(params, lr=self.init_lr, momentum=0.9, weight_decay=0.00005)

    def lr_scheduler(self, epoch):
        if epoch >= 0.5 * self.num_epochs and not self.decay_time[0]:
            self.decay_time[0] = True
            lr = self.init_lr * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        if epoch >= 0.8 * self.num_epochs and not self.decay_time[1]:
            self.decay_time[1] = True
            lr = self.init_lr * self.decay_rate * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def compute_local_cluster_statistics(self, global_centroids):
        self.model.cuda()
        self.model.is_train = False
        centroids = F.normalize(global_centroids.to(self.gpuid, dtype=torch.float32), p=2, dim=1)
        local_sum = torch.zeros(self.args.n_clusters, self.args.feature_dim, dtype=torch.float32)
        local_count = torch.zeros(self.args.n_clusters, dtype=torch.float32)
        self.cached_domain_labels = {}

        self.train_loader.generator.manual_seed(self.cur_epoch)
        for _, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader), disable=True):
            with torch.no_grad():
                features = self._extract_assignment_features(data)
                features = F.normalize(torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0), p=2, dim=1)
                logits = torch.matmul(features, centroids.t())
                domain_labels = torch.argmax(logits, dim=1)

                sample_ids = self._normalize_sample_ids(data["id"])
                for sample_id, label in zip(sample_ids, domain_labels.tolist()):
                    self.cached_domain_labels[sample_id] = int(label)

                cpu_features = features.cpu()
                cpu_labels = domain_labels.cpu()
                local_sum.index_add_(0, cpu_labels, cpu_features)
                local_count += torch.bincount(cpu_labels, minlength=self.args.n_clusters).float()

        self.model.is_train = True
        self.model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        return {'sum': local_sum, 'count': local_count}, self.dset_name

    def tra(self, global_centroids):
        def printnreset(name):
            self.logger.log(
                'Epoch: [{0}] {1}\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\tPrec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    self.local_epoch, name, loss=self.losses, top1=self.top1, top5=self.top5
                )
            )
            self.losses = AverageMeter()
            self.top1 = AverageMeter()
            self.top5 = AverageMeter()

        self.model.train()
        centroids = F.normalize(global_centroids.to(self.gpuid, dtype=torch.float32), p=2, dim=1)

        self.train_loader.generator.manual_seed(self.cur_epoch)
        for _, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if self.dset_name == 'image':
                    inputs_bt = data["processed_img"]
                    labels_bt = data["class_id"]
                    if isinstance(labels_bt, list):
                        labels_bt = torch.tensor(labels_bt, dtype=torch.long)
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    labels_var = torch.autograd.Variable(labels_bt).to(self.gpuid)

                    if self.args.model == 'resnet':
                        fvec, _, class_weight, _ = self.model(inputs_var)
                        self.model.phase = "extract_conv_feature"
                        self.model.is_train = False
                        local_features = self.model(inputs_var)
                        self.model.phase = "None"
                        self.model.is_train = True
                    else:
                        fvec, class_weight, local_features = self.model(inputs_var)
                else:
                    inputs_bt = data["cap_tokens"]
                    labels_bt = data["class_id"]
                    if isinstance(labels_bt, list):
                        labels_bt = torch.tensor(labels_bt, dtype=torch.long)
                    inputs_bt, labels_bt = map(
                        lambda t: torch.cat(t) if not isinstance(t, torch.Tensor) else t,
                        (inputs_bt, labels_bt),
                    )
                    inputs_bt, labels_var = map(lambda t: t.to(self.gpuid).contiguous(), (inputs_bt, labels_bt))

                    if self.args.model == 'resnet':
                        fvec, _, class_weight, _ = self.model(inputs_bt)
                        self.model.phase = "extract_conv_feature"
                        self.model.is_train = False
                        local_features = self.model(inputs_bt).squeeze()
                        self.model.phase = "None"
                        self.model.is_train = True
                    else:
                        fvec, class_weight, local_features = self.model(inputs_bt)

                domain_targets = self._cached_targets_from_batch(data)
                if domain_targets is None:
                    continue

                local_features = F.normalize(torch.nan_to_num(local_features.float(), nan=0.0, posinf=0.0, neginf=0.0), p=2, dim=1)
                domain_logits = torch.matmul(local_features, centroids.t()) / self.args.tau
                loss_cluster = F.cross_entropy(domain_logits, domain_targets)

                loss = self.criterion(fvec, labels_var)
                total_loss = loss + self.args.cluster_weight * loss_cluster
                prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 5))
                self.top1.update(prec1[0], inputs_bt.size(0))
                self.top5.update(prec5[0], inputs_bt.size(0))

                self.losses.update(total_loss.item(), inputs_bt.size(0))
                total_loss.backward()
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 2)
                self.optimizer.step()

            if is_test:
                break

        printnreset(self.dset_name)

    def test(self):
        def printnreset(name):
            self.logger.log('TEST:  Epoch: [{0}] {1}\tPrec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'.format(
                self.local_epoch, name, top1=self.test_top1, top5=self.test_top5
            ))
            self.losses = AverageMeter()
            self.test_top1 = AverageMeter()
            self.test_top5 = AverageMeter()

        self.model.eval()
        self.model.cuda()

        with torch.no_grad():
            for _, data in enumerate(self.test_loader):
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
                else:
                    inputs_bt = data["cap_tokens"]
                    labels_bt = data["class_id"]
                    inputs_bt = inputs_bt.to(self.gpuid)
                    fvec, _, _, _ = self.model(inputs_bt)

                prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 5))
                self.test_top1.update(prec1[0], inputs_bt.size(0))
                self.test_top5.update(prec5[0], inputs_bt.size(0))

        if self.wandb is not None:
            self.wandb.log(
                {
                    f"client_{self.client_id}_val_top1": self.test_top1.avg,
                    f"client_{self.client_id}_val_top5": self.test_top5.avg,
                },
                step=self.local_epoch,
            )

        printnreset(self.dset_name)
        self.model.train()
        return self.losses.avg, self.test_top1.avg, self.test_top5.avg

    def to_half(self):
        _, self.optimizer = amp.initialize(models=[], optimizers=self.optimizer, opt_level='O2')

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError
        return getattr(self.model, k)
