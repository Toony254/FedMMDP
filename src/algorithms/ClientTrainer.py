import copy
import gc
import operator
import torch
import torch.optim as optim
import torch.nn as nn

from apex import amp
from sklearn.metrics import pairwise_distances

from src import losses
from src.networks.clip_model import ClientTextEncoder
from src.networks.clip_model import ClientImageEncoder
from src.utils.Utils import to_one_hot

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
        self.selected_cluster = None
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
                if idx > 10:
                    break
                self.optimizer.zero_grad()
                if self.dset_name == 'image':
                    inputs_bt = data["processed_img"]
                    labels_bt = data["class_id"]
                    if isinstance(labels_bt, list):
                        labels_bt = torch.tensor(labels_bt, dtype=torch.long)
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    labels_var = torch.autograd.Variable(labels_bt).to(self.gpuid)
                    fvec, _, _ = self.model(inputs_var)
                elif self.dset_name == 'text':
                    inputs_bt = data["cap_tokens"]
                    labels_bt = data["class_id"]
                    if isinstance(labels_bt, list):
                        labels_bt = torch.tensor(labels_bt, dtype=torch.long)
                    inputs_bt, labels_bt = map(lambda t: torch.cat(t) if type(t) != torch.Tensor else t,
                                               (inputs_bt, labels_bt))
                    inputs_bt, labels_var = map(lambda t: t.to(self.gpuid).contiguous(), (inputs_bt, labels_bt))
                    fvec, _, _ = self.model(inputs_bt)
                loss = self.criterion(fvec, labels_var)
                # === Proximal term ===
                common_keys = set(dict(self.model.named_parameters()).keys()) & set(global_params.keys())
                prox_loss = 0.0
                for name in common_keys:
                    param = dict(self.model.named_parameters())[name]
                    prox_loss += ((param - global_params[name].to(param.device)) ** 2).sum()
                total_loss = loss + 0.5 * mu * prox_loss
                print(f'Client {self.client_id} - Epoch {self.local_epoch}, Step {idx}, Loss: {total_loss:.4f}, Prox Loss: {prox_loss:.4f}')
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
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.old_model.cuda()

        self.lr_scheduler(self.cur_epoch)

        for i in range(self.local_epochs):
            self.local_epoch += 1
            self.tra()

        self.test()

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/{self.dset_name}/Client{self.client_id}-model_{self.local_epoch}.pth')

        self.model.cpu()
        self.old_model.cpu()

        del self.old_model
        import gc
        gc.collect()
        
    def run_with_moon(self, global_model, prev_models=None, temperature=0.5, mu=1.0):
        self.model.train()
        self.model.cuda()
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
                if self.dset_name == 'image':
                    inputs = data["processed_img"].to(self.gpuid)
                    labels = data["class_id"]
                    if isinstance(labels, list):
                        labels = torch.tensor(labels,dtype=torch.long)
                    labels = labels.to(self.gpuid)
                    fvec, _, _ = self.model(inputs)
                    local_logits = self.model.clip_visual(inputs)
                    with torch.no_grad():
                        fvec_global = global_model.img_enc(inputs)["embedding"]
                        fvec_prev = [m(inputs)[0] for m in prev_models] if prev_models else []
                elif self.dset_name == 'text':
                    inputs = data["cap_tokens"].to(self.gpuid)
                    labels = data["class_id"]
                    if isinstance(labels, list):
                        labels = torch.tensor(labels,dtype=torch.long)
                    labels = labels.to(self.gpuid)
                    fvec, _, _ = self.model(inputs)
                    local_logits = self.model.clip_text(inputs)
                    with torch.no_grad():
                        fvec_global = global_model.txt_enc(inputs)
                        fvec_prev = [m(inputs)[0] for m in prev_models] if prev_models else []
                # 分类损失
                loss_cls = self.criterion(fvec, labels)
                # MOON对比损失
                cos = nn.CosineSimilarity(dim=-1)
                posi = cos(local_logits, fvec_global)
                logits = posi.reshape(-1, 1)
                if prev_models:
                    for fvec_p in fvec_prev:
                        nega = cos(fvec, fvec_p)
                        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                logits /= temperature
                contrastive_labels = torch.zeros(inputs.size(0)).long().to(self.gpuid)
                loss_con = mu * nn.CrossEntropyLoss()(logits, contrastive_labels)
                loss = loss_cls + loss_con
                loss.backward()
                self.optimizer.step()

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
        if self.dset_name == 'image':
            self.model = ClientImageEncoder(num_class=self.classSize, embed_dim=self.args.feature_dim, 
                                        mlp_local=self.args.mlp_local, is_train=True)
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
        elif self.dset_name == 'text':
            self.model = ClientTextEncoder(num_class=self.classSize, embed_dim=self.args.feature_dim,
                                        mlp_local=self.args.mlp_local)
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
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
            if i > 10:
                break
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

                    fvec, class_weight, local_features = self.model(inputs_var)

                elif self.dset_name == 'text':
                    inputs_bt = data["cap_tokens"]
                    labels_bt = data["class_id"]
                    if isinstance(labels_bt, list):
                        labels_bt = torch.tensor(labels_bt,dtype=torch.long)
                    
                    inputs_bt, labels_bt = map(lambda t: torch.cat(t) if type(t) != torch.Tensor else t,
                                               (inputs_bt, labels_bt))
                    inputs_bt, labels_var = map(lambda t: t.to(self.gpuid).contiguous(), (inputs_bt, labels_bt))
                    
                    fvec, class_weight, local_features = self.model(inputs_bt)

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
                if self.dset_name == 'image':
                    inputs_bt = data["processed_img"]
                    labels_bt = data["class_id"]
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    fvec, _, _ = self.model(inputs_var)
                elif self.dset_name == 'text':
                    inputs_bt = data["cap_tokens"]
                    labels_bt = data["class_id"]
                    
                    inputs_bt = inputs_bt.to(self.gpuid)
                    fvec, _, _ = self.model(inputs_bt)

                prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 5))
                self.test_top1.update(prec1[0], inputs_bt.size(0))
                self.test_top5.update(prec5[0], inputs_bt.size(0))

        if self.wandb is not None:
            self.wandb.log({f"client_{self.client_id}_val_top1": self.test_top1.avg,
                            f"client_{self.client_id}_val_top5": self.test_top5.avg}, step=self.local_epoch)

        printnreset(self.dset_name)
        self.model.train()
        
    def predict_logits(self, dataloader):
        """用公共对齐数据输出logits"""
        self.model.cuda()
        self.model.eval()
        logits_list = []
        with torch.no_grad():
            for i, (images, captions, _, _, a_, b_, index) in enumerate(dataloader):
                if self.dset_name == 'image':
                    inputs = images.to(self.gpuid)
                    output = self.model.clip_visual(inputs)
                elif self.dset_name == 'text':
                    inputs = captions.to(self.gpuid)
                    output = self.model.clip_text(inputs)
                logits_list.append(output.cpu().numpy())
        return np.concatenate(logits_list, axis=0)

    def distill_with_logits(self, dataloader, avg_img_logits, avg_txt_logits):
        """用聚合soft label对齐训练"""
        self.model.cuda()
        self.model.train()
        idx = 0
        for i, (images, captions, _, _, a_, b_, index) in enumerate(dataloader):
            if self.dset_name == 'image':
                inputs = images.to(self.gpuid)
                batch_size = inputs.size(0)
                img_soft_label = torch.tensor(avg_img_logits[idx:idx+batch_size]).to(self.gpuid)
                idx += batch_size
                self.optimizer.zero_grad()
                output = self.model.clip_visual(inputs)
                loss = nn.MSELoss()(output, img_soft_label)
                loss.backward()
                self.optimizer.step()
            elif self.dset_name == 'text':
                inputs = captions.to(self.gpuid)
                batch_size = inputs.size(0)
                txt_soft_label = torch.tensor(avg_txt_logits[idx:idx+batch_size]).to(self.gpuid)
                idx += batch_size
                self.optimizer.zero_grad()
                output = self.model.clip_text(inputs)
                loss = nn.MSELoss()(output, txt_soft_label)
                loss.backward()
                self.optimizer.step()

    def train_on_private_data(self):
        """用私有数据常规训练一轮"""
        self.model.cuda()
        self.model.train()
        for i in range(self.local_epochs):
            for idx, data in enumerate(self.train_loader):
                if idx > 10:
                    break
                self.optimizer.zero_grad()
                if self.dset_name == 'image':
                    inputs = data["processed_img"].to(self.gpuid)
                    labels = data["class_id"].to(self.gpuid)
                    output, _, _ = self.model(inputs)
                elif self.dset_name == 'text':
                    inputs = data["cap_tokens"].to(self.gpuid)
                    labels = data["class_id"].to(self.gpuid)
                    output, _, _ = self.model(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
        
    def generate_logits(self, dataloader):
        vec, idx = self.extract_pub_feature(dataloader)
        if self.dset_name == 'image':
            return {'img': vec, 'txt': None}, idx
        elif self.dset_name == 'text':
            return {'img': None, 'txt': vec}, idx
        else:
            assert False
    def extract_pub_feature(self, dataloader):
        self.model.cuda()

        self.model.phase = 'extract_conv_feature'
        self.model.is_train = False
        feature = []
        distill_index = []
        # iterate batch
        for idx, (images, captions, _, _, _, _, index) in enumerate(dataloader):
            with torch.no_grad():
                if self.dset_name == 'image':
                    images = images.to(self.gpuid)
                    im_feature = self.model(images)

                elif self.dset_name == 'text':
                    captions = captions.to(self.gpuid)
                    im_feature = self.model(captions).squeeze()

                im_feature = im_feature.cpu().detach()
                feature.append(im_feature)
                distill_index.extend(index)
                # print(f'im_feature {im_feature.shape} labels {labels_var.shape}')
                # if is_test and idx == 1:
                #     break

        feature = torch.cat(feature, dim=0)
        # print(f'feature {feature.shape} labels {labels.shape}')
        self.model.phase = 'None'
        self.model.is_train = True

        self.model.cpu()
        return feature, distill_index
    def to_half(self):
        # Mixed precision
        # https://nvidia.github.io/apex/amp.html
        _, self.optimizer = amp.initialize(models=[], optimizers=self.optimizer,
                                                    opt_level='O2')

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError
