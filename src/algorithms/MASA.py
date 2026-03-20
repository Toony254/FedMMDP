from email.policy import strict
import gc
import random

import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk

import operator
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import copy

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.datasets.transform import collate_fn
from src.datasets.load_FL_datasets import get_FL_trainloader
from src.algorithms.ClientTrainer import ClientTrainer
from src.algorithms.MMClientTrainer import MMClientTrainer

from src.algorithms.eval_coco import COCOEvaluator
from src.algorithms.retrieval_trainer import TrainerEngine
from src.algorithms.mm_eval import MMEvaluator
from src.utils.config import parse_config, apply_runtime_overrides
from src.utils.load_datasets import prepare_coco_dataloaders
from src.utils.logger import PythonLogger
from src.utils.experiment_naming import projector_tag
from src.utils.model_utils import is_embedding_model

try:
    from apex import amp
except ImportError:
    print('failed to import apex')


is_test = False

class DummyRelayNode:
    def __init__(self):
        self.results = {}  # {cluster_label: [dict, dict, ...]}
    def receive(self, cluster_label, component):
        if cluster_label not in self.results:
            self.results[cluster_label] = []
        self.results[cluster_label].append(component)
    def aggregate(self):
        agg_results = {}
        from collections import Counter
        key_counts = Counter(self.results.keys())
        for cluster_label, dict_list in self.results.items():
            if key_counts[cluster_label] == 1:
                agg_results[cluster_label] = {**dict_list[0], **dict_list[1]}
            else:
                if cluster_label not in agg_results:
                    agg_results[cluster_label] = dict_list
                else:
                    agg_results[cluster_label] = agg_results[cluster_label] + dict_list
        for cluster_label, dict_list in agg_results.items():
            if key_counts[cluster_label] != 1:
                keys = set()
                for d in dict_list:
                    keys.update(d.keys())
                avg_dict = {}
                for k in keys:
                    values = [d[k] for d in dict_list]
                    avg = np.mean(values, axis=0)
                    avg_dict[k] = avg
                agg_results[cluster_label] = avg_dict
        return agg_results

class DummyServer:
    def __init__(self, num_clusters):
        self.intermediate = {}  # 支持任意key
    def receive_intermediate_result(self, relay_id, cluster_label, agg_result):
        if cluster_label not in self.intermediate:
            self.intermediate[cluster_label] = []
        self.intermediate[cluster_label].append(agg_result)
    def aggregate_final_models(self, num_clusters):
        # 对每个key分别聚合
        final_models = {}
        for cluster_label, dict_list in self.intermediate.items():
            keys = dict_list[0].keys()
            agg_dict = {}
            for k in keys:
                arrs = [d[k] for d in dict_list]
                agg_dict[k] = np.mean(arrs, axis=0)
            final_models[cluster_label] = agg_dict
        return final_models
def ascc_select_cluster(client, cluster_models):
    """
    基于注意力的自发客户端聚类（简化版）：
    1. 对每个聚类模型，计算其在本地标记数据上的平均损失
    2. 选择损失最小的聚类模型
    """
    # 获取本地标记数据（假设client有train_loader或labeled_dataset）
    if hasattr(client, "train_loader"):
        data_loader = client.train_loader
    else:
        # 若无标记数据，退化为参数最近邻
        local_param = client.get_flat_params()
        best_idx = np.argmax([np.dot(local_param, m) for m in cluster_models])
        return best_idx

    device = next(client.model.parameters()).device
    losses = []
    for idx, flat_param in enumerate(cluster_models):
        # 将聚类模型参数加载到临时模型
        tmp_model = copy.deepcopy(client.model)
        set_state_dict_from_flat(tmp_model, flat_param)
        tmp_model.eval()
        total_loss, total_num = 0.0, 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in data_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                # 兼容单模态/多模态
                if hasattr(tmp_model, "img_enc"):
                    out = tmp_model.img_enc(x)
                elif hasattr(tmp_model, "txt_enc"):
                    out = tmp_model.txt_enc(x)
                else:
                    out = tmp_model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * x.size(0)
                total_num += x.size(0)
        avg_loss = total_loss / (total_num + 1e-8)
        losses.append(avg_loss)
    best_idx = int(np.argmin(losses))
    return best_idx

def set_state_dict_from_flat(model, flat_params):
    state_dict = model.state_dict()
    shapes = [v.shape for v in state_dict.values()]
    sizes = [np.prod(s) for s in shapes]
    splits = np.split(flat_params, np.cumsum(sizes)[:-1])
    new_state = {}
    for k, arr, shape in zip(state_dict.keys(), splits, shapes):
        new_state[k] = torch.tensor(arr.reshape(shape), dtype=state_dict[k].dtype)
    model.load_state_dict(new_state)

def get_cluster_label(client):
    if hasattr(client, 'selected_cluster'):
        return client.selected_cluster
    return getattr(client, 'cluster_label', client.client_idx % client.args.num_clusters)

def split_model_parameters(state_dict, num_relays):
    # state_dict: dict of {key: tensor}
    # 返回num_relays个dict，每个dict结构与state_dict一致，值为分量
    keys = list(state_dict.keys())
    components = [{k: None for k in keys} for _ in range(num_relays)]
    for k in keys:
        param = state_dict[k].cpu().numpy()
        shape = param.shape
        splits = []
        for _ in range(num_relays - 1):
            if shape == ():  # 标量
                comp = float(np.random.randn())
            else:
                comp = np.random.randn(*shape).astype(param.dtype)
            splits.append(comp)
        last = param - sum(splits)
        splits.append(last)
        for i in range(num_relays):
            components[i][k] = splits[i]
    return components  # list of dict
def get_modalities(client):
    if client.dset_name == 'mm':
        return ['img', 'txt']
    if client.dset_name == 'image':
        return ['img']
    return ['txt']


def get_default_cluster_label(client, num_clusters):
    client_key = getattr(client, 'client_idx', getattr(client, 'client_id', getattr(client, 'client', 0)))
    return int(client_key) % max(int(num_clusters), 1)


def get_client_cluster_label(client, modality, num_clusters):
    default_label = get_default_cluster_label(client, num_clusters)
    if client.dset_name == 'mm':
        selected_clusters = getattr(client, 'selected_clusters', None)
        if isinstance(selected_clusters, dict):
            return int(selected_clusters.get(modality, default_label))
    return int(getattr(client, 'selected_cluster', default_label))


def get_client_encoder_state(client, modality):
    if hasattr(client, 'get_masa_encoder_state'):
        try:
            return client.get_masa_encoder_state(modality)
        except TypeError:
            return client.get_masa_encoder_state()
    return get_model_parameters(client, modality)


def add_state_dict_(accumulator, state_dict):
    for key, value in state_dict.items():
        tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if key not in accumulator:
            accumulator[key] = tensor.clone()
        else:
            accumulator[key] += tensor


def split_state_dict_for_relays(state_dict, num_relays):
    shares = [dict() for _ in range(num_relays)]
    for key, value in state_dict.items():
        tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        running = torch.zeros_like(tensor)
        for relay_idx in range(num_relays - 1):
            random_share = torch.randn_like(tensor)
            shares[relay_idx][key] = random_share
            running += random_share
        shares[-1][key] = tensor - running
    return shares


def average_state_dict(state_dict, divisor):
    divisor = max(int(divisor), 1)
    return {key: value / divisor for key, value in state_dict.items()}

def get_model_parameters(client, modality):
    # 返回 state_dict
    if hasattr(client.model, 'img_enc') and modality == 'img':
        return client.model.img_enc.state_dict()
    elif hasattr(client.model, 'txt_enc') and modality == 'txt':
        return client.model.txt_enc.state_dict()
    elif modality == 'img':
        return client.model.visual_projector.state_dict()
    elif modality == 'txt':
        return client.model.text_projector.state_dict()
    else:
        return client.model.state_dict()

def set_model_parameters(client, modality, param_dict):
    # param_dict: {key: np.ndarray}
    if hasattr(client.model, 'img_enc') and modality == 'img':
        model = client.model.img_enc
    elif hasattr(client.model, 'txt_enc') and modality == 'txt':
        model = client.model.txt_enc
    elif modality == 'img':
        model = client.model.visual_projector
    elif modality == 'txt':
        model = client.model.text_projector
    else:
        model = client.model
    state_dict = model.state_dict()
    new_state = {}
    for k in state_dict.keys():
        arr = param_dict[k]
        new_state[k] = torch.tensor(arr, dtype=state_dict[k].dtype)
    model.load_state_dict(new_state)

class MMFL(object):
    def __init__(self, args, wandb=None):
        self.args = args
        self.wandb = wandb

        self.device = None
        self.img_local_trainers = None
        self.txt_local_trainers = None
        self.mm_local_trainers = None
        # self.engine = None
        if self.args.dataset == 'imagenet':
            self.class_size = 50
        elif self.args.dataset == 'fashion':
            self.class_size = 48
        elif self.args.dataset == 'food':
            self.class_size = 101
        elif self.args.dataset == 'iapr':
            self.class_size = 30  # 5 domains x 6 classes
        self.engine = None
        self.best_score = 0
        self.cur_epoch = 0
        self.best_metadata = None

        # img & txt local dataloaders
        self.img_train_loaders, self.txt_train_loaders = None, None

        # coco global dataloaders
        self.dataloaders_global = None
        # universal test dataloader
        self.test_loader = None

        self.config = None
        self.set_config()

        self.logger = PythonLogger(output_file=self.config.train.output_file)
        self.img_vec, self.txt_vec = None, None
        self.global_img_feature = None
        self.global_txt_feature = None
        self.distill_index = None


    def set_config(self, img='image', txt='text'):
        if self.args.dataset == 'imagenet':
            yaml_name = 'imageNet_cap.yaml'
        elif self.args.dataset == 'fashion':
            yaml_name = 'fashion_gen.yaml'
        elif self.args.dataset == 'food':
            yaml_name = 'umpc_food.yaml'
        elif self.args.dataset == 'iapr':
            yaml_name = 'iapr.yaml'
        self.config = parse_config("./src/" + yaml_name, strict_cast=False)
        self.config = apply_runtime_overrides(self.args, self.config)
        self.config.train.model_save_path = 'model_last_no_prob'
        self.config.train.best_model_save_path = 'model_best_no_prob'
        self.config.train.output_file = f'{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_{projector_tag(self.args)}_model_noprob'
        self.config.model.name = self.args.model
        self.config.model.img_client = img
        self.config.model.txt_client = txt
        self.config.train.model_save_path = self.config.train.model_save_path + '.pth'
        self.config.train.best_model_save_path = self.config.train.best_model_save_path + '.pth'
        self.config.train.output_file = self.config.train.output_file + '.log'

        self.config.model.embed_dim = self.args.feature_dim  # set global model dim
    
    def load_dataset(self, args):
        self.engine = TrainerEngine()
        self.engine.set_logger(self.logger)

        self.config.optimizer.learning_rate = self.args.server_lr
        
        self.evaluator = MMEvaluator(model_name=self.args.model,
                                       dataset=self.args.dataset,
                                       eval_method='matmul',
                                       verbose=False,
                                       eval_device='cuda',
                                       n_crossfolds=1, 
                                       class_size=self.class_size,
                                        feature_dim=self.args.feature_dim,
                                        data_root=self.args.data_root)
        self.engine.create(self.config, self.evaluator, self.args.mlp_local)

        self.engine.model_to_device()
        torch.backends.cudnn.enabled = True
        if self.config.train.get('use_fp16'):
            self.engine.logger.log('Train with half precision')
            self.engine.to_half()
            
        self.val_dataloader = {}
        for i in range(args.num_img_clients):
            val_dataset = load_from_disk(os.path.join(self.args.data_root, f'domain_dataset_{i}', 'test'))
            self.val_dataloader[i] = torch.utils.data.DataLoader(val_dataset, 
                                                            batch_size=self.args.batch_size, 
                                                            shuffle=False, 
                                                            num_workers=0,
                                                            collate_fn=collate_fn,
                                                            # timeout=300
                                                            )
    def create_model(self, args):
        self.logger.log('start creating model and partition datasets')
        self.device = torch.device("cuda:%d" % args.device)

        os.makedirs('/home/bd/data/zs' + f'/data/yClient', exist_ok=True)

        # Create Client Models
        self.img_local_trainers, self.txt_local_trainers, self.mm_local_trainers = [], [], []
        # img clients
        if args.num_img_clients > 0:
            dataset = 'image'
            self.img_trainloaders, test_loaders = get_FL_trainloader(dataset, self.args.data_root,
                                                                 args.num_img_clients, self.args.partition, self.args.alpha, self.args.batch_size)
            self.img_local_trainers = []
            for i in range(args.num_img_clients):
                self.img_local_trainers.append(
                    ClientTrainer(args, dataset, self.class_size, self.logger,
                                  inter_distance=4, client_id=i, wandb=self.wandb))
                self.img_local_trainers[i].train_loader = self.img_trainloaders[i]
                self.img_local_trainers[i].test_loader = test_loaders[i]
                if is_test and i == 0:
                    break
        # txt clients
        if args.num_txt_clients > 0:
            dataset = 'text'
            self.txt_trainloaders, test_loaders = get_FL_trainloader(dataset, self.args.data_root,
                                                                 args.num_txt_clients, self.args.partition, self.args.alpha, self.args.batch_size)
            self.txt_local_trainers = []
            for i in range(args.num_txt_clients):
                self.txt_local_trainers.append(
                    ClientTrainer(args, dataset, self.class_size, self.logger,
                                  inter_distance=4, client_id=i, wandb=self.wandb))
                self.txt_local_trainers[i].train_loader = self.txt_trainloaders[i]
                self.txt_local_trainers[i].test_loader = test_loaders[i]
                if is_test and i == 0:
                    break
        # mm clients
        if args.num_mm_clients > 0:
            # mm img models
            if self.args.dataset == 'imagenet':
                yaml_name = 'imageNet_cap.yaml'
            elif self.args.dataset == 'fashion':
                yaml_name = 'fashion_gen.yaml'
            elif self.args.dataset == 'food':
                yaml_name = 'umpc_food.yaml'
            elif self.args.dataset == 'iapr':
                yaml_name = 'iapr.yaml'
            config = parse_config("./src/" + yaml_name, strict_cast=False)
            config.model.cache_dir = config.model.cache_dir + '-' + config.train.server_dataset
            config.train.output_file = os.path.join(config.model.cache_dir, config.train.output_file)
            config.train.best_model_save_path = os.path.join(config.model.cache_dir, config.train.best_model_save_path)
            config.train.model_save_path = os.path.join(config.model.cache_dir, config.train.model_save_path)
            config.model.embed_dim = self.args.feature_dim
            config.model.name = self.args.model
            self.mm_local_trainers = []
            for client_id in range(args.num_mm_clients):
                self.mm_local_trainers.append(
                    MMClientTrainer(args, config, self.class_size, self.logger, client=client_id, dset_name="mm",
                                    device='cuda', mlp_local=self.args.mlp_local, wandb=self.wandb))
                if is_test and client_id == 0:
                    break
            print(f"Samples Num: {[len(i.train_loader.dataset) for i in self.mm_local_trainers]}")

        self.total_local_trainers = self.img_local_trainers + self.txt_local_trainers + self.mm_local_trainers

        for i in range(len(self.total_local_trainers)):
            self.total_local_trainers[i].client_idx = i
            default_cluster = i % max(int(getattr(self.args, 'num_clusters', 1)), 1)
            self.total_local_trainers[i].selected_cluster = default_cluster
            if self.total_local_trainers[i].dset_name == 'mm':
                self.total_local_trainers[i].selected_clusters = {'img': default_cluster, 'txt': default_cluster}
            
    def aggregate_with_ssm_ascc(self, trainers):
        from collections import defaultdict

        num_relays = max(int(getattr(self.args, 'num_relays', 1)), 1)
        num_clusters = max(int(getattr(self.args, 'num_clusters', 1)), 1)
        relay_buffers = [defaultdict(dict) for _ in range(num_relays)]
        cluster_counts = defaultdict(int)

        for client in trainers:
            for modality in get_modalities(client):
                state_dict = get_client_encoder_state(client, modality)
                if not state_dict:
                    continue
                cluster_label = get_client_cluster_label(client, modality, num_clusters)
                cluster_key = (modality, cluster_label)
                shares = split_state_dict_for_relays(state_dict, num_relays)
                relay_order = random.sample(range(num_relays), num_relays) if num_relays > 1 else [0]
                for share, relay_id in zip(shares, relay_order):
                    add_state_dict_(relay_buffers[relay_id][cluster_key], share)
                cluster_counts[cluster_key] += 1

        modality_cluster_models = {}
        for relay_buffer in relay_buffers:
            for (modality, cluster_label), share_sum in relay_buffer.items():
                modality_cluster_models.setdefault(modality, {}).setdefault(cluster_label, {})
                add_state_dict_(modality_cluster_models[modality][cluster_label], share_sum)

        for modality, cluster_map in modality_cluster_models.items():
            for cluster_label, total_state in list(cluster_map.items()):
                cluster_map[cluster_label] = average_state_dict(total_state, cluster_counts[(modality, cluster_label)])

        if self.logger is not None:
            summary = {f'{modality}_{cluster}': cluster_counts[(modality, cluster)] for modality, clusters in modality_cluster_models.items() for cluster in clusters.keys()}
            self.logger.log(f'MASA secure aggregation cluster counts: {summary}')

        ft_epochs = int(getattr(self.args, 'ascc_ft_epoch', 1))
        for client in trainers:
            for modality in get_modalities(client):
                cluster_map = modality_cluster_models.get(modality, {})
                if not cluster_map:
                    continue
                cluster_labels = sorted(cluster_map.keys())
                cluster_states = [cluster_map[label] for label in cluster_labels]
                if client.dset_name == 'mm':
                    client.masa_personalize_from_clusters(modality, cluster_states, cluster_labels)
                else:
                    client.masa_personalize_from_clusters(cluster_states, cluster_labels)
            if hasattr(client, 'masa_finetune_local'):
                client.masa_finetune_local(ft_epochs=ft_epochs)

    def train(self, round_n):
        self.cur_epoch = round_n
        self.cur_trainers = self.total_local_trainers

        if not is_test:
            self.logger.log(f"Round {round_n + 1}!")
            if len(self.total_local_trainers) != 0:
                self.cur_trainers = random.sample(self.total_local_trainers, self.args.client_num_per_round)

        # local training
        for idx, trainer in enumerate(self.cur_trainers):
            self.logger.log(f"Training Client {trainer.client_idx}!")
            trainer.cur_epoch = round_n
            trainer.run()
            
        self.aggregate_with_ssm_ascc(self.cur_trainers)

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        
        # test in own domain
        import csv

        if not hasattr(self, 'img_txt_results'):
            self.img_txt_results = []
        if not hasattr(self, 'mm_results'):
            self.mm_results = []
        if not hasattr(self, 'rsum_history'):
            self.rsum_history = []

        img_txt_rows = []
        mm_rows = []
        
        print("Testing...")
        rsum = 0
        for idx, trainer in enumerate(self.total_local_trainers):
            if trainer.dset_name == "image":
                domain_idx = idx
                trainer.test_loader = self.val_dataloader[idx]
                print(f"Client {trainer.dset_name} {idx} tests in domain {idx}:")
                losses, test_top1, test_top5 = trainer.test()
                img_txt_rows.append([
                    round_n, trainer.client_idx, domain_idx,
                    losses, test_top1, test_top5
                ])
            elif trainer.dset_name == "text":
                domain_idx = idx - self.args.num_img_clients
                trainer.test_loader = self.val_dataloader[domain_idx]
                print(f"Client {trainer.dset_name} {idx} tests in domain {domain_idx}:")
                losses, test_top1, test_top5 = trainer.test()
                img_txt_rows.append([
                    round_n, trainer.client_idx, domain_idx,
                    losses, test_top1, test_top5
                ])
            else:
                # for domain_idx in range(self.args.num_domains):
                domain_idx = idx - self.args.num_img_clients - self.args.num_txt_clients
                print(f"Client {trainer.dset_name} {idx} tests in domain {domain_idx}:")
                test_scores = trainer.evaluate({'test': self.val_dataloader[domain_idx]})
                metadata = trainer.metadata.copy()
                metadata['cur_epoch'] = round_n + 1
                metadata['lr'] = get_lr(trainer.optimizer)
                trainer.report_scores(step=round_n + 1,
                                        scores=test_scores,
                                        metadata=metadata)
                rsum_i = test_scores['test']['i2t']['recall_1'] + test_scores['test']['t2i']['recall_1'] + \
                    test_scores['test']['i2t']['recall_5'] + test_scores['test']['t2i']['recall_5']
                if domain_idx == idx - self.args.num_img_clients - self.args.num_txt_clients:
                    rsum += rsum_i
                mm_rows.append([
                    round_n, trainer.client_idx, domain_idx, rsum_i,
                    test_scores['test']['n_fold']['i2t']['recall_1'],
                    test_scores['test']['n_fold']['t2i']['recall_1'],
                    test_scores['test']['i2t']['recall_1'],
                    test_scores['test']['t2i']['recall_1'],
                    test_scores['test']['n_fold']['i2t']['recall_5'],
                    test_scores['test']['n_fold']['t2i']['recall_5'],
                    test_scores['test']['i2t']['recall_5'],
                    test_scores['test']['t2i']['recall_5'],
                ])
                mm_client_idx = idx - self.args.num_img_clients - self.args.num_txt_clients
                if self.wandb is not None:
                    self.wandb.log({f"Multimodal_{mm_client_idx} rsum_r1": rsum_i}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{mm_client_idx} n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{mm_client_idx} n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{mm_client_idx} i2t_r1": test_scores['test']['i2t']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{mm_client_idx} t2i_r1": test_scores['test']['t2i']['recall_1']}, step=self.cur_epoch)
        
        self.rsum_history.append(rsum)
        if self.best_metadata is None or self.best_score < rsum:
            best_score = rsum
            metadata['best_score'] = best_score
            metadata['best_epoch'] = round_n + 1
            self.best_metadata, self.best_score = metadata, best_score
            print(f"Best score updated: {best_score} at epoch {round_n + 1}")
            # torch.save({'net': trainer.model.state_dict()}, self.args.name + '-best_model.pt')

        if self.best_metadata is None:
            self.best_metadata = {'best_epoch': round_n + 1}
        if round_n == self.args.comm_rounds - 1:
            print(f"Final best score: {self.best_score} at epoch {self.best_metadata['best_epoch']}")
        #     torch.save({'net': trainer.model.state_dict()}, self.args.name + '-last_model.pt')
        
        os.makedirs('results', exist_ok=True)
        mm_csv = f'results/{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_mm_{self.args.FL_algorithm}_{projector_tag(self.args)}.csv'
        if mm_rows:
            write_header = not os.path.exists(mm_csv)
            with open(mm_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'round', 'client_id', 'domain_idx', 'rsum_i',
                        'n_fold_i2t_r1', 'n_fold_t2i_r1', 'i2t_r1', 't2i_r1',
                        'n_fold_i2t_r5', 'n_fold_t2i_r5', 'i2t_r5', 't2i_r5'
                    ])
                writer.writerows(mm_rows)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, len(self.rsum_history)+1), self.rsum_history, marker='o')
        plt.xlabel('Round')
        plt.ylabel('rsum')
        best_epoch = self.best_metadata.get("best_epoch", round_n + 1) if self.best_metadata else round_n + 1
        plt.title(f'rsum Curve (Best: {self.best_score} at epoch {best_epoch})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/rsum_{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_{projector_tag(self.args)}.png')
        plt.close()
        print("Rsum at round {} is {}".format(round_n, self.rsum_history[-1]))
        gc.collect()
