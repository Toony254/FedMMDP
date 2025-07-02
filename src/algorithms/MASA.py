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
from src.datasets.load_FL_datasets import get_FL_trainloader, get_class_size
from src.algorithms.ClientTrainer import ClientTrainer
from src.algorithms.MMClientTrainer import MMClientTrainer
from src.algorithms.ssm_algorithm import split_model_parameters, select_relay_nodes

from src.algorithms.eval_coco import COCOEvaluator
from src.algorithms.retrieval_trainer import TrainerEngine
from src.algorithms.mm_eval import MMEvaluator
from src.utils.config import parse_config
from src.utils.load_datasets import prepare_coco_dataloaders
from src.utils.logger import PythonLogger

try:
    from apex import amp
except ImportError:
    print('failed to import apex')


is_test = False

class DummyRelayNode:
    def __init__(self):
        self.results = {}
    def receive(self, cluster_label, component):
        if cluster_label not in self.results:
            self.results[cluster_label] = []
        self.results[cluster_label].append(component)
    def aggregate(self):
        return {k: np.sum(v, axis=0) for k, v in self.results.items()}

class DummyServer:
    def __init__(self, num_clusters):
        self.intermediate = {c: [] for c in range(num_clusters)}
    def receive_intermediate_result(self, relay_id, cluster_label, agg_result):
        self.intermediate[cluster_label].append(agg_result)
    def aggregate_final_models(self, num_clusters):
        # 对每个聚类聚合所有中继节点的结果
        return {c: np.mean(self.intermediate[c], axis=0) for c in range(num_clusters) if self.intermediate[c]}

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

def get_flat_params_from_state_dict(state_dict):
    params = []
    for v in state_dict.values():
        params.append(v.cpu().numpy().reshape(-1))
    return np.concatenate(params, axis=0)

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

def get_modalities(client):
    # 返回客户端拥有的模态列表
    if hasattr(client, 'dset_name') and client.dset_name == 'mm':
        return ['img', 'txt']
    return [client.dset_name]

def get_model_parameters(client, modality):
    # 获取指定模态的参数向量
    if modality == 'img':
        return get_flat_params_from_state_dict(client.model.img_enc.state_dict())
    elif modality == 'txt':
        return get_flat_params_from_state_dict(client.model.txt_enc.state_dict())
    else:
        return get_flat_params_from_state_dict(client.model.state_dict())

def set_model_parameters(client, modality, flat_params):
    if modality == 'img':
        set_state_dict_from_flat(client.model.img_enc, flat_params)
    elif modality == 'txt':
        set_state_dict_from_flat(client.model.txt_enc, flat_params)
    else:
        set_state_dict_from_flat(client.model, flat_params)

class MMFL(object):
    def __init__(self, args, wandb=None):
        self.args = args
        self.wandb = wandb

        self.device = None
        self.img_local_trainers = None
        self.txt_local_trainers = None
        self.mm_local_trainers = None
        # self.engine = None
        self.class_size = get_class_size('/home/bd/data/zs/FedMMDP/data/processed_datasets/domain_dataset_0') * 5
        self.engine = None
        self.best_score = 0
        self.cur_epoch = 0

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
        self.config = parse_config("./src/imageNet_cap.yaml", strict_cast=False)
        self.config.train.model_save_path = 'model_last_no_prob'
        self.config.train.best_model_save_path = 'model_best_no_prob'
        self.config.train.output_file = 'model_noprob'
        self.config.model.img_client = img
        self.config.model.txt_client = txt
        self.config.train.model_save_path = self.config.train.model_save_path + '.pth'
        self.config.train.best_model_save_path = self.config.train.best_model_save_path + '.pth'
        self.config.train.output_file = self.config.train.output_file + '.log'

        self.config.model.embed_dim = self.args.feature_dim  # set global model dim
    
    def load_dataset(self, args):
        dataset_root = '/home/bd/data/zs' + '/data/mmdata/MSCOCO/2014'
        vocab_path = './src/datasets/vocabs/coco_vocab.pkl'
        self.dataloaders_global, self.vocab = prepare_coco_dataloaders(self.config.dataloader, dataset_root, vocab_path)

        self.engine = TrainerEngine()
        self.engine.set_logger(self.logger)

        self.config.optimizer.learning_rate = self.args.server_lr

        self._dataloaders = self.dataloaders_global.copy()
        self.evaluator = MMEvaluator(eval_method='matmul',
                                       verbose=False,
                                       eval_device='cuda',
                                       n_crossfolds=5, 
                                       class_size=self.class_size)
        self.engine.create(self.config, self.evaluator, self.args.mlp_local)

        self.train_eval_dataloader = self._dataloaders.pop(
            'train_subset_eval' + f'_{self.args.pub_data_num}') if self._dataloaders is not None else None

        self.engine.model_to_device()
        torch.backends.cudnn.enabled = True
        if self.config.train.get('use_fp16'):
            self.engine.logger.log('Train with half precision')
            self.engine.to_half()
            
        self.val_dataloader = {}
        for i in range(args.num_img_clients):
            val_dataset = load_from_disk(f'/home/bd/data/zs/FedMMDP/data/processed_datasets/domain_dataset_{i}_test')
            self.val_dataloader[i] = torch.utils.data.DataLoader(val_dataset, 
                                                            batch_size=self.args.batch_size, 
                                                            shuffle=False, 
                                                            num_workers=4,
                                                            collate_fn=collate_fn
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
            self.img_trainloaders, test_loaders = get_FL_trainloader(dataset, '/home/bd/data/zs/FedMMDP/data/processed_datasets/',
                                                                 args.num_img_clients, "hetero", 0.1, self.args.batch_size)
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
            self.txt_trainloaders, test_loaders = get_FL_trainloader(dataset, '/home/bd/data/zs/FedMMDP/data/processed_datasets/',
                                                                 args.num_txt_clients, "hetero", 0.1, self.args.batch_size)
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
            config = parse_config("./src/imageNet_cap.yaml", strict_cast=False)
            config.model.cache_dir = config.model.cache_dir + '-' + config.train.server_dataset
            config.train.output_file = os.path.join(config.model.cache_dir, config.train.output_file)
            config.train.best_model_save_path = os.path.join(config.model.cache_dir, config.train.best_model_save_path)
            config.train.model_save_path = os.path.join(config.model.cache_dir, config.train.model_save_path)
            config.model.embed_dim = self.args.feature_dim
            config.model.name = 'clip'
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
            
    def aggregate_with_ssm_ascc(self, trainers):
        # 1. SSM分割与匿名分发
        num_relays = self.args.num_relays
        num_clusters = self.args.num_clusters
        max_components = self.args.max_ssm_components
        relay_nodes = [DummyRelayNode() for _ in range(num_relays)]
        for client in trainers:
            cluster_label = get_cluster_label(client)
            for modality in get_modalities(client):
                params = get_model_parameters(client, modality)
                num_components = np.random.randint(1, max_components + 1)
                components = split_model_parameters(params, num_components)
                selected_relays = select_relay_nodes(num_relays, num_components)
                for comp, relay_id in zip(components, selected_relays):
                    relay_nodes[relay_id].receive(cluster_label, comp)
        # 2. 中继节点聚合并上传服务器
        server = DummyServer(num_clusters)
        for relay_id, relay in enumerate(relay_nodes):
            agg = relay.aggregate()
            for cluster_label, agg_result in agg.items():
                server.receive_intermediate_result(relay_id, cluster_label, agg_result)
        # 3. 服务器端最终聚合
        final_models = server.aggregate_final_models(num_clusters)
        # 4. ASCC分发：每个客户端选择最优聚类模型
        cluster_models = [final_models[c] for c in sorted(final_models.keys())]
        for client in trainers:
            best_idx = ascc_select_cluster(client, cluster_models)
            client.selected_cluster = best_idx
            # 只分发对应模态参数
            for modality in get_modalities(client):
                set_model_parameters(client, modality, cluster_models[best_idx])
            
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
            
        self.aggregate_with_ssm_ascc(self, self.cur_trainers)

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        
        # test in own domain
        
        print("Testing...")
        for domain_idx in range(self.args.num_domains):
            print(f"Server tests in domain {domain_idx}:")
            test_scores = self.engine.evaluate({'test': self.val_dataloader[domain_idx]})
            metadata = self.engine.metadata.copy()
            metadata['cur_epoch'] = round_n + 1
            metadata['lr'] = get_lr(self.engine.optimizer)
            
            self.engine.report_scores(step=round_n + 1,
                                    scores=test_scores,
                                    metadata=metadata)
            rsum = test_scores['test']['n_fold']['i2t']['recall_1'] + test_scores['test']['n_fold']['t2i']['recall_1'] + \
                test_scores['test']['i2t']['recall_1'] + test_scores['test']['t2i']['recall_1']
            self.wandb.log({f"Multimodal rsum_r1": rsum}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal i2t_r1": test_scores['test']['i2t']['recall_1']}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal t2i_r1": test_scores['test']['t2i']['recall_1']}, step=self.cur_epoch)

            if self.best_score > rsum:
                best_score = rsum
                metadata['best_score'] = best_score
                metadata['best_epoch'] = round_n + 1
                self.best_metadata, self.best_score = metadata, best_score
                print(f"Best score updated: {best_score} at epoch {round_n + 1}")
                # torch.save({'net': self.engine.model.state_dict()}, self.args.name + '-best_model.pt')

            if round_n == self.args.comm_rounds - 1:
                print(f"Final best score: {self.best_score} at epoch {self.best_metadata['best_epoch']}")
                # torch.save({'net': self.engine.model.state_dict()}, self.args.name + '-last_model.pt')
            
        self.engine.lr_scheduler.step()
        gc.collect()
