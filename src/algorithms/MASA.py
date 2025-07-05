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

def split_model_parameters(params, num_relays):
    # params: numpy array
    # 返回num_relays个 shape一致的分量，满足 sum(components) == params
    shape = params.shape
    components = []
    for _ in range(num_relays - 1):
        comp = np.random.randn(*shape).astype(params.dtype)
        components.append(comp)
    last = params - sum(components)
    components.append(last)
    # 确保所有分量 shape 一致
    assert all(isinstance(c, np.ndarray) and c.shape == shape for c in components)
    return components
def get_modalities(client):
    # 返回客户端拥有的模态列表
    if client.dset_name == 'mm':
        return ['img', 'txt']
    elif client.dset_name == 'image':
        return ['img']
    else:
        return ['txt']

def get_model_parameters(client, modality):
    # 获取指定模态的参数向量
    if hasattr(client.model, 'img_enc') and modality == 'img':
        return get_flat_params_from_state_dict(client.model.img_enc.state_dict())
    elif hasattr(client.model, 'txt_enc') and modality == 'txt':
        return get_flat_params_from_state_dict(client.model.txt_enc.state_dict())
    else:
        return get_flat_params_from_state_dict(client.model.state_dict())

def set_model_parameters(client, modality, flat_params):
    if hasattr(client.model, 'img_enc') and modality == 'img':
        set_state_dict_from_flat(client.model.img_enc, flat_params)
    elif hasattr(client.model, 'txt_enc') and modality == 'txt':
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
        import torch.nn.functional as F

        num_relays = self.args.num_relays
        num_clusters = self.args.num_clusters
        relay_nodes = [DummyRelayNode() for _ in range(num_relays)]

        # 1. SSM分割与匿名分发
        for client in trainers:
            cluster_label = get_cluster_label(client)
            for modality in get_modalities(client):
                params = get_model_parameters(client, modality)
                components = split_model_parameters(params, num_relays)
                relay_ids = random.sample(range(num_relays), num_relays)
                for comp, relay_id in zip(components, relay_ids):
                    relay_nodes[relay_id].receive(cluster_label, comp)

        # 2. 中继节点聚合并上传服务器
        server = DummyServer(num_clusters)
        for relay_id, relay in enumerate(relay_nodes):
            agg = relay.aggregate()
            for cluster_label, agg_result in agg.items():
                server.receive_intermediate_result(relay_id, cluster_label, agg_result)

        # 3. 服务器端最终聚合
        final_models = server.aggregate_final_models(num_clusters)
        cluster_models = [final_models[c] for c in sorted(final_models.keys())]

        # 4. ASCC分层注意力个性化与本地微调
        def split_layers(flat_params, layer_shapes):
            splits = np.split(flat_params, np.cumsum([np.prod(s) for s in layer_shapes])[:-1])
            return [s.reshape(shape) for s, shape in zip(splits, layer_shapes)]

        for client in trainers:
            for modality in get_modalities(client):
                # 1. 获取分层结构
                if hasattr(client.model, 'img_enc') and modality == 'img':
                    ref_module = client.model.img_enc
                elif hasattr(client.model, 'txt_enc') and modality == 'txt':
                    ref_module = client.model.txt_enc
                elif hasattr(client.model, 'img_enc') and hasattr(client.model, 'txt_enc'):
                    # PCME多模态
                    ref_module = client.model
                ref_state = ref_module.state_dict()
                layer_shapes = [v.shape for v in ref_state.values()]
                L_m = len(layer_shapes)

                # 2. 冻结所有集群编码器参数（实际训练时应设置 requires_grad=False）

                # 3. 初始化分层注意力参数
                A = [torch.zeros(num_clusters, requires_grad=True, device=self.device) for _ in range(L_m)]
                optimizer = torch.optim.Adam(A, lr=getattr(self.args, "ascc_lr", 0.05))

                # 4. 查询数据采样
                if not hasattr(client, "train_loader"):
                    continue
                data_loader = client.train_loader
                beta = getattr(self.args, "ascc_query_ratio", 0.2)
                query_samples = []
                for batch in data_loader:
                    if modality == 'img':
                        x = batch[0]
                        y = batch[1]
                    elif modality == 'txt':
                        x = batch[1]
                        y = batch[0]
                    else:
                        x, y = batch[0], batch[1]
                    n = int(len(x) * beta)
                    if n == 0: n = 1
                    idx = torch.randperm(len(x))[:n]
                    query_samples.append((x[idx], y[idx]))
                    if len(query_samples) > 5:
                        break

                # 5. 分层注意力训练
                for _ in range(getattr(self.args, "ascc_attn_epoch", 3)):
                    total_loss = 0
                    for x, y in query_samples:
                        q = x.to(self.device)
                        y = y.to(self.device)
                        # 分层前向
                        for l in range(L_m):
                            attn = F.softmax(A[l], dim=0)
                            layer_outputs = []
                            for k in range(num_clusters):
                                # 获取第k个聚类模型的第l层参数
                                cluster_layer_params = split_layers(cluster_models[k], layer_shapes)[l]
                                # 构造临时模型，仅替换当前层参数
                                tmp_module = copy.deepcopy(ref_module)
                                tmp_state = tmp_module.state_dict()
                                keys = list(tmp_state.keys())
                                tmp_state[keys[l]] = torch.tensor(cluster_layer_params, dtype=tmp_state[keys[l]].dtype)
                                tmp_module.load_state_dict(tmp_state)
                                tmp_module.eval()
                                with torch.no_grad():
                                    if modality == 'img':
                                        out = tmp_module(q)
                                    elif modality == 'txt':
                                        out = tmp_module(q)
                                    else:
                                        out = tmp_module(q)
                                layer_outputs.append(out)
                            # 加权聚合
                            q = sum(attn[k] * layer_outputs[k] for k in range(num_clusters))
                        # 最终输出送入分类器
                        if hasattr(client.model, "classifier"):
                            logits = client.model.classifier(q)
                        else:
                            logits = q
                        loss = F.cross_entropy(logits, y)
                        total_loss += loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                # 6. 分层聚合生成个性化编码器
                personalized_layers = []
                for l in range(L_m):
                    attn = F.softmax(A[l].detach(), dim=0).cpu().numpy()
                    agg_layer = sum(attn[k] * split_layers(cluster_models[k], layer_shapes)[l] for k in range(num_clusters))
                    personalized_layers.append(agg_layer)
                personalized_flat = np.concatenate([l.reshape(-1) for l in personalized_layers], axis=0)
                set_model_parameters(client, modality, personalized_flat)

                # 7. 选择最优聚类
                attn_sum = np.array([sum(F.softmax(A[l], dim=0)[k].item() for l in range(L_m)) for k in range(num_clusters)])
                best_idx = int(np.argmax(attn_sum))
                client.selected_cluster = best_idx

                # 8. 本地微调
                model = ref_module
                model.train()
                if hasattr(client.model, "classifier"):
                    classifier = client.model.classifier
                    classifier.train()
                    params = list(model.parameters()) + list(classifier.parameters())
                else:
                    params = list(model.parameters())
                optimizer_finetune = torch.optim.Adam(params, lr=getattr(self.args, "ascc_ft_lr", 0.001))
                for _ in range(getattr(self.args, "ascc_ft_epoch", 1)):
                    for batch in data_loader:
                        if modality == 'img':
                            x = batch[0].to(self.device)
                            y = batch[1].to(self.device)
                        elif modality == 'txt':
                            x = batch[1].to(self.device)
                            y = batch[0].to(self.device)
                        else:
                            x, y = batch[0].to(self.device), batch[1].to(self.device)
                        optimizer_finetune.zero_grad()
                        out = model(x)
                        if hasattr(client.model, "classifier"):
                            logits = classifier(out)
                        else:
                            logits = out
                        loss = F.cross_entropy(logits, y)
                        loss.backward()
                        optimizer_finetune.step()
                        break  # 只做一小步
            
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
            # trainer.run()
            
        self.aggregate_with_ssm_ascc(self.cur_trainers)

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        
        # test in own domain
        import csv
        
        if not hasattr(self, 'mm_results'):
            self.mm_results = []
        if not hasattr(self, 'rsum_history'):
            self.rsum_history = []
            
        mm_rows = []
        
        print("Testing...")
        rsum = 0
        for domain_idx in range(self.args.num_domains):
            print(f"Server tests in domain {domain_idx}:")
            test_scores = self.engine.evaluate({'test': self.val_dataloader[domain_idx]})
            metadata = self.engine.metadata.copy()
            metadata['cur_epoch'] = round_n + 1
            metadata['lr'] = get_lr(self.engine.optimizer)
            
            self.engine.report_scores(step=round_n + 1,
                                    scores=test_scores,
                                    metadata=metadata)
            rsum_i = test_scores['test']['n_fold']['i2t']['recall_1'] + test_scores['test']['n_fold']['t2i']['recall_1'] + \
                test_scores['test']['i2t']['recall_1'] + test_scores['test']['t2i']['recall_1']
            rsum += rsum_i
            mm_rows.append([
                round_n, domain_idx, rsum_i,
                test_scores['test']['n_fold']['i2t']['recall_1'],
                test_scores['test']['n_fold']['t2i']['recall_1'],
                test_scores['test']['i2t']['recall_1'],
                test_scores['test']['t2i']['recall_1'],
                test_scores['test']['n_fold']['i2t']['recall_5'],
                test_scores['test']['n_fold']['t2i']['recall_5'],
                test_scores['test']['i2t']['recall_5'],
                test_scores['test']['t2i']['recall_5'],
            ])
            self.wandb.log({f"Multimodal rsum_r1": rsum_i}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal i2t_r1": test_scores['test']['i2t']['recall_1']}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal t2i_r1": test_scores['test']['t2i']['recall_1']}, step=self.cur_epoch)
        
        self.rsum_history.append(rsum)
        if self.best_score < rsum:
            best_score = rsum
            metadata['best_score'] = best_score
            metadata['best_epoch'] = round_n + 1
            self.best_metadata, self.best_score = metadata, best_score
            print(f"Best score updated: {best_score} at epoch {round_n + 1}")
            # torch.save({'net': self.engine.model.state_dict()}, self.args.name + '-best_model.pt')

        if round_n == self.args.comm_rounds - 1:
            print(f"Final best score: {self.best_score} at epoch {self.best_metadata['best_epoch']}")
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(range(1, len(self.rsum_history)+1), self.rsum_history, marker='o')
            plt.xlabel('Round')
            plt.ylabel('rsum')
            plt.title(f'rsum Curve (Best: {self.best_score} at epoch {self.best_metadata["best_epoch"]})')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'rsum_MASA.png')
            plt.close()
            # torch.save({'net': self.engine.model.state_dict()}, self.args.name + '-last_model.pt')
        
        mm_csv = f'mm_MASA.csv'
        if mm_rows:
            write_header = not os.path.exists(mm_csv)
            with open(mm_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'round', 'domain_idx', 'rsum_i',
                        'n_fold_i2t_r1', 'n_fold_t2i_r1', 'i2t_r1', 't2i_r1',
                        'n_fold_i2t_r5', 'n_fold_t2i_r5', 'i2t_r5', 't2i_r5'
                    ])
                writer.writerows(mm_rows)
            
        self.engine.lr_scheduler.step()
        gc.collect()
