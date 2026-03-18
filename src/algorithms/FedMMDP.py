import gc
import random
from collections import OrderedDict
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import copy

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.datasets.transform import collate_fn
from src.datasets.load_FL_datasets import get_FL_trainloader
from src.algorithms.FedMMDPClientTrainer import ClientTrainer
from src.algorithms.FedMMDPMMClientTrainer import MMClientTrainer

from src.algorithms.mm_eval import MMEvaluator
from src.utils.config import parse_config
from src.utils.logger import PythonLogger

try:
    from apex import amp
except ImportError:
    print('failed to import apex')


is_test = False


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
        self.best_score = 0
        self.cur_epoch = 0
        self.best_metadata = None

        # img & txt local dataloaders
        self.img_train_loaders, self.txt_train_loaders = None, None

        # universal test dataloader
        self.test_loader = None

        self.config = None
        self.set_config()

        self.logger = PythonLogger(output_file=self.config.train.output_file)


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
        # self.engine = TrainerEngine()
        # self.engine.set_logger(self.logger)
        self.val_dataloader = {}
        for i in range(args.num_img_clients):
            val_dataset = load_from_disk(os.path.join(self.args.data_root, f'domain_dataset_{i}', 'test'))
            self.val_dataloader[i] = torch.utils.data.DataLoader(val_dataset, 
                                                            batch_size=self.args.batch_size, 
                                                            shuffle=False, 
                                                            num_workers=0,
                                                            collate_fn=collate_fn
                                                            )

        # self._dataloaders = self.dataloaders_global.copy()
        self.evaluator = MMEvaluator(model_name=self.args.model,
                                       dataset=self.args.dataset,
                                       eval_method='matmul',
                                       verbose=False,
                                       eval_device='cuda',
                                       n_crossfolds=1, 
                                       class_size=self.class_size,
                                       feature_dim=self.args.feature_dim,
                                       data_root=self.args.data_root)
        # self.engine.create(self.config, self.evaluator, self.args.mlp_local)

        # self.engine.model_to_device()
        torch.backends.cudnn.enabled = True
        # if self.config.train.get('use_fp16'):
            # self.engine.logger.log('Train with half precision using AMP')
            # self.engine.to_half()

    def create_model(self, args):
        self.logger.log('start creating model and partition datasets')
        self.device = torch.device("cuda:%d" % args.device)

        os.makedirs('/home/bd/data/zs' + f'/data/yClient', exist_ok=True)

        # Create Client Models
        self.img_local_trainers, self.txt_local_trainers, self.mm_local_trainers = [], [], []
        # img clients
        if args.num_img_clients > 0:
            dataset = 'image'
            self.img_trainloaders, self.img_test_loaders = get_FL_trainloader(dataset, self.args.data_root,
                                                                 args.num_img_clients, "hetero", self.args.alpha, self.args.batch_size)
            self.img_local_trainers = []
            for i in range(args.num_img_clients):
                self.img_local_trainers.append(
                    ClientTrainer(args, dataset, self.class_size, self.logger,
                                  inter_distance=4, client_id=i, wandb=self.wandb))
                self.img_local_trainers[i].train_loader = self.img_trainloaders[i]
                self.img_local_trainers[i].test_loader = self.img_test_loaders[i]
                if is_test and i == 0:
                    break
        # txt clients
        if args.num_txt_clients > 0:
            dataset = 'text'
            self.txt_trainloaders, self.txt_test_loaders = get_FL_trainloader(dataset, self.args.data_root,
                                                                 args.num_txt_clients, "hetero", self.args.alpha, self.args.batch_size)
            self.txt_local_trainers = []
            for i in range(args.num_txt_clients):
                self.txt_local_trainers.append(
                    ClientTrainer(args, dataset, self.class_size, self.logger,
                                  inter_distance=4, client_id=i, wandb=self.wandb))
                self.txt_local_trainers[i].train_loader = self.txt_trainloaders[i]
                self.txt_local_trainers[i].test_loader = self.txt_test_loaders[i]
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

    def aggregate_clip_models(self, local_image_models, local_text_models, local_mm_models):
        server_model = self.engine.model
        
        image_encoder_params = OrderedDict()
        
        all_image_encoders = []
        for model in local_image_models:
            all_image_encoders.append({
                'visual_projector': model.visual_projector.state_dict(),
                'clip_visual': model.clip_visual.state_dict()
            })
        
        for model in local_mm_models:
            all_image_encoders.append({
                'visual_projector': model.img_enc.visual_projector.state_dict(),
                'clip_visual': model.img_enc.clip_visual.state_dict()
            })
        
        for key in all_image_encoders[0]['visual_projector'].keys():
            param_name = f'visual_projector.{key}'
            params = [enc['visual_projector'][key] for enc in all_image_encoders]
            orig_dtype = params[0].dtype
            avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
            image_encoder_params[param_name] = avg_param.to(orig_dtype)

        for key in all_image_encoders[0]['clip_visual'].keys():
            param_name = f'clip_visual.{key}'
            params = [enc['clip_visual'][key] for enc in all_image_encoders]
            orig_dtype = params[0].dtype
            avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
            image_encoder_params[param_name] = avg_param.to(orig_dtype)
        
        server_model.img_enc.load_state_dict(image_encoder_params)
        
        text_encoder_params = OrderedDict()
        
        all_text_encoders = []
        for model in local_text_models:
            all_text_encoders.append({
                'text_projector': model.text_projector.state_dict(),
                'clip_text': model.clip_text.state_dict()
            })
        
        for model in local_mm_models:
            all_text_encoders.append({
                'text_projector': model.txt_enc.text_projector.state_dict(),
                'clip_text': model.txt_enc.clip_text.state_dict()
            })
        
        for key in all_text_encoders[0]['text_projector'].keys():
            param_name = f'text_projector.{key}'
            params = [enc['text_projector'][key] for enc in all_text_encoders]
            orig_dtype = params[0].dtype
            avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
            text_encoder_params[param_name] = avg_param.to(orig_dtype)

        for key in all_text_encoders[0]['clip_text'].keys():
            param_name = f'clip_text.{key}'
            params = [enc['clip_text'][key] for enc in all_text_encoders]
            orig_dtype = params[0].dtype
            avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
            text_encoder_params[param_name] = avg_param.to(orig_dtype)
        
        server_model.txt_enc.load_state_dict(text_encoder_params)
        
        return server_model
    
    def aggregate_resnet_models(self, local_image_models, local_text_models, local_mm_models):
        server_model = copy.deepcopy(local_mm_models[0])
        
        image_encoder_params = OrderedDict()
        
        all_image_encoders = []
        for model in local_image_models:
            model.to(self.device)
            all_image_encoders.append({
                'resnet': model.state_dict()
            })
        
        for model in local_mm_models:
            model.to(self.device)
            all_image_encoders.append({
                'resnet': model.img_enc.cnn.state_dict()
            })
        for key in all_image_encoders[0]['resnet'].keys():
            param_name = f'{key}'
            if all(key in enc['resnet'] for enc in all_image_encoders):
                params = [enc['resnet'][key] for enc in all_image_encoders]
                orig_dtype = params[0].dtype
                avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
                image_encoder_params[param_name] = avg_param.to(orig_dtype)
        
        server_model.img_enc.cnn.load_state_dict(image_encoder_params)
        
        text_encoder_params = OrderedDict()
        
        all_text_encoders = []
        for model in local_text_models:
            model.to(self.device)
            all_text_encoders.append({
                'text': model.state_dict()
            })
        
        for model in local_mm_models:
            all_text_encoders.append({
                'text': model.txt_enc.state_dict()
            })
            
        for key in all_text_encoders[0]['text'].keys():
            param_name = f'{key}'
            if all(key in enc['text'] for enc in all_text_encoders):
                params = [enc['text'][key] for enc in all_text_encoders]
                orig_dtype = params[0].dtype
                avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
                text_encoder_params[param_name] = avg_param.to(orig_dtype)
        
        server_model.txt_enc.load_state_dict(text_encoder_params)
        
        return server_model

    def train(self, round_n):
        self.cur_epoch = round_n
        self.cur_trainers = self.total_local_trainers

        if not is_test:
            self.logger.log(f"Round {round_n + 1}!")
            
            if len(self.total_local_trainers) != 0:
                self.cur_trainers = random.sample(self.total_local_trainers, self.args.client_num_per_round)
        # Generate random key using QR decomposition
        n_features = self.args.feature_dim
        seed = self.args.seed + round_n
        np.random.seed(seed)
        random_matrix = np.random.randn(n_features, n_features)
        key, _ = np.linalg.qr(random_matrix)
        
        # Get local representations
        features = []
        client_idx_list = []
        for idx, trainer in enumerate(self.cur_trainers):
            trainer.cur_epoch = round_n
            local_vec, dataset_name = trainer.generate_logits(key)
            client_idx = trainer.client_idx
            self.logger.log(f"Generate {dataset_name} Client {client_idx} Representations!")
            features.append(local_vec)
            client_idx_list += [client_idx] * local_vec.shape[0]

        print(f"Client_idx_list Length: {len(client_idx_list)}")
        print(f"Client numbers: {len(features)}")
        
        combined_feats = np.concatenate(features, axis=0)
        if not np.isfinite(combined_feats).all():
            invalid_count = np.size(combined_feats) - np.isfinite(combined_feats).sum()
            self.logger.log(f"Sanitizing {invalid_count} non-finite feature values before clustering")
            combined_feats = np.nan_to_num(combined_feats, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Features Shape: {combined_feats.shape}")
        
        # Cluster representations
        if self.args.cluster_method == 'finch':
            from finch import FINCH
            print("Running FINCH clustering...")
            cluster, _, _ = FINCH(combined_feats)
            partition_level = cluster.shape[1] - self.args.partition_level
            # partition_level = self.args.partition_level
            cluster_labels = cluster[:, partition_level]
            n_clusters = len(np.unique(cluster_labels))
            print(f"FINCH found {n_clusters} communities at partition level {partition_level}")
            
        elif self.args.cluster_method == 'spectral':
            from sklearn.cluster import SpectralClustering
            affinity_matrix = cosine_similarity(combined_feats) + 1
            n_clusters = self.args.n_clusters
            spectral = SpectralClustering(n_clusters=n_clusters, 
                              affinity='precomputed',
                              random_state=42,
                              assign_labels='kmeans')
            cluster_labels = spectral.fit_predict(affinity_matrix)
            
        elif self.args.cluster_method == 'kmeans':
            from sklearn.cluster import KMeans
            n_clusters = self.args.n_clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels= kmeans.fit_predict(combined_feats)
            
        elif self.args.cluster_method == 'dbscan':
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=self.args.eps, min_samples=self.args.min_samples)
            cluster_labels = dbscan.fit_predict(combined_feats)
            n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
            
        else:
            raise ValueError("Unsupported clustering method. Choose 'finch' or 'spectral'.")
        
        # Compute cluster centers
        cluster_centers = {}
        cluster_centers_enc = {}
        for label in range(n_clusters):
            # Get indices for samples in this cluster
            cluster_mask = cluster_labels == label
            if np.any(cluster_mask):
                # Calculate mean of features for this cluster
                center = np.mean(combined_feats[cluster_mask], axis=0)
                cluster_centers[label] = np.dot(center, key.T)  # get the original space representation
                cluster_centers_enc[label] = center

        # Log cluster sizes
        for label in range(n_clusters):
            cluster_size = np.sum(cluster_labels == label)
            self.logger.log(f"Cluster {label} size: {cluster_size}")
        
        # Count cluster labels for each client_idx
        client_cluster_list = {}
        for client_idx, cluster_label in zip(client_idx_list, cluster_labels):
            if client_idx not in client_cluster_list:
                client_cluster_list[client_idx] = []
            client_cluster_list[client_idx].append(cluster_label)
        if not getattr(self.args, "disable_tsne", False):
            tsne_visualize(combined_feats, client_idx_list, cluster_labels, round_n)
        del features, client_idx_list, combined_feats, cluster_labels
        gc.collect()
        
        local_image_model = []
        local_text_model = []
        local_mm_model = []
        # local training
        for idx, trainer in enumerate(self.cur_trainers):
            self.logger.log(f"Training Client {trainer.client_idx}!")
            trainer.run(cluster_centers, client_cluster_list)
            
            if trainer.dset_name == 'image':
                local_image_model.append(trainer.model)
            elif trainer.dset_name == 'text':
                local_text_model.append(trainer.model)
            elif trainer.dset_name == 'mm':
                local_mm_model.append(trainer.model)
        
        # aggregate local models
        if self.args.aggregate == True:
            if self.args.model == 'clip':
                server_model = self.aggregate_clip_models(local_image_model, local_text_model, local_mm_model)
                for trainer in self.cur_trainers:
                    if hasattr(trainer.model, "img_enc") and hasattr(trainer.model, "txt_enc"):
                        trainer.model.load_state_dict(server_model.state_dict())
                    elif hasattr(trainer.model, "clip_visual"):
                        for name, param in server_model.img_enc.state_dict().items():
                            if name in trainer.model.state_dict():
                                trainer.model.state_dict()[name].copy_(param)
                        if hasattr(trainer.model, "visual_projector") and hasattr(server_model.img_enc, "visual_projector"):
                            for name, param in server_model.img_enc.visual_projector.state_dict().items():
                                if name in trainer.model.visual_projector.state_dict():
                                    trainer.model.visual_projector.state_dict()[name].copy_(param)
                    elif hasattr(trainer.model, "clip_text"):
                        for name, param in server_model.txt_enc.state_dict().items():
                            if name in trainer.model.state_dict():
                                trainer.model.state_dict()[name].copy_(param)
                        if hasattr(trainer.model, "text_projector") and hasattr(server_model.txt_enc, "text_projector"):
                            for name, param in server_model.txt_enc.text_projector.state_dict().items():
                                if name in trainer.model.text_projector.state_dict():
                                    trainer.model.text_projector.state_dict()[name].copy_(param)
            elif self.args.model == 'resnet':
                server_model = self.aggregate_resnet_models(local_image_model, local_text_model, local_mm_model)
                for trainer in self.cur_trainers:
                    if hasattr(trainer.model, "img_enc") and hasattr(trainer.model, "txt_enc"):
                        trainer.model.load_state_dict(server_model.state_dict())
                    elif hasattr(trainer.model, "ResNet"):
                        for name, param in server_model.img_enc.cnn.state_dict().items():
                            if name in trainer.model.state_dict():
                                trainer.model.state_dict()[name].copy_(param)
                    elif hasattr(trainer.model, "EncoderText"):
                        for name, param in server_model.txt_enc.cnn.state_dict().items():
                            if name in trainer.model.state_dict():
                                trainer.model.state_dict()[name].copy_(param)
                            
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        # metadata = self.engine.metadata.copy()
        # metadata['cur_epoch'] = round_n + 1
        # metadata['lr'] = get_lr(self.engine.optimizer)
        
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
        rsum_domain = 0
        for idx, trainer in enumerate(self.total_local_trainers):
            if trainer.dset_name == "image":
                domain_idx = idx
                trainer.test_loader = self.val_dataloader[domain_idx]
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
                for domain_idx in range(self.args.num_domains):
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
                        rsum_domain += rsum_i
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
                    self.wandb.log({f"Multimodal_{mm_client_idx} rsum_r1": rsum_i}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{mm_client_idx} n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{mm_client_idx} n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{mm_client_idx} i2t_r1": test_scores['test']['i2t']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{mm_client_idx} t2i_r1": test_scores['test']['t2i']['recall_1']}, step=self.cur_epoch)
        
        self.rsum_history.append(rsum_domain)
        if self.best_score < rsum_domain:
            rsum /= 5
            best_score = rsum_domain
            metadata['best_score'] = best_score
            metadata['best_epoch'] = round_n + 1
            self.best_metadata, self.best_score = metadata, best_score
            print(f"Best score updated: {best_score} at epoch {round_n + 1}, rsum: {rsum}")
            # torch.save({'net': trainer.model.state_dict()}, self.args.name + '-best_model.pt')

        if round_n == self.args.comm_rounds - 1:
            print(f"Final best score: {self.best_score} at epoch {self.best_metadata['best_epoch']}")
        #     torch.save({'net': trainer.model.state_dict()}, self.args.name + '-last_model.pt')
        
        os.makedirs('results', exist_ok=True)
        mm_csv = f'results/mm_{self.args.dataset}_{self.args.alpha}_{self.args.cluster_method}_{str(self.args.cluster_weight)}_{self.args.model}_{self.args.local_epochs}_{self.args.comm_rounds}_{self.args.lr}_{self.args.n_clusters}_{self.args.rmg_weight}.csv'

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
        plt.title(f'rsum Curve ({str(self.args.cluster_weight)}, {self.args.model}), best rsum: {self.best_score} at round {self.best_metadata["best_epoch"]}')
        plt.grid(True)
        plt.tight_layout()
        if self.args.aggregate:
            plt.savefig(f'results/rsum_{self.args.dataset}_{self.args.alpha}_{self.args.cluster_method}_{str(self.args.cluster_weight)}_{self.args.model}_{self.args.local_epochs}_{self.args.comm_rounds}_{self.args.lr}_{self.args.n_clusters}_aggregate.png')
        else:
            plt.savefig(f'results/rsum_{self.args.dataset}_{self.args.alpha}_{self.args.cluster_method}_{str(self.args.cluster_weight)}_{self.args.model}_{self.args.local_epochs}_{self.args.comm_rounds}_{self.args.lr}_{self.args.n_clusters}_{self.args.rmg_weight}.png')
        plt.close()

        gc.collect()
        
def tsne_visualize(combined_feats, client_idx_list, cluster_labels, round_n):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(combined_feats)

    tsne_x = tsne_results[:, 0]
    tsne_y = tsne_results[:, 1]
    unique_clients = np.unique(client_idx_list)
    unique_clusters = np.unique(cluster_labels)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'H', '<', '>', 'd', 'p', 'h', '8']
    if len(unique_clients) > len(markers):
        raise ValueError('Client number exceeds available markers. Please add more markers.')

    plt.figure(figsize=(12, 10))
    client_handles = []
    cluster_handles = []

    for client_idx in unique_clients:
        for cluster_idx in unique_clusters:
            mask = (np.array(client_idx_list) == client_idx) & (cluster_labels == cluster_idx)
            if np.any(mask):
                plt.scatter(
                    tsne_x[mask],
                    tsne_y[mask],
                    color=colors[cluster_idx],
                    marker=markers[client_idx % len(markers)],
                    alpha=0.6,
                    s=20,
                )

                if cluster_idx == unique_clusters[0]:
                    client_handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker=markers[client_idx % len(markers)],
                            color='w',
                            markerfacecolor='gray',
                            markersize=10,
                            label=f'Client {client_idx}',
                        )
                    )

                if client_idx == unique_clients[0]:
                    cluster_handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker='o',
                            color='w',
                            markerfacecolor=colors[cluster_idx],
                            markersize=10,
                            label=f'Cluster {cluster_idx}',
                        )
                    )

    plt.title(f't-SNE Visualization of Clustering Results (Round {round_n})', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)

    first_legend = plt.legend(handles=cluster_handles, title='Clusters', loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.gca().add_artist(first_legend)
    plt.legend(handles=client_handles, title='Clients', loc='upper right', bbox_to_anchor=(1.3, 0.7))

    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_dir = 'visualization_results/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'tsne_round_{round_n}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f't-SNE visualization saved to {save_path}')
    plt.close()
