import copy
import csv
import gc
import os
import random
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.algorithms.FedMMDPClientTrainer import ClientTrainer
from src.algorithms.FedMMDPMMClientTrainer import MMClientTrainer
from src.algorithms.mm_eval import MMEvaluator
from src.algorithms.secure_agg import SecureAggregator
from src.datasets.load_FL_datasets import get_FL_trainloader
from src.datasets.transform import collate_fn
from src.utils.config import parse_config, apply_runtime_overrides
from src.utils.logger import PythonLogger
from src.utils.experiment_naming import projector_tag
from src.utils.model_utils import is_embedding_model


is_test = False


class MMFL(object):
    def __init__(self, args, wandb=None):
        self.args = args
        self.wandb = wandb

        self.device = None
        self.img_local_trainers = None
        self.txt_local_trainers = None
        self.mm_local_trainers = None
        if self.args.dataset == 'imagenet':
            self.class_size = 50
        elif self.args.dataset == 'fashion':
            self.class_size = 48
        elif self.args.dataset == 'food':
            self.class_size = 101
        elif self.args.dataset == 'iapr':
            self.class_size = 30
        else:
            raise ValueError(f'Unsupported dataset: {self.args.dataset}')
        self.best_score = 0
        self.cur_epoch = 0
        self.best_metadata = None
        self.img_train_loaders, self.txt_train_loaders = None, None
        self.test_loader = None
        self.global_centroids = None

        self.config = None
        self.set_config()
        self.logger = PythonLogger(output_file=self.config.train.output_file)
        self._validate_fedmmdp_args()

    def _validate_fedmmdp_args(self):
        if self.args.cluster_method != 'kmeans':
            self.logger.log(
                f"FedMMDP secure aggregation only supports Lloyd k-means; overriding cluster_method "
                f"from {self.args.cluster_method} to kmeans."
            )
            self.args.cluster_method = 'kmeans'
        if self.args.n_clusters <= 0:
            raise ValueError('FedMMDP requires n_clusters > 0.')
        if self.args.tau <= 0:
            raise ValueError('FedMMDP requires tau > 0.')
        if self.args.secure_agg_mode != 'plaintext':
            raise ValueError(f'Unsupported secure aggregation mode: {self.args.secure_agg_mode}')
        if self.args.centroid_init != 'random_unit':
            raise ValueError(f'Unsupported centroid initialization mode: {self.args.centroid_init}')

    def _artifact_tag(self):
        return f'{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_{projector_tag(self.args)}_FedMMDP_secureagg_{self.args.secure_agg_mode}_{self.args.cluster_method}'

    def _initialize_global_centroids(self):
        if self.global_centroids is not None:
            return
        rng = np.random.default_rng(self.args.seed)
        centroids = rng.standard_normal((self.args.n_clusters, self.args.feature_dim)).astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        centroids = centroids / norms
        self.global_centroids = torch.from_numpy(centroids)
        self.logger.log(
            f'Initialized {self.args.n_clusters} FedMMDP centroids with random_unit strategy '
            f'(seed={self.args.seed}).'
        )

    def _lloyd_update(self, aggregated_stats):
        global_sum = aggregated_stats['sum']
        global_count = aggregated_stats['count']
        if self.global_centroids is None:
            self._initialize_global_centroids()

        updated_centroids = self.global_centroids.clone().float()
        for cluster_idx in range(self.args.n_clusters):
            count = float(global_count[cluster_idx].item())
            if count > 0:
                center = global_sum[cluster_idx] / count
                center = torch.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
                center_norm = center.norm(p=2).clamp_min(1e-12)
                updated_centroids[cluster_idx] = center / center_norm
        self.global_centroids = updated_centroids.cpu()

        for cluster_idx in range(self.args.n_clusters):
            self.logger.log(
                f"Secure-agg Lloyd cluster {cluster_idx} count: {int(global_count[cluster_idx].item())}"
            )

    def set_config(self, img='image', txt='text'):
        if self.args.dataset == 'imagenet':
            yaml_name = 'imageNet_cap.yaml'
        elif self.args.dataset == 'fashion':
            yaml_name = 'fashion_gen.yaml'
        elif self.args.dataset == 'food':
            yaml_name = 'umpc_food.yaml'
        else:
            yaml_name = 'iapr.yaml'
        self.config = parse_config("./src/" + yaml_name, strict_cast=False)
        self.config = apply_runtime_overrides(self.args, self.config)
        self.config.train.model_save_path = 'model_last_no_prob.pth'
        self.config.train.best_model_save_path = 'model_best_no_prob.pth'
        self.config.train.output_file = f'{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_{projector_tag(self.args)}_model_noprob.log'
        self.config.model.img_client = img
        self.config.model.txt_client = txt
        self.config.model.embed_dim = self.args.feature_dim

    def load_dataset(self, args):
        self.val_dataloader = {}
        for i in range(args.num_img_clients):
            val_dataset = load_from_disk(os.path.join(self.args.data_root, f'domain_dataset_{i}', 'test'))
            self.val_dataloader[i] = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )

        self.evaluator = MMEvaluator(
            model_name=self.args.model,
            dataset=self.args.dataset,
            eval_method='matmul',
            verbose=False,
            eval_device='cuda',
            n_crossfolds=1,
            class_size=self.class_size,
            feature_dim=self.args.feature_dim,
            data_root=self.args.data_root,
        )
        torch.backends.cudnn.enabled = True

    def create_model(self, args):
        self.logger.log('start creating model and partition datasets')
        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

        os.makedirs('/home/bd/data/zs/data/yClient', exist_ok=True)

        self.img_local_trainers, self.txt_local_trainers, self.mm_local_trainers = [], [], []
        if args.num_img_clients > 0:
            dataset = 'image'
            self.img_trainloaders, self.img_test_loaders = get_FL_trainloader(
                dataset, self.args.data_root, args.num_img_clients, self.args.partition, self.args.alpha, self.args.batch_size
            )
            for i in range(args.num_img_clients):
                trainer = ClientTrainer(args, dataset, self.class_size, self.logger, inter_distance=4, client_id=i, wandb=self.wandb)
                trainer.train_loader = self.img_trainloaders[i]
                trainer.test_loader = self.img_test_loaders[i]
                self.img_local_trainers.append(trainer)
                if is_test and i == 0:
                    break
        if args.num_txt_clients > 0:
            dataset = 'text'
            self.txt_trainloaders, self.txt_test_loaders = get_FL_trainloader(
                dataset, self.args.data_root, args.num_txt_clients, self.args.partition, self.args.alpha, self.args.batch_size
            )
            for i in range(args.num_txt_clients):
                trainer = ClientTrainer(args, dataset, self.class_size, self.logger, inter_distance=4, client_id=i, wandb=self.wandb)
                trainer.train_loader = self.txt_trainloaders[i]
                trainer.test_loader = self.txt_test_loaders[i]
                self.txt_local_trainers.append(trainer)
                if is_test and i == 0:
                    break
        if args.num_mm_clients > 0:
            if self.args.dataset == 'imagenet':
                yaml_name = 'imageNet_cap.yaml'
            elif self.args.dataset == 'fashion':
                yaml_name = 'fashion_gen.yaml'
            elif self.args.dataset == 'food':
                yaml_name = 'umpc_food.yaml'
            else:
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
                trainer = MMClientTrainer(
                    args, config, self.class_size, self.logger, client=client_id, dset_name="mm",
                    device='cuda', mlp_local=self.args.mlp_local, wandb=self.wandb,
                )
                self.mm_local_trainers.append(trainer)
                if is_test and client_id == 0:
                    break
            print(f"Samples Num: {[len(i.train_loader.dataset) for i in self.mm_local_trainers]}")

        self.total_local_trainers = self.img_local_trainers + self.txt_local_trainers + self.mm_local_trainers
        for i, trainer in enumerate(self.total_local_trainers):
            trainer.client_idx = i

    def aggregate_clip_models(self, local_image_models, local_text_models, local_mm_models):
        server_model = self.engine.model

        image_encoder_params = OrderedDict()
        all_image_encoders = []
        for model in local_image_models:
            all_image_encoders.append({
                'visual_projector': model.visual_projector.state_dict(),
                'clip_visual': model.clip_visual.state_dict(),
            })
        for model in local_mm_models:
            all_image_encoders.append({
                'visual_projector': model.img_enc.visual_projector.state_dict(),
                'clip_visual': model.img_enc.clip_visual.state_dict(),
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
                'clip_text': model.clip_text.state_dict(),
            })
        for model in local_mm_models:
            all_text_encoders.append({
                'text_projector': model.txt_enc.text_projector.state_dict(),
                'clip_text': model.txt_enc.clip_text.state_dict(),
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

        all_image_encoders = []
        for model in local_image_models:
            all_image_encoders.append({'image': model.state_dict()})
        for model in local_mm_models:
            all_image_encoders.append({'image': model.img_enc.state_dict()})

        image_encoder_params = OrderedDict()
        for key in all_image_encoders[0]['image'].keys():
            if all(key in enc['image'] for enc in all_image_encoders):
                params = [enc['image'][key] for enc in all_image_encoders]
                orig_dtype = params[0].dtype
                avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
                image_encoder_params[key] = avg_param.to(orig_dtype)
        server_model.img_enc.load_state_dict(image_encoder_params)

        all_text_encoders = []
        for model in local_text_models:
            all_text_encoders.append({'text': model.state_dict()})
        for model in local_mm_models:
            all_text_encoders.append({'text': model.txt_enc.state_dict()})

        text_encoder_params = OrderedDict()
        for key in all_text_encoders[0]['text'].keys():
            if all(key in enc['text'] for enc in all_text_encoders):
                params = [enc['text'][key] for enc in all_text_encoders]
                orig_dtype = params[0].dtype
                avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
                text_encoder_params[key] = avg_param.to(orig_dtype)
        server_model.txt_enc.load_state_dict(text_encoder_params)
        return server_model

    def _get_active_trainers(self):
        self.cur_trainers = self.total_local_trainers
        if not is_test and len(self.total_local_trainers) != 0:
            if self.args.client_num_per_round >= len(self.total_local_trainers):
                self.cur_trainers = self.total_local_trainers
            else:
                self.cur_trainers = random.sample(self.total_local_trainers, self.args.client_num_per_round)
        return self.cur_trainers

    def _run_secure_agg_lloyd_round(self, round_n):
        self._initialize_global_centroids()
        aggregator = SecureAggregator(
            num_clusters=self.args.n_clusters,
            feature_dim=self.args.feature_dim,
            mode=self.args.secure_agg_mode,
        )
        for trainer in self.cur_trainers:
            trainer.cur_epoch = round_n
            local_stats, dataset_name = trainer.compute_local_cluster_statistics(self.global_centroids)
            self.logger.log(f"Secure-agg statistics collected from {dataset_name} client {trainer.client_idx}.")
            aggregator.collect(trainer.client_idx, local_stats)
        aggregated_stats = aggregator.finalize()
        self._lloyd_update(aggregated_stats)
        self.logger.log(
            f"Secure aggregation finalized for {aggregated_stats['num_clients']} clients in round {round_n + 1}."
        )

    def train(self, round_n):
        self.cur_epoch = round_n
        if not is_test:
            self.logger.log(f"Round {round_n + 1}!")
        self._get_active_trainers()
        self._run_secure_agg_lloyd_round(round_n)

        local_image_model = []
        local_text_model = []
        local_mm_model = []
        for trainer in self.cur_trainers:
            self.logger.log(f"Training Client {trainer.client_idx} with secure-agg centroids.")
            trainer.run(self.global_centroids)
            if trainer.dset_name == 'image':
                local_image_model.append(trainer.model)
            elif trainer.dset_name == 'text':
                local_text_model.append(trainer.model)
            elif trainer.dset_name == 'mm':
                local_mm_model.append(trainer.model)

        if self.args.aggregate is True:
            if is_embedding_model(self.args.model):
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
            return 0.0

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
        metadata = {'best_epoch': 0, 'best_score': 0.0}
        for idx, trainer in enumerate(self.total_local_trainers):
            if trainer.dset_name == "image":
                domain_idx = idx
                trainer.test_loader = self.val_dataloader[domain_idx]
                print(f"Client {trainer.dset_name} {idx} tests in domain {idx}:")
                losses, test_top1, test_top5 = trainer.test()
                img_txt_rows.append([round_n, trainer.client_idx, domain_idx, losses, test_top1, test_top5])
            elif trainer.dset_name == "text":
                domain_idx = idx - self.args.num_img_clients
                trainer.test_loader = self.val_dataloader[domain_idx]
                print(f"Client {trainer.dset_name} {idx} tests in domain {domain_idx}:")
                losses, test_top1, test_top5 = trainer.test()
                img_txt_rows.append([round_n, trainer.client_idx, domain_idx, losses, test_top1, test_top5])
            else:
                for domain_idx in range(self.args.num_domains):
                    print(f"Client {trainer.dset_name} {idx} tests in domain {domain_idx}:")
                    test_scores = trainer.evaluate({'test': self.val_dataloader[domain_idx]})
                    metadata = trainer.metadata.copy()
                    metadata['cur_epoch'] = round_n + 1
                    metadata['lr'] = get_lr(trainer.optimizer)
                    trainer.report_scores(step=round_n + 1, scores=test_scores, metadata=metadata)
                    rsum_i = (
                        test_scores['test']['i2t']['recall_1'] +
                        test_scores['test']['t2i']['recall_1'] +
                        test_scores['test']['i2t']['recall_5'] +
                        test_scores['test']['t2i']['recall_5']
                    )
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

        if round_n == self.args.comm_rounds - 1 and self.best_metadata is not None:
            print(f"Final best score: {self.best_score} at epoch {self.best_metadata['best_epoch']}")

        os.makedirs('results', exist_ok=True)
        artifact_tag = self._artifact_tag()
        mm_csv = (
            f'results/mm_{artifact_tag}_{self.args.dataset}_{self.args.alpha}_{str(self.args.cluster_weight)}_'
            f'{self.args.model}_{self.args.local_epochs}_{self.args.comm_rounds}_{self.args.lr}_'
            f'{self.args.n_clusters}_{self.args.rmg_weight}.csv'
        )

        if mm_rows:
            write_header = not os.path.exists(mm_csv)
            with open(mm_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'round', 'client_id', 'domain_idx', 'rsum_i',
                        'n_fold_i2t_r1', 'n_fold_t2i_r1', 'i2t_r1', 't2i_r1',
                        'n_fold_i2t_r5', 'n_fold_t2i_r5', 'i2t_r5', 't2i_r5',
                    ])
                writer.writerows(mm_rows)

        plt.figure()
        plt.plot(range(1, len(self.rsum_history) + 1), self.rsum_history, marker='o')
        plt.xlabel('Round')
        plt.ylabel('rsum')
        if self.best_metadata is not None:
            title = (
                f'rsum Curve ({artifact_tag}, {self.args.model}), best rsum: {self.best_score} '
                f'at round {self.best_metadata["best_epoch"]}'
            )
        else:
            title = f'rsum Curve ({artifact_tag}, {self.args.model})'
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        if self.args.aggregate:
            plt.savefig(
                f'results/rsum_{artifact_tag}_{self.args.dataset}_{self.args.alpha}_{str(self.args.cluster_weight)}_'
                f'{self.args.model}_{self.args.local_epochs}_{self.args.comm_rounds}_{self.args.lr}_{self.args.n_clusters}_aggregate.png'
            )
        else:
            plt.savefig(
                f'results/rsum_{artifact_tag}_{self.args.dataset}_{self.args.alpha}_{str(self.args.cluster_weight)}_'
                f'{self.args.model}_{self.args.local_epochs}_{self.args.comm_rounds}_{self.args.lr}_'
                f'{self.args.n_clusters}_{self.args.rmg_weight}.png'
            )
        plt.close()
        gc.collect()
