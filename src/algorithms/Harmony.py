import gc
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.cluster import KMeans

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.datasets.transform import collate_fn
from src.datasets.load_FL_datasets import get_FL_trainloader
from src.algorithms.ClientTrainer import ClientTrainer
from src.algorithms.MMClientTrainer import MMClientTrainer
from src.algorithms.retrieval_trainer import TrainerEngine
from src.algorithms.mm_eval import MMEvaluator
from src.utils.config import parse_config, apply_runtime_overrides
from src.utils.logger import PythonLogger
from src.utils.model_utils import is_embedding_model

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
        if self.args.dataset == 'imagenet':
            self.class_size = 50
        elif self.args.dataset == 'fashion':
            self.class_size = 48
        elif self.args.dataset == 'food':
            self.class_size = 101
        elif self.args.dataset == 'iapr':
            self.class_size = 30
        else:
            raise ValueError(f'Unsupported dataset for Harmony: {self.args.dataset}')

        self.engine = None
        self.best_score = 0
        self.cur_epoch = 0
        self.best_metadata = None

        self.img_train_loaders, self.txt_train_loaders = None, None
        self.dataloaders_global = None
        self.test_loader = None

        self.config = None
        self.set_config()

        self.logger = PythonLogger(output_file=self.config.train.output_file)
        self.img_vec, self.txt_vec = None, None
        self.global_img_feature = None
        self.global_txt_feature = None
        self.distill_index = None

        self.mm_benchmark_img_states = []
        self.mm_benchmark_txt_states = []
        self.last_cluster_labels = []
        self.last_discrepancy_matrix = []

        self.harmony_stage1_rounds = self._resolve_stage1_rounds()
        self.harmony_benchmark_ready = False

    def _resolve_stage1_rounds(self):
        requested = getattr(self.args, 'harmony_stage1_rounds', 0)
        if requested and requested > 0:
            stage1_rounds = requested
        else:
            stage1_rounds = max(1, self.args.comm_rounds // 2)
            if self.args.comm_rounds > 1:
                stage1_rounds = min(stage1_rounds, self.args.comm_rounds - 1)
        return max(1, min(stage1_rounds, self.args.comm_rounds))

    def _current_stage_name(self, round_n):
        if round_n < self.harmony_stage1_rounds or not self.harmony_benchmark_ready:
            return 'stage1'
        return 'stage2'

    def set_config(self, img='image', txt='text'):
        if self.args.dataset == 'imagenet':
            yaml_name = 'imageNet_cap.yaml'
        elif self.args.dataset == 'fashion':
            yaml_name = 'fashion_gen.yaml'
        elif self.args.dataset == 'food':
            yaml_name = 'umpc_food.yaml'
        elif self.args.dataset == 'iapr':
            yaml_name = 'iapr.yaml'
        self.config = parse_config('./src/' + yaml_name, strict_cast=False)
        self.config = apply_runtime_overrides(self.args, self.config)
        self.config.train.model_save_path = 'model_last_no_prob'
        self.config.train.best_model_save_path = 'model_best_no_prob'
        self.config.train.output_file = f'{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_model_noprob'
        self.config.model.name = self.args.model
        self.config.model.img_client = img
        self.config.model.txt_client = txt
        self.config.train.model_save_path = self.config.train.model_save_path + '.pth'
        self.config.train.best_model_save_path = self.config.train.best_model_save_path + '.pth'
        self.config.train.output_file = self.config.train.output_file + '.log'
        self.config.model.embed_dim = self.args.feature_dim

    def load_dataset(self, args):
        self.engine = TrainerEngine()
        self.engine.set_logger(self.logger)

        self.config.optimizer.learning_rate = self.args.server_lr

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
        self.engine.create(self.config, self.evaluator, self.args.mlp_local)

        self.engine.model_to_device()
        torch.backends.cudnn.enabled = True
        if self.config.train.get('use_fp16'):
            self.engine.logger.log('Train with half precision')
            self.engine.to_half()

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

    def create_model(self, args):
        self.logger.log('start creating model and partition datasets')
        self.device = torch.device('cuda:%d' % args.device)

        os.makedirs('/home/bd/data/zs/data/yClient', exist_ok=True)

        self.img_local_trainers, self.txt_local_trainers, self.mm_local_trainers = [], [], []
        if args.num_img_clients > 0:
            dataset = 'image'
            self.img_trainloaders, test_loaders = get_FL_trainloader(
                dataset,
                self.args.data_root,
                args.num_img_clients,
                self.args.partition,
                self.args.alpha,
                self.args.batch_size,
            )
            for i in range(args.num_img_clients):
                trainer = ClientTrainer(
                    args,
                    dataset,
                    self.class_size,
                    self.logger,
                    inter_distance=4,
                    client_id=i,
                    wandb=self.wandb,
                )
                trainer.train_loader = self.img_trainloaders[i]
                trainer.test_loader = test_loaders[i]
                self.img_local_trainers.append(trainer)
                if is_test and i == 0:
                    break

        if args.num_txt_clients > 0:
            dataset = 'text'
            self.txt_trainloaders, test_loaders = get_FL_trainloader(
                dataset,
                self.args.data_root,
                args.num_txt_clients,
                self.args.partition,
                self.args.alpha,
                self.args.batch_size,
            )
            for i in range(args.num_txt_clients):
                trainer = ClientTrainer(
                    args,
                    dataset,
                    self.class_size,
                    self.logger,
                    inter_distance=4,
                    client_id=i,
                    wandb=self.wandb,
                )
                trainer.train_loader = self.txt_trainloaders[i]
                trainer.test_loader = test_loaders[i]
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
            elif self.args.dataset == 'iapr':
                yaml_name = 'iapr.yaml'
            config = parse_config('./src/' + yaml_name, strict_cast=False)
            config.model.cache_dir = config.model.cache_dir + '-' + config.train.server_dataset
            config.train.output_file = os.path.join(config.model.cache_dir, config.train.output_file)
            config.train.best_model_save_path = os.path.join(config.model.cache_dir, config.train.best_model_save_path)
            config.train.model_save_path = os.path.join(config.model.cache_dir, config.train.model_save_path)
            config.model.embed_dim = self.args.feature_dim
            config.model.name = self.args.model
            for client_id in range(args.num_mm_clients):
                trainer = MMClientTrainer(
                    args,
                    config,
                    self.class_size,
                    self.logger,
                    client=client_id,
                    dset_name='mm',
                    device='cuda',
                    mlp_local=self.args.mlp_local,
                    wandb=self.wandb,
                )
                self.mm_local_trainers.append(trainer)
                if is_test and client_id == 0:
                    break
            print(f"Samples Num: {[len(i.train_loader.dataset) for i in self.mm_local_trainers]}")

        self.total_local_trainers = self.img_local_trainers + self.txt_local_trainers + self.mm_local_trainers
        for i, trainer in enumerate(self.total_local_trainers):
            trainer.client_idx = i

    @staticmethod
    def _average_state_dicts(state_dicts):
        avg_state = OrderedDict()
        if not state_dicts:
            return avg_state
        for key in state_dicts[0].keys():
            params = [state[key] for state in state_dicts if key in state]
            if not params:
                continue
            orig_dtype = params[0].dtype
            avg_state[key] = torch.mean(torch.stack([param.float() for param in params]), dim=0).to(orig_dtype)
        return avg_state

    @staticmethod
    def _cpu_state_dict(state_dict):
        return OrderedDict((key, value.detach().cpu().clone()) for key, value in state_dict.items())

    def aggregate_clip_models(self, local_image_models, local_text_models, local_mm_models):
        server_model = self.engine.model

        image_states = [model.visual_projector.state_dict() for model in local_image_models]
        image_states.extend(model.img_enc.visual_projector.state_dict() for model in local_mm_models)
        if image_states:
            avg_image_state = self._average_state_dicts(image_states)
            prefixed_image_state = OrderedDict((f'visual_projector.{key}', value) for key, value in avg_image_state.items())
            server_model.img_enc.load_state_dict(prefixed_image_state, strict=False)

        text_states = [model.text_projector.state_dict() for model in local_text_models]
        text_states.extend(model.txt_enc.text_projector.state_dict() for model in local_mm_models)
        if text_states:
            avg_text_state = self._average_state_dicts(text_states)
            prefixed_text_state = OrderedDict((f'text_projector.{key}', value) for key, value in avg_text_state.items())
            server_model.txt_enc.load_state_dict(prefixed_text_state, strict=False)

        return server_model

    def aggregate_resnet_models(self, local_image_models, local_text_models, local_mm_models):
        server_model = self.engine.model

        image_states = [model.state_dict() for model in local_image_models]
        image_states.extend(model.img_enc.cnn.state_dict() for model in local_mm_models)
        if image_states:
            avg_image_state = self._average_state_dicts(image_states)
            server_model.img_enc.cnn.load_state_dict(avg_image_state, strict=False)

        text_states = [model.state_dict() for model in local_text_models]
        text_states.extend(model.txt_enc.state_dict() for model in local_mm_models)
        if text_states:
            avg_text_state = self._average_state_dicts(text_states)
            server_model.txt_enc.load_state_dict(avg_text_state, strict=False)

        return server_model

    def _sync_server_to_trainers(self, trainers):
        server_state = self._cpu_state_dict(self.engine.model.state_dict())
        if is_embedding_model(self.args.model):
            img_state = OrderedDict(
                (key.split('img_enc.visual_projector.', 1)[1], value)
                for key, value in server_state.items()
                if key.startswith('img_enc.visual_projector.')
            )
            txt_state = OrderedDict(
                (key.split('txt_enc.text_projector.', 1)[1], value)
                for key, value in server_state.items()
                if key.startswith('txt_enc.text_projector.')
            )
            for trainer in trainers:
                if hasattr(trainer.model, 'img_enc') and hasattr(trainer.model, 'txt_enc'):
                    trainer.model.load_state_dict(server_state, strict=False)
                elif hasattr(trainer.model, 'visual_projector') and img_state:
                    trainer.model.visual_projector.load_state_dict(img_state, strict=False)
                elif hasattr(trainer.model, 'text_projector') and txt_state:
                    trainer.model.text_projector.load_state_dict(txt_state, strict=False)
        elif self.args.model == 'resnet':
            img_state = self._cpu_state_dict(self.engine.model.img_enc.cnn.state_dict())
            txt_state = self._cpu_state_dict(self.engine.model.txt_enc.state_dict())
            for trainer in trainers:
                if hasattr(trainer.model, 'img_enc') and hasattr(trainer.model, 'txt_enc'):
                    trainer.model.load_state_dict(server_state, strict=False)
                elif hasattr(trainer.model, 'state_dict') and hasattr(trainer.model, 'ResNet'):
                    trainer.model.load_state_dict(img_state, strict=False)
                elif hasattr(trainer.model, 'state_dict') and hasattr(trainer.model, 'EncoderText'):
                    trainer.model.load_state_dict(txt_state, strict=False)

    def _snapshot_mm_benchmarks(self):
        self.mm_benchmark_img_states = []
        self.mm_benchmark_txt_states = []
        for trainer in self.mm_local_trainers:
            if is_embedding_model(self.args.model):
                img_state = trainer.model.img_enc.visual_projector.state_dict()
                txt_state = trainer.model.txt_enc.text_projector.state_dict()
            else:
                img_state = trainer.model.img_enc.cnn.state_dict()
                txt_state = trainer.model.txt_enc.state_dict()
            self.mm_benchmark_img_states.append(self._cpu_state_dict(img_state))
            self.mm_benchmark_txt_states.append(self._cpu_state_dict(txt_state))

    @staticmethod
    def _flatten_state_dict(state_dict):
        if not state_dict:
            return np.zeros((0,), dtype=np.float32)
        flat_tensors = [state_dict[key].detach().float().reshape(-1).cpu() for key in sorted(state_dict.keys())]
        if not flat_tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(flat_tensors, dim=0).numpy().astype(np.float32, copy=False)

    def _compute_discrepancy_matrix(self):
        discrepancies = []
        for idx, trainer in enumerate(self.mm_local_trainers):
            if is_embedding_model(self.args.model):
                cur_img = trainer.model.img_enc.visual_projector.state_dict()
                cur_txt = trainer.model.txt_enc.text_projector.state_dict()
            else:
                cur_img = trainer.model.img_enc.cnn.state_dict()
                cur_txt = trainer.model.txt_enc.state_dict()

            bench_img = self.mm_benchmark_img_states[idx]
            bench_txt = self.mm_benchmark_txt_states[idx]

            img_vec = self._flatten_state_dict(cur_img)
            txt_vec = self._flatten_state_dict(cur_txt)
            bench_img_vec = self._flatten_state_dict(bench_img)
            bench_txt_vec = self._flatten_state_dict(bench_txt)

            img_norm = np.linalg.norm(img_vec) * np.linalg.norm(bench_img_vec)
            txt_norm = np.linalg.norm(txt_vec) * np.linalg.norm(bench_txt_vec)
            img_bias = 0.0 if img_norm == 0 else 1.0 - float(np.dot(img_vec, bench_img_vec) / img_norm)
            txt_bias = 0.0 if txt_norm == 0 else 1.0 - float(np.dot(txt_vec, bench_txt_vec) / txt_norm)
            discrepancies.append([img_bias, txt_bias])
        if not discrepancies:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(discrepancies, dtype=np.float32)

    def _normalize_discrepancies(self, discrepancy_matrix):
        if discrepancy_matrix.size == 0:
            return discrepancy_matrix
        normalized = discrepancy_matrix.copy()
        for col_idx in range(normalized.shape[1]):
            max_val = float(np.max(normalized[:, col_idx]))
            if max_val > 1e-12:
                normalized[:, col_idx] /= max_val
            else:
                normalized[:, col_idx] = 0.0
        return normalized

    def _resolve_harmony_clusters(self, normalized_discrepancies):
        if normalized_discrepancies.shape[0] <= 1:
            return 1

        distinct_points = len(np.unique(np.round(normalized_discrepancies, decimals=8), axis=0))
        if distinct_points <= 1:
            return 1

        if getattr(self.args, 'harmony_cluster_mode', 'fixed') == 'svd':
            singular_values = np.linalg.svd(normalized_discrepancies, compute_uv=False)
            if singular_values.size == 0:
                requested = 1
            else:
                threshold = max(float(getattr(self.args, 'harmony_svd_threshold', 0.1)) * singular_values[0], 1e-8)
                requested = int(np.sum(singular_values > threshold))
                requested = max(1, requested)
        else:
            requested = int(getattr(self.args, 'harmony_clusters', 2))

        return max(1, min(requested, normalized_discrepancies.shape[0], distinct_points))

    def _cluster_mm_trainers(self, discrepancy_matrix):
        if discrepancy_matrix.shape[0] == 0:
            return np.empty((0,), dtype=np.int64), discrepancy_matrix
        normalized = self._normalize_discrepancies(discrepancy_matrix)
        n_clusters = self._resolve_harmony_clusters(normalized)
        if n_clusters <= 1:
            labels = np.zeros((normalized.shape[0],), dtype=np.int64)
        else:
            labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(normalized)
        return labels, normalized

    def _apply_clusterwise_mm_aggregation(self, labels):
        if len(self.mm_local_trainers) == 0:
            return
        unique_labels = sorted(set(int(label) for label in labels.tolist()))
        for label in unique_labels:
            members = [trainer for trainer, trainer_label in zip(self.mm_local_trainers, labels.tolist()) if int(trainer_label) == label]
            if not members:
                continue
            if is_embedding_model(self.args.model):
                avg_img_state = self._average_state_dicts([trainer.model.img_enc.visual_projector.state_dict() for trainer in members])
                avg_txt_state = self._average_state_dicts([trainer.model.txt_enc.text_projector.state_dict() for trainer in members])
                for trainer in members:
                    trainer.model.img_enc.visual_projector.load_state_dict(avg_img_state, strict=False)
                    trainer.model.txt_enc.text_projector.load_state_dict(avg_txt_state, strict=False)
            else:
                avg_img_state = self._average_state_dicts([trainer.model.img_enc.cnn.state_dict() for trainer in members])
                avg_txt_state = self._average_state_dicts([trainer.model.txt_enc.state_dict() for trainer in members])
                for trainer in members:
                    trainer.model.img_enc.cnn.load_state_dict(avg_img_state, strict=False)
                    trainer.model.txt_enc.load_state_dict(avg_txt_state, strict=False)

    def _refresh_server_from_mm_clients(self):
        if not self.mm_local_trainers:
            return
        if is_embedding_model(self.args.model):
            avg_img_state = self._average_state_dicts([trainer.model.img_enc.visual_projector.state_dict() for trainer in self.mm_local_trainers])
            avg_txt_state = self._average_state_dicts([trainer.model.txt_enc.text_projector.state_dict() for trainer in self.mm_local_trainers])
            self.engine.model.img_enc.load_state_dict(OrderedDict((f'visual_projector.{key}', value) for key, value in avg_img_state.items()), strict=False)
            self.engine.model.txt_enc.load_state_dict(OrderedDict((f'text_projector.{key}', value) for key, value in avg_txt_state.items()), strict=False)
        elif self.args.model == 'resnet':
            avg_img_state = self._average_state_dicts([trainer.model.img_enc.cnn.state_dict() for trainer in self.mm_local_trainers])
            avg_txt_state = self._average_state_dicts([trainer.model.txt_enc.state_dict() for trainer in self.mm_local_trainers])
            self.engine.model.img_enc.cnn.load_state_dict(avg_img_state, strict=False)
            self.engine.model.txt_enc.load_state_dict(avg_txt_state, strict=False)

    def _run_stage1(self, round_n):
        local_image_models, local_text_models, local_mm_models = [], [], []
        for trainer in self.total_local_trainers:
            self.logger.log(f'Training Client {trainer.client_idx}!')
            trainer.cur_epoch = round_n
            trainer.run()
            if trainer.dset_name == 'image':
                local_image_models.append(trainer.model)
            elif trainer.dset_name == 'text':
                local_text_models.append(trainer.model)
            else:
                local_mm_models.append(trainer.model)

        if is_embedding_model(self.args.model):
            self.engine.model = self.aggregate_clip_models(local_image_models, local_text_models, local_mm_models)
        elif self.args.model == 'resnet':
            self.engine.model = self.aggregate_resnet_models(local_image_models, local_text_models, local_mm_models)

        self._sync_server_to_trainers(self.total_local_trainers)
        self._snapshot_mm_benchmarks()
        self.harmony_benchmark_ready = True
        self.logger.log(f'Harmony stage1 completed at round {round_n + 1}; benchmark encoders refreshed.')

    def _run_stage2(self, round_n):
        for trainer in self.mm_local_trainers:
            self.logger.log(f'Training Client {trainer.client_idx}!')
            trainer.cur_epoch = round_n
            trainer.run()

        discrepancy_matrix = self._compute_discrepancy_matrix()
        labels, normalized_discrepancies = self._cluster_mm_trainers(discrepancy_matrix)
        self.last_cluster_labels = labels.tolist()
        self.last_discrepancy_matrix = normalized_discrepancies.tolist()
        if labels.size > 0:
            self.logger.log(
                f'Harmony stage2 clustering round {round_n + 1}: labels={self.last_cluster_labels}, '
                f'discrepancies={self.last_discrepancy_matrix}'
            )
        self._apply_clusterwise_mm_aggregation(labels)
        self._refresh_server_from_mm_clients()

    @staticmethod
    def _get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
        return None

    def _evaluate(self, round_n):
        import csv
        import matplotlib.pyplot as plt

        if not hasattr(self, 'img_txt_results'):
            self.img_txt_results = []
        if not hasattr(self, 'mm_results'):
            self.mm_results = []
        if not hasattr(self, 'rsum_history'):
            self.rsum_history = []

        img_txt_rows = []
        mm_rows = []

        print('Testing...')
        rsum = 0
        last_metadata = {'stage': self._current_stage_name(round_n)}
        for idx, trainer in enumerate(self.total_local_trainers):
            if trainer.dset_name == 'image':
                domain_idx = idx
                trainer.test_loader = self.val_dataloader[idx]
                print(f'Client {trainer.dset_name} {idx} tests in domain {idx}:')
                losses, test_top1, test_top5 = trainer.test()
                img_txt_rows.append([round_n, trainer.client_idx, domain_idx, losses, test_top1, test_top5])
            elif trainer.dset_name == 'text':
                domain_idx = idx - self.args.num_img_clients
                trainer.test_loader = self.val_dataloader[domain_idx]
                print(f'Client {trainer.dset_name} {idx} tests in domain {domain_idx}:')
                losses, test_top1, test_top5 = trainer.test()
                img_txt_rows.append([round_n, trainer.client_idx, domain_idx, losses, test_top1, test_top5])
            else:
                for domain_idx in range(self.args.num_domains):
                    print(f'Client {trainer.dset_name} {idx} tests in domain {domain_idx}:')
                    test_scores = trainer.evaluate({'test': self.val_dataloader[domain_idx]})
                    metadata = trainer.metadata.copy()
                    metadata['cur_epoch'] = round_n + 1
                    metadata['lr'] = self._get_lr(trainer.optimizer)
                    metadata['stage'] = self._current_stage_name(round_n)
                    mm_idx = idx - self.args.num_img_clients - self.args.num_txt_clients
                    if self.last_cluster_labels and 0 <= mm_idx < len(self.last_cluster_labels):
                        metadata['cluster_label'] = self.last_cluster_labels[mm_idx]
                    trainer.report_scores(step=round_n + 1, scores=test_scores, metadata=metadata)
                    last_metadata = metadata
                    rsum_i = (
                        test_scores['test']['i2t']['recall_1']
                        + test_scores['test']['t2i']['recall_1']
                        + test_scores['test']['i2t']['recall_5']
                        + test_scores['test']['t2i']['recall_5']
                    )
                    if domain_idx == mm_idx:
                        rsum += rsum_i
                    mm_rows.append([
                        round_n,
                        trainer.client_idx,
                        domain_idx,
                        rsum_i,
                        test_scores['test']['n_fold']['i2t']['recall_1'],
                        test_scores['test']['n_fold']['t2i']['recall_1'],
                        test_scores['test']['i2t']['recall_1'],
                        test_scores['test']['t2i']['recall_1'],
                        test_scores['test']['n_fold']['i2t']['recall_5'],
                        test_scores['test']['n_fold']['t2i']['recall_5'],
                        test_scores['test']['i2t']['recall_5'],
                        test_scores['test']['t2i']['recall_5'],
                    ])
                    self.wandb.log({f'Multimodal_{mm_idx} rsum_r1': rsum_i}, step=self.cur_epoch)
                    self.wandb.log({f'Multimodal_{mm_idx} n_fold_i2t_r1': test_scores['test']['n_fold']['i2t']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f'Multimodal_{mm_idx} n_fold_t2i_r1': test_scores['test']['n_fold']['t2i']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f'Multimodal_{mm_idx} i2t_r1': test_scores['test']['i2t']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f'Multimodal_{mm_idx} t2i_r1': test_scores['test']['t2i']['recall_1']}, step=self.cur_epoch)

        self.rsum_history.append(rsum)
        if self.best_score < rsum or self.best_metadata is None:
            best_score = rsum
            best_metadata = dict(last_metadata)
            best_metadata['best_score'] = best_score
            best_metadata['best_epoch'] = round_n + 1
            self.best_metadata, self.best_score = best_metadata, best_score
            print(f'Best score updated: {best_score} at epoch {round_n + 1}')

        if round_n == self.args.comm_rounds - 1 and self.best_metadata is not None:
            print(f"Final best score: {self.best_score} at epoch {self.best_metadata['best_epoch']}")

        os.makedirs('results', exist_ok=True)
        mm_csv = f'results/{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_mm_{self.args.FL_algorithm}.csv'
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

        plt.figure()
        plt.plot(range(1, len(self.rsum_history) + 1), self.rsum_history, marker='o')
        plt.xlabel('Round')
        plt.ylabel('rsum')
        best_epoch = self.best_metadata['best_epoch'] if self.best_metadata is not None else 'N/A'
        plt.title(f'rsum Curve (Best: {self.best_score} at epoch {best_epoch})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/rsum_{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}.png')
        plt.close()
        print(f'Rsum at round {round_n} is {self.rsum_history[-1]}')
        gc.collect()

    def train(self, round_n):
        self.cur_epoch = round_n
        if not is_test:
            self.logger.log(f'Round {round_n + 1}!')
            self.logger.log(f'Harmony enters {self._current_stage_name(round_n)} at round {round_n + 1}.')

        if round_n < self.harmony_stage1_rounds or not self.harmony_benchmark_ready:
            self._run_stage1(round_n)
        else:
            self._run_stage2(round_n)

        self._evaluate(round_n)
