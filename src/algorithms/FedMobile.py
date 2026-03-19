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

try:
    from src.algorithms.fedmobileClientTrainer import FedMobileClientTrainer
    from src.algorithms.fedmobileMMClientTrainer import FedMobileMMClientTrainer
    from src.algorithms.fedmobile_utils import (
        LabelConditionalGenerator,
        clone_state_dict,
        cluster_generator_states,
        sanitize_module_parameters,
        shapley_values,
        weighted_average_state_dicts,
    )
    from src.algorithms.mm_eval import MMEvaluator
    from src.algorithms.retrieval_trainer import TrainerEngine
    from src.datasets.load_FL_datasets import get_FL_trainloader
    from src.datasets.transform import collate_fn
    from src.utils.config import parse_config, apply_runtime_overrides
    from src.utils.logger import PythonLogger
except ImportError:
    from algorithms.fedmobileClientTrainer import FedMobileClientTrainer
    from algorithms.fedmobileMMClientTrainer import FedMobileMMClientTrainer
    from algorithms.fedmobile_utils import (
        LabelConditionalGenerator,
        clone_state_dict,
        cluster_generator_states,
        sanitize_module_parameters,
        shapley_values,
        weighted_average_state_dicts,
    )
    from algorithms.mm_eval import MMEvaluator
    from algorithms.retrieval_trainer import TrainerEngine
    from datasets.load_FL_datasets import get_FL_trainloader
    from datasets.transform import collate_fn
    from utils.config import parse_config
    from utils.logger import PythonLogger


class MMFL(object):
    def __init__(self, args, wandb=None):
        self.args = args
        self.wandb = wandb
        if self.args.model != 'clip':
            raise NotImplementedError('FedMobile is implemented for the CLIP pathway in this repository.')

        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        self.img_local_trainers = []
        self.txt_local_trainers = []
        self.mm_local_trainers = []
        self.total_local_trainers = []
        self.cur_trainers = []
        self.engine = None
        self.val_dataloader = {}
        self.best_score = 0
        self.best_metadata = None
        self.rsum_history = []
        self.cur_epoch = 0

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

        self.config = None
        self.set_config()
        self.logger = PythonLogger(output_file=self.config.train.output_file)
        self.global_generator_state = clone_state_dict(self._generator_factory().state_dict())
        self.global_image_head_state = None
        self.global_text_head_state = None

    def _generator_factory(self):
        return LabelConditionalGenerator(
            num_classes=self.class_size,
            embed_dim=self.args.feature_dim,
            noise_dim=getattr(self.args, 'fedmobile_noise_dim', 128),
            hidden_dim=getattr(self.args, 'fedmobile_hidden_dim', self.args.feature_dim * 2),
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
        self.config.train.output_file = 'model_noprob.log'
        self.config.train.use_fp16 = False
        self.config.model.name = self.args.model
        self.config.model.img_client = img
        self.config.model.txt_client = txt
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

        for domain_idx in range(args.num_domains):
            val_dataset = load_from_disk(os.path.join(self.args.data_root, f'domain_dataset_{domain_idx}', 'test'))
            self.val_dataloader[domain_idx] = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )

    def create_model(self, args):
        self.logger.log('start creating model and partition datasets')

        if args.num_img_clients > 0:
            dataset = 'image'
            self.img_trainloaders, test_loaders = get_FL_trainloader(
                dataset, self.args.data_root, args.num_img_clients, self.args.partition, self.args.alpha, self.args.batch_size
            )
            for i in range(args.num_img_clients):
                trainer = FedMobileClientTrainer(
                    args, dataset, self.class_size, self.logger, inter_distance=4, client_id=i, wandb=self.wandb
                )
                trainer.train_loader = self.img_trainloaders[i]
                trainer.test_loader = test_loaders[i]
                self.img_local_trainers.append(trainer)

        if args.num_txt_clients > 0:
            dataset = 'text'
            self.txt_trainloaders, test_loaders = get_FL_trainloader(
                dataset, self.args.data_root, args.num_txt_clients, self.args.partition, self.args.alpha, self.args.batch_size
            )
            for i in range(args.num_txt_clients):
                trainer = FedMobileClientTrainer(
                    args, dataset, self.class_size, self.logger, inter_distance=4, client_id=i, wandb=self.wandb
                )
                trainer.train_loader = self.txt_trainloaders[i]
                trainer.test_loader = test_loaders[i]
                self.txt_local_trainers.append(trainer)

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
            config.train.use_fp16 = False
            config.dataloader.batch_size = self.args.batch_size
            config.dataloader.eval_batch_size = self.args.batch_size
            config.optimizer.name = 'adam'
            config.optimizer.learning_rate = getattr(self.args, 'fedmobile_mm_lr', self.args.fedmobile_gen_lr)
            config.optimizer.weight_decay = self.args.weight_decay
            config.model.embed_dim = self.args.feature_dim
            config.model.name = self.args.model
            for client_id in range(args.num_mm_clients):
                self.mm_local_trainers.append(
                    FedMobileMMClientTrainer(
                        args, config, self.class_size, self.logger, client=client_id,
                        dset_name="mm", device='cuda', mlp_local=self.args.mlp_local, wandb=self.wandb,
                    )
                )

        self.total_local_trainers = self.img_local_trainers + self.txt_local_trainers + self.mm_local_trainers
        for idx, trainer in enumerate(self.total_local_trainers):
            trainer.client_idx = idx
        if self.img_local_trainers:
            self.global_image_head_state = clone_state_dict(self.img_local_trainers[0].model.class_fc_2.state_dict())
        if self.txt_local_trainers:
            self.global_text_head_state = clone_state_dict(self.txt_local_trainers[0].model.class_fc_2.state_dict())

    def _sync_server_model_to_client(self, trainer):
        server_model = self.engine.model
        if trainer.dset_name == 'image':
            trainer.model.visual_projector.load_state_dict(server_model.img_enc.visual_projector.state_dict(), strict=True)
            if self.global_image_head_state is not None:
                trainer.model.class_fc_2.load_state_dict(self.global_image_head_state, strict=True)
            sanitize_module_parameters(trainer.model.visual_projector)
            sanitize_module_parameters(trainer.model.class_fc_2)
        elif trainer.dset_name == 'text':
            trainer.model.text_projector.load_state_dict(server_model.txt_enc.text_projector.state_dict(), strict=True)
            if self.global_text_head_state is not None:
                trainer.model.class_fc_2.load_state_dict(self.global_text_head_state, strict=True)
            sanitize_module_parameters(trainer.model.text_projector)
            sanitize_module_parameters(trainer.model.class_fc_2)
        else:
            trainer.model.img_enc.load_state_dict(server_model.img_enc.state_dict(), strict=False)
            trainer.model.txt_enc.load_state_dict(server_model.txt_enc.state_dict(), strict=False)
            sanitize_module_parameters(trainer.model.img_enc)
            sanitize_module_parameters(trainer.model.txt_enc)

    def _extract_image_projector_state(self, trainer):
        if trainer.dset_name == 'image':
            return clone_state_dict(trainer.model.visual_projector.state_dict())
        if trainer.dset_name == 'mm':
            return clone_state_dict(trainer.model.img_enc.visual_projector.state_dict())
        return None

    def _extract_text_projector_state(self, trainer):
        if trainer.dset_name == 'text':
            return clone_state_dict(trainer.model.text_projector.state_dict())
        if trainer.dset_name == 'mm':
            return clone_state_dict(trainer.model.txt_enc.text_projector.state_dict())
        return None

    def _extract_image_head_state(self, trainer):
        if trainer.dset_name == 'image':
            return clone_state_dict(trainer.model.class_fc_2.state_dict())
        return None

    def _extract_text_head_state(self, trainer):
        if trainer.dset_name == 'text':
            return clone_state_dict(trainer.model.class_fc_2.state_dict())
        return None

    def _aggregate_projector_branches(self, selected_trainers, client_weights):
        image_states, image_weights = [], []
        text_states, text_weights = [], []
        image_head_states, image_head_weights = [], []
        text_head_states, text_head_weights = [], []

        for trainer in selected_trainers:
            weight = client_weights.get(trainer.client_idx, 0.0)
            if weight <= 0:
                continue
            image_state = self._extract_image_projector_state(trainer)
            if image_state is not None:
                image_states.append(image_state)
                image_weights.append(weight)
            text_state = self._extract_text_projector_state(trainer)
            if text_state is not None:
                text_states.append(text_state)
                text_weights.append(weight)
            image_head_state = self._extract_image_head_state(trainer)
            if image_head_state is not None:
                image_head_states.append(image_head_state)
                image_head_weights.append(weight)
            text_head_state = self._extract_text_head_state(trainer)
            if text_head_state is not None:
                text_head_states.append(text_head_state)
                text_head_weights.append(weight)

        if image_states:
            averaged_image = weighted_average_state_dicts(image_states, image_weights)
            self.engine.model.img_enc.visual_projector.load_state_dict(averaged_image, strict=True)
            sanitize_module_parameters(self.engine.model.img_enc.visual_projector)
        if text_states:
            averaged_text = weighted_average_state_dicts(text_states, text_weights)
            self.engine.model.txt_enc.text_projector.load_state_dict(averaged_text, strict=True)
            sanitize_module_parameters(self.engine.model.txt_enc.text_projector)
        if image_head_states:
            self.global_image_head_state = weighted_average_state_dicts(image_head_states, image_head_weights)
        if text_head_states:
            self.global_text_head_state = weighted_average_state_dicts(text_head_states, text_head_weights)

    def _compute_client_contributions(self, selected_trainers):
        generator_states = [trainer.get_generator_state() for trainer in selected_trainers]
        if len(generator_states) == 1:
            trainer = selected_trainers[0]
            weight = float(len(trainer.train_loader.dataset))
            self.global_generator_state = generator_states[0]
            return {trainer.client_idx: weight}

        probe_count = min(self.class_size, getattr(self.args, 'fedmobile_probe_labels', self.class_size))
        probe_labels = torch.arange(probe_count, dtype=torch.long)
        _, clusters = cluster_generator_states(
            generator_states,
            self._generator_factory,
            probe_labels,
            self.device,
            getattr(self.args, 'fedmobile_num_clusters', 3),
        )

        cluster_states = []
        cluster_members = []
        for _, members in sorted(clusters.items(), key=lambda item: item[0]):
            cluster_states.append(weighted_average_state_dicts([generator_states[idx] for idx in members]))
            cluster_members.append(members)

        utility_cache = {}

        def utility(subset):
            if not subset:
                return 0.0
            if subset not in utility_cache:
                aggregated_generator = weighted_average_state_dicts([cluster_states[idx] for idx in subset])
                scores = [
                    trainer.evaluate_generator_utility(
                        aggregated_generator,
                        max_batches=getattr(self.args, 'fedmobile_eval_batches', 1),
                    )
                    for trainer in self.total_local_trainers
                ]
                scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                utility_cache[subset] = float(np.mean(scores))
            return utility_cache[subset]

        cluster_weights = shapley_values(
            len(cluster_states),
            utility,
            max_permutations=getattr(self.args, 'fedmobile_shapley_samples', 24),
            seed=self.args.seed,
        )

        self.global_generator_state = weighted_average_state_dicts(cluster_states, cluster_weights)

        client_weights = {}
        for cluster_idx, members in enumerate(cluster_members):
            base_weight = cluster_weights[cluster_idx] / max(1, len(members))
            for member in members:
                trainer = selected_trainers[member]
                data_size = len(trainer.train_loader.dataset) if hasattr(trainer.train_loader, 'dataset') else 1
                client_weights[trainer.client_idx] = base_weight * float(data_size)
        return client_weights

    def _normalize_weights(self, weights):
        total = sum(weights.values())
        if total <= 0:
            uniform = 1.0 / max(1, len(weights))
            return {key: uniform for key in weights}
        return {key: value / total for key, value in weights.items()}

    def _evaluate_round(self, round_n):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        import csv

        mm_rows = []
        img_txt_rows = []
        rsum = 0

        for idx, trainer in enumerate(self.total_local_trainers):
            if trainer.dset_name == "image":
                domain_idx = idx
                trainer.test_loader = self.val_dataloader[domain_idx]
                losses, test_top1, test_top5 = trainer.test()
                img_txt_rows.append([round_n, trainer.client_idx, domain_idx, losses, test_top1, test_top5])
            elif trainer.dset_name == "text":
                domain_idx = idx - self.args.num_img_clients
                trainer.test_loader = self.val_dataloader[domain_idx]
                losses, test_top1, test_top5 = trainer.test()
                img_txt_rows.append([round_n, trainer.client_idx, domain_idx, losses, test_top1, test_top5])

        for domain_idx in range(self.args.num_domains):
            test_scores = self.engine.evaluate({'test': self.val_dataloader[domain_idx]})
            metadata = self.engine.metadata.copy()
            metadata['cur_epoch'] = round_n + 1
            metadata['lr'] = get_lr(self.engine.optimizer)
            self.engine.report_scores(step=round_n + 1, scores=test_scores, metadata=metadata)

            rsum_i = (
                test_scores['test']['i2t']['recall_1'] +
                test_scores['test']['t2i']['recall_1'] +
                test_scores['test']['i2t']['recall_5'] +
                test_scores['test']['t2i']['recall_5']
            )
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

        self.rsum_history.append(rsum)
        if self.best_score < rsum:
            metadata['best_score'] = rsum
            metadata['best_epoch'] = round_n + 1
            self.best_metadata = metadata
            self.best_score = rsum

        os.makedirs('results', exist_ok=True)

        mm_csv = f'results/server_{self.args.FL_algorithm}.csv'
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

        plt.figure()
        plt.plot(range(1, len(self.rsum_history) + 1), self.rsum_history, marker='o')
        plt.xlabel('Round')
        plt.ylabel('rsum')
        if self.best_metadata is not None:
            plt.title(f'rsum Curve (Best: {self.best_score} at epoch {self.best_metadata["best_epoch"]})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f'results/rsum_{self.args.FL_algorithm}_{self.args.dataset}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}.png'
        )
        plt.close()

    def train(self, round_n):
        self.cur_epoch = round_n
        self.cur_trainers = self.total_local_trainers
        self.logger.log(f"Round {round_n + 1}!")

        if len(self.total_local_trainers) > self.args.client_num_per_round:
            self.cur_trainers = random.sample(self.total_local_trainers, self.args.client_num_per_round)

        for trainer in self.cur_trainers:
            self._sync_server_model_to_client(trainer)
            trainer.cur_epoch = round_n
            trainer.run(server_generator_state=self.global_generator_state)

        active_trainers = []
        for trainer in self.cur_trainers:
            stats = getattr(trainer, 'last_epoch_stats', {})
            optimized = stats.get('optimized_batches', 0)
            total = max(1, stats.get('total_batches', 0))
            if optimized <= 0:
                continue
            if optimized / total < 0.5:
                self.logger.log(
                    f"FedMobile skipping client {trainer.client_idx} from aggregation due to low valid-batch ratio: {optimized}/{total}"
                )
                continue
            active_trainers.append(trainer)
        if not active_trainers:
            self.logger.log('FedMobile warning: no active trainers with finite updates in this round; keeping previous global state.')
            self._evaluate_round(round_n)
            gc.collect()
            return

        client_weights = self._compute_client_contributions(active_trainers)
        client_weights = self._normalize_weights(client_weights)
        self.logger.log(f'FedMobile active clients: {[trainer.client_idx for trainer in active_trainers]}')
        self.logger.log(f'FedMobile client weights: {client_weights}')
        self._aggregate_projector_branches(active_trainers, client_weights)
        self._evaluate_round(round_n)
        gc.collect()
