import gc
import os
import random
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.utils.model_utils import is_embedding_model

try:
    from src.algorithms.fedmektClientTrainer import FedMEKTClientTrainer
    from src.algorithms.fedmektMMClientTrainer import FedMEKTMMClientTrainer
    from src.algorithms.fedmobile_utils import sanitize_module_parameters
    from src.algorithms.mm_eval import MMEvaluator
    from src.algorithms.retrieval_trainer import TrainerEngine
    from src.datasets.load_FL_datasets import get_FL_trainloader
    from src.datasets.transform import collate_fn
    from src.utils.config import parse_config, apply_runtime_overrides
    from src.utils.load_datasets import prepare_coco_dataloaders
    from src.utils.logger import PythonLogger
except ImportError:
    from algorithms.fedmektClientTrainer import FedMEKTClientTrainer
    from algorithms.fedmektMMClientTrainer import FedMEKTMMClientTrainer
    from algorithms.fedmobile_utils import sanitize_module_parameters
    from algorithms.mm_eval import MMEvaluator
    from algorithms.retrieval_trainer import TrainerEngine
    from datasets.load_FL_datasets import get_FL_trainloader
    from datasets.transform import collate_fn
    from utils.config import parse_config
    from utils.load_datasets import prepare_coco_dataloaders
    from utils.logger import PythonLogger


class MMFL(object):
    def __init__(self, args, wandb=None):
        self.args = args
        self.wandb = wandb
        if not is_embedding_model(self.args.model):
            raise NotImplementedError('FedMEKT currently supports precomputed embedding pathways: clip, align, siglip.')

        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        self.img_local_trainers = []
        self.txt_local_trainers = []
        self.mm_local_trainers = []
        self.total_local_trainers = []
        self.cur_trainers = []
        self.engine = None
        self.val_dataloader = {}
        self.proxy_loader = None
        self.server_distill_loader = None
        self.proxy_size = 0
        self.proxy_index_lookup = None
        self.server_proxy_knowledge = {'image': None, 'text': None}
        self.best_score = 0
        self.best_metadata = None
        self.rsum_history = []
        self.cur_epoch = 0
        self.mse_loss = nn.MSELoss()

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
        self.config.train.output_file = f'{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_model_noprob.log'
        self.config.train.use_fp16 = False
        self.config.model.name = self.args.model
        self.config.model.img_client = img
        self.config.model.txt_client = txt
        self.config.model.embed_dim = self.args.feature_dim

    def _build_proxy_loader(self):
        coco_root = '/home/bd/data/zs/data/mmdata/MSCOCO/2014'
        vocab_path = './src/datasets/vocabs/coco_vocab.pkl'
        self.proxy_index_lookup = None
        if os.path.exists(coco_root) and os.path.exists(vocab_path):
            self.dataloaders_global, self.vocab = prepare_coco_dataloaders(
                self.config.dataloader,
                coco_root,
                vocab_path,
                pub_data_num=self.args.pub_data_num,
                feature_dim=self.args.feature_dim,
                clip_model_name=getattr(self.config.model, 'clip_model', 'RN50'),
                model_name=self.args.model,
                cache_device=str(self.device),
            )
            proxy_key = 'train_subset_eval' + f'_{self.args.pub_data_num}'
            train_key = 'train_subset' + f'_{self.args.pub_data_num}'
            self.proxy_loader = self.dataloaders_global[proxy_key]
            self.server_distill_loader = self.dataloaders_global[train_key]
            try:
                self.proxy_size = len(self.proxy_loader.dataset)
            except TypeError:
                self.proxy_size = len(self.proxy_loader)
            dataset = getattr(self.proxy_loader, 'dataset', None)
            subset_indices = getattr(dataset, 'indices', None)
            if subset_indices is not None:
                self.proxy_index_lookup = {
                    int(original_index): mapped_index
                    for mapped_index, original_index in enumerate(subset_indices)
                }
            return

        proxy_datasets = []
        for domain_idx in range(self.args.num_domains):
            train_dataset = load_from_disk(os.path.join(self.args.data_root, f'domain_dataset_{domain_idx}', 'train'))
            proxy_datasets.append(train_dataset)

        proxy_dataset = concatenate_datasets(proxy_datasets)
        proxy_num = min(len(proxy_dataset), max(1, self.args.pub_data_num))
        proxy_dataset = proxy_dataset.shuffle(seed=self.args.seed).select(range(proxy_num))
        proxy_dataset = proxy_dataset.add_column('proxy_index', list(range(proxy_num)))
        self.proxy_size = len(proxy_dataset)
        self.proxy_loader = torch.utils.data.DataLoader(
            proxy_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        self.server_distill_loader = self.proxy_loader

    def _normalize_proxy_index(self, proxy_index):
        if self.proxy_index_lookup is None:
            return proxy_index

        if isinstance(proxy_index, torch.Tensor):
            raw_indices = proxy_index.view(-1).tolist()
            mapped_indices = [self.proxy_index_lookup[int(index)] for index in raw_indices]
            return torch.tensor(mapped_indices, dtype=torch.long)

        return torch.tensor(
            [self.proxy_index_lookup[int(index)] for index in proxy_index],
            dtype=torch.long,
        )

    def _proxy_batch_to_inputs(self, data):
        if isinstance(data, dict):
            images = data["processed_img"].to(self.device)
            captions = data["cap_tokens"].to(self.device)
            proxy_index = torch.tensor(data["proxy_index"], dtype=torch.long)
            return images, captions, proxy_index

        images, captions, _, _, _, _, index = data
        proxy_index = self._normalize_proxy_index(torch.as_tensor(index, dtype=torch.long))
        return images.to(self.device), captions.to(self.device), proxy_index

    def _extract_server_proxy_knowledge(self):
        image_embeddings = torch.zeros(self.proxy_size, self.args.feature_dim, dtype=torch.float32)
        text_embeddings = torch.zeros(self.proxy_size, self.args.feature_dim, dtype=torch.float32)

        self.engine.model_to_device()
        self.engine.model.eval()
        with torch.no_grad():
            for data in self.proxy_loader:
                images, captions, proxy_index = self._proxy_batch_to_inputs(data)
                output = self.engine.model(images, captions)
                image_features = F.normalize(
                    torch.nan_to_num(output['image_features'].float(), nan=0.0, posinf=1e4, neginf=-1e4),
                    dim=-1,
                ).cpu()
                text_features = F.normalize(
                    torch.nan_to_num(output['caption_features'].float(), nan=0.0, posinf=1e4, neginf=-1e4),
                    dim=-1,
                ).cpu()
                image_embeddings[proxy_index] = image_features
                text_embeddings[proxy_index] = text_features

        self.engine.model.train()
        return {'image': image_embeddings, 'text': text_embeddings}

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

        self._build_proxy_loader()
        for trainer in self.total_local_trainers:
            trainer.set_proxy_loader(self.proxy_loader)
        self.server_proxy_knowledge = self._extract_server_proxy_knowledge()

    def create_model(self, args):
        self.logger.log('start creating model and partition datasets')

        if args.num_img_clients > 0:
            dataset = 'image'
            self.img_trainloaders, test_loaders = get_FL_trainloader(
                dataset, self.args.data_root, args.num_img_clients, self.args.partition, self.args.alpha, self.args.batch_size
            )
            for i in range(args.num_img_clients):
                trainer = FedMEKTClientTrainer(
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
                trainer = FedMEKTClientTrainer(
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
            config.model.embed_dim = self.args.feature_dim
            config.model.name = self.args.model
            for client_id in range(args.num_mm_clients):
                self.mm_local_trainers.append(
                    FedMEKTMMClientTrainer(
                        args, config, self.class_size, self.logger, client=client_id,
                        dset_name="mm", device='cuda', mlp_local=self.args.mlp_local, wandb=self.wandb,
                    )
                )

        self.total_local_trainers = self.img_local_trainers + self.txt_local_trainers + self.mm_local_trainers
        for idx, trainer in enumerate(self.total_local_trainers):
            trainer.client_idx = idx
            trainer.set_proxy_loader(self.proxy_loader)

    def _aggregate_client_proxy_knowledge(self, selected_trainers):
        image_sum = torch.zeros(self.proxy_size, self.args.feature_dim, dtype=torch.float32)
        text_sum = torch.zeros(self.proxy_size, self.args.feature_dim, dtype=torch.float32)
        image_weight = 0.0
        text_weight = 0.0

        for trainer in selected_trainers:
            upload = trainer.export_proxy_embeddings()
            client_weight = float(len(trainer.train_loader.dataset)) if hasattr(trainer.train_loader, 'dataset') else 1.0
            if upload['image'] is not None:
                image_sum += upload['image'].float() * client_weight
                image_weight += client_weight
            if upload['text'] is not None:
                text_sum += upload['text'].float() * client_weight
                text_weight += client_weight

        if image_weight > 0:
            image_target = F.normalize(image_sum / image_weight, dim=-1)
        else:
            image_target = self.server_proxy_knowledge['image']
        if text_weight > 0:
            text_target = F.normalize(text_sum / text_weight, dim=-1)
        else:
            text_target = self.server_proxy_knowledge['text']

        return {'image': image_target.cpu(), 'text': text_target.cpu()}

    def _server_alignment_loss(self, source_embeddings, target_embeddings):
        source_norm = F.normalize(source_embeddings.float(), dim=-1)
        target_norm = F.normalize(target_embeddings.float(), dim=-1)
        return self.mse_loss(source_norm, target_norm)

    def _server_distill_round(self, aggregated_targets):
        self.engine.model_to_device()
        self.engine.model.train()
        server_epochs = max(1, getattr(self.args, 'fedmekt_server_proxy_epochs', 1))

        for _ in range(server_epochs):
            for data in self.server_distill_loader:
                images, captions, proxy_index_cpu = self._proxy_batch_to_inputs(data)
                proxy_index = proxy_index_cpu.to(self.device)

                output = self.engine.model(images, captions)
                retrieval_loss, _ = self.engine.criterion(**output)
                image_target = aggregated_targets['image'][proxy_index_cpu].to(self.device)
                text_target = aggregated_targets['text'][proxy_index_cpu].to(self.device)
                align_loss = self._server_alignment_loss(output['image_features'], image_target)
                align_loss = align_loss + self._server_alignment_loss(output['caption_features'], text_target)

                loss = getattr(self.args, 'fedmekt_server_retrieval_weight', 1.0) * retrieval_loss
                loss = loss + getattr(self.args, 'fedmekt_server_align_weight', 1.0) * align_loss
                if not torch.isfinite(loss):
                    continue

                self.engine.optimizer.zero_grad()
                loss.backward()
                if self.config.train.grad_clip > 0:
                    nn.utils.clip_grad.clip_grad_norm_(self.engine.model.parameters(), self.config.train.grad_clip)
                self.engine.optimizer.step()
                sanitize_module_parameters(self.engine.model)

        self.server_proxy_knowledge = self._extract_server_proxy_knowledge()

    def _evaluate_round(self, round_n):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        import csv

        mm_rows = []
        rsum = 0

        for idx, trainer in enumerate(self.total_local_trainers):
            if trainer.dset_name == "image":
                domain_idx = idx
                trainer.test_loader = self.val_dataloader[domain_idx]
                trainer.test()
            elif trainer.dset_name == "text":
                domain_idx = idx - self.args.num_img_clients
                trainer.test_loader = self.val_dataloader[domain_idx]
                trainer.test()

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
        mm_csv = f'results/{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_server_{self.args.FL_algorithm}.csv'
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
        plt.savefig(f'results/rsum_{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}.png')
        plt.close()

    def train(self, round_n):
        self.cur_epoch = round_n
        self.cur_trainers = self.total_local_trainers
        self.logger.log(f"Round {round_n + 1}!")

        if len(self.total_local_trainers) > self.args.client_num_per_round:
            self.cur_trainers = random.sample(self.total_local_trainers, self.args.client_num_per_round)

        for trainer in self.cur_trainers:
            trainer.cur_epoch = round_n
            trainer.set_server_proxy_targets(
                image_targets=self.server_proxy_knowledge['image'],
                text_targets=self.server_proxy_knowledge['text'],
            )
            trainer.run()

        aggregated_targets = self._aggregate_client_proxy_knowledge(self.cur_trainers)
        self._server_distill_round(aggregated_targets)
        self._evaluate_round(round_n)
        gc.collect()
