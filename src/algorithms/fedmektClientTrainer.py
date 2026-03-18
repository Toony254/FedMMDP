import gc
import itertools
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

try:
    from src.algorithms.ClientTrainer import ClientTrainer
    from src.algorithms.fedmobile_utils import sanitize_module_parameters
except ImportError:
    from algorithms.ClientTrainer import ClientTrainer
    from algorithms.fedmobile_utils import sanitize_module_parameters


class FedMEKTClientTrainer(ClientTrainer):
    def __init__(self, args, dataset, class_size, logger, inter_distance=4, loss='softmax',
                 gpuid='cuda:0', num_epochs=30, init_lr=0.0001, decay=0.1,
                 num_workers=4, print_freq=10, save_step=10, scale=128,
                 pool_type='max_avg', client_id=-1, wandb=None):
        super().__init__(
            args, dataset, class_size, logger, inter_distance=inter_distance, loss=loss, gpuid=gpuid,
            num_epochs=num_epochs, init_lr=init_lr, decay=decay, num_workers=num_workers,
            print_freq=print_freq, save_step=save_step, scale=scale, pool_type=pool_type,
            client_id=client_id, wandb=wandb,
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = getattr(self.args, 'fedmekt_local_lr', self.init_lr)
            param_group['weight_decay'] = self.args.weight_decay
        self.proxy_loader = None
        self.proxy_size = 0
        self.proxy_index_lookup = None
        self.server_image_targets = None
        self.server_text_targets = None
        self.proxy_upload = {'image': None, 'text': None}
        self.mse_loss = nn.MSELoss()
        self.last_epoch_stats = {}

    def set_proxy_loader(self, proxy_loader):
        self.proxy_loader = proxy_loader
        self.proxy_size = len(proxy_loader.dataset) if proxy_loader is not None else 0
        self.proxy_index_lookup = None
        dataset = getattr(proxy_loader, 'dataset', None) if proxy_loader is not None else None
        subset_indices = getattr(dataset, 'indices', None)
        if subset_indices is not None:
            self.proxy_index_lookup = {
                int(original_index): mapped_index
                for mapped_index, original_index in enumerate(subset_indices)
            }

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

    def set_server_proxy_targets(self, image_targets=None, text_targets=None):
        self.server_image_targets = image_targets
        self.server_text_targets = text_targets

    def _proxy_batch_to_embedding_input(self, data):
        if isinstance(data, dict):
            proxy_index = torch.tensor(data["proxy_index"], dtype=torch.long)
            if self.dset_name == 'image':
                inputs = data["processed_img"].to(self.gpuid)
            else:
                inputs = data["cap_tokens"]
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.cat(inputs)
                inputs = inputs.to(self.gpuid).contiguous()
            return inputs, proxy_index

        images, captions, _, _, _, _, index = data
        proxy_index = self._normalize_proxy_index(torch.as_tensor(index, dtype=torch.long))
        if self.dset_name == 'image':
            return images.to(self.gpuid), proxy_index
        return captions.to(self.gpuid), proxy_index

    def _forward_local(self, data):
        if self.dset_name == 'image':
            inputs = data["processed_img"].to(self.gpuid)
            labels = data["class_id"]
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(self.gpuid)
            logits, _, embedding = self.model(inputs)
        else:
            inputs = data["cap_tokens"]
            labels = data["class_id"]
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
            inputs, labels = map(
                lambda tensor: torch.cat(tensor) if not isinstance(tensor, torch.Tensor) else tensor,
                (inputs, labels),
            )
            inputs = inputs.to(self.gpuid).contiguous()
            labels = labels.to(self.gpuid).contiguous()
            logits, _, embedding = self.model(inputs)

        if embedding.dim() > 2:
            embedding = embedding.view(embedding.size(0), -1)
        return inputs, labels, logits, embedding

    def _forward_proxy_embedding(self, data):
        inputs, _ = self._proxy_batch_to_embedding_input(data)
        _, _, embedding = self.model(inputs)
        if embedding.dim() > 2:
            embedding = embedding.view(embedding.size(0), -1)
        return embedding

    def _normalized_mse(self, source_embeddings, target_embeddings):
        source_norm = F.normalize(source_embeddings.float(), dim=-1)
        target_norm = F.normalize(target_embeddings.float(), dim=-1)
        return self.mse_loss(source_norm, target_norm)

    def _get_target_bank(self):
        if self.dset_name == 'image':
            return self.server_image_targets
        return self.server_text_targets

    def _train_private_epoch(self):
        total_loss = 0.0
        total_batches = 0
        skipped_nonfinite_output = 0
        skipped_nonfinite_loss = 0
        skipped_nonfinite_grad = 0

        self.model.train()
        for data in self.train_loader:
            self.optimizer.zero_grad()
            _, labels, logits, embedding = self._forward_local(data)
            if not torch.isfinite(logits).all() or not torch.isfinite(embedding).all():
                skipped_nonfinite_output += 1
                continue

            loss = self.criterion(logits, labels)
            if not torch.isfinite(loss):
                skipped_nonfinite_loss += 1
                continue

            loss.backward()
            grad_finite = True
            for param in self.model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    grad_finite = False
                    break
            if not grad_finite:
                skipped_nonfinite_grad += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            sanitize_module_parameters(self.model)

            total_loss += float(loss.item())
            total_batches += 1

        return {
            'private_batches': total_batches,
            'private_mean_loss': total_loss / max(1, total_batches),
            'private_skipped_nonfinite_output': skipped_nonfinite_output,
            'private_skipped_nonfinite_loss': skipped_nonfinite_loss,
            'private_skipped_nonfinite_grad': skipped_nonfinite_grad,
        }

    def _train_proxy_alignment_epoch(self):
        target_bank = self._get_target_bank()
        if self.proxy_loader is None or target_bank is None:
            return {
                'proxy_batches': 0,
                'proxy_mean_loss': 0.0,
                'proxy_skipped_nonfinite_output': 0,
                'proxy_skipped_nonfinite_loss': 0,
                'proxy_skipped_nonfinite_grad': 0,
            }

        total_loss = 0.0
        total_batches = 0
        skipped_nonfinite_output = 0
        skipped_nonfinite_loss = 0
        skipped_nonfinite_grad = 0
        max_steps = max(0, getattr(self.args, 'fedmekt_proxy_steps', 16))

        self.model.train()
        for data in itertools.islice(self.proxy_loader, max_steps):
            self.optimizer.zero_grad()
            _, proxy_index_cpu = self._proxy_batch_to_embedding_input(data)
            proxy_index = proxy_index_cpu.to(self.gpuid)
            embedding = self._forward_proxy_embedding(data)
            if not torch.isfinite(embedding).all():
                skipped_nonfinite_output += 1
                continue

            target = target_bank[proxy_index_cpu].to(self.gpuid)
            loss = getattr(self.args, 'fedmekt_local_align_weight', 0.5) * self._normalized_mse(embedding, target)
            if not torch.isfinite(loss):
                skipped_nonfinite_loss += 1
                continue

            loss.backward()
            grad_finite = True
            for param in self.model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    grad_finite = False
                    break
            if not grad_finite:
                skipped_nonfinite_grad += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            sanitize_module_parameters(self.model)

            total_loss += float(loss.item())
            total_batches += 1

        return {
            'proxy_batches': total_batches,
            'proxy_mean_loss': total_loss / max(1, total_batches),
            'proxy_skipped_nonfinite_output': skipped_nonfinite_output,
            'proxy_skipped_nonfinite_loss': skipped_nonfinite_loss,
            'proxy_skipped_nonfinite_grad': skipped_nonfinite_grad,
        }

    def train_fedmekt_epoch(self):
        private_stats = self._train_private_epoch()
        proxy_stats = self._train_proxy_alignment_epoch()
        self.last_epoch_stats = {}
        self.last_epoch_stats.update(private_stats)
        self.last_epoch_stats.update(proxy_stats)
        self.last_epoch_stats['optimized_batches'] = (
            private_stats['private_batches'] + proxy_stats['proxy_batches']
        )
        self.last_epoch_stats['total_batches'] = self.last_epoch_stats['optimized_batches']

    def _compute_proxy_upload(self):
        if self.proxy_loader is None or self.proxy_size <= 0:
            self.proxy_upload = {'image': None, 'text': None}
            return

        embeddings = torch.zeros(self.proxy_size, self.args.feature_dim, dtype=torch.float32)
        self.model.to(self.gpuid)
        self.model.eval()
        with torch.no_grad():
            for data in self.proxy_loader:
                _, proxy_index = self._proxy_batch_to_embedding_input(data)
                embedding = self._forward_proxy_embedding(data)
                embedding = F.normalize(
                    torch.nan_to_num(embedding.float(), nan=0.0, posinf=1e4, neginf=-1e4),
                    dim=-1,
                ).cpu()
                embeddings[proxy_index] = embedding
        self.model.train()

        if self.dset_name == 'image':
            self.proxy_upload = {'image': embeddings, 'text': None}
        else:
            self.proxy_upload = {'image': None, 'text': embeddings}

    def export_proxy_embeddings(self):
        return self.proxy_upload

    def run(self):
        self.model.to(self.gpuid)
        self.lr_scheduler(self.cur_epoch)

        for _ in range(self.local_epochs):
            self.local_epoch += 1
            self.train_fedmekt_epoch()
            if self.logger is not None:
                self.logger.log(f"FedMEKT client {self.client_id} stats: {self.last_epoch_stats}")

        self.test()
        self._compute_proxy_upload()

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/{self.dset_name}/Client{self.client_id}-model_{self.local_epoch}.pth')

        self.model.cpu()
        gc.collect()
