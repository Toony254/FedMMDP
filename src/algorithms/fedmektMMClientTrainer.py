import gc
import itertools
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

try:
    from src.algorithms.MMClientTrainer import MMClientTrainer
    from src.algorithms.fedmobile_utils import sanitize_module_parameters
except ImportError:
    from algorithms.MMClientTrainer import MMClientTrainer
    from algorithms.fedmobile_utils import sanitize_module_parameters

try:
    from apex import amp
except ImportError:
    amp = None


class FedMEKTMMClientTrainer(MMClientTrainer):
    def __init__(self, args, config, class_size, logger, client=-1, dset_name="mm",
                 device='cuda', mlp_local=False, wandb=None):
        super().__init__(args, config, class_size, logger, client=client, dset_name=dset_name,
                         device=device, mlp_local=mlp_local, wandb=wandb)
        mm_params = [param for param in self.model.parameters() if param.requires_grad]
        mm_params += [param for param in self.criterion.parameters() if param.requires_grad]
        self.optimizer = optim.SGD(
            mm_params,
            lr=getattr(self.args, 'fedmekt_mm_lr', self.config.optimizer.learning_rate),
            momentum=getattr(self.args, 'momentum', 0.9),
            weight_decay=self.args.weight_decay,
        )
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

    def _proxy_batch_to_inputs(self, data):
        if isinstance(data, dict):
            images = data["processed_img"].to(self.device)
            captions = data["cap_tokens"].to(self.device)
            proxy_index = torch.tensor(data["proxy_index"], dtype=torch.long)
            return images, captions, proxy_index

        images, captions, _, _, _, _, index = data
        proxy_index = self._normalize_proxy_index(torch.as_tensor(index, dtype=torch.long))
        return images.to(self.device), captions.to(self.device), proxy_index

    def _normalized_mse(self, source_embeddings, target_embeddings):
        source_norm = F.normalize(source_embeddings.float(), dim=-1)
        target_norm = F.normalize(target_embeddings.float(), dim=-1)
        return self.mse_loss(source_norm, target_norm)

    def _train_private_epoch(self):
        total_loss = 0.0
        total_batches = 0
        skipped_nonfinite_output = 0
        skipped_nonfinite_loss = 0
        skipped_nonfinite_grad = 0

        self.model.train()
        for data in self.train_loader:
            images = data["processed_img"].to(self.device)
            captions = data["cap_tokens"].to(self.device)
            output = self.model(images, captions)
            if not torch.isfinite(output['image_features']).all() or not torch.isfinite(output['caption_features']).all():
                skipped_nonfinite_output += 1
                continue

            loss, _ = self.criterion(**output)
            self.optimizer.zero_grad()
            if not torch.isfinite(loss):
                skipped_nonfinite_loss += 1
                continue

            if self.config.train.get('use_fp16') and amp is not None:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            grad_finite = True
            for param in list(self.model.parameters()) + list(self.criterion.parameters()):
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    grad_finite = False
                    break
            if not grad_finite:
                skipped_nonfinite_grad += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
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
        if self.proxy_loader is None or self.server_image_targets is None or self.server_text_targets is None:
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
            images, captions, proxy_index_cpu = self._proxy_batch_to_inputs(data)
            proxy_index = proxy_index_cpu.to(self.device)

            output = self.model(images, captions)
            img_features = output['image_features']
            txt_features = output['caption_features']
            if not torch.isfinite(img_features).all() or not torch.isfinite(txt_features).all():
                skipped_nonfinite_output += 1
                continue

            target_img = self.server_image_targets[proxy_index_cpu].to(self.device)
            target_txt = self.server_text_targets[proxy_index_cpu].to(self.device)
            loss = getattr(self.args, 'fedmekt_local_align_weight', 0.5) * (
                self._normalized_mse(img_features, target_img) +
                self._normalized_mse(txt_features, target_txt)
            )

            self.optimizer.zero_grad()
            if not torch.isfinite(loss):
                skipped_nonfinite_loss += 1
                continue

            loss.backward()
            grad_finite = True
            for param in list(self.model.parameters()) + list(self.criterion.parameters()):
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    grad_finite = False
                    break
            if not grad_finite:
                skipped_nonfinite_grad += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
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

        image_embeddings = torch.zeros(self.proxy_size, self.args.feature_dim, dtype=torch.float32)
        text_embeddings = torch.zeros(self.proxy_size, self.args.feature_dim, dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for data in self.proxy_loader:
                images, captions, proxy_index = self._proxy_batch_to_inputs(data)
                output = self.model(images, captions)
                img_features = F.normalize(
                    torch.nan_to_num(output['image_features'].float(), nan=0.0, posinf=1e4, neginf=-1e4),
                    dim=-1,
                ).cpu()
                txt_features = F.normalize(
                    torch.nan_to_num(output['caption_features'].float(), nan=0.0, posinf=1e4, neginf=-1e4),
                    dim=-1,
                ).cpu()
                image_embeddings[proxy_index] = img_features
                text_embeddings[proxy_index] = txt_features
        self.model.train()
        self.proxy_upload = {'image': image_embeddings, 'text': text_embeddings}

    def export_proxy_embeddings(self):
        return self.proxy_upload

    def run(self, prefix=''):
        self.model.cuda()
        if self.local_epoch == 0 and self.config.train.get('use_fp16') and amp is not None:
            _, self.optimizer = amp.initialize([], self.optimizer, opt_level='O2')
        self.model.train()

        for _ in range(self.local_epochs):
            self.local_epoch += 1
            if self.logger is not None:
                self.logger.log(f"Epoch {self.local_epoch}")
            self.train_fedmekt_epoch()
            if self.logger is not None:
                self.logger.log(f"FedMEKT mm client {self.client} stats: {self.last_epoch_stats}")

        self._compute_proxy_upload()

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/mm/Client{self.client}-model_{self.local_epoch}.pth')

        self.model.cpu()
        gc.collect()
