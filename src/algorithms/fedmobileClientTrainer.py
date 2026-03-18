import copy
import gc
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

try:
    from src.algorithms.ClientTrainer import ClientTrainer
    from src.algorithms.fedmobile_utils import LabelConditionalGenerator, clone_state_dict, sanitize_module_parameters
except ImportError:
    from algorithms.ClientTrainer import ClientTrainer
    from algorithms.fedmobile_utils import LabelConditionalGenerator, clone_state_dict, sanitize_module_parameters


class FedMobileClientTrainer(ClientTrainer):
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
        self.generator = LabelConditionalGenerator(
            num_classes=self.classSize,
            embed_dim=self.args.feature_dim,
            noise_dim=getattr(self.args, 'fedmobile_noise_dim', 128),
            hidden_dim=getattr(self.args, 'fedmobile_hidden_dim', self.args.feature_dim * 2),
        )
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=getattr(self.args, 'fedmobile_gen_lr', self.init_lr),
            weight_decay=self.args.weight_decay,
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = getattr(self.args, 'fedmobile_local_lr', self.init_lr)
            param_group['weight_decay'] = self.args.weight_decay
        self.reconstruction_loss = nn.MSELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.last_epoch_stats = {}

    def set_generator_state(self, state_dict):
        if state_dict is not None:
            self.generator.load_state_dict(state_dict, strict=True)

    def get_generator_state(self):
        return clone_state_dict(self.generator.state_dict())

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

    def _compute_generator_logits(self, generated_embeddings):
        linear = self.model.class_fc_2
        if hasattr(self.model, 'relu'):
            linear.weight.data = self.model.relu(linear.weight.data)
        return linear(generated_embeddings.to(linear.weight.dtype))

    def _normalized_alignment_loss(self, source_embeddings, target_embeddings):
        source_norm = F.normalize(source_embeddings.float(), dim=-1)
        target_norm = F.normalize(target_embeddings.float(), dim=-1)
        labels = torch.ones(source_norm.size(0), device=source_norm.device, dtype=torch.float32)
        return self.cosine_loss(source_norm, target_norm, labels)

    def train_fedmobile_epoch(self):
        self.model.train()
        self.generator.train()
        total_loss = 0.0
        total_batches = 0
        skipped_batches = 0
        finite_grad_batches = 0
        skipped_nonfinite_output = 0
        skipped_nonfinite_loss = 0
        skipped_nonfinite_grad = 0

        for data in self.train_loader:
            self.optimizer.zero_grad()
            self.generator_optimizer.zero_grad()

            _, labels, logits, embedding = self._forward_local(data)
            if not torch.isfinite(logits).all() or not torch.isfinite(embedding).all():
                skipped_batches += 1
                skipped_nonfinite_output += 1
                continue

            cls_loss = self.criterion(logits, labels)
            generated_embeddings = torch.nan_to_num(self.generator(labels).float(), nan=0.0, posinf=1e4, neginf=-1e4)
            target_embeddings = torch.nan_to_num(embedding.float(), nan=0.0, posinf=1e4, neginf=-1e4)

            gen_recon = self._normalized_alignment_loss(generated_embeddings, target_embeddings.detach())
            model_align = self._normalized_alignment_loss(target_embeddings, generated_embeddings.detach())
            gen_logits = self._compute_generator_logits(generated_embeddings).float()
            gen_aux = self.auxiliary_loss(gen_logits, labels)

            loss = cls_loss
            loss = loss + getattr(self.args, 'fedmobile_gen_weight', 0.5) * gen_recon
            loss = loss + getattr(self.args, 'fedmobile_align_weight', 0.1) * model_align
            loss = loss + getattr(self.args, 'fedmobile_aux_ce_weight', 0.5) * gen_aux

            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches += 1
                skipped_nonfinite_loss += 1
                continue
            loss.backward()
            grad_finite = True
            for param in list(self.model.parameters()) + list(self.generator.parameters()):
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    grad_finite = False
                    break
            if not grad_finite:
                skipped_batches += 1
                skipped_nonfinite_grad += 1
                self.optimizer.zero_grad(set_to_none=True)
                self.generator_optimizer.zero_grad(set_to_none=True)
                continue
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=2.0)
            self.optimizer.step()
            self.generator_optimizer.step()
            sanitize_module_parameters(self.model)
            sanitize_module_parameters(self.generator)
            total_loss += float(loss.item())
            total_batches += 1
            finite_grad_batches += 1

        self.last_epoch_stats = {
            'total_batches': total_batches + skipped_batches,
            'optimized_batches': total_batches,
            'skipped_batches': skipped_batches,
            'finite_grad_batches': finite_grad_batches,
            'mean_loss': total_loss / max(1, total_batches),
            'skipped_nonfinite_output': skipped_nonfinite_output,
            'skipped_nonfinite_loss': skipped_nonfinite_loss,
            'skipped_nonfinite_grad': skipped_nonfinite_grad,
        }

    def run(self, server_generator_state=None):
        self.model.to(self.gpuid)
        self.generator.to(self.gpuid)
        self.lr_scheduler(self.cur_epoch)
        self.set_generator_state(server_generator_state)

        for _ in range(self.local_epochs):
            self.local_epoch += 1
            self.train_fedmobile_epoch()
            if self.logger is not None:
                self.logger.log(f"FedMobile client {self.client_id} stats: {self.last_epoch_stats}")

        self.test()

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/{self.dset_name}/Client{self.client_id}-model_{self.local_epoch}.pth')

        self.model.cpu()
        self.generator.cpu()
        gc.collect()

    def evaluate_generator_utility(self, generator_state, max_batches=1):
        generator = LabelConditionalGenerator(
            num_classes=self.classSize,
            embed_dim=self.args.feature_dim,
            noise_dim=getattr(self.args, 'fedmobile_noise_dim', 128),
            hidden_dim=getattr(self.args, 'fedmobile_hidden_dim', self.args.feature_dim * 2),
        )
        generator.load_state_dict(generator_state, strict=True)
        generator.to(self.gpuid)
        generator.eval()

        self.model.to(self.gpuid)
        self.model.eval()

        score = 0.0
        batches = 0
        with torch.no_grad():
            for data in self.test_loader:
                _, labels, _, embedding = self._forward_local(data)
                noise = torch.zeros(labels.size(0), generator.noise_dim, device=self.gpuid, dtype=torch.float32)
                generated_embeddings = torch.nan_to_num(generator(labels, noise=noise).float(), nan=0.0, posinf=1e4, neginf=-1e4)
                generated_logits = self._compute_generator_logits(generated_embeddings).float()
                predicted = generated_logits.argmax(dim=1)
                score += float((predicted == labels).float().mean().item())
                batches += 1
                if batches >= max_batches:
                    break

        generator.cpu()
        self.model.cpu()
        self.model.train()
        return score / max(1, batches)

    def generator_signature(self, probe_labels, device):
        self.generator.to(device)
        self.generator.eval()
        with torch.no_grad():
            noise = torch.zeros(probe_labels.size(0), self.generator.noise_dim, device=device, dtype=torch.float32)
            signature = self.generator(probe_labels.to(device), noise=noise).detach().cpu()
            signature = torch.nan_to_num(signature, nan=0.0, posinf=1e4, neginf=-1e4).reshape(-1).numpy()
        self.generator.cpu()
        return signature
