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
    from src.algorithms.MMClientTrainer import MMClientTrainer
    from src.algorithms.fedmobile_utils import LabelConditionalGenerator, clone_state_dict, sanitize_module_parameters
except ImportError:
    from algorithms.MMClientTrainer import MMClientTrainer
    from algorithms.fedmobile_utils import LabelConditionalGenerator, clone_state_dict, sanitize_module_parameters
try:
    from apex import amp
except ImportError:
    amp = None


class FedMobileMMClientTrainer(MMClientTrainer):
    def __init__(self, args, config, class_size, logger, client=-1, dset_name="mm",
                 device='cuda', mlp_local=False, wandb=None):
        super().__init__(args, config, class_size, logger, client=client, dset_name=dset_name,
                         device=device, mlp_local=mlp_local, wandb=wandb)
        self.generator = LabelConditionalGenerator(
            num_classes=self.class_size,
            embed_dim=self.args.feature_dim,
            noise_dim=getattr(self.args, 'fedmobile_noise_dim', 128),
            hidden_dim=getattr(self.args, 'fedmobile_hidden_dim', self.args.feature_dim * 2),
        )
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=getattr(self.args, 'fedmobile_gen_lr', self.config.optimizer.learning_rate),
            weight_decay=self.args.weight_decay,
        )
        mm_params = [param for param in self.model.parameters() if param.requires_grad]
        mm_params += [param for param in self.criterion.parameters() if param.requires_grad]
        self.optimizer = optim.SGD(
            mm_params,
            lr=getattr(self.args, 'fedmobile_mm_lr', self.config.optimizer.learning_rate),
            momentum=getattr(self.args, 'momentum', 0.9),
            weight_decay=self.args.weight_decay,
        )
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.last_epoch_stats = {}

    def set_generator_state(self, state_dict):
        if state_dict is not None:
            self.generator.load_state_dict(state_dict, strict=True)

    def get_generator_state(self):
        return clone_state_dict(self.generator.state_dict())

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
            images = data["processed_img"].to(self.device)
            captions = data["cap_tokens"].to(self.device)
            labels = data["class_id"]
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(self.device)

            output = self.model(images, captions)
            if not torch.isfinite(output['image_features']).all() or not torch.isfinite(output['caption_features']).all():
                skipped_batches += 1
                skipped_nonfinite_output += 1
                continue
            retrieval_loss, _ = self.criterion(**output)

            img_features = torch.nan_to_num(output['image_features'].float(), nan=0.0, posinf=1e4, neginf=-1e4)
            txt_features = torch.nan_to_num(output['caption_features'].float(), nan=0.0, posinf=1e4, neginf=-1e4)
            shared_features = 0.5 * (img_features + txt_features)
            generated_embeddings = torch.nan_to_num(self.generator(labels).float(), nan=0.0, posinf=1e4, neginf=-1e4)

            gen_recon = self._normalized_alignment_loss(generated_embeddings, shared_features.detach())
            gen_recon = gen_recon + 0.5 * self._normalized_alignment_loss(generated_embeddings, img_features.detach())
            gen_recon = gen_recon + 0.5 * self._normalized_alignment_loss(generated_embeddings, txt_features.detach())

            model_align = 0.5 * self._normalized_alignment_loss(img_features, generated_embeddings.detach())
            model_align = model_align + 0.5 * self._normalized_alignment_loss(txt_features, generated_embeddings.detach())

            loss = retrieval_loss
            loss = loss + getattr(self.args, 'fedmobile_gen_weight', 0.5) * gen_recon
            loss = loss + getattr(self.args, 'fedmobile_align_weight', 0.1) * model_align

            self.optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches += 1
                skipped_nonfinite_loss += 1
                continue
            if self.config.train.get('use_fp16') and amp is not None:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
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

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                nn.utils.clip_grad.clip_grad_norm_(self.generator.parameters(), self.config.train.grad_clip)

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

    def run(self, server_generator_state=None, prefix=''):
        self.model.cuda()
        self.generator.to(self.device)
        self.set_generator_state(server_generator_state)
        if self.local_epoch == 0 and self.config.train.get('use_fp16') and amp is not None:
            _, self.optimizer = amp.initialize([], self.optimizer, opt_level='O2')
        self.model.train()

        for _ in range(self.local_epochs):
            self.local_epoch += 1
            if self.logger is not None:
                self.logger.log(f"Epoch {self.local_epoch}")
            self.train_fedmobile_epoch()
            if self.logger is not None:
                self.logger.log(f"FedMobile mm client {self.client} stats: {self.last_epoch_stats}")

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/mm/Client{self.client}-model_{self.local_epoch}.pth')

        self.model.cpu()
        self.generator.cpu()
        gc.collect()

    def evaluate_generator_utility(self, generator_state, max_batches=1):
        generator = LabelConditionalGenerator(
            num_classes=self.class_size,
            embed_dim=self.args.feature_dim,
            noise_dim=getattr(self.args, 'fedmobile_noise_dim', 128),
            hidden_dim=getattr(self.args, 'fedmobile_hidden_dim', self.args.feature_dim * 2),
        )
        generator.load_state_dict(generator_state, strict=True)
        generator.to(self.device)
        generator.eval()

        self.model.to(self.device)
        self.model.eval()

        score = 0.0
        batches = 0
        with torch.no_grad():
            for data in self.val_loader:
                images = data["processed_img"].to(self.device)
                captions = data["cap_tokens"].to(self.device)
                labels = data["class_id"]
                if isinstance(labels, list):
                    labels = torch.tensor(labels, dtype=torch.long)
                labels = labels.to(self.device)

                output = self.model(images, captions)
                shared_features = 0.5 * (
                    torch.nan_to_num(output['image_features'].float(), nan=0.0, posinf=1e4, neginf=-1e4) +
                    torch.nan_to_num(output['caption_features'].float(), nan=0.0, posinf=1e4, neginf=-1e4)
                )
                noise = torch.zeros(labels.size(0), generator.noise_dim, device=self.device, dtype=torch.float32)
                generated_embeddings = torch.nan_to_num(generator(labels, noise=noise).float(), nan=0.0, posinf=1e4, neginf=-1e4)
                generated_embeddings = F.normalize(generated_embeddings, dim=-1)
                shared_features = F.normalize(shared_features, dim=-1)
                sim_matrix = generated_embeddings @ shared_features.T
                predicted = sim_matrix.argmax(dim=1)
                targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
                score += float((predicted == targets).float().mean().item())
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
