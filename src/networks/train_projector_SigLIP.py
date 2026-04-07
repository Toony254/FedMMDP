import argparse
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from pycocotools.coco import COCO
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import SiglipModel, SiglipProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_COCO_ROOT = Path(
    os.environ.get("FEDMMDP_COCO_ROOT", str(DEFAULT_DATA_DIR / "COCO"))
)
DEFAULT_FLICKR_ROOT = Path(
    os.environ.get("FEDMMDP_FLICKR30K_ROOT", str(DEFAULT_DATA_DIR / "flickr30k" / "flickr30k-images"))
)
LEGACY_FLICKR_ROOTS = (
    "/data/mmdata/Flick30k/flickr30k-images/",
)
for candidate in (ROOT_DIR, SRC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

try:
    from src.utils.projector_utils import (
        normalize_projector_variant,
        projector_checkpoint_name,
        projector_variant_label,
    )
except ImportError:
    from utils.projector_utils import (
        normalize_projector_variant,
        projector_checkpoint_name,
        projector_variant_label,
    )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = False
RANDOM_SEED = 42
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class F30kCaptionsCap(Dataset):
    def __init__(self, ann_file="dataset_k_split.pkl", split="test", transform=None):
        self.transform = transform
        self.data = pickle.load(open(ann_file, "rb"))
        if split not in self.data:
            raise ValueError(f"Invalid split: {split}")
        self.data = self.data[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, caption = self.data[index][0], self.data[index][1]
        image_path = resolve_flickr_image_path(image_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, caption


def resolve_flickr_image_path(image_path: str) -> str:
    candidate = Path(image_path)
    if candidate.exists():
        return str(candidate)
    normalized = image_path.replace("\\", "/")
    for legacy_root in LEGACY_FLICKR_ROOTS:
        if normalized.startswith(legacy_root):
            rel_path = normalized[len(legacy_root):].lstrip("/")
            return str(DEFAULT_FLICKR_ROOT / rel_path)
    return str(DEFAULT_FLICKR_ROOT / candidate.name)


class MSCOCODataset(Dataset):
    def __init__(self, coco_json, img_dir, img_transform=None):
        self.coco = COCO(coco_json)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.img_transform = img_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.imgs[img_id]
        image_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        if self.img_transform is not None:
            image = self.img_transform(image)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        captions = [ann["caption"] for ann in self.coco.loadAnns(ann_ids)]
        caption = random.choice(captions)
        return image, caption


class SelfAttentionProjector(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.proj(attn @ v)


class ResidualProjector(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.proj(x)


def build_projector(projector_name: str, emb_dim: int) -> nn.Module:
    if projector_name == "linear":
        return nn.Linear(emb_dim, emb_dim)
    if projector_name == "mlp":
        return nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
    if projector_name == "residual":
        return ResidualProjector(emb_dim)
    if projector_name == "norm":
        return nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )
    if projector_name == "mlp+norm":
        return nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )
    if projector_name == "bottleneck":
        return nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, emb_dim),
        )
    if projector_name == "attention":
        return SelfAttentionProjector(emb_dim)
    raise ValueError(f"Unsupported projector: {projector_name}")


class CombinedModel(nn.Module):
    def __init__(self, siglip_model, siglip_processor, args):
        super().__init__()
        self.siglip_model = siglip_model
        self.siglip_processor = siglip_processor
        emb_dim = siglip_model.config.vision_config.hidden_size
        self.visual_projector = build_projector(args.projector, emb_dim)
        self.text_projector = build_projector(args.projector, emb_dim)

    def get_image_features(self, images):
        vision_outputs = self.siglip_model.vision_model(pixel_values=images)
        return self.visual_projector(vision_outputs.pooler_output)

    def get_text_features(self, texts):
        text_inputs = self.siglip_processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {key: value.to(DEVICE) for key, value in text_inputs.items()}
        text_outputs = self.siglip_model.text_model(**text_inputs)
        return self.text_projector(text_outputs.pooler_output)

    def forward(self, images, texts):
        return self.get_image_features(images), self.get_text_features(texts)


def contrastive_loss(img_feats, text_feats, temperature=0.07):
    img_feats = F.normalize(img_feats, p=2, dim=1)
    text_feats = F.normalize(text_feats, p=2, dim=1)
    logits = torch.matmul(img_feats, text_feats.t()) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def compute_rmg(image_features, text_features):
    image_to_text_map = torch.arange(image_features.shape[0], device=image_features.device).reshape(image_features.shape[0], 1)
    text_to_image_map = torch.arange(image_features.shape[0], device=image_features.device)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features_original = image_features.clone()
    image_features = torch.stack([image_features[l] for l in text_to_image_map], dim=0)
    text_feature_per_image = text_features
    labels_to_idx_map = torch.ones((text_features.size(0), text_features.size(0)), dtype=torch.bool, device=image_features.device)
    for txt_idcs in image_to_text_map:
        for i, j in [(x.item(), y.item()) for x in txt_idcs for y in txt_idcs]:
            labels_to_idx_map[i, j] = False

    image_features_matching = torch.sum(image_features * text_feature_per_image, dim=1).mean()
    image_features_matching = 1 - (image_features_matching + 1) / 2
    image_features_matching = torch.where(
        image_features_matching > 0,
        image_features_matching,
        torch.ones_like(image_features_matching) * 1e-3,
    )

    i_x_i = image_features_original @ image_features_original.T
    i_x_i.fill_diagonal_(0)
    mean_img_similarity = i_x_i.sum() / (math.prod(i_x_i.shape) - i_x_i.shape[0])
    mean_img_similarity = 1 - (mean_img_similarity + 1) / 2

    t_x_t = text_features @ text_features.T
    t_x_t.fill_diagonal_(0)
    mean_txt_similarity = t_x_t.sum() / (math.prod(t_x_t.shape) - t_x_t.shape[0])
    mean_txt_similarity = 1 - (mean_txt_similarity + 1) / 2

    normalizer = image_features_matching.mean() + (mean_img_similarity.mean() + mean_txt_similarity.mean()) / 2
    return image_features_matching.mean() / normalizer.clamp_min(1e-8)


def max_margin_loss(img_feats, text_feats, margin=0.2):
    img_feats = F.normalize(img_feats, p=2, dim=1)
    text_feats = F.normalize(text_feats, p=2, dim=1)
    sims = img_feats @ text_feats.t()
    batch_size = sims.size(0)
    if batch_size <= 1:
        return torch.zeros((), device=sims.device, dtype=sims.dtype)
    pos = sims.diag()
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=sims.device)
    i2t = F.relu(margin + sims - pos.unsqueeze(1))[mask].mean()
    t2i = F.relu(margin + sims - pos.unsqueeze(0))[mask].mean()
    return 0.5 * (i2t + t2i)


def calculate_accuracy(image_features, text_features, labels, threshold=0.5):
    if isinstance(image_features, torch.Tensor):
        image_features = image_features.detach().cpu().numpy()
        text_features = text_features.detach().cpu().numpy()
    similarity_matrix = cosine_similarity(image_features, text_features)
    predictions = (similarity_matrix > threshold).astype(int)
    return float(np.mean(predictions == labels))


def compute_total_loss(img_feats, text_feats, loss_mode, temperature, margin):
    normalized_mode = normalize_projector_variant(loss_mode)
    cl_loss = contrastive_loss(img_feats, text_feats, temperature=temperature)
    rmg_loss = compute_rmg(img_feats, text_feats)
    mm_loss = max_margin_loss(img_feats, text_feats, margin=margin)

    if normalized_mode == "clonly":
        total_loss = cl_loss
    elif normalized_mode == "rmgonly":
        total_loss = rmg_loss
    elif normalized_mode == "maxmargin":
        total_loss = mm_loss
    else:
        total_loss = cl_loss + rmg_loss

    stats = {
        "loss": float(total_loss.detach().item()),
        "cl_loss": float(cl_loss.detach().item()),
        "rmg_loss": float(rmg_loss.detach().item()),
        "max_margin_loss": float(mm_loss.detach().item()),
    }
    return total_loss, stats


def evaluate_loader(model, data_loader, temperature, margin):
    model.eval()
    total_loss = 0.0
    total_cl = 0.0
    total_rmg = 0.0
    total_mm = 0.0
    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():
        progress = tqdm(data_loader, desc="eval", leave=False)
        for images, texts in progress:
            images = images.to(DEVICE)
            img_feats = model.get_image_features(images)
            text_feats = model.get_text_features(texts)
            _, stats = compute_total_loss(img_feats, text_feats, "", temperature, margin)
            batch_size = len(texts)
            accuracy = calculate_accuracy(img_feats, text_feats, np.eye(batch_size))

            total_loss += stats["loss"] * batch_size
            total_cl += stats["cl_loss"] * batch_size
            total_rmg += stats["rmg_loss"] * batch_size
            total_mm += stats["max_margin_loss"] * batch_size
            total_accuracy += accuracy * batch_size
            total_samples += batch_size
            progress.set_postfix(loss=f"{stats['loss']:.4f}", rmg=f"{stats['rmg_loss']:.4f}", acc=f"{accuracy * 100:.2f}%")

    return {
        "loss": total_loss / max(total_samples, 1),
        "cl_loss": total_cl / max(total_samples, 1),
        "rmg": total_rmg / max(total_samples, 1),
        "max_margin_loss": total_mm / max(total_samples, 1),
        "accuracy": total_accuracy / max(total_samples, 1),
    }


def train(args, train_loader, val_loader, writer):
    normalized_mode = normalize_projector_variant(args.loss_mode)
    print(f"Using device: {DEVICE}")
    print(f"Projector: {args.projector}, loss_mode: {projector_variant_label(normalized_mode)}")

    siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    siglip_model = SiglipModel.from_pretrained(SIGLIP_MODEL_NAME).to(DEVICE)
    combined_model = CombinedModel(siglip_model, siglip_processor, args).to(DEVICE)

    for param in combined_model.siglip_model.parameters():
        param.requires_grad = False
    for param in combined_model.visual_projector.parameters():
        param.requires_grad = True
    for param in combined_model.text_projector.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, combined_model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    best_state = None
    best_metrics = None
    best_loss = float("inf")

    for epoch in range(args.epochs):
        combined_model.train()
        train_loss_sum = 0.0
        train_samples = 0
        progress = tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}")

        for step, (images, texts) in enumerate(progress):
            optimizer.zero_grad()
            images = images.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                img_feats = combined_model.get_image_features(images)
                text_feats = combined_model.get_text_features(texts)
                loss, stats = compute_total_loss(
                    img_feats,
                    text_feats,
                    args.loss_mode,
                    args.temperature,
                    args.margin,
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(combined_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = len(texts)
            train_loss_sum += stats["loss"] * batch_size
            train_samples += batch_size
            global_step = epoch * len(train_loader) + step
            writer.add_scalar("train/loss", stats["loss"], global_step)
            writer.add_scalar("train/cl_loss", stats["cl_loss"], global_step)
            writer.add_scalar("train/rmg_loss", stats["rmg_loss"], global_step)
            writer.add_scalar("train/max_margin_loss", stats["max_margin_loss"], global_step)
            progress.set_postfix(
                loss=f"{stats['loss']:.4f}",
                cl=f"{stats['cl_loss']:.4f}",
                rmg=f"{stats['rmg_loss']:.4f}",
                mm=f"{stats['max_margin_loss']:.4f}",
            )

        avg_train_loss = train_loss_sum / max(train_samples, 1)
        print(f"[Epoch {epoch + 1}/{args.epochs}] Train Loss: {avg_train_loss:.4f}")

        if val_loader is None:
            continue

        val_metrics = evaluate_loader(combined_model, val_loader, args.temperature, args.margin)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/rmg", val_metrics["rmg"], epoch)
        writer.add_scalar("val/max_margin_loss", val_metrics["max_margin_loss"], epoch)
        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Accuracy: {val_metrics['accuracy'] * 100:.2f}%, "
            f"Val RMG: {val_metrics['rmg']:.4f}"
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_metrics = {"epoch": epoch + 1, **val_metrics}
            best_state = {key: value.detach().cpu().clone() for key, value in combined_model.state_dict().items()}

    if best_state is not None:
        combined_model.load_state_dict(best_state)
    else:
        best_metrics = {"epoch": args.epochs}

    emb_dim = siglip_model.config.vision_config.hidden_size
    checkpoint_name = projector_checkpoint_name(args.projector, "siglip", emb_dim, normalized_mode)
    checkpoint_path = Path(args.save_path) / checkpoint_name
    torch.save(
        {
            "visual_projector": combined_model.visual_projector.state_dict(),
            "text_projector": combined_model.text_projector.state_dict(),
            "metadata": {
                "model": "siglip",
                "projector": args.projector,
                "loss_mode": projector_variant_label(normalized_mode),
                "temperature": args.temperature,
                "margin": args.margin,
                "best_metrics": best_metrics,
            },
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    return combined_model, checkpoint_path, best_metrics


def write_metrics_json(json_path: Path, payload: dict) -> None:
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def build_dataloaders(args):
    siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_NAME)

    def siglip_preprocess(image):
        return siglip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    train_dataset = MSCOCODataset(args.coco_json, args.coco_img_dir, img_transform=siglip_preprocess)
    val_dataset = MSCOCODataset(args.coco_val_json, args.coco_val_img_dir, img_transform=siglip_preprocess)
    flickr_dataset = F30kCaptionsCap(args.flickr_split, split=args.flickr_eval_split, transform=siglip_preprocess)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    flickr_loader = DataLoader(
        flickr_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, flickr_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain SigLIP projector with configurable loss variants")
    parser.add_argument("--save_path", type=str, default="saved/projector_weights/")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--projector", type=str, default="mlp+norm",
                        choices=["linear", "mlp", "residual", "norm", "mlp+norm", "bottleneck", "attention"])
    parser.add_argument("--loss_mode", type=str, default="cl_rmg",
                        choices=["cl_rmg", "cl_only", "rmg_only", "max_margin", "clonly", "rmgonly", "maxmargin"])
    parser.add_argument("--coco_json", type=str, default=str(DEFAULT_COCO_ROOT / "annotations" / "captions_train2017.json"))
    parser.add_argument("--coco_img_dir", type=str, default=str(DEFAULT_COCO_ROOT / "train2017"))
    parser.add_argument("--coco_val_json", type=str, default=str(DEFAULT_COCO_ROOT / "annotations" / "captions_val2017.json"))
    parser.add_argument("--coco_val_img_dir", type=str, default=str(DEFAULT_COCO_ROOT / "val2017"))
    parser.add_argument("--flickr_split", type=str, default="dataset_k_split.pkl")
    parser.add_argument("--flickr_eval_split", type=str, default="test")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    variant_label = projector_variant_label(args.loss_mode)
    writer = SummaryWriter(log_dir=os.path.join("runs", f"projector_siglip_{args.projector}_{variant_label}"))
    train_loader, val_loader, flickr_loader = build_dataloaders(args)
    best_model, checkpoint_path, best_val_metrics = train(args, train_loader, val_loader, writer)
    flickr_metrics = evaluate_loader(best_model, flickr_loader, args.temperature, args.margin)
    writer.close()

    print(f"Flickr30k Accuracy: {flickr_metrics['accuracy'] * 100:.2f}%")
    print(f"Flickr30k RMG: {flickr_metrics['rmg']:.4f}")

    metrics_payload = {
        "checkpoint": str(checkpoint_path),
        "model": "siglip",
        "projector": args.projector,
        "loss_mode": variant_label,
        "selection_metric": "val_loss",
        "mscoco_val": best_val_metrics,
        "flickr30k": flickr_metrics,
    }
    metrics_path = checkpoint_path.with_suffix(".json")
    write_metrics_json(metrics_path, metrics_payload)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
