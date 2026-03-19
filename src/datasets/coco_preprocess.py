"""Precompute public MS-COCO embeddings for distillation baselines."""

import argparse
import hashlib
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from transformers import AutoModel, AutoTokenizer, SiglipModel, SiglipProcessor

from src.utils.model_utils import normalize_model_name

try:
    import ujson as json
except ImportError:
    import json


def _merge_coco(ann_file: str, extra_ann_file: Optional[str]) -> COCO:
    if not extra_ann_file:
        return COCO(ann_file)

    coco = COCO()
    with open(ann_file, "r") as fin1, open(extra_ann_file, "r") as fin2:
        dataset = json.load(fin1)
        extra_dataset = json.load(fin2)
        if not isinstance(dataset, dict) or not isinstance(extra_dataset, dict):
            raise TypeError(f"invalid type {type(dataset)} {type(extra_dataset)}")
        if set(dataset.keys()) != set(extra_dataset.keys()):
            raise KeyError(f"key mismatch {list(dataset.keys())} != {list(extra_dataset.keys())}")
        for key in ["images", "annotations"]:
            dataset[key].extend(extra_dataset[key])
    coco.dataset = dataset
    coco.createIndex()
    return coco


def _default_model_variant(model_name: str, clip_model_name: str = "RN50") -> str:
    model_name = normalize_model_name(model_name)
    if model_name == "clip":
        return clip_model_name
    if model_name == "align":
        return "kakaobrain/align-base"
    if model_name == "siglip":
        return "google/siglip-base-patch16-224"
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_cache_path(
    ann_file: str,
    extra_ann_file: Optional[str] = None,
    model_name: str = "clip",
    clip_model_name: str = "RN50",
    feature_dim: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> str:
    base_dir = cache_dir or os.path.dirname(os.path.abspath(ann_file))
    model_name = normalize_model_name(model_name)
    variant = _default_model_variant(model_name, clip_model_name).replace("/", "-")
    key = os.path.abspath(ann_file)
    if extra_ann_file:
        key += "|" + os.path.abspath(extra_ann_file)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    dim_suffix = f"_{feature_dim}" if feature_dim is not None else ""
    filename = f"coco_emb_{model_name}{dim_suffix}_{variant}_{digest}.pt"
    return os.path.join(base_dir, filename)


def load_clip_cache(path: str, map_location: Union[str, torch.device] = "cpu") -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    cache = torch.load(path, map_location=map_location)
    required_keys = ["image_ids", "image_features", "ann_ids", "caption_features"]
    missing = [k for k in required_keys if k not in cache]
    if missing:
        raise KeyError(f"Cache file {path} missing keys: {missing}")
    return cache


def _chunk(seq: Sequence, size: int) -> Iterable[Sequence]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _encode_clip_images(coco: COCO, root: str, clip_model, preprocess, device: str, batch_size: int):
    image_ids = sorted({ann["image_id"] for ann in coco.anns.values()})
    encoded: List[torch.Tensor] = []
    for batch_ids in _chunk(image_ids, batch_size):
        images = []
        for image_id in batch_ids:
            info = coco.loadImgs(image_id)[0]
            image_path = os.path.join(root, info["file_name"])
            with Image.open(image_path).convert("RGB") as img:
                images.append(preprocess(img))
        with torch.no_grad():
            feats = clip_model.encode_image(torch.stack(images).to(device))
        encoded.append(feats.cpu())
    return image_ids, torch.cat(encoded, dim=0)


def _encode_clip_captions(coco: COCO, clip_model, device: str, batch_size: int):
    import clip

    ann_ids = list(coco.anns.keys())
    captions = [coco.anns[ann_id]["caption"] for ann_id in ann_ids]
    encoded: List[torch.Tensor] = []
    for batch_captions in _chunk(captions, batch_size):
        tokens = clip.tokenize(batch_captions, truncate=True).to(device)
        with torch.no_grad():
            feats = clip_model.encode_text(tokens)
        encoded.append(feats.cpu())
    return ann_ids, captions, torch.cat(encoded, dim=0)


def _align_preprocess():
    return T.Compose([
        T.Resize(256, interpolation=Image.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def _encode_align_images(coco: COCO, root: str, align_model, preprocess, device: str, batch_size: int):
    image_ids = sorted({ann["image_id"] for ann in coco.anns.values()})
    encoded: List[torch.Tensor] = []
    for batch_ids in _chunk(image_ids, batch_size):
        images = []
        for image_id in batch_ids:
            info = coco.loadImgs(image_id)[0]
            image_path = os.path.join(root, info["file_name"])
            with Image.open(image_path).convert("RGB") as img:
                images.append(preprocess(img))
        with torch.no_grad():
            feats = align_model.get_image_features(torch.stack(images).to(device))
        encoded.append(feats.cpu())
    return image_ids, torch.cat(encoded, dim=0)


def _encode_align_captions(coco: COCO, align_model, tokenizer, device: str, batch_size: int):
    ann_ids = list(coco.anns.keys())
    captions = [coco.anns[ann_id]["caption"] for ann_id in ann_ids]
    encoded: List[torch.Tensor] = []
    for batch_captions in _chunk(captions, batch_size):
        tokens = tokenizer(
            list(batch_captions),
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            feats = align_model.get_text_features(**tokens)
        encoded.append(feats.cpu())
    return ann_ids, captions, torch.cat(encoded, dim=0)


def _encode_siglip_images(coco: COCO, root: str, siglip_model, processor, device: str, batch_size: int):
    image_ids = sorted({ann["image_id"] for ann in coco.anns.values()})
    encoded: List[torch.Tensor] = []
    for batch_ids in _chunk(image_ids, batch_size):
        images = []
        for image_id in batch_ids:
            info = coco.loadImgs(image_id)[0]
            image_path = os.path.join(root, info["file_name"])
            with Image.open(image_path).convert("RGB") as img:
                images.append(img.copy())
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            feats = siglip_model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
        encoded.append(feats.cpu())
    return image_ids, torch.cat(encoded, dim=0)


def _encode_siglip_captions(coco: COCO, siglip_model, processor, device: str, batch_size: int):
    ann_ids = list(coco.anns.keys())
    captions = [coco.anns[ann_id]["caption"] for ann_id in ann_ids]
    encoded: List[torch.Tensor] = []
    for batch_captions in _chunk(captions, batch_size):
        inputs = processor(text=list(batch_captions), padding=True, truncation=True, return_tensors="pt")
        model_inputs = {k: v.to(device) for k, v in inputs.items() if k in {"input_ids", "attention_mask"}}
        with torch.no_grad():
            feats = siglip_model.get_text_features(**model_inputs)
        encoded.append(feats.cpu())
    return ann_ids, captions, torch.cat(encoded, dim=0)


def precompute_coco_embeddings(
    root: str,
    ann_file: str,
    extra_ann_file: Optional[str] = None,
    output_path: Optional[str] = None,
    model_name: str = "clip",
    clip_model_name: str = "RN50",
    batch_size: int = 256,
    device: Optional[str] = None,
    feature_dim: Optional[int] = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = normalize_model_name(model_name)
    coco = _merge_coco(ann_file, extra_ann_file)
    output_path = output_path or build_cache_path(ann_file, extra_ann_file, model_name=model_name, clip_model_name=clip_model_name, feature_dim=feature_dim)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if model_name == "clip":
        import clip

        model_variant = clip_model_name
        model, preprocess = clip.load(model_variant, device=device)
        model.eval()
        image_ids, image_features = _encode_clip_images(coco, root, model, preprocess, device, batch_size)
        ann_ids, captions, caption_features = _encode_clip_captions(coco, model, device, batch_size)
    elif model_name == "align":
        model_variant = _default_model_variant(model_name, clip_model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_variant)
        model = AutoModel.from_pretrained(model_variant).to(device)
        model.eval()
        preprocess = _align_preprocess()
        image_ids, image_features = _encode_align_images(coco, root, model, preprocess, device, batch_size)
        ann_ids, captions, caption_features = _encode_align_captions(coco, model, tokenizer, device, batch_size)
    elif model_name == "siglip":
        model_variant = _default_model_variant(model_name, clip_model_name)
        processor = SiglipProcessor.from_pretrained(model_variant)
        model = SiglipModel.from_pretrained(model_variant).to(device)
        model.eval()
        image_ids, image_features = _encode_siglip_images(coco, root, model, processor, device, batch_size)
        ann_ids, captions, caption_features = _encode_siglip_captions(coco, model, processor, device, batch_size)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    cache = {
        "model_name": model_name,
        "model_variant": model_variant,
        "feature_dim": int(image_features.shape[-1]),
        "image_ids": image_ids,
        "ann_ids": ann_ids,
        "captions": captions,
        "image_features": image_features.half(),
        "caption_features": caption_features.half(),
    }
    torch.save(cache, output_path)
    print(f"Saved cache to {output_path}")
    return output_path


def precompute_coco_clip(
    root: str,
    ann_file: str,
    extra_ann_file: Optional[str] = None,
    output_path: Optional[str] = None,
    clip_model_name: str = "RN50",
    batch_size: int = 256,
    device: Optional[str] = None,
) -> str:
    return precompute_coco_embeddings(
        root=root,
        ann_file=ann_file,
        extra_ann_file=extra_ann_file,
        output_path=output_path,
        model_name="clip",
        clip_model_name=clip_model_name,
        batch_size=batch_size,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute embeddings for COCO public distillation data")
    parser.add_argument("--root", required=True, help="COCO image root directory")
    parser.add_argument("--ann", required=True, help="COCO annotation file")
    parser.add_argument("--extra-ann", default=None, help="Extra annotation file to merge")
    parser.add_argument("--out", default=None, help="Optional output cache path")
    parser.add_argument("--model", default="clip", choices=["clip", "align", "siglip"], help="Embedding model family")
    parser.add_argument("--clip-model", default="RN50", help="CLIP model variant when --model clip")
    parser.add_argument("--feature-dim", type=int, default=None, help="Optional expected embedding dimension")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    precompute_coco_embeddings(
        root=os.path.expanduser(args.root),
        ann_file=args.ann,
        extra_ann_file=args.extra_ann,
        output_path=args.out,
        model_name=args.model,
        clip_model_name=args.clip_model,
        batch_size=args.batch_size,
        device=args.device,
        feature_dim=args.feature_dim,
    )
