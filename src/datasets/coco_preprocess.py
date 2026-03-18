"""CLIP preprocessing utilities for MS-COCO.

This script precomputes image and caption embeddings with CLIP and stores
lightweight cache files that can be consumed directly by `CocoCaptionsCap`.
"""

import argparse
import hashlib
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from pycocotools.coco import COCO

try:
    import ujson as json
except ImportError:
    import json


def _merge_coco(ann_file: str, extra_ann_file: Optional[str]) -> COCO:
    """Load COCO annotations, optionally merging an extra annotation file."""
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


def build_cache_path(
    ann_file: str,
    extra_ann_file: Optional[str] = None,
    clip_model_name: str = "RN50",
    cache_dir: Optional[str] = None,
) -> str:
    """Deterministically build a cache path for the given annotation files."""
    base_dir = cache_dir or os.path.dirname(os.path.abspath(ann_file))
    key = os.path.abspath(ann_file)
    if extra_ann_file:
        key += "|" + os.path.abspath(extra_ann_file)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    filename = f"coco_clip_{clip_model_name.lower()}_{digest}.pt"
    return os.path.join(base_dir, filename)


def load_clip_cache(path: str, map_location: Union[str, torch.device] = "cpu") -> Dict:
    """Load a cached embedding file with basic validation."""
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


def _encode_images(
    coco: COCO,
    root: str,
    preprocess,
    clip_model,
    device: str,
    batch_size: int,
) -> Tuple[List[int], torch.Tensor]:
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


def _encode_captions(
    coco: COCO,
    clip_model,
    device: str,
    batch_size: int,
) -> Tuple[List[int], List[str], torch.Tensor]:
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


def precompute_coco_clip(
    root: str,
    ann_file: str,
    extra_ann_file: Optional[str] = None,
    output_path: Optional[str] = None,
    clip_model_name: str = "RN50",
    batch_size: int = 256,
    device: Optional[str] = None,
) -> str:
    """Encode COCO images and captions with CLIP and save to a cache file."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    coco = _merge_coco(ann_file, extra_ann_file)
    output_path = output_path or build_cache_path(ann_file, extra_ann_file, clip_model_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    import clip

    clip_model, preprocess = clip.load(clip_model_name, device=device)
    clip_model.eval()

    image_ids, image_features = _encode_images(coco, root, preprocess, clip_model, device, batch_size)
    ann_ids, captions, caption_features = _encode_captions(coco, clip_model, device, batch_size)

    cache = {
        "clip_model": clip_model_name,
        "image_ids": image_ids,
        "ann_ids": ann_ids,
        "captions": captions,
        "image_features": image_features.half(),
        "caption_features": caption_features.half(),
    }
    torch.save(cache, output_path)
    print(f"Saved cache to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute CLIP embeddings for COCO")
    parser.add_argument("--root", required=True, help="COCO image root directory")
    parser.add_argument("--ann", required=True, help="COCO annotation file")
    parser.add_argument("--extra-ann", default=None, help="Extra annotation file to merge")
    parser.add_argument("--out", default=None, help="Optional output cache path")
    parser.add_argument("--clip-model", default="RN50", help="CLIP model name")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    precompute_coco_clip(
        root=os.path.expanduser(args.root),
        ann_file=args.ann,
        extra_ann_file=args.extra_ann,
        output_path=args.out,
        clip_model_name=args.clip_model,
        batch_size=args.batch_size,
        device=args.device,
    )
