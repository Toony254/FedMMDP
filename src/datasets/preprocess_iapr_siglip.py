"""
IAPR TC-12 SigLIP preprocessing script.

This script preserves the exact split/schema/layout of an existing CLIP IAPR
dataset and only re-encodes image/text features with SigLIP.

Input:
  - IAPR train/validation JSON files
  - an existing CLIP-preprocessed IAPR domain dataset

Output:
  - a SigLIP-preprocessed IAPR domain dataset with the same split layout
  - the corresponding `label2id.json`

Saved sample fields are identical to the CLIP IAPR dataset:
  id, class_id, processed_img, cap_tokens, domain_id
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import Dataset, load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import SiglipModel, SiglipProcessor


sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


CODE_BASE_DIR = Path(__file__).resolve().parents[2]
DATA_BASE_DIR = Path(os.environ.get("FEDMMDP_PREPROCESS_ROOT", str(CODE_BASE_DIR)))
TRAIN_JSON = Path(os.environ.get("FEDMMDP_IAPR_JSON_TRAIN", str(CODE_BASE_DIR / "data" / "iapr_tc12" / "train.json")))
VAL_JSON = Path(os.environ.get("FEDMMDP_IAPR_JSON_VAL", str(CODE_BASE_DIR / "data" / "iapr_tc12" / "validation.json")))
CLIP_ROOT = Path(os.environ.get("FEDMMDP_IAPR_CLIP_ROOT", str(CODE_BASE_DIR / "data" / "iapr")))
CLIP_DOMAIN_ROOT = CLIP_ROOT / "domain_datasets"
OUTPUT_ROOT = Path(
    os.environ.get("FEDMMDP_IAPR_SIGLIP_PREPROCESS_ROOT", str(CODE_BASE_DIR / "data" / "iapr_siglip"))
)
OUTPUT_DIR = OUTPUT_ROOT / "domain_datasets"

NUM_DOMAINS = 5
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"

logger.info("Device: %s", DEVICE)
logger.info("Input JSONs: %s , %s", TRAIN_JSON, VAL_JSON)
logger.info("CLIP template root: %s", CLIP_DOMAIN_ROOT)
logger.info("Output root: %s", OUTPUT_ROOT)


DOMAIN_TRIGGER = {
    3: ["bird", "eagle", "parrot", "flamingo", "heron", "pelican", "stork", "penguin",
        "seagull", "dove", "hawk", "owl", "peacock", "pigeon", "sparrow", "dog", "cat",
        "puppy", "kitten", "horse", "pony", "donkey", "mule", "cow", "sheep", "goat",
        "pig", "chicken", "llama", "alpaca", "camel", "bull", "cattle", "ox", "ram",
        "lamb", "rooster", "lion", "tiger", "elephant", "deer", "bear", "monkey", "fox",
        "wolf", "rabbit", "squirrel", "dolphin", "whale", "seal", "giraffe", "zebra",
        "antelope", "butterfly", "insect", "bee", "snake", "lizard", "crocodile",
        "fish", "frog", "turtle", "duck"],
    2: ["man", "woman", "person", "people", "child", "children", "kid", "kids", "boy",
        "girl", "baby", "infant", "group", "crowd", "cyclist", "athlete", "tourist",
        "visitor", "monk", "priest", "soldier", "farmer", "worker", "vendor", "couple",
        "family", "spectator"],
    1: ["church", "cathedral", "mosque", "temple", "chapel", "monastery", "convent",
        "basilica", "shrine", "castle", "palace", "fortress", "fort", "ruins", "ruin",
        "ancient", "medieval", "bridge", "tower", "arch", "gate", "monument", "statue",
        "house", "building", "barn", "cottage", "villa", "farmhouse", "street", "road",
        "city", "town", "village", "alley", "square", "plaza", "wall", "facade"],
    0: ["mountain", "hill", "peak", "summit", "alpine", "volcano", "glacier", "cliff",
        "canyon", "cave", "rock", "stone", "gorge", "forest", "tree", "bush", "jungle",
        "woodland", "vegetation", "river", "lake", "sea", "ocean", "waterfall",
        "stream", "pond", "beach", "coast", "shore", "bay", "harbour", "desert", "sand",
        "dune", "arid", "landscape", "valley", "meadow", "field", "plain", "sky",
        "cloud"],
}

DOMAIN_CLASSES = {
    0: [
        (0, "mountain",          ["mountain", "hill", "peak", "summit", "alpine", "volcano", "glacier", "snowy", "ridge", "slope"]),
        (1, "forest_vegetation", ["forest", "tree", "trees", "bush", "jungle", "woodland", "woods", "vegetation", "palm"]),
        (2, "water_scene",       ["river", "lake", "sea", "ocean", "waterfall", "stream", "pond", "beach", "coast", "shore", "bay", "harbour", "water"]),
        (3, "desert_arid",       ["desert", "sand", "dune", "arid", "dry", "barren"]),
        (4, "rock_cliff_canyon", ["rock", "cliff", "canyon", "stone", "cave", "gorge", "formation", "cliffs", "rocks"]),
        (5, "general_landscape", ["landscape", "valley", "meadow", "field", "plain", "sky", "cloud", "view", "scenic", "panorama"]),
    ],
    1: [
        (0, "religious",         ["church", "cathedral", "mosque", "temple", "chapel", "monastery", "convent", "basilica", "shrine", "altar", "nave", "steeple"]),
        (1, "historic",          ["castle", "palace", "fortress", "fort", "ruins", "ruin", "ancient", "archaeological", "historic", "medieval"]),
        (2, "residential",       ["house", "home", "cottage", "villa", "barn", "farmhouse", "residential", "farm"]),
        (3, "infrastructure",    ["bridge", "tower", "arch", "gate", "monument", "statue", "column", "aqueduct", "dam", "lighthouse"]),
        (4, "urban_street",      ["street", "road", "city", "town", "village", "alley", "square", "plaza", "urban"]),
        (5, "general_building",  ["building", "wall", "window", "facade", "structure", "architecture", "complex"]),
    ],
    2: [
        (0, "sports_outdoor",    ["sport", "cycling", "cycle", "bicycle", "football", "tennis", "swim", "ski", "run", "climb", "hike", "surf", "athlete", "player", "race", "competition", "fitness"]),
        (1, "cultural_events",   ["festival", "parade", "ceremony", "dance", "performance", "music", "concert", "celebration", "traditional", "ritual", "costume", "carnival"]),
        (2, "group_social",      ["group", "couple", "family", "crowd", "gathering", "party", "meeting", "together", "several people", "spectator"]),
        (3, "children",          ["child", "children", "kid", "kids", "boy", "girl", "baby", "infant", "young"]),
        (4, "individual",        ["man", "woman", "person", "tourist", "visitor", "traveler", "worker", "soldier", "farmer", "priest", "monk", "vendor"]),
        (5, "general_people",    ["people", "human", "adult", "figure"]),
    ],
    3: [
        (0, "birds",             ["bird", "eagle", "parrot", "flamingo", "heron", "pelican", "stork", "penguin", "seagull", "dove", "hawk", "owl", "peacock", "pigeon", "sparrow", "duck", "goose", "swan", "crane"]),
        (1, "dogs_cats",         ["dog", "cat", "puppy", "kitten", "hound", "canine", "feline"]),
        (2, "horses_equine",     ["horse", "pony", "donkey", "mule", "equine", "stallion", "mare"]),
        (3, "farm_livestock",    ["cow", "sheep", "goat", "pig", "chicken", "llama", "alpaca", "camel", "bull", "cattle", "ox", "ram", "lamb", "ewe", "rooster", "livestock"]),
        (4, "wild_animals",      ["lion", "tiger", "elephant", "deer", "bear", "monkey", "fox", "wolf", "rabbit", "squirrel", "dolphin", "whale", "seal", "giraffe", "zebra", "antelope", "leopard", "cheetah", "jaguar"]),
        (5, "other_animals",     ["butterfly", "insect", "bee", "snake", "lizard", "crocodile", "fish", "frog", "turtle", "spider"]),
    ],
    4: [
        (0, "food_drinks",       ["food", "drink", "meal", "dish", "bread", "cake", "fruit", "vegetable", "restaurant", "cafe", "wine", "coffee", "cuisine"]),
        (1, "vehicles_transport",["car", "bus", "truck", "train", "boat", "ship", "motorcycle", "tram", "airplane", "vehicle", "transport", "ferry"]),
        (2, "indoor_spaces",     ["room", "bedroom", "kitchen", "hall", "interior", "ceiling", "sofa", "furniture", "table", "chair", "corridor"]),
        (3, "art_culture",       ["painting", "sculpture", "art", "museum", "exhibition", "gallery", "artifact", "instrument", "carving"]),
        (4, "market_commercial", ["market", "shop", "store", "souvenir", "craft", "stall", "vendor"]),
        (5, "miscellaneous",     []),
    ],
}

DOMAIN_NAMES = {
    0: "nature_landscape",
    1: "architecture_urban",
    2: "people_activities",
    3: "animals_wildlife",
    4: "objects_misc",
}


def assign_domain_and_class(caption: str) -> Tuple[int, int]:
    text = caption.lower()
    for domain_id in [3, 2, 1, 0, 4]:
        triggers = DOMAIN_TRIGGER.get(domain_id, [])
        if triggers and not any(kw in text for kw in triggers):
            continue
        for class_id, _, class_kws in DOMAIN_CLASSES[domain_id]:
            if class_kws and any(kw in text for kw in class_kws):
                return domain_id, class_id
        return domain_id, 5
    return 4, 5


def normalize_image_path(raw_path: str) -> str:
    if raw_path.startswith("./"):
        return str(DATA_BASE_DIR / raw_path[2:])
    if Path(raw_path).is_absolute():
        return raw_path
    return str(DATA_BASE_DIR / raw_path)


def load_all_raw_samples() -> Dict[str, Dict[str, object]]:
    logger.info("=" * 70)
    logger.info("Step 1: Loading raw IAPR JSON metadata")
    logger.info("=" * 70)

    all_samples: List[Dict[str, object]] = []
    for json_path in (TRAIN_JSON, VAL_JSON):
        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        logger.info("Loaded %d samples from %s", len(samples), json_path)
        all_samples.extend(samples)

    samples_by_id: Dict[str, Dict[str, object]] = {}
    for sample in all_samples:
        image_path = normalize_image_path(sample["image_path"])
        sample_id = Path(image_path).name
        domain_id, class_id = assign_domain_and_class(sample["caption"])
        record = {
            "id": sample_id,
            "image_path": image_path,
            "caption": sample["caption"],
            "domain_id": domain_id,
            "class_id": domain_id * 6 + class_id,
        }
        if sample_id in samples_by_id:
            raise RuntimeError(f"Duplicate sample id detected: {sample_id}")
        samples_by_id[sample_id] = record

    logger.info("Prepared %d raw samples", len(samples_by_id))
    return samples_by_id


def load_clip_template(samples_by_id: Dict[str, Dict[str, object]]) -> Dict[int, Dict[str, Dataset]]:
    logger.info("=" * 70)
    logger.info("Step 2: Loading CLIP IAPR template splits")
    logger.info("=" * 70)

    template: Dict[int, Dict[str, Dataset]] = {}
    for domain_idx in range(NUM_DOMAINS):
        domain_dir = CLIP_DOMAIN_ROOT / f"domain_dataset_{domain_idx}"
        train_ds = load_from_disk(str(domain_dir / "train"))
        test_ds = load_from_disk(str(domain_dir / "test"))

        for split_name, ds in (("train", train_ds), ("test", test_ds)):
            for row in ds:
                sample_id = row["id"]
                if sample_id not in samples_by_id:
                    raise RuntimeError(f"Sample id {sample_id} from CLIP dataset not found in raw JSON")
                raw_meta = samples_by_id[sample_id]
                if row["class_id"] != raw_meta["class_id"]:
                    raise RuntimeError(
                        f"class_id mismatch for {sample_id}: clip={row['class_id']} raw={raw_meta['class_id']}"
                    )
                if row["domain_id"] != raw_meta["domain_id"]:
                    raise RuntimeError(
                        f"domain_id mismatch for {sample_id}: clip={row['domain_id']} raw={raw_meta['domain_id']}"
                    )

        template[domain_idx] = {"train": train_ds, "test": test_ds}
        logger.info(
            "Domain %d: template train=%d, test=%d",
            domain_idx,
            train_ds.num_rows,
            test_ds.num_rows,
        )
    return template


def load_siglip() -> Tuple[SiglipModel, SiglipProcessor]:
    logger.info("=" * 70)
    logger.info("Step 3: Loading SigLIP model")
    logger.info("=" * 70)

    model = SiglipModel.from_pretrained(SIGLIP_MODEL_NAME).to(DEVICE)
    model.eval()
    processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    logger.info("SigLIP model loaded: %s", SIGLIP_MODEL_NAME)
    return model, processor


def encode_batch(
    model: SiglipModel,
    processor: SiglipProcessor,
    batch_samples: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    images = []
    texts = []
    valid_samples = []

    for sample in batch_samples:
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open image {sample['image_path']}: {exc}") from exc
        images.append(img)
        texts.append(sample["caption"])
        valid_samples.append(sample)

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-10)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)

    image_features = image_features.cpu().tolist()
    text_features = text_features.cpu().tolist()

    encoded = []
    for idx, sample in enumerate(valid_samples):
        encoded.append(
            {
                "id": sample["id"],
                "class_id": sample["class_id"],
                "processed_img": image_features[idx],
                "cap_tokens": text_features[idx],
                "domain_id": sample["domain_id"],
            }
        )
    return encoded


def build_split_records(
    clip_split: Dataset,
    samples_by_id: Dict[str, Dict[str, object]],
    model: SiglipModel,
    processor: SiglipProcessor,
) -> List[Dict[str, object]]:
    ordered_samples = [samples_by_id[row["id"]] for row in clip_split]
    encoded_records: List[Dict[str, object]] = []

    for start in tqdm(range(0, len(ordered_samples), BATCH_SIZE), desc="Encoding", leave=False):
        batch = ordered_samples[start:start + BATCH_SIZE]
        encoded_records.extend(encode_batch(model, processor, batch))
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Keep exact template order and verify ids/class/domain remain aligned.
    for encoded_row, clip_row in zip(encoded_records, clip_split):
        if encoded_row["id"] != clip_row["id"]:
            raise RuntimeError(f"id order mismatch for {encoded_row['id']} vs {clip_row['id']}")
        if encoded_row["class_id"] != clip_row["class_id"]:
            raise RuntimeError(f"class_id mismatch for {encoded_row['id']}")
        if encoded_row["domain_id"] != clip_row["domain_id"]:
            raise RuntimeError(f"domain_id mismatch for {encoded_row['id']}")

    return encoded_records


def save_outputs(template: Dict[int, Dict[str, Dataset]], samples_by_id: Dict[str, Dict[str, object]]) -> None:
    model, processor = load_siglip()

    logger.info("=" * 70)
    logger.info("Step 4: Encoding splits with SigLIP and saving datasets")
    logger.info("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for domain_idx in range(NUM_DOMAINS):
        out_domain_dir = OUTPUT_DIR / f"domain_dataset_{domain_idx}"
        out_domain_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ("train", "test"):
            clip_split = template[domain_idx][split_name]
            logger.info(
                "Domain %d %s: encoding %d samples",
                domain_idx,
                split_name,
                clip_split.num_rows,
            )
            split_records = build_split_records(clip_split, samples_by_id, model, processor)
            split_path = out_domain_dir / split_name
            if split_path.exists():
                shutil.rmtree(split_path)
            Dataset.from_list(split_records).save_to_disk(str(split_path))

        src_mapping = CLIP_DOMAIN_ROOT / f"domain_dataset_{domain_idx}" / "class_mapping.pkl"
        dst_mapping = out_domain_dir / "class_mapping.pkl"
        shutil.copy2(src_mapping, dst_mapping)

    label2id_src = CLIP_ROOT / "label2id.json"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label2id_src, OUTPUT_ROOT / "label2id.json")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def validate_saved_outputs(template: Dict[int, Dict[str, Dataset]]) -> None:
    logger.info("=" * 70)
    logger.info("Step 5: Validating saved SigLIP dataset against CLIP template")
    logger.info("=" * 70)

    dim_counter = Counter()
    for domain_idx in range(NUM_DOMAINS):
        for split_name in ("train", "test"):
            clip_split = template[domain_idx][split_name]
            siglip_split = load_from_disk(str(OUTPUT_DIR / f"domain_dataset_{domain_idx}" / split_name))

            if clip_split.num_rows != siglip_split.num_rows:
                raise RuntimeError(
                    f"Row count mismatch in domain {domain_idx} {split_name}: "
                    f"clip={clip_split.num_rows}, siglip={siglip_split.num_rows}"
                )
            if clip_split.column_names != siglip_split.column_names:
                raise RuntimeError(
                    f"Column mismatch in domain {domain_idx} {split_name}: "
                    f"clip={clip_split.column_names}, siglip={siglip_split.column_names}"
                )

            for clip_row, siglip_row in zip(clip_split, siglip_split):
                for field in ("id", "class_id", "domain_id"):
                    if clip_row[field] != siglip_row[field]:
                        raise RuntimeError(
                            f"Field mismatch in domain {domain_idx} {split_name} for {field}: "
                            f"{clip_row[field]} vs {siglip_row[field]}"
                        )
                dim_counter["processed_img"] = len(siglip_row["processed_img"])
                dim_counter["cap_tokens"] = len(siglip_row["cap_tokens"])

        with open(CLIP_DOMAIN_ROOT / f"domain_dataset_{domain_idx}" / "class_mapping.pkl", "rb") as f:
            clip_mapping = pickle.load(f)
        with open(OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "class_mapping.pkl", "rb") as f:
            siglip_mapping = pickle.load(f)
        if clip_mapping != siglip_mapping:
            raise RuntimeError(f"class_mapping mismatch in domain {domain_idx}")

    with open(CLIP_ROOT / "label2id.json", "r", encoding="utf-8") as f:
        clip_label2id = json.load(f)
    with open(OUTPUT_ROOT / "label2id.json", "r", encoding="utf-8") as f:
        siglip_label2id = json.load(f)
    if clip_label2id != siglip_label2id:
        raise RuntimeError("label2id.json mismatch")

    logger.info("Validation passed. SigLIP feature dims: %s", dict(dim_counter))


def main() -> None:
    samples_by_id = load_all_raw_samples()
    template = load_clip_template(samples_by_id)
    save_outputs(template, samples_by_id)
    validate_saved_outputs(template)

    logger.info("=" * 70)
    logger.info("All done!")
    for domain_idx in range(NUM_DOMAINS):
        train_ds = load_from_disk(str(OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "train"))
        test_ds = load_from_disk(str(OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "test"))
        logger.info(
            "Domain %d (%s): train=%d, test=%d",
            domain_idx,
            DOMAIN_NAMES[domain_idx],
            train_ds.num_rows,
            test_ds.num_rows,
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
