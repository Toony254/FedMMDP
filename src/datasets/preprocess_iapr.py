"""
IAPR TC-12 数据集预处理脚本（参照 preprocess_imagenet.py / preprocess_food.py）

输入：data/iapr_tc12/train.json + validation.json（image_path + caption）
输出：preprocessed_iapr/domain_datasets/domain_dataset_{0-4}/train|test
字段：id, class_id, processed_img (CLIP RN50 1024-dim), cap_tokens (CLIP RN50 1024-dim), domain_id

5 个语义域（基于 caption 关键词匹配，优先级 Domain3>Domain2>Domain1>Domain0>Domain4）：
  Domain 0 – Nature & Landscape
  Domain 1 – Architecture & Urban
  Domain 2 – People & Activities
  Domain 3 – Animals & Wildlife
  Domain 4 – Objects & Misc (catch-all)
"""

import os
import sys
import json
import pickle
import logging
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import torch
import clip
from PIL import Image
from tqdm import tqdm
from datasets import Dataset

# ============================================================
# 配置
# ============================================================
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(os.getcwd())
TRAIN_JSON = BASE_DIR / "data/iapr_tc12/train.json"
VAL_JSON   = BASE_DIR / "data/iapr_tc12/validation.json"
OUTPUT_DIR = BASE_DIR / "preprocessed_iapr/domain_datasets"
NUM_DOMAINS = 5
TEST_RATIO  = 0.1
BATCH_SIZE  = 64
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

logger.info(f"Device: {DEVICE}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# ============================================================
# 语义域 + 类别分类表
# ============================================================
DOMAIN_TRIGGER = {
    3: ["bird","eagle","parrot","flamingo","heron","pelican","stork","penguin",
        "seagull","dove","hawk","owl","peacock","pigeon","sparrow","dog","cat",
        "puppy","kitten","horse","pony","donkey","mule","cow","sheep","goat",
        "pig","chicken","llama","alpaca","camel","bull","cattle","ox","ram",
        "lamb","rooster","lion","tiger","elephant","deer","bear","monkey","fox",
        "wolf","rabbit","squirrel","dolphin","whale","seal","giraffe","zebra",
        "antelope","butterfly","insect","bee","snake","lizard","crocodile",
        "fish","frog","turtle","duck"],
    2: ["man","woman","person","people","child","children","kid","kids","boy",
        "girl","baby","infant","group","crowd","cyclist","athlete","tourist",
        "visitor","monk","priest","soldier","farmer","worker","vendor","couple",
        "family","spectator"],
    1: ["church","cathedral","mosque","temple","chapel","monastery","convent",
        "basilica","shrine","castle","palace","fortress","fort","ruins","ruin",
        "ancient","medieval","bridge","tower","arch","gate","monument","statue",
        "house","building","barn","cottage","villa","farmhouse","street","road",
        "city","town","village","alley","square","plaza","wall","facade"],
    0: ["mountain","hill","peak","summit","alpine","volcano","glacier","cliff",
        "canyon","cave","rock","stone","gorge","forest","tree","bush","jungle",
        "woodland","vegetation","river","lake","sea","ocean","waterfall",
        "stream","pond","beach","coast","shore","bay","harbour","desert","sand",
        "dune","arid","landscape","valley","meadow","field","plain","sky",
        "cloud"],
}

DOMAIN_CLASSES = {
    0: [
        (0,"mountain",          ["mountain","hill","peak","summit","alpine","volcano","glacier","snowy","ridge","slope"]),
        (1,"forest_vegetation", ["forest","tree","trees","bush","jungle","woodland","woods","vegetation","palm"]),
        (2,"water_scene",       ["river","lake","sea","ocean","waterfall","stream","pond","beach","coast","shore","bay","harbour","water"]),
        (3,"desert_arid",       ["desert","sand","dune","arid","dry","barren"]),
        (4,"rock_cliff_canyon", ["rock","cliff","canyon","stone","cave","gorge","formation","cliffs","rocks"]),
        (5,"general_landscape", ["landscape","valley","meadow","field","plain","sky","cloud","view","scenic","panorama"]),
    ],
    1: [
        (0,"religious",         ["church","cathedral","mosque","temple","chapel","monastery","convent","basilica","shrine","altar","nave","steeple"]),
        (1,"historic",          ["castle","palace","fortress","fort","ruins","ruin","ancient","archaeological","historic","medieval"]),
        (2,"residential",       ["house","home","cottage","villa","barn","farmhouse","residential","farm"]),
        (3,"infrastructure",    ["bridge","tower","arch","gate","monument","statue","column","aqueduct","dam","lighthouse"]),
        (4,"urban_street",      ["street","road","city","town","village","alley","square","plaza","urban"]),
        (5,"general_building",  ["building","wall","window","facade","structure","architecture","complex"]),
    ],
    2: [
        (0,"sports_outdoor",    ["sport","cycling","cycle","bicycle","football","tennis","swim","ski","run","climb","hike","surf","athlete","player","race","competition","fitness"]),
        (1,"cultural_events",   ["festival","parade","ceremony","dance","performance","music","concert","celebration","traditional","ritual","costume","carnival"]),
        (2,"group_social",      ["group","couple","family","crowd","gathering","party","meeting","together","several people","spectator"]),
        (3,"children",          ["child","children","kid","kids","boy","girl","baby","infant","young"]),
        (4,"individual",        ["man","woman","person","tourist","visitor","traveler","worker","soldier","farmer","priest","monk","vendor"]),
        (5,"general_people",    ["people","human","adult","figure"]),
    ],
    3: [
        (0,"birds",             ["bird","eagle","parrot","flamingo","heron","pelican","stork","penguin","seagull","dove","hawk","owl","peacock","pigeon","sparrow","duck","goose","swan","crane"]),
        (1,"dogs_cats",         ["dog","cat","puppy","kitten","hound","canine","feline"]),
        (2,"horses_equine",     ["horse","pony","donkey","mule","equine","stallion","mare"]),
        (3,"farm_livestock",    ["cow","sheep","goat","pig","chicken","llama","alpaca","camel","bull","cattle","ox","ram","lamb","ewe","rooster","livestock"]),
        (4,"wild_animals",      ["lion","tiger","elephant","deer","bear","monkey","fox","wolf","rabbit","squirrel","dolphin","whale","seal","giraffe","zebra","antelope","leopard","cheetah","jaguar"]),
        (5,"other_animals",     ["butterfly","insect","bee","snake","lizard","crocodile","fish","frog","turtle","spider"]),
    ],
    4: [
        (0,"food_drinks",       ["food","drink","meal","dish","bread","cake","fruit","vegetable","restaurant","cafe","wine","coffee","cuisine"]),
        (1,"vehicles_transport",["car","bus","truck","train","boat","ship","motorcycle","tram","airplane","vehicle","transport","ferry"]),
        (2,"indoor_spaces",     ["room","bedroom","kitchen","hall","interior","ceiling","sofa","furniture","table","chair","corridor"]),
        (3,"art_culture",       ["painting","sculpture","art","museum","exhibition","gallery","artifact","instrument","carving"]),
        (4,"market_commercial", ["market","shop","store","souvenir","craft","stall","vendor"]),
        (5,"miscellaneous",     []),
    ],
}

DOMAIN_NAMES = {
    0:"nature_landscape", 1:"architecture_urban",
    2:"people_activities", 3:"animals_wildlife", 4:"objects_misc",
}


def assign_domain_and_class(caption: str):
    text = caption.lower()
    for domain_id in [3, 2, 1, 0, 4]:
        triggers = DOMAIN_TRIGGER.get(domain_id, [])
        if triggers and not any(kw in text for kw in triggers):
            continue
        for (class_id, class_name, class_kws) in DOMAIN_CLASSES[domain_id]:
            if class_kws and any(kw in text for kw in class_kws):
                return domain_id, class_id
        return domain_id, 5
    return 4, 5


# ============================================================
# Step 1: Load JSON data
# ============================================================
logger.info("=" * 70)
logger.info("Step 1: Loading IAPR TC-12 JSON data")
logger.info("=" * 70)

all_samples = []
for json_path in [TRAIN_JSON, VAL_JSON]:
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    logger.info(f"Loaded {len(samples)} samples from {json_path}")
    all_samples.extend(samples)

logger.info(f"Total samples: {len(all_samples)}")

for s in all_samples:
    raw_path = s["image_path"]
    if raw_path.startswith("./"):
        s["image_path"] = str(BASE_DIR / raw_path[2:])
    elif not os.path.isabs(raw_path):
        s["image_path"] = str(BASE_DIR / raw_path)

# ============================================================
# Step 2: Assign domains and classes
# ============================================================
logger.info("=" * 70)
logger.info("Step 2: Assigning semantic domains and class labels")
logger.info("=" * 70)

for s in all_samples:
    s["domain_id"], s["class_id"] = assign_domain_and_class(s["caption"])

domain_counter = Counter(s["domain_id"] for s in all_samples)
for d in range(NUM_DOMAINS):
    class_counter = Counter(s["class_id"] for s in all_samples if s["domain_id"] == d)
    logger.info(
        f"Domain {d} ({DOMAIN_NAMES[d]}): {domain_counter[d]} samples, "
        f"classes: {dict(sorted(class_counter.items()))}"
    )

# ============================================================
# Step 3: Load CLIP RN50
# ============================================================
logger.info("=" * 70)
logger.info("Step 3: Loading CLIP RN50 model")
logger.info("=" * 70)

model, preprocess = clip.load("RN50", device=DEVICE)
model.eval()
tokenizer = clip.tokenize
logger.info("CLIP model loaded")

# ============================================================
# Step 4: Batch encode images + captions
# ============================================================
logger.info("=" * 70)
logger.info("Step 4: Encoding with CLIP")
logger.info("=" * 70)


def encode_batch(image_paths, captions):
    imgs, valid_indices = [], []
    for i, p in enumerate(image_paths):
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
            valid_indices.append(i)
        except Exception as e:
            logger.warning(f"Failed to load image {p}: {e}")
    if not imgs:
        return None, None, valid_indices
    img_tensor = torch.stack(imgs).to(DEVICE)
    valid_captions = [captions[i] for i in valid_indices]
    text_tokens = tokenizer(valid_captions, truncate=True).to(DEVICE)
    with torch.no_grad():
        img_feats = model.encode_image(img_tensor)
        txt_feats = model.encode_text(text_tokens)
        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-10)
        txt_feats = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-10)
    return img_feats.cpu().numpy(), txt_feats.cpu().numpy(), valid_indices


encoded_records = []

for batch_start in tqdm(range(0, len(all_samples), BATCH_SIZE), desc="Encoding"):
    batch = all_samples[batch_start : batch_start + BATCH_SIZE]
    img_feats, txt_feats, valid_idx = encode_batch(
        [s["image_path"] for s in batch],
        [s["caption"]    for s in batch],
    )
    if img_feats is None:
        continue
    for rank, orig_idx in enumerate(valid_idx):
        s = batch[orig_idx]
        encoded_records.append({
            "id":            os.path.basename(s["image_path"]),
            "class_id":      s["domain_id"] * 6 + s["class_id"],   # global unique: domain*6+local
            "processed_img": img_feats[rank].tolist(),
            "cap_tokens":    txt_feats[rank].tolist(),
            "domain_id":     s["domain_id"],
        })
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

logger.info(f"Encoded {len(encoded_records)} / {len(all_samples)} samples")

# ============================================================
# Step 5: Split by domain → train/test → save HF Datasets
# ============================================================
logger.info("=" * 70)
logger.info("Step 5: Saving per-domain HuggingFace Datasets")
logger.info("=" * 70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

domain_records = defaultdict(list)
for rec in encoded_records:
    domain_records[rec["domain_id"]].append(rec)

rng = random.Random(RANDOM_SEED)
all_domain_info = {}

for domain_idx in range(NUM_DOMAINS):
    records = domain_records[domain_idx]
    if not records:
        logger.warning(f"Domain {domain_idx}: no samples, skipping")
        continue
    rng.shuffle(records)
    split_at      = max(1, int(len(records) * (1 - TEST_RATIO)))
    train_records = records[:split_at]
    test_records  = records[split_at:]

    dom_dir = OUTPUT_DIR / f"domain_dataset_{domain_idx}"
    dom_dir.mkdir(exist_ok=True)

    Dataset.from_list(train_records).save_to_disk(str(dom_dir / "train"))
    Dataset.from_list(test_records).save_to_disk(str(dom_dir  / "test"))

    class_counter  = Counter(r["class_id"] for r in records)
    class_names_map = {cid: cname for (cid, cname, _) in DOMAIN_CLASSES[domain_idx]}
    mapping = {
        "domain_name":      DOMAIN_NAMES[domain_idx],
        "class_id_to_name": class_names_map,
        "class_counts":     dict(class_counter),
        "num_classes":      len(class_counter),
        "num_train":        len(train_records),
        "num_test":         len(test_records),
    }
    with open(dom_dir / "class_mapping.pkl", "wb") as f:
        pickle.dump(mapping, f)
    all_domain_info[domain_idx] = mapping

    logger.info(
        f"Domain {domain_idx} ({DOMAIN_NAMES[domain_idx]}): "
        f"train={len(train_records)}, test={len(test_records)}, "
        f"classes={mapping['num_classes']}"
    )

# ============================================================
# Save global label2id
# ============================================================
label2id = {}
global_offset = 0
for domain_idx in range(NUM_DOMAINS):
    for (cid, cname, _) in DOMAIN_CLASSES[domain_idx]:
        label2id[f"domain{domain_idx}_{cname}"] = global_offset + cid
    global_offset += len(DOMAIN_CLASSES[domain_idx])

label2id_path = BASE_DIR / "preprocessed_iapr" / "label2id.json"
label2id_path.parent.mkdir(exist_ok=True)
with open(label2id_path, "w", encoding="utf-8") as f:
    json.dump(label2id, f, indent=2, ensure_ascii=False)
logger.info(f"Saved label2id to {label2id_path}")

# ============================================================
# Summary
# ============================================================
logger.info("=" * 70)
logger.info("All done!")
for domain_idx in range(NUM_DOMAINS):
    if domain_idx in all_domain_info:
        info = all_domain_info[domain_idx]
        logger.info(
            f"  Domain {domain_idx} ({info['domain_name']}): "
            f"train={info['num_train']}, test={info['num_test']}, "
            f"classes={info['num_classes']}"
        )
logger.info("=" * 70)
