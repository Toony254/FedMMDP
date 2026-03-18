import io
import sys
import shutil
import logging
import pickle
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import Dataset
from transformers import SiglipProcessor, SiglipModel

sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# ============================================
# 配置参数
# ============================================
RAW_DIR = Path("preprocessed_imagenet/domain_datasets/raw")           # 已分割的 raw 数据集目录
OUTPUT_DIR = Path("preprocessed_imagenet/domain_datasets/siglip") # 输出目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODE_BATCH_SIZE = 64   # SigLIP 编码时的批次大小
NUM_DOMAINS = 5

logger.info(f"Device: {DEVICE}")
logger.info(f"Raw data directory: {RAW_DIR}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# ============================================
# 步骤1: 加载 SigLIP 模型
# ============================================
logger.info("=" * 70)
logger.info("Step 1: Loading SigLIP model")
logger.info("=" * 70)

SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
siglip_model = SiglipModel.from_pretrained(SIGLIP_MODEL_NAME).to(DEVICE)
siglip_model.eval()
siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_NAME)
logger.info("SigLIP model loaded")

# ============================================
# 步骤2: 定义批量编码函数
# ============================================

def encode_batch(examples):
    """
    HuggingFace Dataset.map 批量编码函数。
    - 若数据集含 "image" 列（原始字节），则用 SigLIP 重新编码图像。
    - 若不含 "image" 列（已有 processed_img 特征），则保留原图像特征，仅用 SigLIP 重新编码文本。
    """
    texts = examples["recaption_short"]
    result = {}

    has_image_col = "image" in examples

    if has_image_col:
        # 有原始图像字节：同时编码图像和文本
        images = []
        for img_bytes in examples["image"]:
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            images.append(img)

        inputs = siglip_processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = siglip_model(**inputs)
            img_feats = outputs.image_embeds
            txt_feats = outputs.text_embeds
            img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-10)
            txt_feats = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-10)

        result["processed_img"] = img_feats.cpu().numpy().tolist()
        result["cap_tokens"] = txt_feats.cpu().numpy().tolist()
    else:
        # 无原始图像：保留已有 processed_img，仅用 SigLIP 重新编码文本
        text_inputs = siglip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

        with torch.no_grad():
            txt_feats = siglip_model.text_model(**text_inputs).pooler_output
            txt_feats = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-10)

        result["cap_tokens"] = txt_feats.cpu().numpy().tolist()

    return result

# ============================================
# 步骤3: 逐域处理 train / test
# ============================================
logger.info("=" * 70)
logger.info("Step 2: Encoding datasets with SigLIP")
logger.info("=" * 70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for domain_idx in range(NUM_DOMAINS):
    raw_domain_dir = RAW_DIR / f"domain_dataset_{domain_idx}"
    out_domain_dir = OUTPUT_DIR / f"domain_dataset_{domain_idx}"
    out_domain_dir.mkdir(exist_ok=True)

    for split in ("train", "test"):
        raw_split_path = raw_domain_dir / split
        out_split_path = out_domain_dir / split

        if not raw_split_path.exists():
            logger.warning(f"Domain {domain_idx} {split}: path not found ({raw_split_path}), skipping")
            continue

        if out_split_path.exists():
            logger.info(f"Domain {domain_idx} {split}: already exists, skipping")
            continue

        logger.info(f"Domain {domain_idx} {split}: loading from {raw_split_path}")
        ds = Dataset.load_from_disk(str(raw_split_path))
        logger.info(f"  {len(ds)} samples, columns: {ds.column_names}")

        logger.info(f"  Encoding with SigLIP (batch_size={ENCODE_BATCH_SIZE})...")
        cols_to_remove = [c for c in ["image"] if c in ds.column_names]
        ds_encoded = ds.map(
            encode_batch,
            batched=True,
            batch_size=ENCODE_BATCH_SIZE,
            remove_columns=cols_to_remove,  # 仅当 image 列存在时才移除
            desc=f"domain_{domain_idx}/{split}",
        )

        ds_encoded.save_to_disk(str(out_split_path))
        logger.info(f"  Saved {len(ds_encoded)} samples to {out_split_path}")

        del ds, ds_encoded
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # 复制 class_mapping.pkl（如果存在）
    src_mapping = raw_domain_dir / "class_mapping.pkl"
    dst_mapping = out_domain_dir / "class_mapping.pkl"
    if src_mapping.exists() and not dst_mapping.exists():
        shutil.copy2(str(src_mapping), str(dst_mapping))
        logger.info(f"Domain {domain_idx}: copied class_mapping.pkl")

# ============================================
# 完成，打印统计
# ============================================
logger.info("=" * 70)
logger.info("All processing completed!")
logger.info("=" * 70)

for domain_idx in range(NUM_DOMAINS):
    train_path = OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "train"
    test_path  = OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "test"
    mapping_path = OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "class_mapping.pkl"

    if train_path.exists() and test_path.exists():
        train_ds = Dataset.load_from_disk(str(train_path))
        test_ds  = Dataset.load_from_disk(str(test_path))
        num_classes = "?"
        if mapping_path.exists():
            with open(mapping_path, "rb") as f:
                num_classes = pickle.load(f).get("num_classes", "?")
        logger.info(f"Domain {domain_idx}: train={len(train_ds)}, test={len(test_ds)}, classes={num_classes}")
    else:
        logger.info(f"Domain {domain_idx}: NOT FOUND")

logger.info("=" * 70)
