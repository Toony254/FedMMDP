import os
import json
import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from datasets import Dataset, Features, Value, Image as HFImage, Sequence, ClassLabel

# 1. 配置路径与参�?Dataset roots are configured below.
ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_DATA_DIR = Path(os.environ.get("FEDMMDP_FOOD101_ROOT", str(ROOT_DIR / "data" / "UMPC-FOOD-101")))
OUTPUT_DIR = Path("preprocessed_food")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

print(f"Processing UMPC-FOOD-101 from {SOURCE_DATA_DIR} to {OUTPUT_DIR}")
print(f"Using device: {DEVICE}")

# 2. 加载 CLIP 模型
print("Loading CLIP RN50...")
model, preprocess = clip.load("RN50", device=DEVICE)
model.eval()

# 3. 准备类别映射 (label2id)
print("Scanning classes...")
train_img_dir = SOURCE_DATA_DIR / "images" / "train"
test_img_dir = SOURCE_DATA_DIR / "images" / "test"

classes = set()
for d in [train_img_dir, test_img_dir]:
    if d.exists():
        for p in d.iterdir():
            if p.is_dir():
                classes.add(p.name)

class_names = sorted(list(classes))
label2id = {c: i for i, c in enumerate(class_names)}
print(f"Found {len(class_names)} classes.")

# 4. 定义数据生成�?def load_metadata_map(csv_path):
    meta = {}
    if not csv_path.exists():
        return meta
    try:
        df = pd.read_csv(csv_path)
        # 简单推断列名：假设包含 filename, description
        cols = df.columns
        fname_col = next((c for c in cols if c.lower() in ['filename', 'file', 'image']), cols[0])
        desc_col = next((c for c in cols if c.lower() in ['description', 'desc', 'text', 'caption', 'title']), cols[1] if len(cols)>1 else None)
        
        for _, row in df.iterrows():
            fname = str(row[fname_col]).strip()
            if fname:
                meta[fname] = str(row[desc_col]).strip() if desc_col else ""
    except Exception as e:
        print(f"Warning: Failed to load metadata from {csv_path}: {e}")
    return meta

def dataset_generator(split):
    img_dir = SOURCE_DATA_DIR / "images" / split
    csv_path = SOURCE_DATA_DIR / "texts" / f"{split}_titles.csv"
    
    if not img_dir.exists():
        return

    meta_map = load_metadata_map(csv_path)
    
    for class_dir in sorted(img_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        label = class_dir.name
        label_id = label2id[label]
        
        # 遍历图片
        for img_path in sorted(class_dir.glob("*.*")):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                continue
                
            fname = img_path.name
            text = meta_map.get(fname, "")
            # 若无文本，使用类别名作为描述
            if not text:
                text = label.replace("_", " ")
            
            yield {
                "id": fname,
                "image": str(img_path),
                "text": text,
                "label": label,
                "label_id": label_id
            }

# 5. 创建 HF Dataset
features = Features({
    "id": Value("string"),
    "image": HFImage(),
    "text": Value("string"),
    "label": Value("string"),
    "label_id": ClassLabel(names=class_names)
})

print("Generating train dataset...")
train_ds = Dataset.from_generator(dataset_generator, gen_kwargs={"split": "train"}, features=features)
print(f"Train samples: {len(train_ds)}")

print("Generating test dataset...")
test_ds = Dataset.from_generator(dataset_generator, gen_kwargs={"split": "test"}, features=features)
print(f"Test samples: {len(test_ds)}")

# 6. 计算 Embeddings
def compute_embeddings(batch):
    # batch["image"] 已经�?PIL Image 对象列表（由 HF Dataset 自动加载�?    images = [img.convert("RGB") for img in batch["image"]]
    texts = batch["text"]
    
    # 预处�?    image_inputs = torch.stack([preprocess(img) for img in images]).to(DEVICE)
    text_inputs = clip.tokenize(texts, truncate=True).to(DEVICE)
    
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)
        
        # 归一�?        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
    return {
        "image_emb": image_features.cpu().numpy(),
        "text_emb": text_features.cpu().numpy()
    }

print("Computing embeddings for train set (this may take a while)...")
train_ds = train_ds.map(compute_embeddings, batched=True, batch_size=BATCH_SIZE)

print("Computing embeddings for test set...")
test_ds = test_ds.map(compute_embeddings, batched=True, batch_size=BATCH_SIZE)

# 7. 保存结果
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
print(f"Saving datasets to {OUTPUT_DIR}...")
train_ds.save_to_disk(OUTPUT_DIR / "train")
test_ds.save_to_disk(OUTPUT_DIR / "test")

# 保存 label2id
with open(OUTPUT_DIR / "label2id.json", "w", encoding="utf-8") as f:
    json.dump(label2id, f, indent=2, ensure_ascii=False)

print("Preprocessing complete!")

# ============================================================
# 按标签划�?5 个领域并重命名字段，生成精简版领域数据集
# ============================================================
from datasets import load_from_disk

preprocessed_root = os.path.join(os.getcwd(), "preprocessed_food")
domain_out_root = os.environ.get(
    "FEDMMDP_FOOD_DOMAIN_ROOT",
    str(ROOT_DIR / "data" / "preprocessed_food" / "domain_datasets"),
)
os.makedirs(domain_out_root, exist_ok=True)

print("Loading saved datasets for domain split...")
train_ds_loaded = load_from_disk(os.path.join(preprocessed_root, "train"))
test_ds_loaded = load_from_disk(os.path.join(preprocessed_root, "test"))

def prepare(ds):
    # domain_id: 依据 class_id 均匀映射�?5 个领域，互不重叠
    ds = ds.map(lambda x: {"domain_id": x["label_id"] % 5})
    # 重命名表征字�?    ds = ds.rename_column("image_emb", "processed_img")
    ds = ds.rename_column("text_emb", "cap_tokens")
    ds = ds.rename_column("label_id", "class_id")
    # 删除原始图像/描述字段
    drop_cols = [c for c in ["image", "text", "label"] if c in ds.column_names]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    return ds

train_ds_prepped = prepare(train_ds_loaded)
test_ds_prepped = prepare(test_ds_loaded)

for dom in range(5):
    dom_dir = os.path.join(domain_out_root, f"domain_dataset_{dom}")
    os.makedirs(dom_dir, exist_ok=True)
    dom_train = train_ds_prepped.filter(lambda x: x["domain_id"] == dom)
    dom_test = test_ds_prepped.filter(lambda x: x["domain_id"] == dom)
    dom_train.save_to_disk(os.path.join(dom_dir, "train"))
    dom_test.save_to_disk(os.path.join(dom_dir, "test"))
    print(f"Saved domain_dataset_{dom}: train {len(dom_train)}, test {len(dom_test)}")

