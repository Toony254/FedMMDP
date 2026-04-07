import os, json, h5py, numpy as np, torch
import clip
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage, Sequence, ClassLabel
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 路径与模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, img_preprocess = clip.load("RN50", device=device)
clip_model.eval()
clip_dim = clip_model.text_projection.shape[1]

if 'dir_fashion_gen' not in globals():
    dir_fashion_gen = os.environ.get("FEDMMDP_FASHION_GEN_ROOT", os.path.join(ROOT_DIR, "data", "fashion-gen"))
train_path = os.path.join(dir_fashion_gen, "fashiongen_256_256_train.h5")

def pick_split(options):
    for fname in options:
        cand = os.path.join(dir_fashion_gen, fname)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("No validation/test split found in dir_fashion_gen")

test_path = pick_split([
    "fashiongen_256_256_validation.h5",
    "fashiongen_256_256_valid.h5",
    "fashiongen_256_256_test.h5",
])

# 收集类别映射
def collect_categories(paths, key="input_category", step=50000):
    cats = set()
    for path in paths:
        with h5py.File(path, "r") as f:
            data = f[key]
            total = len(data)
            for start in range(0, total, step):
                end = min(start + step, total)
                batch = data[start:end]
                for v in batch:
                    if isinstance(v, bytes):
                        cats.add(v.decode("utf-8"))
                    else:
                        cats.add(str(v))
    return sorted(cats)

class_names = collect_categories([train_path, test_path])
label2id = {c: i for i, c in enumerate(class_names)}

base_features = Features({
    "index": Value("int64"),
    "input_image": HFImage(),
    "input_description": Value("string"),
    "input_category": Value("string"),
})

def gen_examples(path):
    def _gen():
        with h5py.File(path, "r") as f:
            imgs = f["input_image"]
            descs = f["input_description"]
            cats = f["input_category"]
            idxs = f["index"] if "index" in f else None
            total = len(imgs)
            for i in range(total):
                desc = descs[i]
                cat = cats[i]
                yield {
                    "index": int(idxs[i]) if idxs is not None else i,
                    "input_image": imgs[i],
                    "input_description": desc.decode("utf-8") if isinstance(desc, bytes) else str(desc),
                    "input_category": cat.decode("utf-8") if isinstance(cat, bytes) else str(cat),
                }
    return _gen

train_ds = Dataset.from_generator(gen_examples(train_path), features=base_features)
test_ds = Dataset.from_generator(gen_examples(test_path), features=base_features)

def add_embeds(batch):
    pil_imgs = [img if isinstance(img, Image.Image) else Image.fromarray(np.array(img)) for img in batch["input_image"]]
    pixel_values = torch.stack([img_preprocess(im) for im in pil_imgs]).to(device)
    text_tokens = clip.tokenize(batch["input_description"], truncate=True).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(pixel_values)
        txt_feat = clip_model.encode_text(text_tokens)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    img_feat = img_feat.cpu().numpy().astype("float32")
    txt_feat = txt_feat.cpu().numpy().astype("float32")
    class_ids = [label2id[c] for c in batch["input_category"]]
    return {"image_emb": img_feat, "discri_emb": txt_feat, "class_id": class_ids}

train_ds = train_ds.map(add_embeds, batched=True, batch_size=32)
test_ds = test_ds.map(add_embeds, batched=True, batch_size=32)

final_features = Features({
    "index": Value("int64"),
    "input_image": HFImage(),
    "input_description": Value("string"),
    "input_category": Value("string"),
    "class_id": ClassLabel(names=class_names),
    "image_emb": Sequence(Value("float32"), length=clip_dim),
    "discri_emb": Sequence(Value("float32"), length=clip_dim),
})
train_ds = train_ds.cast(final_features)
test_ds = test_ds.cast(final_features)

output_dir = os.path.join(os.getcwd(), "preprocessed_fashion")
os.makedirs(output_dir, exist_ok=True)
train_out = os.path.join(output_dir, "train")
test_out = os.path.join(output_dir, "test")
train_ds.save_to_disk(train_out)
test_ds.save_to_disk(test_out)
with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as f:
    json.dump(label2id, f, ensure_ascii=False, indent=2)

print(f"Saved train to {train_out} ({len(train_ds)} samples)")
print(f"Saved test to {test_out} ({len(test_ds)} samples)")
print(f"Classes: {len(class_names)}; mapping stored at {os.path.join(output_dir, 'label2id.json')}")

# ============================================================
# 按标签划分 5 个领域并重命名字段，生成精简版领域数据集
# ============================================================
from datasets import load_from_disk

preprocessed_root = os.path.join(os.getcwd(), "preprocessed_fashion")
domain_out_root = os.environ.get(
    "FEDMMDP_FASHION_DOMAIN_ROOT",
    os.path.join(ROOT_DIR, "data", "preprocessed_fashion", "domain_datasets"),
)
os.makedirs(domain_out_root, exist_ok=True)

print("Loading saved datasets for domain split...")
train_ds_loaded = load_from_disk(os.path.join(preprocessed_root, "train"))
test_ds_loaded = load_from_disk(os.path.join(preprocessed_root, "test"))

def prepare(ds):
    # domain_id: 依据 class_id 均匀映射到 5 个领域，互不重叠
    ds = ds.map(lambda x: {"domain_id": x["class_id"] % 5})
    # 重命名表征字段
    ds = ds.rename_column("image_emb", "processed_img")
    ds = ds.rename_column("discri_emb", "cap_tokens")
    ds = ds.rename_column("index", "id")
    # 删除原始图像/描述字段
    drop_cols = [c for c in ["input_image", "input_description", "input_category"] if c in ds.column_names]
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
