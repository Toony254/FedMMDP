import os

from datasets import load_dataset, Dataset
from imagenet_label import ImageNetAnalyzer
from nltk.corpus import wordnet as wn
import networkx as nx
from PIL import Image
from tqdm import tqdm
import io
import sys
import logging
import torch
import numpy as np
from pathlib import Path
import pickle

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
OUTPUT_DIR = Path("preprocessed_imagenet/domain_datasets/raw")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1000  # 每个域累积多少样本后保存一次
NUM_DOMAINS = 5
MIN_DEPTH = 5
TEST_RATIO = 0.1

logger.info(f"Device: {DEVICE}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# ============================================
# 步骤1: 确定域划分
# ============================================
logger.info("=" * 70)
logger.info("Step 1: Analyzing ImageNet hierarchy and determining domain splits")
logger.info("=" * 70)

query_labels = []
with open("src/datasets/imagenet_synsets.txt", "r") as f:
    for line in f:
        if line.startswith("n"):
            wnid = line.split()[0]
            query_labels.append(wnid)

synsets = [wn.synset_from_pos_and_offset('n', int(wnid[1:])) for wnid in query_labels]

analyzer = ImageNetAnalyzer(synsets, query_wnids=query_labels)
hierarchy_graph = analyzer.build_hierarchy_graph(max_depth=20)
subtrees = analyzer.find_deep_subtrees(min_depth=MIN_DEPTH, num_subtrees=NUM_DOMAINS)

# 构建域的wnid列表
domain_subtrees = []
domain_wnid_to_classid = {}  # 每个域内的wnid到class_id的映射

for idx, (root, size, level) in enumerate(subtrees):
    descendants = nx.descendants(analyzer.hierarchy_graph, root)
    subtree_nodes = [root] + list(descendants)
    # 提取wnid
    wnid_list = [node.split(".")[1] for node in subtree_nodes]
    domain_subtrees.append(wnid_list)
    
    # 为当前域创建类别映射（从0开始）
    domain_class_map = {wnid: class_idx for class_idx, wnid in enumerate(wnid_list)}
    domain_wnid_to_classid[idx] = domain_class_map
    
    logger.info(f"Domain {idx}: {len(wnid_list)} classes, root level={level}")

# ============================================
# 步骤3: 处理数据并编码
# ============================================
logger.info("=" * 70)
logger.info("Step 3: Processing and encoding dataset")
logger.info("=" * 70)

def process_and_encode_example(example, domain_idx, class_id):
    """
    处理单个样本：提取图像和文本，使用SigLIP编码
    """
    try:
        return {
            "id": example["id"],
            "recaption_short": example["recaption_short"],
            "image": example["image"],
            "class_id": class_id
        }
    except Exception as e:
        logger.warning(f"Failed to process {example['id']}: {str(e)}")
        return None

# 初始化每个域的数据收集器
domain_data = {i: [] for i in range(NUM_DOMAINS)}
domain_batch_counts = {i: 0 for i in range(NUM_DOMAINS)}  # 记录每个域保存的批次数
skipped_count = 0
processed_count = 0

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for domain_idx in range(NUM_DOMAINS):
    domain_dir = OUTPUT_DIR / f"domain_dataset_{domain_idx}"
    domain_dir.mkdir(exist_ok=True)
    (domain_dir / "temp_batches").mkdir(exist_ok=True)

def save_batch(domain_idx, data_list, batch_idx):
    """保存一个批次的数据到临时文件"""
    if not data_list:
        return
    
    temp_dir = OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "temp_batches"
    batch_path = temp_dir / f"batch_{batch_idx}"
    
    batch_dataset = Dataset.from_list(data_list)
    batch_dataset.save_to_disk(str(batch_path))
    logger.info(f"Domain {domain_idx}: saved batch {batch_idx} with {len(data_list)} samples to {batch_path}")

# 流式加载数据集
logger.info("Loading dataset in streaming mode...")
dataset = load_dataset("dataset/data", split="train", streaming=True, cache_dir="./cache")

# 处理每个样本
logger.info("Processing examples...")
for example in tqdm(dataset, desc="Processing", miniters=10000):
    # 确定样本属于哪个域
    assigned = False
    for domain_idx, wnid_list in enumerate(domain_subtrees):
        if any(example["id"].startswith(sid) for sid in wnid_list):
            # 找到该样本的wnid
            wnid = example["id"].split("_")[0]
            
            # 使用域内的类别映射获取class_id
            if wnid not in domain_wnid_to_classid[domain_idx]:
                logger.warning(f"WNID {wnid} not found in domain {domain_idx} mapping, skipping")
                skipped_count += 1
                break
            
            class_id = domain_wnid_to_classid[domain_idx][wnid]
            
            # 处理并编码
            processed = process_and_encode_example(example, domain_idx, class_id)
            
            if processed is None:
                skipped_count += 1
                continue
            
            # 添加到域数据
            domain_data[domain_idx].append(processed)
            processed_count += 1
            assigned = True
            
            # 当累积到BATCH_SIZE时，保存一次（避免内存溢出）
            if len(domain_data[domain_idx]) >= BATCH_SIZE:
                save_batch(domain_idx, domain_data[domain_idx], domain_batch_counts[domain_idx])
                domain_batch_counts[domain_idx] += 1
                domain_data[domain_idx] = []  # 清空缓存
                
                # 清理GPU缓存
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            
            break
    
    if not assigned:
        skipped_count += 1
    
    # 每处理10000个样本打印一次统计
    if (processed_count + skipped_count) % 10000 == 0:
        logger.info(f"Processed: {processed_count}, Skipped: {skipped_count}")
        for d_idx in range(NUM_DOMAINS):
            logger.info(f"  Domain {d_idx}: {domain_batch_counts[d_idx]} batches saved, {len(domain_data[d_idx])} samples in buffer")

# 保存剩余数据
logger.info("Saving remaining data...")
for domain_idx in range(NUM_DOMAINS):
    if domain_data[domain_idx]:
        save_batch(domain_idx, domain_data[domain_idx], domain_batch_counts[domain_idx])
        domain_batch_counts[domain_idx] += 1
        domain_data[domain_idx] = []

logger.info(f"✓ Data processing completed")
logger.info(f"  Total processed: {processed_count}")
logger.info(f"  Total skipped: {skipped_count}")

# ============================================
# 步骤4: 合并批次、分割训练集和测试集、保存
# ============================================
logger.info("=" * 70)
logger.info("Step 4: Merging batches, splitting and saving datasets")
logger.info("=" * 70)

import shutil
from collections import Counter

for domain_idx in range(NUM_DOMAINS):
    temp_dir = OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "temp_batches"
    
    # 检查是否有批次文件
    if not temp_dir.exists():
        logger.warning(f"Domain {domain_idx}: Temp directory not found, skipping...")
        continue
        
    batch_files = sorted(temp_dir.glob("batch_*"))
    if not batch_files:
        logger.warning(f"Domain {domain_idx}: No data, skipping...")
        continue
    
    logger.info(f"Domain {domain_idx}: Merging {len(batch_files)} batches")
    
    try:
        # 逐个加载并合并批次
        all_data = []
        for batch_path in batch_files:
            batch_ds = Dataset.load_from_disk(str(batch_path))
            all_data.extend(batch_ds.to_list())
            logger.info(f"  Loaded {batch_path.name}: {len(batch_ds)} samples")
        
        if not all_data:
            logger.warning(f"Domain {domain_idx}: No data after merging, skipping...")
            continue
            
        logger.info(f"Domain {domain_idx}: Total {len(all_data)} samples after merging")
        
        # 统计类别分布（类似 preprocess_datasets.py）
        class_counts = Counter(sample["class_id"] for sample in all_data)
        logger.info(f"Domain {domain_idx}: Class distribution:")
        logger.info(f"  Total classes: {len(class_counts)}")
        logger.info(f"  Samples per class: min={min(class_counts.values())}, max={max(class_counts.values())}, avg={sum(class_counts.values())/len(class_counts):.1f}")
        
        # 可选：过滤样本数量过少的类别（参考 preprocess_datasets.py 的做法）
        # MIN_SAMPLES_PER_CLASS = 50  # 可根据需要调整
        # popular_classes = {cls: count for cls, count in class_counts.items() if count >= MIN_SAMPLES_PER_CLASS}
        # if len(popular_classes) < len(class_counts):
        #     logger.info(f"  Filtering classes with < {MIN_SAMPLES_PER_CLASS} samples")
        #     all_data = [sample for sample in all_data if sample["class_id"] in popular_classes]
        #     logger.info(f"  After filtering: {len(all_data)} samples, {len(popular_classes)} classes")
        
        # 转换为Dataset
        full_dataset = Dataset.from_list(all_data)
        
        # 分割训练集和测试集
        split_dataset = full_dataset.train_test_split(test_size=TEST_RATIO, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        
        # 保存
        train_path = OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "train"
        test_path = OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "test"
        
        train_dataset.save_to_disk(str(train_path))
        test_dataset.save_to_disk(str(test_path))
        
        logger.info(f"  ✓ Saved train: {len(train_dataset)} samples")
        logger.info(f"  ✓ Saved test: {len(test_dataset)} samples")
        
        # 保存类别映射信息
        mapping_path = OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "class_mapping.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'wnid_to_classid': domain_wnid_to_classid[domain_idx],
                'class_counts': dict(class_counts),
                'num_classes': len(class_counts)
            }, f)
        logger.info(f"  ✓ Saved class mapping to {mapping_path}")
        
        # 清理临时批次文件
        shutil.rmtree(str(temp_dir))
        logger.info(f"  ✓ Cleaned up temporary batches")
        
        # 释放内存
        del all_data, full_dataset, split_dataset, train_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Domain {domain_idx}: Error during merging/splitting: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# ============================================
# 完成
# ============================================
logger.info("=" * 70)
logger.info("✓✓✓ All processing completed successfully! ✓✓✓")
logger.info("=" * 70)
logger.info(f"Output directory: {OUTPUT_DIR}")

# 统计结果
for domain_idx in range(NUM_DOMAINS):
    train_path = OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "train"
    test_path = OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "test"
    mapping_path = OUTPUT_DIR / f"domain_dataset_{domain_idx}" / "class_mapping.pkl"
    
    if train_path.exists():
        train_ds = Dataset.load_from_disk(str(train_path))
        test_ds = Dataset.load_from_disk(str(test_path))
        
        # 加载并显示类别映射信息
        if mapping_path.exists():
            with open(mapping_path, 'rb') as f:
                mapping_info = pickle.load(f)
            logger.info(f"Domain {domain_idx}: train={len(train_ds)}, test={len(test_ds)}, classes={mapping_info['num_classes']}")
        else:
            logger.info(f"Domain {domain_idx}: train={len(train_ds)}, test={len(test_ds)}")
    else:
        logger.info(f"Domain {domain_idx}: NOT FOUND")

logger.info("=" * 70)
