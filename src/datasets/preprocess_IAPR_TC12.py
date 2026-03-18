"""
IAPR TC-12 数据集下载和预处理脚本

数据集包含约20,000张图像及其对应的多语言描述
下载地址: http://www-i6.informatik.rwth-aachen.de/imageclef/resources/iaprtc12.tgz
"""

import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import shutil


DATASET_URL = "http://www-i6.informatik.rwth-aachen.de/imageclef/resources/iaprtc12.tgz"


def download_dataset(save_dir, url=DATASET_URL):
    """
    下载IAPR TC-12数据集
    
    Args:
        save_dir: 保存数据集的目录
        url: 数据集下载地址
    """
    os.makedirs(save_dir, exist_ok=True)
    
    tgz_path = os.path.join(save_dir, "iaprtc12.tgz")
    
    if os.path.exists(tgz_path):
        print(f"Archive already exists at {tgz_path}")
    else:
        print(f"Downloading IAPR TC-12 dataset from {url}...")
        print("This may take a while (~1.8GB)...")
        
        # 使用 tqdm 显示下载进度
        def reporthook(block_num, block_size, total_size):
            if not hasattr(reporthook, 'pbar'):
                reporthook.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
            downloaded = block_num * block_size
            reporthook.pbar.update(block_size)
            if downloaded >= total_size:
                reporthook.pbar.close()
                delattr(reporthook, 'pbar')
        
        urllib.request.urlretrieve(url, tgz_path, reporthook)
        print(f"Downloaded to {tgz_path}")
    
    return tgz_path


def extract_dataset(tgz_path, extract_dir):
    """
    解压数据集
    
    Args:
        tgz_path: tgz文件路径
        extract_dir: 解压目录
    """
    if os.path.exists(os.path.join(extract_dir, "images")):
        print(f"Dataset already extracted at {extract_dir}")
        return
    
    print(f"Extracting dataset to {extract_dir}...")
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    print("Extraction complete!")


def parse_annotation(annotation_path):
    """
    解析单个XML注释文件，提取英文描述
    
    Args:
        annotation_path: XML文件路径
        
    Returns:
        英文描述文本，如果解析失败返回None
    """
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # 查找英文描述
        description = None
        for desc in root.findall('.//DESCRIPTION'):
            description = desc.text
            break
        
        if description is None:
            # 尝试其他可能的标签
            for desc in root.findall('.//NOTES'):
                description = desc.text
                break
        
        if description:
            # 清理文本
            description = description.strip()
            description = ' '.join(description.split())  # 规范化空白字符
            
        return description
    except Exception as e:
        return None


def convert_image_to_rgb(image_path):
    """
    读取图像并转换为RGB格式，处理各种边缘情况
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        PIL Image (RGB格式) 或 None（如果处理失败）
    """
    try:
        image = Image.open(image_path)
        
        # 处理各种图像模式
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode == 'P':
            image = image.convert('RGBA')
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode == 'LA':
            image = image.convert('RGBA')
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 创建干净的RGB图像（清除元数据）
        clean_image = Image.new('RGB', image.size)
        clean_image.putdata(list(image.getdata()))
        
        return clean_image
    except Exception as e:
        return None


def preprocess_and_save_dataset(save_dir, data_dir=None, train_ratio=0.9):
    """
    预处理IAPR TC-12数据集并保存
    
    Args:
        save_dir: 保存处理后数据集的目录
        data_dir: 原始数据集目录（如果为None，则下载数据集）
        train_ratio: 训练集比例
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 下载和解压数据集
    if data_dir is None:
        raw_dir = os.path.join(save_dir, "raw")
        tgz_path = download_dataset(raw_dir)
        extract_dataset(tgz_path, raw_dir)
        data_dir = os.path.join(raw_dir, "iaprtc12")
    
    # 数据集目录结构
    images_dir = os.path.join(data_dir, "images")
    annotations_dir = os.path.join(data_dir, "annotations_complete_eng")
    
    if not os.path.exists(images_dir):
        # 尝试其他可能的路径
        for subdir in os.listdir(data_dir):
            potential_images = os.path.join(data_dir, subdir, "images")
            if os.path.exists(potential_images):
                images_dir = potential_images
                annotations_dir = os.path.join(data_dir, subdir, "annotations_complete_eng")
                break
    
    print(f"Images directory: {images_dir}")
    print(f"Annotations directory: {annotations_dir}")
    
    # 收集所有图像-描述对
    data_pairs = []
    success_count = 0
    failed_count = 0
    no_caption_count = 0
    
    print("Processing images and annotations...")
    
    # 遍历图像目录
    for root, dirs, files in os.walk(images_dir):
        for file in tqdm(files, desc="Processing images"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image_path = os.path.join(root, file)
                
                # 构建对应的注释文件路径
                # 图像路径: images/00/00001.jpg
                # 注释路径: annotations_complete_eng/00/00001.eng
                rel_path = os.path.relpath(image_path, images_dir)
                base_name = os.path.splitext(rel_path)[0]
                annotation_path = os.path.join(annotations_dir, base_name + ".eng")
                
                # 读取注释
                caption = None
                if os.path.exists(annotation_path):
                    caption = parse_annotation(annotation_path)
                
                if caption is None or len(caption.strip()) == 0:
                    no_caption_count += 1
                    continue
                
                # 验证图像可读
                image = convert_image_to_rgb(image_path)
                if image is None:
                    failed_count += 1
                    continue
                
                data_pairs.append({
                    'image_path': image_path,
                    'caption': caption
                })
                success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Processing Statistics:")
    print(f"{'='*50}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed to load image:   {failed_count}")
    print(f"  No caption found:       {no_caption_count}")
    print(f"  Total valid pairs:      {len(data_pairs)}")
    print(f"{'='*50}")
    
    if len(data_pairs) == 0:
        print("No valid image-caption pairs found!")
        return None
    
    # 划分训练集和验证集
    import random
    random.seed(42)
    random.shuffle(data_pairs)
    
    split_idx = int(len(data_pairs) * train_ratio)
    train_data = data_pairs[:split_idx]
    val_data = data_pairs[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Training set:   {len(train_data)}")
    print(f"  Validation set: {len(val_data)}")
    
    # 保存为简单的格式（图像路径和描述）
    import json
    
    train_save_path = os.path.join(save_dir, "train.json")
    val_save_path = os.path.join(save_dir, "validation.json")
    
    with open(train_save_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_save_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDataset saved to:")
    print(f"  Train: {train_save_path}")
    print(f"  Val:   {val_save_path}")
    
    return train_data, val_data


class IAPRTC12Dataset:
    """IAPR TC-12 数据集的 PyTorch Dataset 包装器"""
    
    def __init__(self, data_dir, split='train', img_transform=None):
        """
        Args:
            data_dir: 保存数据集的目录
            split: 'train' 或 'validation'
            img_transform: 图像变换（如CLIP预处理）
        """
        import json
        
        json_path = os.path.join(data_dir, f"{split}.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.img_transform = img_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        caption = item['caption']
        
        # 读取图像
        image = Image.open(image_path).convert('RGB')
        
        if self.img_transform:
            image = self.img_transform(image)
            
        return image, caption


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and preprocess IAPR TC-12 dataset")
    parser.add_argument('--save_dir', type=str, default="./data/iapr_tc12",
                        help="Directory to save the processed dataset")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Directory containing the raw dataset (if already downloaded)")
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help="Ratio of training data")
    
    args = parser.parse_args()
    
    preprocess_and_save_dataset(args.save_dir, args.data_dir, args.train_ratio)
