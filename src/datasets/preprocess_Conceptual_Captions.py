from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import os

import PIL.Image
from PIL import Image

from datasets import load_dataset, Dataset
from datasets.utils.file_utils import get_datasets_user_agent


USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url, timeout=10, retries=2):
    """下载单张图片，失败返回None"""
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
                # 处理各种图像模式，安全地转换为RGB
                if image.mode == 'RGBA':
                    # 创建白色背景并合成
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])  # 使用alpha通道作为mask
                    image = background
                elif image.mode == 'P':
                    # 调色板模式，先转为RGBA再转RGB（处理透明度）
                    image = image.convert('RGBA')
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                elif image.mode == 'LA':
                    # 灰度+透明度模式
                    image = image.convert('RGBA')
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                elif image.mode != 'RGB':
                    # 其他模式直接转换
                    image = image.convert('RGB')
                
                # 创建一个全新的RGB图像，彻底清除所有元数据（包括透明度信息）
                clean_image = Image.new('RGB', image.size)
                clean_image.putdata(list(image.getdata()))
                image = clean_image
            break
        except Exception as e:
            image = None
    return image


def fetch_images(batch, num_threads, timeout=10, retries=2):
    """批量下载图片"""
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch


def filter_valid_images(example):
    """过滤掉下载失败的图片"""
    return example["image"] is not None


def preprocess_and_save_dataset(save_dir, num_threads=20, split='train'):
    """
    下载、预处理并保存 Conceptual Captions 数据集
    
    Args:
        save_dir: 保存数据集的目录
        num_threads: 下载图片的线程数
        split: 数据集分割 ('train' 或 'validation')
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading Conceptual Captions dataset ({split} split)...")
    dset = load_dataset("google-research-datasets/conceptual_captions", split=split)
    
    original_size = len(dset)
    print(f"Original dataset size: {original_size}")
    
    print("Fetching images from URLs...")
    dset = dset.map(
        fetch_images, 
        batched=True, 
        batch_size=100, 
        fn_kwargs={"num_threads": num_threads, "timeout": 10, "retries": 2},
        desc="Downloading images"
    )
    
    print("Filtering out failed downloads...")
    dset = dset.filter(filter_valid_images, desc="Filtering valid images")
    
    # 统计下载成功和失败的数量
    success_count = len(dset)
    failed_count = original_size - success_count
    success_rate = (success_count / original_size) * 100
    
    print(f"\n{'='*50}")
    print(f"Download Statistics for {split} split:")
    print(f"{'='*50}")
    print(f"  Total images:      {original_size}")
    print(f"  Successfully downloaded: {success_count} ({success_rate:.2f}%)")
    print(f"  Failed downloads:  {failed_count} ({100-success_rate:.2f}%)")
    print(f"{'='*50}")
    print(f"Dataset size after filtering: {success_count}")
    
    # 保存数据集
    save_path = os.path.join(save_dir, split)
    print(f"Saving dataset to {save_path}...")
    dset.save_to_disk(save_path)
    
    print(f"Dataset saved successfully!")
    return dset


class ConceptualCaptionsDataset:
    """Conceptual Captions 数据集的 PyTorch Dataset 包装器"""
    
    def __init__(self, data_dir, split='train', img_transform=None):
        """
        Args:
            data_dir: 保存数据集的目录
            split: 'train' 或 'validation'
            img_transform: 图像变换（如CLIP预处理）
        """
        from datasets import load_from_disk
        
        self.data_path = os.path.join(data_dir, split)
        self.dataset = load_from_disk(self.data_path)
        self.img_transform = img_transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        caption = item['caption']
        
        # 确保图像是PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # 确保是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.img_transform:
            image = self.img_transform(image)
            
        return image, caption


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and preprocess Conceptual Captions dataset")
    parser.add_argument('--save_dir', type=str, default="./data/conceptual_captions",
                        help="Directory to save the processed dataset")
    parser.add_argument('--num_threads', type=int, default=20,
                        help="Number of threads for downloading images")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'validation'],
                        help="Dataset split to download")
    parser.add_argument('--download_all', action='store_true',
                        help="Download both train and validation splits")
    
    args = parser.parse_args()
    
    if args.download_all:
        for split in ['train', 'validation']:
            print(f"\n{'='*50}")
            print(f"Processing {split} split...")
            print(f"{'='*50}")
            preprocess_and_save_dataset(args.save_dir, args.num_threads, split)
    else:
        preprocess_and_save_dataset(args.save_dir, args.num_threads, args.split)
