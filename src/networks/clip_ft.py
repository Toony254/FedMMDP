# $ conda activate zs_vita
# $ python src/networks/train_projector.py &
# $ ps -ef|grep train_projector
# $ kill [...]
import os
# 导入所需的库和模块
import datetime
import clip
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

# 设置设备(GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import sys
import torch.nn.functional as F
import argparse
import random
import os

# 设置混合精度训练
use_amp = True
# 初始化tensorboard
writer = SummaryWriter(comment="runs/clip")
# 设置CUDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
RANDOM_SEED = 42
# 设置随机种子函数
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class MSCOCODataset(Dataset):
    def __init__(self, coco_json, img_dir, img_transform=None):
        """
        Args:
            coco_json: COCO注释json文件
            img_dir: 包含所有图像的目录
            img_transform: 应用于图像的变换
            txt_transform: 应用于文本的变换
        """
        self.coco = COCO(coco_json)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.img_transform = img_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)
        
        # 获取该图像的所有描述
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in annotations]
        
        # 随机选择一个描述
        caption = random.choice(captions)
        return image, caption


def contrastive_loss(img_feats, text_feats, temperature=0.07):
    """
    计算图文对比损失
    """
    # 归一化特征
    img_feats = F.normalize(img_feats, p=2, dim=1)
    text_feats = F.normalize(text_feats, p=2, dim=1)
    
    # 计算余弦相似度矩阵 (batch_size x batch_size)
    logits = torch.matmul(img_feats, text_feats.t()) / temperature
    
    # 对角线上的元素是正样本对
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # 计算图像->文本和文本->图像方向的损失
    i2t_loss = F.cross_entropy(logits, labels)
    t2i_loss = F.cross_entropy(logits.t(), labels)
    
    # 总损失是两个方向损失的平均
    total_loss = (i2t_loss + t2i_loss) / 2
    return total_loss

def compute_rmg(image_features, text_features):
    image_to_text_map = torch.arange(image_features.shape[0]).reshape(image_features.shape[0], 1) # [batch_size, 1]
    text_to_image_map = torch.arange(image_features.shape[0]) # [batch_size * 1]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features_original = image_features.clone()
    image_features = torch.stack([image_features[l] for l in text_to_image_map], dim=0)
    text_feature_per_image = text_features
    labels_to_idx_map = torch.ones((text_features.size(0),text_features.size(0))).bool()
    for txt_idcs in image_to_text_map:
        for i,j in [(x.item(), y.item()) for x in txt_idcs for y in txt_idcs]:
            labels_to_idx_map[i,j] = False
    
    image_features_matching = torch.sum(image_features*text_feature_per_image, dim=1).mean()
    image_features_matching = 1-(image_features_matching+1)/2 # [0, 1] & flip
    image_features_matching = torch.where(image_features_matching > 0, image_features_matching, torch.ones_like(image_features_matching)*1e-3) # [1e-3, 1]

    i_x_i = image_features_original @ image_features_original.T

    i_x_i.fill_diagonal_(0)
    mean_img_similarity = i_x_i.sum() / (math.prod(i_x_i.shape)-i_x_i.shape[0]) # [-1, 1]
    mean_img_similarity = 1-(mean_img_similarity+1)/2 # [0,1] & flip

    t_x_t = text_features @ text_features.T
    t_x_t.fill_diagonal_(0)
    mean_txt_similarity = t_x_t.sum() / (math.prod(t_x_t.shape)-t_x_t.shape[0]) # [-1, 1]
    mean_txt_similarity = 1-(mean_txt_similarity+1)/2 # [0,1] & flip

    normalizer = image_features_matching.mean() + (mean_img_similarity.mean() + mean_txt_similarity.mean()) / 2
    dist = image_features_matching.mean().item() / normalizer.item()

    return torch.tensor(dist)

# 修改训练函数
def train(hyper_dict, train_loader, val_loader=None):
    print(hyper_dict)
    print("using {} device.".format(device))
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()
    
    for param in clip_model.parameters():
        param.requires_grad = True

    # 只优化可训练参数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=hyper_dict.lr)
    
    # 初始化
    epochs = hyper_dict.epochs
    best_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    have_saved = None
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        clip_model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        
        for step, (images, texts) in enumerate(train_bar):
            optimizer.zero_grad()
            images = images.to(device)
            texts = clip.tokenize(texts).to(device)
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast(enabled=use_amp):
                img_feats = clip_model.encode_image(images)
                text_feats = clip_model.encode_text(texts)
                cl_loss = contrastive_loss(img_feats, text_feats, temperature=hyper_dict.temperature)
                rmg_loss = compute_rmg(img_feats, text_feats)
                loss = cl_loss + rmg_loss

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            train_bar.desc = "train epoch[{}/{}] cl_loss:{:.3f} rmg_loss:{:.3f}".format(epoch + 1, epochs, cl_loss, rmg_loss)
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + step)
            writer.add_scalar("CL_Loss/train", cl_loss.item(), epoch * len(train_loader) + step)
            writer.add_scalar("RMG_Loss/train", rmg_loss.item(), epoch * len(train_loader) + step)
        
        avg_loss = running_loss / len(train_loader)
        print(f'[Epoch {epoch+1}/{epochs}] Train Loss: {avg_loss:.3f}')
        
        # 验证阶段（如果有验证集）
        if val_loader is not None:
            clip_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout)
                for val_images, val_texts in val_bar:
                    val_images = val_images.to(device)
                    val_texts = clip.tokenize(val_texts).to(device)
                    
                    img_feats = clip_model.encode_image(val_images)
                    text_feats = clip_model.encode_text(val_texts)
                    cl_loss = contrastive_loss(img_feats, text_feats, temperature=hyper_dict.temperature)
                    rmg_loss = compute_rmg(img_feats, text_feats)
                    loss = cl_loss + rmg_loss
                    
                    val_loss += loss.item()
                    val_bar.desc = f"val epoch[{epoch+1}/{epochs}] loss:{loss:.3f} cl_loss:{cl_loss:.3f} rmg_loss:{rmg_loss:.3f}"
            
            avg_val_loss = val_loss / len(val_loader)
            print(f'[Epoch {epoch+1}/{epochs}] Val Loss: {avg_val_loss:.3f}')
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            
            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                if have_saved:
                    os.remove(have_saved)
                
                now = datetime.datetime.now()
                filename = now.strftime("%Y%m%d_%H%M%S.pth")
                save_path_file = os.path.join(hyper_dict.save_path, filename)
                torch.save(clip_model, save_path_file)
                have_saved = save_path_file
                print(f"Model saved to {save_path_file}")
        else:
            # 没有验证集时，每个epoch保存一次模型
            if epoch % 5 == 0 or epoch == epochs - 1:
                now = datetime.datetime.now()
                filename = now.strftime("%Y%m%d_%H%M%S.pth")
                save_path_file = os.path.join(hyper_dict.save_path, filename)
                torch.save(clip_model, save_path_file)
                print(f"Model saved to {save_path_file}")
    
    print('Finished Training')
    if val_loader is not None:
        print(f'Best Loss: {best_loss:.3f}')

# 修改主函数
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--save_path', type=str, default="saved/FT_CLIP/")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--coco_json', type=str, default="/home/bd/data/zs/data/COCO/annotations/captions_train2017.json")
    parser.add_argument('--coco_img_dir', type=str, default="/home/bd/data/zs/data/COCO/train2017")
    parser.add_argument('--coco_val_json', type=str, default="/home/bd/data/zs/data/COCO/annotations/captions_val2017.json")
    parser.add_argument('--coco_val_img_dir', type=str, default="/home/bd/data/zs/data/COCO/val2017")
    parser.add_argument('--val_split', type=float, default=0.1)
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    # 确保保存路径存在
    os.makedirs(args.save_path, exist_ok=True)
    
    # 加载CLIP图像预处理
    _, preprocess = clip.load("ViT-B/32", device=device)
    
    # 创建MSCOCO数据集
    train_dataset = MSCOCODataset(
        args.coco_json,
        args.coco_img_dir,
        img_transform=preprocess
    )
    
    # 创建验证集
    val_dataset = MSCOCODataset(
        args.coco_val_json,
        args.coco_val_img_dir,
        img_transform=preprocess
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 开始训练
    train(args, train_loader, val_loader)