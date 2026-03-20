# 导入所需的库和模块
import argparse
import datetime
import math
import os
import pickle
import random
import sys

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# 设置设备(GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
from tqdm import tqdm

# 导入 Conceptual Captions 数据集
from src.datasets.preprocess_Conceptual_Captions import ConceptualCaptionsDataset

# 设置混合精度训练
use_amp = True
# 初始化tensorboard
writer = SummaryWriter(comment="runs/clip_cc")
# 设置CUDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
RANDOM_SEED = 42


class F30kCaptionsCap(Dataset):
    def __init__(self, annFile='dataset_k_split.pkl', split='train',
                 transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = pickle.load(open(annFile, 'rb'))
        if split not in self.data.keys():
            assert False, f'split wrong {split}'
        self.data = self.data[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        caption = data[1]

        path = data[0].replace('/data/mmdata/Flick30k/flickr30k-images/',
                       '/home/bd/data/zs/data/flickr30k/flickr30k-images/')

        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(caption, truncate=True)[0]
            return img, target, caption
        return img, caption

    
# 设置随机种子函数
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class SelfAttentionProjector(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return self.proj(out)


class ResidualProjector(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.proj(x)
            

class CombinedModel(nn.Module):
    def __init__(self, clip_model, args):
        super(CombinedModel, self).__init__()
        self.clip_model = clip_model
        emb_dim = clip_model.visual.output_dim
        if args.projector == "linear":
            self.visual_projector = nn.Linear(emb_dim, emb_dim)
            self.text_projector = nn.Linear(emb_dim, emb_dim)
            
        elif args.projector == "mlp":
            self.visual_projector = nn.Sequential(
                nn.Linear(emb_dim, 2*emb_dim),
                nn.ReLU(),
                nn.Linear(2*emb_dim, emb_dim)
            )
            self.text_projector = nn.Sequential(
                nn.Linear(emb_dim, 2*emb_dim),
                nn.ReLU(),
                nn.Linear(2*emb_dim, emb_dim)
            )
            
        elif args.projector == "residual":
            self.visual_projector = ResidualProjector(emb_dim)
            self.text_projector = ResidualProjector(emb_dim)
            
        elif args.projector == "norm":
            self.visual_projector = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU()
            )
            self.text_projector = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU()
            )
            
        elif args.projector == "mlp+norm":
            self.visual_projector = nn.Sequential(
                nn.Linear(emb_dim, 2*emb_dim),
                nn.ReLU(),
                nn.Linear(2*emb_dim, emb_dim),
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU()
            )
            self.text_projector = nn.Sequential(
                nn.Linear(emb_dim, 2*emb_dim),
                nn.ReLU(),
                nn.Linear(2*emb_dim, emb_dim),
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU()
            )
            
        elif args.projector == "bottleneck":
            self.visual_projector = nn.Sequential(
                nn.Linear(emb_dim, emb_dim//2),
                nn.ReLU(),
                nn.Linear(emb_dim//2, emb_dim)
            )
            self.text_projector = nn.Sequential(
                nn.Linear(emb_dim, emb_dim//2),
                nn.ReLU(),
                nn.Linear(emb_dim//2, emb_dim)
            )
            
        elif args.projector == "attention":
            self.visual_projector = SelfAttentionProjector(emb_dim)
            self.text_projector = SelfAttentionProjector(emb_dim)
            
        elif args.projector == "clip":
            self.visual_projector = nn.Identity()
            self.text_projector = nn.Identity()

    def forward(self, img_input, text_input):
        img_feats = self.clip_model.encode_image(img_input)
        img_feats = self.visual_projector(img_feats)
        text_feats = self.clip_model.encode_text(text_input)
        text_feats = self.text_projector(text_feats)
        return img_feats, text_feats
    
    def get_image_features(self, img_input):
        img_feats = self.clip_model.encode_image(img_input)
        img_feats = self.visual_projector(img_feats)
        return img_feats
    
    def get_text_features(self, text_input):
        text_feats = self.clip_model.encode_text(text_input)
        text_feats = self.text_projector(text_feats)
        return text_feats


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
    image_to_text_map = torch.arange(image_features.shape[0]).reshape(image_features.shape[0], 1)
    text_to_image_map = torch.arange(image_features.shape[0])
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features_original = image_features.clone()
    image_features = torch.stack([image_features[l] for l in text_to_image_map], dim=0)
    text_feature_per_image = text_features
    labels_to_idx_map = torch.ones((text_features.size(0), text_features.size(0))).bool()
    for txt_idcs in image_to_text_map:
        for i, j in [(x.item(), y.item()) for x in txt_idcs for y in txt_idcs]:
            labels_to_idx_map[i, j] = False
    
    image_features_matching = torch.sum(image_features * text_feature_per_image, dim=1).mean()
    image_features_matching = 1 - (image_features_matching + 1) / 2
    image_features_matching = torch.where(image_features_matching > 0, image_features_matching, 
                                          torch.ones_like(image_features_matching) * 1e-3)

    i_x_i = image_features_original @ image_features_original.T
    i_x_i.fill_diagonal_(0)
    mean_img_similarity = i_x_i.sum() / (math.prod(i_x_i.shape) - i_x_i.shape[0])
    mean_img_similarity = 1 - (mean_img_similarity + 1) / 2

    t_x_t = text_features @ text_features.T
    t_x_t.fill_diagonal_(0)
    mean_txt_similarity = t_x_t.sum() / (math.prod(t_x_t.shape) - t_x_t.shape[0])
    mean_txt_similarity = 1 - (mean_txt_similarity + 1) / 2

    normalizer = image_features_matching.mean() + (mean_img_similarity.mean() + mean_txt_similarity.mean()) / 2
    dist = image_features_matching.mean().item() / normalizer.item()

    return torch.tensor(dist)


# 修改训练函数
def train(hyper_dict, train_loader, val_loader=None):
    print(hyper_dict)
    print("using {} device.".format(device))
    clip_model, _ = clip.load(hyper_dict.model, device=device)
    clip_model = clip_model.float()

    # 创建只包含CLIP和投影层的模型
    combined_model = CombinedModel(clip_model, hyper_dict).to(device)
    if hyper_dict.projector == "clip":
        for param in combined_model.parameters():
            param.requires_grad = False

        for param in combined_model.clip_model.visual.layer4.parameters():
            param.requires_grad = True

        for param in combined_model.clip_model.transformer.resblocks[-1].mlp.parameters():
            param.requires_grad = True

    else:
        for param in combined_model.clip_model.parameters():
            param.requires_grad = False
        
        for param in combined_model.visual_projector.parameters():
            param.requires_grad = True
        
        for param in combined_model.text_projector.parameters():
            param.requires_grad = True

    # 只优化可训练参数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, combined_model.parameters()), lr=hyper_dict.lr)
    
    # 初始化
    epochs = hyper_dict.epochs
    best_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    have_saved = None
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        combined_model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        
        for step, (images, texts) in enumerate(train_bar):
            optimizer.zero_grad()
            images = images.to(device)
            texts = clip.tokenize(texts, truncate=True).to(device)
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast(enabled=use_amp):
                img_feats = combined_model.get_image_features(images)
                text_feats = combined_model.get_text_features(texts)
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
            combined_model.eval()
            val_loss = 0.0
            total_accuracy = 0
            total_RMG = 0
            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout)
                for val_images, val_texts in val_bar:
                    val_images = val_images.to(device)
                    val_texts = clip.tokenize(val_texts, truncate=True).to(device)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        img_feats = combined_model.get_image_features(val_images)
                        text_feats = combined_model.get_text_features(val_texts)
                        cl_loss = contrastive_loss(img_feats, text_feats, temperature=hyper_dict.temperature)
                        rmg_loss = compute_rmg(img_feats, text_feats)
                        loss = cl_loss + rmg_loss
                    
                    val_loss += loss.item()
                    
                    # 计算准确率
                    img_feats = F.normalize(img_feats, p=2, dim=1)
                    text_feats = F.normalize(text_feats, p=2, dim=1)
                    total_accuracy += calculate_accuracy(img_feats, text_feats, 
                                                         np.eye(img_feats.shape[0]), threshold=0.5) * img_feats.shape[0]
                    total_RMG += compute_rmg(img_feats, text_feats).item() * img_feats.shape[0]
                    
                    val_bar.desc = f"val epoch[{epoch+1}/{epochs}] loss:{loss:.3f}"
            
            avg_val_loss = val_loss / len(val_loader)
            avg_accuracy = total_accuracy / len(val_loader.dataset)
            avg_RMG = total_RMG / len(val_loader.dataset)
            print(f'[Epoch {epoch+1}/{epochs}] Val Loss: {avg_val_loss:.3f}')
            print(f'[Epoch {epoch+1}/{epochs}] Val Accuracy: {avg_accuracy*100:.3f}%')
            print(f'[Epoch {epoch+1}/{epochs}] Val RMG: {avg_RMG:.3f}')
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/val", avg_accuracy, epoch)
            writer.add_scalar("RMG/val", avg_RMG, epoch)
            
            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_RMG = avg_RMG
                best_accuracy = avg_accuracy
                print(f"Best Loss: {best_loss:.3f}, Best RMG: {best_RMG:.3f}, Best Accuracy: {best_accuracy*100:.3f}%")
                if have_saved:
                    os.remove(have_saved)
                
                now = datetime.datetime.now()
                filename = f"{hyper_dict.projector}_clip_{clip_model.visual.output_dim}.pth"
                save_path_file = os.path.join(hyper_dict.save_path, filename)
                best_model = combined_model
                torch.save({'visual_projector': combined_model.visual_projector,
                            'text_projector': combined_model.text_projector}, save_path_file)
                have_saved = save_path_file
                print(f"Model saved to {save_path_file}")
        else:
            # 没有验证集时，每个epoch保存一次模型
            if epoch % 5 == 0 or epoch == epochs - 1:
                now = datetime.datetime.now()
                filename = now.strftime("%Y%m%d_%H%M%S.pth")
                save_path_file = os.path.join(hyper_dict.save_path, filename)
                torch.save({'visual_projector': combined_model.visual_projector,
                            'text_projector': combined_model.text_projector}, save_path_file)
                print(f"Model saved to {save_path_file}")

    
    print('Finished Training')
    if val_loader is not None:
        print(f'Best Loss: {best_loss:.3f}')
        return best_model
    return combined_model


def calculate_accuracy(image_features, text_features, labels, threshold=0.5):
    if isinstance(image_features, torch.Tensor):
        image_features = image_features.cpu().numpy()
        text_features = text_features.cpu().numpy()
    similarity_matrix = cosine_similarity(image_features, text_features)
    
    # 根据阈值判断匹配情况
    predictions = (similarity_matrix > threshold).astype(int)
    
    # 计算准确率
    accuracy = np.mean(predictions == labels)
    
    return accuracy


# 修改主函数
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="training with Conceptual Captions dataset")
    parser.add_argument('--save_path', type=str, default="saved/projector_weights/")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.07)
    # Conceptual Captions 数据集参数
    parser.add_argument('--cc_data_dir', type=str, default="./data/conceptual_captions",
                        help="Directory containing the processed Conceptual Captions dataset")
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--projector', type=str, default="mlp+norm", 
                        choices=["linear", "mlp", "residual", "norm", "mlp+norm", "bottleneck", "attention", "clip"])
    parser.add_argument('--model', type=str, default="RN50",
                        choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    # 确保保存路径存在
    os.makedirs(args.save_path, exist_ok=True)
    
    # 加载CLIP图像预处理
    clip_model, preprocess = clip.load(args.model, device=device)
    clip_model = clip_model.float()
    
    # 创建 Conceptual Captions 数据集
    print("Loading Conceptual Captions training dataset...")
    train_dataset = ConceptualCaptionsDataset(
        data_dir=args.cc_data_dir,
        split='train',
        img_transform=preprocess
    )
    print(f"Training dataset size: {len(train_dataset)}")
    
    # 创建验证集
    print("Loading Conceptual Captions validation dataset...")
    val_dataset = ConceptualCaptionsDataset(
        data_dir=args.cc_data_dir,
        split='validation',
        img_transform=preprocess
    )
    print(f"Validation dataset size: {len(val_dataset)}")
    
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
    combined_model = train(args, train_loader, val_loader)
    
    # 在 Flickr30k 上验证
    FLICKR_PATH = 'dataset_k_split.pkl'
    if os.path.exists(FLICKR_PATH):
        flickr30k_dataset = F30kCaptionsCap(FLICKR_PATH, transform=preprocess,
                                            target_transform=clip.tokenize)
        print(f"Loaded Flickr30k dataset with {len(flickr30k_dataset)} samples")

        batch_size = 32
        dataloader = DataLoader(flickr30k_dataset, batch_size=batch_size, shuffle=False)
        
        combined_model.eval()
        
        total_rmg = 0
        total_accuracy = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing Flickr30k"):
                images, text_tokens, captions = batch
                
                images = images.to(device)
                text_tokens = text_tokens.to(device)
                
                img_feats, txt_feats = combined_model(images, text_tokens)
                
                img_feats = F.normalize(img_feats, p=2, dim=1)
                txt_feats = F.normalize(txt_feats, p=2, dim=1)
                total_rmg += compute_rmg(img_feats, txt_feats) * img_feats.shape[0]
                total_accuracy += calculate_accuracy(img_feats, txt_feats, np.eye(img_feats.shape[0]), threshold=0.5) * img_feats.shape[0]
        
        flickr30k_rmg = total_rmg / len(flickr30k_dataset)
        flickr30k_accuracy = total_accuracy / len(flickr30k_dataset)
        print(f"Flickr30k RMG: {flickr30k_rmg:.3f}")
        print(f"Flickr30k Accuracy: {flickr30k_accuracy*100:.3f}%")
    else:
        print(f"Flickr30k dataset not found at {FLICKR_PATH}, skipping evaluation.")
