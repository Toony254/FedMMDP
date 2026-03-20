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
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel

# 设置设备(GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import sys
import torch.nn.functional as F
import argparse
import random
import os

# 设置混合精度训练
use_amp = False
# 初始化tensorboard
writer = SummaryWriter(comment="runs/align")
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
    def __init__(self, align_model, align_tokenizer, args):
        super(CombinedModel, self).__init__()
        self.align_model = align_model
        self.align_tokenizer = align_tokenizer
        # 获取ALIGN输出维度
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 224, 224).to(device)
            dummy_text = align_tokenizer("test", return_tensors="pt", padding="max_length", truncation=True, max_length=align_tokenizer.model_max_length)
            dummy_text = {k: v.to(device) for k, v in dummy_text.items()}
            img_feat = align_model.get_image_features(dummy_img)
            txt_feat = align_model.get_text_features(**dummy_text)
            emb_dim = img_feat.shape[-1]
        # 投影器
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
        img_feats = self.align_model.get_image_features(img_input)
        img_feats = self.visual_projector(img_feats)
        text_feats = self.align_model.get_text_features(**text_input)
        text_feats = self.text_projector(text_feats)
        return img_feats, text_feats

    def get_image_features(self, img_input):
        img_feats = self.align_model.get_image_features(img_input)
        img_feats = self.visual_projector(img_feats)
        return img_feats

    def get_text_features(self, text_input):
        text_feats = self.align_model.get_text_features(**text_input)
        text_feats = self.text_projector(text_feats)
        return text_feats

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
    # 加载ALIGN模型和tokenizer
    ALIGN_MODEL_NAME = "kakaobrain/align-base"
    align_tokenizer = AutoTokenizer.from_pretrained(ALIGN_MODEL_NAME)
    align_model = AutoModel.from_pretrained(ALIGN_MODEL_NAME).to(device)
    align_model.eval()

    # 创建只包含ALIGN和投影层的模型
    combined_model = CombinedModel(align_model, align_tokenizer, hyper_dict).to(device)
    for param in combined_model.align_model.parameters():
        param.requires_grad = False
    for param in combined_model.visual_projector.parameters():
        param.requires_grad = True
    for param in combined_model.text_projector.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, combined_model.parameters()), lr=hyper_dict.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_dict.epochs, eta_min=1e-6)
    epochs = hyper_dict.epochs
    best_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    have_saved = None

    for epoch in range(epochs):
        combined_model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, (images, texts) in enumerate(train_bar):
            optimizer.zero_grad()
            images = images.to(device)
            # 文本tokenize
            text_inputs = align_tokenizer(
                list(texts),
                padding="max_length",
                truncation=True,
                max_length=align_tokenizer.model_max_length,
                return_tensors="pt"
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                img_feats = combined_model.get_image_features(images)
                text_feats = combined_model.get_text_features(text_inputs)
                cl_loss = contrastive_loss(img_feats, text_feats, temperature=hyper_dict.temperature)
                rmg_loss = compute_rmg(img_feats, text_feats)
                loss = cl_loss + rmg_loss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(combined_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] cl_loss:{:.3f} rmg_loss:{:.3f}".format(epoch + 1, epochs, cl_loss, rmg_loss)
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + step)
            writer.add_scalar("CL_Loss/train", cl_loss.item(), epoch * len(train_loader) + step)
            writer.add_scalar("RMG_Loss/train", rmg_loss.item(), epoch * len(train_loader) + step)
        avg_loss = running_loss / len(train_loader)
        print(f'[Epoch {epoch+1}/{epochs}] Train Loss: {avg_loss:.3f}')
        if val_loader is not None:
            combined_model.eval()
            val_loss = 0.0
            total_accuracy = 0
            total_RMG = 0
            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout)
                for val_images, val_texts in val_bar:
                    val_images = val_images.to(device)
                    text_inputs = align_tokenizer(
                        list(val_texts),
                        padding="max_length",
                        truncation=True,
                        max_length=align_tokenizer.model_max_length,
                        return_tensors="pt"
                    )
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                    img_feats = combined_model.get_image_features(val_images)
                    text_feats = combined_model.get_text_features(text_inputs)
                    cl_loss = contrastive_loss(img_feats, text_feats, temperature=hyper_dict.temperature)
                    rmg_loss = compute_rmg(img_feats, text_feats)
                    loss = cl_loss + rmg_loss
                    total_RMG += rmg_loss*len(val_texts)
                    labels = np.eye(len(val_texts))
                    accuracy = calculate_accuracy(img_feats, text_feats, labels)
                    total_accuracy += accuracy*len(val_texts)
                    val_loss += loss.item()
                    val_bar.desc = f"val epoch[{epoch+1}/{epochs}] loss:{loss:.3f} cl_loss:{cl_loss:.3f} rmg_loss:{rmg_loss:.3f}"
            avg_val_loss = val_loss / len(val_loader)
            avg_accuracy = total_accuracy / len(val_loader.dataset)
            avg_RMG = total_RMG / len(val_loader.dataset)
            print(f'[Epoch {epoch+1}/{epochs}] Val Loss: {avg_val_loss:.3f}')
            print(f'[Epoch {epoch+1}/{epochs}] Val Accuracy: {avg_accuracy*100:.3f}%')
            print(f'[Epoch {epoch+1}/{epochs}] Val RMG: {avg_RMG:.3f}')
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/val", avg_accuracy, epoch)
            writer.add_scalar("RMG/val", avg_RMG, epoch)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_RMG = avg_RMG
                best_accuracy = avg_accuracy
                print(f"Best Loss: {best_loss:.3f}, Best RMG: {best_RMG:.3f}, Best Accuracy: {best_accuracy*100:.3f}%")
                if have_saved:
                    os.remove(have_saved)
                now = datetime.datetime.now()
                filename = f"{hyper_dict.projector}_align_{img_feats.shape[-1]}.pth"
                save_path_file = os.path.join(hyper_dict.save_path, filename)
                best_model = combined_model
                torch.save({'visual_projector': combined_model.visual_projector,
                            'text_projector': combined_model.text_projector}, save_path_file)
                have_saved = save_path_file
                print(f"Model saved to {save_path_file}")
        else:
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
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--save_path', type=str, default="saved/projector_weights/")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--coco_json', type=str, default="/home/bd/data/zs/data/COCO/annotations/captions_train2017.json")
    parser.add_argument('--coco_img_dir', type=str, default="/home/bd/data/zs/data/COCO/train2017")
    parser.add_argument('--coco_val_json', type=str, default="/home/bd/data/zs/data/COCO/annotations/captions_val2017.json")
    parser.add_argument('--coco_val_img_dir', type=str, default="/home/bd/data/zs/data/COCO/val2017")
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--projector', type=str, default="mlp+norm", choices=["linear", "mlp", "residual", "norm", "mlp+norm", "bottleneck", "attention"])
    parser.add_argument('--model', type=str, default="base")
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    # 确保保存路径存在
    os.makedirs(args.save_path, exist_ok=True)
    
    # 加载ALIGN图像预处理
    ALIGN_MODEL_NAME = "kakaobrain/align-base"
    align_tokenizer = AutoTokenizer.from_pretrained(ALIGN_MODEL_NAME)
    align_model = AutoModel.from_pretrained(ALIGN_MODEL_NAME).to(device)
    align_model.eval()
    align_preprocess = T.Compose([
        T.Resize(256, interpolation=Image.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    # 创建MSCOCO数据集
    train_dataset = MSCOCODataset(
        args.coco_json,
        args.coco_img_dir,
        img_transform=align_preprocess
    )
    val_dataset = MSCOCODataset(
        args.coco_val_json,
        args.coco_val_img_dir,
        img_transform=align_preprocess
    )
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
    
    # original_rmg = 0
    # original_accuracy = 0
    # with torch.no_grad():
    #     for batch in tqdm(val_loader, desc="Processing MSCOCO"):
    #         images, text_tokens = batch
    #         images = images.to(device)
    #         text_tokens = clip.tokenize(text_tokens, truncate=True).to(device)
            
    #         # 获取特征
    #         img_feats, txt_feats = clip_model.encode_image(images), clip_model.encode_text(text_tokens)
            
    #         # 归一化特征
    #         img_feats = F.normalize(img_feats, p=2, dim=1)
    #         txt_feats = F.normalize(txt_feats, p=2, dim=1)
    #         original_rmg += compute_rmg(img_feats, txt_feats) * img_feats.shape[0]
    #         original_accuracy += calculate_accuracy(img_feats, txt_feats, np.eye(img_feats.shape[0]), threshold=0.5) * img_feats.shape[0]
    # original_rmg /= len(val_loader.dataset)
    # original_accuracy /= len(val_loader.dataset)
    # print(f"MSCOCO Original RMG: {original_rmg:.3f}")
    # print(f"MSCOCO Original Accuracy: {original_accuracy*100:.3f}%")
    
    # 开始训练
    combined_model = train(args, train_loader, val_loader)
    
    FLICKR_PATH = 'dataset_k_split.pkl'
    flickr30k_dataset = F30kCaptionsCap(FLICKR_PATH, transform=align_preprocess,
                                        target_transform=lambda x, truncate=True: align_tokenizer(x, padding="max_length", truncation=True, max_length=align_tokenizer.model_max_length, return_tensors="pt")["input_ids"])
    print(f"Loaded Flickr30k dataset with {len(flickr30k_dataset)} samples")

    batch_size = 32
    dataloader = DataLoader(flickr30k_dataset, batch_size=batch_size, shuffle=False)
    
    # original_rmg = 0
    # original_accuracy = 0
    # with torch.no_grad():
    #     for batch in tqdm(dataloader, desc="Processing Flickr30k"):
    #         images, _, captions = batch
    #         images = images.to(device)
    #         text_tokens = clip.tokenize(captions, truncate=True).to(device)
            
    #         # 获取特征
    #         img_feats, txt_feats = clip_model.encode_image(images), clip_model.encode_text(text_tokens)
            
    #         # 归一化特征
    #         img_feats = F.normalize(img_feats, p=2, dim=1)
    #         txt_feats = F.normalize(txt_feats, p=2, dim=1)
    #         original_rmg += compute_rmg(img_feats, txt_feats) * img_feats.shape[0]
    #         original_accuracy += calculate_accuracy(img_feats, txt_feats, np.eye(img_feats.shape[0]), threshold=0.5) * img_feats.shape[0]
    # original_rmg /= len(flickr30k_dataset)
    # original_accuracy /= len(flickr30k_dataset)
    # print(f"Flickr30k Original RMG: {original_rmg:.3f}")
    # print(f"Flickr30k Original Accuracy: {original_accuracy*100:.3f}%")
    
    combined_model.eval()  # Ensure the model is in evaluation mode
    
    total_rmg = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Flickr30k"):
            images, text_tokens, captions = batch
            
            # Move data to device
            images = images.to(device)
            # text_tokens 已经是token id tensor
            if isinstance(text_tokens, list):
                text_tokens = torch.stack([t.squeeze(0) for t in text_tokens]).to(device)
            else:
                text_tokens = text_tokens.to(device)
            text_inputs = {"input_ids": text_tokens}
            img_feats, txt_feats = combined_model(images, text_inputs)
            img_feats = F.normalize(img_feats, p=2, dim=1)
            txt_feats = F.normalize(txt_feats, p=2, dim=1)
            total_rmg += compute_rmg(img_feats, txt_feats) * img_feats.shape[0]
            total_accuracy += calculate_accuracy(img_feats, txt_feats, np.eye(img_feats.shape[0]), threshold=0.5) * img_feats.shape[0]
    
    flickr30k_rmg = total_rmg / len(flickr30k_dataset)
    flickr30k_accuracy = total_accuracy / len(flickr30k_dataset)
    print(f"Flickr30k RMG: {flickr30k_rmg:.3f}")
    print(f"Flickr30k Accuracy: {flickr30k_accuracy*100:.3f}%")
