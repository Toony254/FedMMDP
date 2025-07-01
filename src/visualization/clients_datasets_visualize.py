import os
import sys
import random
import warnings
import pickle
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import clip
from datasets import load_from_disk
import gc

from torch.utils.data import DataLoader, Subset
from src.datasets.transform import collate_fn

sys.path.append("../../")
sys.path.append("../../../")


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

vdim, tdim = 512, 512
print("using {} device.".format(device))
clip_model, img_preprocess = clip.load("ViT-B/32", device = device)
clip_model = clip_model.float()
D = 512

class CombinedModel(nn.Module):
    def __init__(self, clip_model):
        super(CombinedModel, self).__init__()
        self.clip_model = clip_model
        self.visual_projector = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        self.text_projector = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, img_input, text_input):
        img_feats = self.clip_model.encode_image(img_input)
        img_feats = self.visual_projector(img_feats)
        text_feats = self.clip_model.encode_text(text_input)
        text_feats = self.text_projector(text_feats)
        return img_feats, text_feats

# 组合CLIP和自定义网络
combined_model = CombinedModel(clip_model).to(device)
param_dict = torch.load("saved/projector_weights/RMG_projector_MLP.pth")
combined_model.visual_projector.load_state_dict(param_dict['visual_projector'].state_dict())
combined_model.text_projector.load_state_dict(param_dict['text_projector'].state_dict())
combined_model.eval()

# 加载ImageNet-Cap数据集
def load_imagenet_cap_dataset(data_root, num_clients=5, partition='hetero', alpha=0.5):
    print(f"Loading ImageNet-Cap dataset from {data_root}...")
    dataset = load_from_disk(data_root)
    
    # 分片数据集，方便处理
    dataset_sm = dataset.shard(num_shards=21, index=0)
    dataset_mm = dataset.shard(num_shards=21, index=1)
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 为可视化创建不同模态的客户端数据集
    client_datasets = {}
    
    # 图像模态数据
    image_set = dataset_sm.remove_columns(["cap_tokens", "recaption_short"])
    image_train_test_split = image_set.train_test_split(test_size=0.1)
    image_train_set = image_train_test_split['train']
    
    # 文本模态数据
    text_set = dataset_sm.remove_columns(["processed_img", "recaption_short"])
    text_train_test_split = text_set.train_test_split(test_size=0.1)
    text_train_set = text_train_test_split['train']
    
    # 多模态数据
    mm_set = dataset_mm.remove_columns(["recaption_short"])
    mm_train_test_split = mm_set.train_test_split(test_size=0.1)
    mm_train_set = mm_train_test_split['train']
    
    # 获取数据集的目标标签和样本数量
    image_targets = image_train_set['class_id']
    text_targets = text_train_set['class_id']
    mm_targets = mm_train_set['class_id']
    
    # 对每个模态进行数据分区
    image_idx_map = data_partitioner('image', image_train_set.num_rows, num_clients, partition, alpha, np.array(image_targets))
    text_idx_map = data_partitioner('text', text_train_set.num_rows, num_clients, partition, alpha, np.array(text_targets))
    mm_idx_map = data_partitioner('mm', mm_train_set.num_rows, num_clients, partition, alpha, np.array(mm_targets))
    
    # 创建客户端数据集
    for i in range(num_clients):
        # 图像模态客户端
        client_datasets[f'image_client_{i}'] = {
            'dataset': Subset(image_train_set, image_idx_map[i]),
            'type': 'image',
            'client_id': i
        }
        
        # 文本模态客户端
        client_datasets[f'text_client_{i}'] = {
            'dataset': Subset(text_train_set, text_idx_map[i]),
            'type': 'text',
            'client_id': i
        }
        
        # 多模态客户端
        client_datasets[f'mm_client_{i}'] = {
            'dataset': Subset(mm_train_set, mm_idx_map[i]),
            'type': 'mm',
            'client_id': i
        }
    
    return client_datasets

# 数据分区函数
def data_partitioner(dataset_name, num_samples, num_clients, partition='homo', alpha=0.5, y_train=None):
    check_dir = f"./data_partition/client_{dataset_name}"

    if partition == "homo":
        check_dir = check_dir + "_iid.pkl"
        if os.path.isfile(check_dir):
            net_dataidx_map = pickle.load(open(check_dir, 'rb'))
        else:
            idxs = np.random.permutation(num_samples)
            batch_idxs = np.array_split(idxs, num_clients)
            net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}
            pickle.dump(net_dataidx_map, open(check_dir, 'wb'))

    elif partition == "hetero":
        check_dir = check_dir + "_noniid.pkl"
        if os.path.isfile(check_dir):
            net_dataidx_map = pickle.load(open(check_dir, 'rb'))
        else:
            min_size = 0
            K = max(y_train) + 1
            net_dataidx_map = {}
            print('Hetero partition')
            while min_size < 40: # min_size of single modal dataset per clients
                idx_batch = [[] for _ in range(num_clients)]
                # for each class in the dataset
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                    ## Balance
                    proportions = np.array(
                        [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(num_clients):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

            pickle.dump(net_dataidx_map, open(check_dir, 'wb'))

    return net_dataidx_map

# 为可视化加载数据集
all_client_datasets = load_imagenet_cap_dataset('data/full_dataset', num_clients=5)
print(f"总共加载了 {len(all_client_datasets)} 个客户端数据集")

# 计算客户端表征
def compute_client_representations(client_datasets, combined_model, device):
    client_representations = {}
    
    for client_name, client_info in tqdm(client_datasets.items(), desc="计算客户端表征"):
        client_dataset = client_info['dataset']
        client_type = client_info['type']
        client_id = client_info['client_id']
        
        if client_dataset is None:
            continue
        
        batch_size = 32
        features_list = []
        modalities = []
        
        with torch.no_grad():
            if client_type == 'image':
                dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                for batch_data in tqdm(dataloader, desc=f"{client_name}", leave=False):
                    # 图像数据处理
                    images = batch_data["processed_img"].to(device)
                    
                    # 使用CLIP模型编码图像
                    img_feats = combined_model.clip_model.encode_image(images)
                    img_feats = combined_model.visual_projector(img_feats)
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    
                    features_list.append(img_feats.cpu())
                    modalities.extend(['image'] * len(images))
            
            elif client_type == 'text':
                dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                for batch_data in tqdm(dataloader, desc=f"{client_name}", leave=False):
                    # 文本数据处理
                    text_tokens = batch_data["cap_tokens"].to(device)
                    
                    # 使用CLIP模型编码文本
                    txt_feats = combined_model.clip_model.encode_text(text_tokens)
                    txt_feats = combined_model.text_projector(txt_feats)
                    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                    
                    features_list.append(txt_feats.cpu())
                    modalities.extend(['text'] * len(text_tokens))
            
            elif client_type == 'mm':
                dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                for batch_data in tqdm(dataloader, desc=f"{client_name}", leave=False):
                    # 多模态数据处理
                    images = batch_data["processed_img"].to(device)
                    text_tokens = batch_data["cap_tokens"].to(device)

                    # 使用combined_model处理图像和文本
                    img_feats, txt_feats = combined_model(images, text_tokens)
                    
                    # 归一化特征
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                    
                    features_list.append(img_feats.cpu())
                    features_list.append(txt_feats.cpu())
                    modalities.extend(['image'] * len(images))
                    modalities.extend(['text'] * len(text_tokens))
        
        if features_list:
            all_features = torch.cat(features_list, dim=0).numpy()
            client_representations[client_name] = {
                'features': all_features,
                'modalities': modalities,
                'client_id': client_id,
                'client_type': client_type
            }
    
    return client_representations

# 设置随机种子以保证结果可重现
np.random.seed(42)
torch.manual_seed(42)

# 计算所有客户端数据的表征
client_representations = compute_client_representations(all_client_datasets, combined_model, device)

print(f"已计算 {len(client_representations)} 个客户端的数据表征")

# 使用t-SNE可视化所有客户端表征
def visualize_client_representations(client_representations, n_components=2, perplexity=30):
    # Collect all features and related information
    all_features = []
    all_modalities = []
    all_client_ids = []
    all_client_types = []
    
    for client_name, rep_info in client_representations.items():
        all_features.append(rep_info['features'])
        all_modalities.extend(rep_info['modalities'])
        all_client_ids.extend([client_name] * len(rep_info['features']))
        all_client_types.extend([rep_info['client_type']] * len(rep_info['features']))
    
    all_features = np.concatenate(all_features, axis=0)
    
    # Use t-SNE for dimensionality reduction
    print(f"Using t-SNE to reduce dimensions for {all_features.shape[0]} samples...")
    tsne = TSNE(n_components=n_components, perplexity=min(perplexity, all_features.shape[0]-1), 
                random_state=42, init='pca', learning_rate='auto')
    features_embedded = tsne.fit_transform(all_features)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Assign a unique color to each client
    unique_clients = np.unique(all_client_ids)
    client_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clients)))
    
    # Set different markers for different modalities
    modality_markers = {'image': 'o', 'text': '^'}
    
    # Create scatter plot
    for i, client_id in enumerate(unique_clients):
        for modality in ['image', 'text']:
            mask = (np.array(all_client_ids) == client_id) & (np.array(all_modalities) == modality)
            if np.any(mask):
                plt.scatter(
                    features_embedded[mask, 0], 
                    features_embedded[mask, 1], 
                    marker=modality_markers[modality],
                    color=client_colors[i],
                    label=f'{client_id} ({modality})',
                    alpha=0.7,
                    s=50
                )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE Visualization of ImageNet-Cap Client Representations')
    plt.tight_layout()
    
    # Create directory for saving
    os.makedirs("visualization_results", exist_ok=True)
    
    # Save the image
    plt.savefig('visualization_results/imagenet_cap_client_representations_tsne.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to visualization_results/imagenet_cap_client_representations_tsne.png")
    plt.show()

# 可视化客户端表征
visualize_client_representations(client_representations)