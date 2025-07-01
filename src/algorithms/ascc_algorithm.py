import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def attention_based_spontaneous_client_clustering(
    client_id,  # 客户端ID
    cluster_encoders,  # 集群编码器列表 [e^1_m, ..., e^K_m]
    cluster_classifiers,  # 集群分类器列表 [c^1_m, ..., c^K_m]
    Lm,  # 编码器层数
    beta,  # 查询数据采样率
    labeled_dataset,  # 标记数据集 D_i^{m,l}
    learning_rate,  # 学习率 μ
    device='cpu'  # 计算设备
):
    """
    基于注意力的自发客户端聚类 (ASCC) 算法实现
    
    参数:
    - client_id: 客户端ID
    - cluster_encoders: 集群编码器列表，每个编码器对应一个集群
    - cluster_classifiers: 集群分类器列表，每个分类器对应一个集群
    - Lm: 编码器层数
    - beta: 查询数据采样率
    - labeled_dataset: 标记数据集
    - learning_rate: 学习率
    - device: 计算设备
    
    返回:
    - 个性化本地编码器 e_i^m
    - 个性化本地分类器 c_i^m
    - 选择的集群索引
    """
    K = len(cluster_encoders)  # 集群数量
    
    # 步骤2: 冻结编码器和分类器参数
    for encoder in cluster_encoders:
        for param in encoder.parameters():
            param.requires_grad = False
    
    for classifier in cluster_classifiers:
        for param in classifier.parameters():
            param.requires_grad = False
    
    # 步骤3: 初始化注意力参数 A_i^m = {A_i^{m,1}, ..., A_i^{m,Lm}}
    # 每个 A_i^{m,l} 是一个包含K个注意力参数的列表
    attention_params = []
    for l in range(Lm):
        layer_attention = [
            torch.rand(1, requires_grad=True, device=device)
            for _ in range(K)
        ]
        attention_params.append(layer_attention)
    
    # 步骤4: 从标记数据集中采样查询数据 Q_i^{m,0}
    num_samples = int(len(labeled_dataset) * beta)
    if num_samples == 0:
        num_samples = 1  # 确保至少有一个样本
    
    # 随机采样
    indices = np.random.choice(len(labeled_dataset), num_samples, replace=False)
    query_dataset = [labeled_dataset[i] for i in indices]
    
    # 创建数据加载器
    query_data = torch.stack([sample[0] for sample in query_dataset]).to(device)
    query_labels = torch.stack([sample[1] for sample in query_dataset]).to(device)
    query_loader = DataLoader(
        TensorDataset(query_data, query_labels),
        batch_size=16,
        shuffle=True
    )
    
    # 步骤5-15: 注意力训练
    optimizer = optim.Adam(
        [param for layer in attention_params for param in layer],
        lr=learning_rate
    )
    cross_entropy = nn.CrossEntropyLoss()
    
    # 假设初始选择的集群为0
    selected_cluster = 0
    
    # 训练注意力参数
    for epoch in range(10):  # 执行多个训练轮次
        total_loss = 0.0
        
        for x, y in query_loader:
            batch_size = x.size(0)
            optimizer.zero_grad()
            batch_loss = 0.0
            
            # 对每个样本执行前向传播
            for i in range(batch_size):
                sample = x[i:i+1]  # 形状: [1, feature_dim]
                
                # 初始化查询为输入样本
                q = sample
                
                # 逐层计算
                for l in range(Lm):
                    # 步骤9: 归一化注意力值
                    att_weights = torch.softmax(
                        torch.cat([param for param in attention_params[l]]), 
                        dim=0
                    )
                    
                    # 步骤11: 计算新查询 q_i,j^{m,l}
                    encoder_outputs = []
                    for k in range(K):
                        encoder_output = cluster_encoders[k].forward_layer(q, l)
                        encoder_outputs.append(encoder_output)
                    
                    # 加权聚合
                    q = torch.zeros_like(encoder_outputs[0])
                    for k in range(K):
                        q += att_weights[k] * encoder_outputs[k]
                
                # 步骤13: 计算注意力损失
                logits = cluster_classifiers[selected_cluster](q)
                loss = cross_entropy(logits, y[i:i+1])
                batch_loss += loss
            
            # 平均批次损失
            batch_loss /= batch_size
            total_loss += batch_loss.item()
            
            # 步骤15: 更新注意力参数
            batch_loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Client {client_id}, Epoch {epoch+1}, Attention Loss: {total_loss/len(query_loader):.4f}")
    
    # 步骤16-18: 聚合和聚类
    # 步骤17: 聚合编码器
    # 计算每个集群的累积注意力值
    accumulated_attention = [0.0] * K
    for l in range(Lm):
        att_weights = torch.softmax(
            torch.cat([param for param in attention_params[l]]), 
            dim=0
        )
        for k in range(K):
            accumulated_attention[k] += att_weights[k].item()
    
    # 步骤18: 根据累积注意力值重新选择集群
    selected_cluster = np.argmax(accumulated_attention)
    print(f"Client {client_id} selected cluster {selected_cluster}")
    
    # 步骤19-28: 本地模型微调
    # 解冻选定集群的编码器和分类器
    selected_encoder = cluster_encoders[selected_cluster]
    selected_classifier = cluster_classifiers[selected_cluster]
    
    for param in selected_encoder.parameters():
        param.requires_grad = True
    
    for param in selected_classifier.parameters():
        param.requires_grad = True
    
    # 创建完整数据集的加载器
    full_data = torch.stack([sample[0] for sample in labeled_dataset]).to(device)
    full_labels = torch.stack([sample[1] for sample in labeled_dataset]).to(device)
    full_loader = DataLoader(
        TensorDataset(full_data, full_labels),
        batch_size=16,
        shuffle=True
    )
    
    # 微调模型
    local_optimizer = optim.Adam(
        list(selected_encoder.parameters()) + list(selected_classifier.parameters()),
        lr=learning_rate
    )
    
    for epoch in range(5):  # 执行几个微调轮次
        total_loss = 0.0
        
        for x, y in full_loader:
            local_optimizer.zero_grad()
            
            # 前向传播
            features = selected_encoder(x)
            logits = selected_classifier(features)
            loss = cross_entropy(logits, y)
            
            # 反向传播和优化
            loss.backward()
            local_optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Client {client_id}, Fine-tuning Epoch {epoch+1}, Loss: {total_loss/len(full_loader):.4f}")
    
    # 返回个性化本地模型和选择的集群
    return selected_encoder, selected_classifier, selected_cluster

# 示例编码器类，包含逐层前向传播功能
class ExampleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.activation = nn.ReLU()
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
    
    def forward_layer(self, x, layer_idx):
        """逐层前向传播"""
        return self.activation(self.layers[layer_idx](x))

# 示例用法
def demo_ascc():
    # 初始化参数
    K = 3  # 集群数量
    Lm = 2  # 编码器层数
    beta = 0.2  # 查询数据采样率
    learning_rate = 0.001  # 学习率
    input_dim = 10  # 输入维度
    hidden_dim = 20  # 隐藏层维度
    output_dim = 5  # 输出维度（类别数）
    num_samples = 100  # 样本数
    
    # 创建示例数据
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    labeled_dataset = list(zip(X, y))
    
    # 初始化集群编码器和分类器
    cluster_encoders = [ExampleEncoder(input_dim, hidden_dim, Lm) for _ in range(K)]
    cluster_classifiers = [nn.Linear(hidden_dim, output_dim) for _ in range(K)]
    
    # 运行ASCC算法
    client_id = 0
    local_encoder, local_classifier, selected_cluster = attention_based_spontaneous_client_clustering(
        client_id, cluster_encoders, cluster_classifiers, Lm, beta, labeled_dataset, learning_rate
    )
    
    print(f"ASCC completed for client {client_id}.")
    print(f"Selected cluster: {selected_cluster}")

if __name__ == "__main__":
    demo_ascc()    