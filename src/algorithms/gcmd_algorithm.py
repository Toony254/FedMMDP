"""
Gated Cross-modal Distillation (GCMD)算法流程的数学表示

符号说明:
- n_i: 第i个客户端
- m: 模态, m ∈ M_i
- D_i^{m,1}: 客户端n_i的模态m的标记数据集
- D_i^{m,u}: 客户端n_i的模态m的未标记数据集
- e_i^m: 客户端n_i的模态m的编码器
- f_i,j^m: 样本x_i,j^m经过编码器e_i^m后的特征
- Y_i^{m,1}: 真实标签集合
- Y_i^{m,1}: 聚类标签集合
- s_i^m: 模态m的互信息分数(MIS)
- m*: 优势模态
- B_i^m: 样本相关矩阵(SCM)
- L_rec: 重建损失
- L_dis: 蒸馏损失
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy

def compute_mutual_information_score(Y_true, Y_pred):
    """计算互信息分数(MIS)"""
    # 计算联合概率分布
    C = len(np.unique(Y_true))
    joint_pmf = np.zeros((C, C))
    for y_true, y_pred in zip(Y_true, Y_pred):
        joint_pmf[y_true, y_pred] += 1
    joint_pmf /= len(Y_true)
    
    # 计算边缘概率分布
    marginal_pmf_true = np.sum(joint_pmf, axis=1)
    marginal_pmf_pred = np.sum(joint_pmf, axis=0)
    
    # 计算互信息
    mi = 0.0
    for c in range(C):
        for c_tilde in range(C):
            if joint_pmf[c, c_tilde] > 0:
                mi += joint_pmf[c, c_tilde] * np.log2(
                    joint_pmf[c, c_tilde] / (marginal_pmf_true[c] * marginal_pmf_pred[c_tilde])
                )
    
    # 计算熵
    H_true = entropy(marginal_pmf_true, base=2)
    H_pred = entropy(marginal_pmf_pred, base=2)
    
    # 计算MIS
    mis = mi / (H_true + H_pred)
    return mis

def compute_sample_correlation_matrix(features):
    """计算样本相关矩阵(SCM)"""
    return np.dot(features, features.T)

def gcmd_algorithm(
    clients,  # 客户端列表
    modalities,  # 模态列表
    num_classes,  # 类别数
    kmeans_n_clusters=None,  # KMeans聚类的簇数
    max_iterations=100  # 最大迭代次数
):
    """GCMD算法主流程"""
    if kmeans_n_clusters is None:
        kmeans_n_clusters = num_classes
    
    for iteration in range(max_iterations):
        for client in clients:
            # 1. 优势模态识别
            mis_scores = {}
            for m in modalities:
                # 1.1 获取特征
                features = client.get_features(m)  # e_i^m(D_i^{m,1})
                
                # 1.2 KMeans聚类
                kmeans = KMeans(n_clusters=kmeans_n_clusters)
                cluster_labels = kmeans.fit_predict(features)
                
                # 1.3 计算MIS
                true_labels = client.get_labels(m)
                mis = compute_mutual_information_score(true_labels, cluster_labels)
                mis_scores[m] = mis
            
            # 1.4 确定优势模态
            superior_modality = max(mis_scores, key=mis_scores.get)
            
            # 2. 样本相关矩阵计算
            scm_matrices = {}
            for m in modalities:
                # 从无标签数据采样
                batch_features = client.sample_unlabeled_batch(m)
                # 计算SCM
                scm_matrices[m] = compute_sample_correlation_matrix(batch_features)
            
            # 3. 计算蒸馏损失
            for m in modalities:
                if m == superior_modality:
                    # 优势模态只计算重建损失
                    loss = client.compute_reconstruction_loss(m)
                else:
                    # 劣势模态计算重建损失+蒸馏损失
                    reconstruction_loss = client.compute_reconstruction_loss(m)
                    distillation_loss = np.linalg.norm(
                        scm_matrices[superior_modality] - scm_matrices[m]
                    )
                    loss = reconstruction_loss + distillation_loss
                
                # 4. 更新编码器
                client.update_encoder(m, loss)
        
        # 5. 模型聚合 (简化表示)
        server.aggregate_models(clients)
    
    return server.global_model    