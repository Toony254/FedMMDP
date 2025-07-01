"""
Split-Shuffle Mechanism (SSM)算法流程的数学表示

符号说明:
- n_i: 第i个客户端
- m: 模态, m ∈ M_i
- w_i^m: 客户端n_i的模态m的本地模型参数
- S_i: 客户端n_i分割的组件数量
- s_i,j^m: 客户端n_i的模态m的第j个分割组件
- λ: 中继节点总数
- r_k: 第k个中继节点
- C: 聚类标签集合
- c_i: 客户端n_i的聚类标签
- χ_k^c: 中继节点r_k聚合的聚类c的中间结果
- W^c: 服务器聚合的聚类c的最终模型
"""

import numpy as np
from typing import List, Dict, Tuple

def split_model_parameters(
    model_params: np.ndarray,  # 模型参数向量
    num_components: int,  # 分割的组件数量
    random_seed: int = None  # 随机种子
) -> List[np.ndarray]:
    """将模型参数分割为多个随机组件"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 创建随机组件
    components = [np.random.randn(*model_params.shape) for _ in range(num_components - 1)]
    
    # 计算最后一个组件，确保所有组件之和等于原始模型参数
    last_component = model_params - sum(components)
    components.append(last_component)
    
    return components

def select_relay_nodes(
    total_relay_nodes: int,  # 中继节点总数
    num_selections: int,  # 需要选择的中继节点数量
    random_seed: int = None  # 随机种子
) -> List[int]:
    """随机选择中继节点"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    return np.random.choice(total_relay_nodes, num_selections, replace=False).tolist()

def ssm_algorithm(
    clients: List,  # 客户端列表
    relay_nodes: List,  # 中继节点列表
    server,  # 服务器
    total_relay_nodes: int,  # 中继节点总数
    max_components_per_client: int,  # 每个客户端最多分割的组件数
    num_clusters: int,  # 聚类数量
    random_seed: int = None  # 随机种子
) -> Dict[int, np.ndarray]:
    """Split-Shuffle Mechanism (SSM)算法主流程"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 初始化中间结果
    intermediate_results = {
        relay_node_id: {cluster: [] for cluster in range(num_clusters)}
        for relay_node_id in range(total_relay_nodes)
    }
    
    # 客户端分割并发送模型组件
    for client_id, client in enumerate(clients):
        # 获取客户端的聚类标签
        cluster_label = client.get_cluster_label()
        
        # 对每个模态执行SSM
        for modality in client.get_modalities():
            # 获取本地模型参数
            model_params = client.get_model_parameters(modality)
            
            # 确定分割的组件数量 (1 ≤ S_i ≤ max_components_per_client)
            num_components = np.random.randint(1, max_components_per_client + 1)
            
            # 分割模型参数
            components = split_model_parameters(model_params, num_components)
            
            # 选择中继节点
            selected_relay_nodes = select_relay_nodes(
                total_relay_nodes, num_components
            )
            
            # 发送组件到选定的中继节点
            for component, relay_node_id in zip(components, selected_relay_nodes):
                intermediate_results[relay_node_id][cluster_label].append(component)
    
    # 中继节点聚合中间结果
    for relay_node_id in range(total_relay_nodes):
        relay_node = relay_nodes[relay_node_id]
        
        for cluster_label, components in intermediate_results[relay_node_id].items():
            if components:  # 如果有该聚类的组件
                # 聚合组件
                aggregated_result = sum(components)
                
                # 中继节点发送中间结果到服务器
                server.receive_intermediate_result(
                    relay_node_id, cluster_label, aggregated_result
                )
    
    # 服务器聚合最终模型
    final_models = server.aggregate_final_models(num_clusters)
    
    return final_models    