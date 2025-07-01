"""
imagenet21k_analysis.py
ImageNet21K标签关联分析与可视化工具
"""

import nltk
from nltk.corpus import wordnet as wn
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyvis.network import Network
from typing import List, Dict

class ImageNetAnalyzer:
    def __init__(self, synsets: list[wn.synset], query_wnids: list[str] = None):
        """
        初始化分析器
        :param synsets: WordNet同义词集列表
        :param query_wnids: 查询标签wnid列表
        """
        self.synsets = synsets
        self.sim_matrix = None
        self.hierarchy_graph = nx.DiGraph()
        self.query_wnids = set(query_wnids) if query_wnids else set()
        self.node_wnid_map = {}  # node_name -> wnid

    def build_hierarchy_graph(self, max_depth: int = 20) -> nx.DiGraph:
        """
        构建层级关系图，确保所有类别都在图中，并记录每个节点的层数
        :param max_depth: 最大层级深度
        :return: NetworkX有向图对象
        """
        # 添加所有类别为节点
        for syn in sorted(self.synsets, key=lambda s: s.name()):
            node_name = self._format_node_name(syn)
            wnid = syn.name().split('.')[0]
            self.node_wnid_map[node_name] = wnid
            self.hierarchy_graph.add_node(node_name, size=15, title=node_name, level=0)

        # 构建层级关系并记录层数
        for syn in sorted(self.synsets, key=lambda s: s.name()):
            try:
                paths = sorted(syn.hypernym_paths(), key=lambda p: [x.name() for x in p])
                if paths:
                    path = paths[0][:max_depth]  # 截断到指定深度
                    for i in range(len(path) - 1):
                        parent = self._format_node_name(path[i])
                        child = self._format_node_name(path[i + 1])
                        self._add_hierarchy_edge(parent, child)
                        # 记录child的level为i+1（根节点level=0）
                        self.hierarchy_graph.nodes[child]['level'] = i + 1
                        # 记录wnid映射
                        self.node_wnid_map[parent] = path[i].name().split('.')[0]
                        self.node_wnid_map[child] = path[i + 1].name().split('.')[0]
            except IndexError:
                continue

        return self.hierarchy_graph

    def _format_node_name(self, syn: wn.synset) -> str:
        """格式化节点名称，保留wnid信息"""
        wnid = syn.name().split('.')[0]
        return f"{wnid}.{syn.pos()}{str(syn.offset()).zfill(8)}"

    def _add_hierarchy_edge(self, parent: str, child: str):
        """添加层级边并维护节点属性"""
        self.hierarchy_graph.add_node(parent, size=20, title=parent)
        self.hierarchy_graph.add_node(child, size=15, title=child)
        self.hierarchy_graph.add_edge(parent, child)

    def compute_similarity_matrix(self) -> np.ndarray:
        """
        计算语义相似度矩阵（Wu-Palmer算法）
        :return: NxN相似度矩阵
        """
        n = len(self.synsets)
        self.sim_matrix = np.identity(n)
        
        for i in range(n):
            for j in range(i+1, n):
                try:
                    sim = self.synsets[i].wup_similarity(self.synsets[j]) or 0
                except:
                    sim = 0
                self.sim_matrix[i][j] = sim
                self.sim_matrix[j][i] = sim
        return self.sim_matrix

    def visualize_hierarchy(self, output_file: str = "hierarchy.html"):
        """生成交互式层级可视化（PyVis），query标签为橙色，其余节点蓝色渐变，层级越高越深，节点越大"""
        net = Network(height="800px", width="100%", directed=True)
        import matplotlib
        from matplotlib import cm

        # 统计最大层级
        levels = [attrs.get('level', 0) for _, attrs in sorted(self.hierarchy_graph.nodes(data=True), key=lambda x: x[0])]
        if levels:
            max_level = max(levels)
            min_level = min(levels)
        else:
            max_level = 1
            min_level = 0

        # 生成蓝色渐变色（层级越高越深）
        n_colors = max_level - min_level + 1
        blues = cm.get_cmap('Blues', n_colors)
        blue_colors = [matplotlib.colors.rgb2hex(blues(i)) for i in range(n_colors)]

        # 节点大小范围
        max_size = 35
        min_size = 10

        orange = "#ff9900"        

        for node, attrs in sorted(self.hierarchy_graph.nodes(data=True), key=lambda x: x[0]):
            wnid = self.node_wnid_map.get(node, "")
            is_query = wnid in self.query_wnids
            level = attrs.get('level', 0)
            if is_query:
                color = orange
                size = 25  # query节点大小可自定义
            else:
                # 层级越高，level越小，颜色越深，节点越大
                color = blue_colors[level - min_level] if 0 <= (level - min_level) < len(blue_colors) else blue_colors[-1]
                # 线性插值大小
                if max_level > min_level:
                    size = max_size - (level - min_level) * (max_size - min_size) / (max_level - min_level)
                else:
                    size = max_size
            net.add_node(node, label=node, color=color, size=size, title=attrs.get('title', node))

        for u, v in sorted(self.hierarchy_graph.edges(), key=lambda x: (x[0], x[1])):
            net.add_edge(u, v)

        net.toggle_physics(True)
        net.show_buttons(filter_=['physics'])
        net.save_graph(output_file)
        print(f"层级可视化已保存至 {output_file}")

    def plot_heatmap(self, labels: List[str], output_file: str = "heatmap.png"):
        """绘制语义相似度热力图"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.sim_matrix,
            annot=True,
            xticklabels=labels,
            yticklabels=labels,
            cmap="YlGnBu"
        )
        plt.title("ImageNet21K 类别语义相似度矩阵")
        plt.savefig(output_file, bbox_inches='tight')
        print(f"热力图已保存至 {output_file}")
        
    def find_deep_subtrees(self, min_depth=5, num_subtrees=10):
        """
        查找深度大于min_depth且子树大小尽量接近的子树根节点
        :return: [(根节点名, 子树大小, 层级), ...]
        """
        candidates = []
        # 保证节点遍历顺序一致
        for node, attrs in sorted(self.hierarchy_graph.nodes(data=True), key=lambda x: x[0]):
            level = attrs.get('level', 0)
            if level >= min_depth:
                descendants = nx.descendants(self.hierarchy_graph, node)
                descendants = sorted(descendants)  # 保证后代顺序一致
                if not descendants:
                    continue
                # 检查所有后代节点都是叶子节点
                # all_leaf = all(self.hierarchy_graph.out_degree(n) == 0 for n in descendants)
                all_leaf = True
                if all_leaf:
                    size = len(descendants) + 1
                    candidates.append((node, size, level, descendants))
        if not candidates:
            return []

        target_size = 10
        # 排序前先按node名排序，保证稳定性
        candidates.sort(key=lambda x: (abs(x[1] - target_size), x[0]))

        selected = []
        excluded = set()
        for node, size, level, descendants in candidates:
            subtree_nodes = set(descendants) | {node}
            # 保证集合操作顺序一致
            if subtree_nodes.isdisjoint(excluded):
                selected.append((node, size, level))
                excluded.update(sorted(subtree_nodes))
            if len(selected) >= num_subtrees:
                break
        return selected

    def visualize_partitioned_hierarchy(self, subtree_roots, output_file="partitioned_hierarchy.html"):
        """
        对选中的子树分区可视化，不同子树用不同颜色
        :param subtree_roots: 子树根节点列表
        """
        net = Network(height="900px", width="100%", directed=True)
        import matplotlib
        from matplotlib import cm

        # 生成10种可区分的颜色
        tab10 = cm.get_cmap('tab10', len(subtree_roots))
        colors = [matplotlib.colors.rgb2hex(tab10(i)) for i in range(len(subtree_roots))]
        node_color_map = {}

        # 标记每个节点属于哪个子树
        for idx, (root, _, _) in enumerate(sorted(subtree_roots, key=lambda x: x[0])):
            descendants = nx.descendants(self.hierarchy_graph, root)
            for node in sorted(descendants):
                node_color_map[node] = colors[idx]
            node_color_map[root] = colors[idx]

        # 其它节点为灰色
        default_color = "#cccccc"

        for node, attrs in sorted(self.hierarchy_graph.nodes(data=True), key=lambda x: x[0]):
            color = node_color_map.get(node, default_color)
            size = attrs.get('size', 15)
            net.add_node(node, label=node, color=color, size=size, title=attrs.get('title', node))

        for u, v in sorted(self.hierarchy_graph.edges(), key=lambda x: (x[0], x[1])):
            net.add_edge(u, v)

        net.toggle_physics(True)
        net.save_graph(output_file)
        print(f"分区层级可视化已保存至 {output_file}")

def main():
    # 示例使用
    # 1. 准备测试数据（实际应替换为ImageNet21K的wnid列表）
    query_labels = []
    with open("src/datasets/imagenet_synsets.txt", "r") as f:
        for line in f:
            if line.startswith("n"):
                wnid = line.split()[0]
                query_labels.append(wnid)

    query_labels = sorted(query_labels)  # 保证顺序一致
    synsets = [wn.synset_from_pos_and_offset('n', int(wnid[1:])) for wnid in query_labels]
    # 2. 初始化分析器，传入query_labels
    analyzer = ImageNetAnalyzer(synsets, query_wnids=query_labels)

    # 3. 构建层级关系
    hierarchy_graph = analyzer.build_hierarchy_graph(max_depth=20)
    print(f"层级图包含 {hierarchy_graph.number_of_nodes()} 个节点, {hierarchy_graph.number_of_edges()} 条边")

    # 4. 可视化层级图
    analyzer.visualize_hierarchy(output_file="hierarchy_with_all_nodes.html")

    # 选取10个深度大于4的子树
    subtrees = analyzer.find_deep_subtrees(min_depth=5, num_subtrees=10)
    print("选中的10个子树根节点及其子树大小：")
    for idx, (root, size, level) in enumerate(subtrees):
        print(f"{idx+1}. 根节点: {root} | 层级: {level} | 子树大小: {size}")
        descendants = nx.descendants(analyzer.hierarchy_graph, root)
        subtree_nodes = [root] + sorted(descendants)
        print(f"    节点列表: {subtree_nodes}")

    # 分区可视化
    analyzer.visualize_partitioned_hierarchy(subtrees, output_file="partitioned_hierarchy.html")


if __name__ == "__main__":
    main()