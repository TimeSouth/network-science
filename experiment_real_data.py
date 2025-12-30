"""
基于结构摄动理论与反事实推演的社会生态相容性匹配方法
使用真实数据集进行实验评估
============================================================
支持数据集：
1. SNAP Facebook Ego Networks (https://snap.stanford.edu/data/)
2. WeChat Data (https://github.com/LiuTian821/WechatData)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import linalg
from collections import defaultdict
import warnings
import os
import urllib.request
import gzip
import tarfile
import shutil

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据集下载与加载
# ============================================================

class DatasetLoader:
    """数据集加载器"""
    
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    # -------------------- SNAP Facebook --------------------
    def download_facebook_dataset(self):
        """下载SNAP Facebook数据集"""
        url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
        gz_path = os.path.join(self.data_dir, "facebook_combined.txt.gz")
        txt_path = os.path.join(self.data_dir, "facebook_combined.txt")
        
        if os.path.exists(txt_path):
            print(f"Facebook数据集已存在: {txt_path}")
            return txt_path
        
        print(f"正在下载Facebook数据集...")
        print(f"URL: {url}")
        
        try:
            urllib.request.urlretrieve(url, gz_path)
            print("下载完成，正在解压...")
            
            with gzip.open(gz_path, 'rb') as f_in:
                with open(txt_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            os.remove(gz_path)
            print(f"Facebook数据集已保存至: {txt_path}")
            return txt_path
        except Exception as e:
            print(f"下载失败: {e}")
            print("\n请手动下载:")
            print(f"1. 访问 https://snap.stanford.edu/data/egonets-Facebook.html")
            print(f"2. 下载 facebook_combined.txt.gz")
            print(f"3. 解压后放到 {self.data_dir} 目录")
            return None
    
    def load_facebook_network(self):
        """加载Facebook网络"""
        txt_path = os.path.join(self.data_dir, "facebook_combined.txt")
        
        if not os.path.exists(txt_path):
            txt_path = self.download_facebook_dataset()
            if txt_path is None:
                return None
        
        print("正在加载Facebook网络...")
        G = nx.read_edgelist(txt_path, nodetype=int)
        print(f"加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        return G
    
    # -------------------- WeChat Data --------------------
    def load_wechat_network(self):
        """
        加载微信网络
        支持的文件名: friends.csv 或 edges.csv
        文件格式: user_id_1, user_id_2, ... (好友关系)
        """
        wechat_dir = os.path.join(self.data_dir, "wechat")
        
        # 检查可能的边文件名
        possible_edge_files = ["friends.csv", "edges.csv"]
        edges_path = None
        
        for filename in possible_edge_files:
            path = os.path.join(wechat_dir, filename)
            if os.path.exists(path):
                edges_path = path
                break
        
        if edges_path is None:
            print(f"微信数据集未找到，请将数据放到: {wechat_dir}")
            print(f"需要的文件: friends.csv (好友关系边文件)")
            print(f"下载地址: https://github.com/LiuTian821/WechatData")
            return None
        
        print(f"微信数据集已存在: {edges_path}")
        print("正在加载微信网络（文件较大，请稍候）...")
        
        # 读取边文件 - 支持字符串格式的用户ID
        G = nx.Graph()
        node_mapping = {}  # 将字符串ID映射为整数
        next_id = 0
        
        try:
            with open(edges_path, 'r', encoding='utf-8') as f:
                header = f.readline()  # 跳过表头
                line_count = 0
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        src_str, dst_str = parts[0].strip(), parts[1].strip()
                        
                        # 映射字符串ID到整数
                        if src_str not in node_mapping:
                            node_mapping[src_str] = next_id
                            next_id += 1
                        if dst_str not in node_mapping:
                            node_mapping[dst_str] = next_id
                            next_id += 1
                        
                        src, dst = node_mapping[src_str], node_mapping[dst_str]
                        G.add_edge(src, dst)
                        
                        line_count += 1
                        if line_count % 500000 == 0:
                            print(f"  已加载 {line_count} 条边...")
                            
        except Exception as e:
            print(f"读取边文件失败: {e}")
            return None
        
        print(f"加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        
        # 保存映射关系以备后用
        self.wechat_node_mapping = node_mapping
        self.wechat_reverse_mapping = {v: k for k, v in node_mapping.items()}
        
        return G
    
    # -------------------- 本地文件加载 --------------------
    def load_from_edgelist(self, filepath, delimiter=None):
        """从边列表文件加载网络"""
        print(f"正在加载网络: {filepath}")
        
        if delimiter:
            G = nx.read_edgelist(filepath, delimiter=delimiter, nodetype=int)
        else:
            G = nx.read_edgelist(filepath, nodetype=int)
        
        print(f"加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        return G
    
    def load_from_csv(self, filepath, src_col=0, dst_col=1, has_header=True):
        """从CSV文件加载网络"""
        print(f"正在加载网络: {filepath}")
        
        G = nx.Graph()
        with open(filepath, 'r', encoding='utf-8') as f:
            if has_header:
                f.readline()
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > max(src_col, dst_col):
                    try:
                        src, dst = int(parts[src_col]), int(parts[dst_col])
                        G.add_edge(src, dst)
                    except ValueError:
                        continue
        
        print(f"加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        return G


# ============================================================
# 核心算法（从experiment.py复用）
# ============================================================

def extract_ego_network(G, node, radius=1):
    """提取节点的Ego网络"""
    ego_nodes = {node}
    current_layer = {node}
    
    for _ in range(radius):
        next_layer = set()
        for n in current_layer:
            next_layer.update(G.neighbors(n))
        ego_nodes.update(next_layer)
        current_layer = next_layer
    
    return G.subgraph(ego_nodes).copy()

def merge_ego_networks(G, node_x, node_y, radius=1):
    """合并两个节点的Ego网络并添加强制连边"""
    ego_x = extract_ego_network(G, node_x, radius)
    ego_y = extract_ego_network(G, node_y, radius)
    
    merged_nodes = set(ego_x.nodes()) | set(ego_y.nodes())
    G_before = G.subgraph(merged_nodes).copy()
    
    G_after = G_before.copy()
    if not G_after.has_edge(node_x, node_y):
        G_after.add_edge(node_x, node_y)
    
    return G_before, G_after

def compute_laplacian_spectrum(G):
    """计算图的拉普拉斯特征值谱"""
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues = np.sort(linalg.eigvalsh(L))
    return eigenvalues

def compute_graph_energy(G):
    """计算图能量"""
    A = nx.adjacency_matrix(G).toarray()
    eigenvalues = linalg.eigvalsh(A)
    return np.sum(np.abs(eigenvalues))

def compute_fiedler_value(G):
    """计算Fiedler值"""
    if not nx.is_connected(G):
        return 0.0
    eigenvalues = compute_laplacian_spectrum(G)
    return eigenvalues[1] if len(eigenvalues) > 1 else 0.0

def compute_spectral_change(G_before, G_after):
    """计算谱变化指标"""
    spec_before = compute_laplacian_spectrum(G_before)
    spec_after = compute_laplacian_spectrum(G_after)
    
    min_len = min(len(spec_before), len(spec_after))
    spec_before = spec_before[:min_len]
    spec_after = spec_after[:min_len]
    
    diff = spec_after - spec_before
    S_spec = np.sqrt(np.sum(diff ** 2))
    
    return S_spec, spec_before, spec_after

def compute_energy_change(G_before, G_after):
    """计算图能量变化"""
    E_before = compute_graph_energy(G_before)
    E_after = compute_graph_energy(G_after)
    return E_after - E_before, E_before, E_after

def compute_path_length_change(G_before, G_after):
    """计算平均路径长度变化"""
    def safe_avg_path(G):
        if G.number_of_nodes() < 2:
            return 0
        if nx.is_connected(G):
            return nx.average_shortest_path_length(G)
        else:
            total = 0
            count = 0
            for cc in nx.connected_components(G):
                if len(cc) > 1:
                    subG = G.subgraph(cc)
                    total += nx.average_shortest_path_length(subG) * len(cc)
                    count += len(cc)
            return total / count if count > 0 else float('inf')
    
    L_before = safe_avg_path(G_before)
    L_after = safe_avg_path(G_after)
    return L_after - L_before, L_before, L_after

def compute_clustering_change(G_before, G_after):
    """计算聚类系数变化"""
    C_before = nx.average_clustering(G_before)
    C_after = nx.average_clustering(G_after)
    return C_after - C_before, C_before, C_after

def compute_common_neighbors(G, node_x, node_y):
    """计算共同邻居数"""
    neighbors_x = set(G.neighbors(node_x))
    neighbors_y = set(G.neighbors(node_y))
    return len(neighbors_x & neighbors_y)

def compute_social_distance(G, node_x, node_y):
    """计算社交距离"""
    try:
        return nx.shortest_path_length(G, node_x, node_y)
    except nx.NetworkXNoPath:
        return float('inf')

def compute_structural_stress(G, node_x, node_y, weights=None):
    """计算综合结构应力指标"""
    if weights is None:
        weights = {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.2, 'delta': 0.1}
    
    G_before, G_after = merge_ego_networks(G, node_x, node_y)
    
    S_spec, _, _ = compute_spectral_change(G_before, G_after)
    delta_E, E_before, _ = compute_energy_change(G_before, G_after)
    delta_L, L_before, _ = compute_path_length_change(G_before, G_after)
    delta_C, _, _ = compute_clustering_change(G_before, G_after)
    
    S_spec_norm = S_spec / (G_before.number_of_nodes() + 1)
    delta_E_norm = abs(delta_E) / (E_before + 1)
    delta_L_norm = abs(delta_L) / (L_before + 0.1) if L_before != float('inf') else 1.0
    
    I_stress = (weights['alpha'] * S_spec_norm + 
                weights['beta'] * delta_E_norm + 
                weights['gamma'] * delta_L_norm)
    
    return {
        'I_stress': I_stress,
        'S_spec': S_spec,
        'S_spec_norm': S_spec_norm,
        'delta_E': delta_E,
        'delta_E_norm': delta_E_norm,
        'delta_L': delta_L,
        'delta_L_norm': delta_L_norm,
        'delta_C': delta_C,
        'G_before': G_before,
        'G_after': G_after
    }

def compute_compatibility_score(I_stress, common_neighbors=0, method='exponential', delta=0.1):
    """计算相容性评分"""
    if method == 'inverse':
        score = 1.0 / (1.0 + I_stress)
    else:
        base_score = np.exp(-I_stress)
        neighbor_bonus = 1 + delta * np.log(1 + common_neighbors)
        score = base_score * neighbor_bonus
    
    return min(max(score, 0), 1)


# ============================================================
# 实验流程
# ============================================================

def get_network_stats(G):
    """计算网络基本统计特征"""
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
    }
    
    if nx.is_connected(G):
        stats['avg_path_length'] = nx.average_shortest_path_length(G)
        stats['diameter'] = nx.diameter(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        stats['avg_path_length'] = nx.average_shortest_path_length(subG)
        stats['diameter'] = nx.diameter(subG)
        stats['num_components'] = nx.number_connected_components(G)
        stats['largest_cc_size'] = len(largest_cc)
    
    return stats

def sample_candidate_pairs(G, n_pairs=100, ensure_diversity=True):
    """从网络中采样候选用户对"""
    nodes = list(G.nodes())
    pairs = []
    
    if ensure_diversity:
        distance_buckets = defaultdict(list)
        
        np.random.seed(42)
        sampled = 0
        max_attempts = n_pairs * 50
        attempts = 0
        
        while sampled < n_pairs * 3 and attempts < max_attempts:
            attempts += 1
            x, y = np.random.choice(nodes, 2, replace=False)
            if G.has_edge(x, y):
                continue
            
            dist = compute_social_distance(G, x, y)
            if dist == float('inf'):
                bucket = 'inf'
            elif dist <= 2:
                bucket = 'close'
            elif dist <= 4:
                bucket = 'medium'
            else:
                bucket = 'far'
            
            distance_buckets[bucket].append((x, y))
            sampled += 1
        
        per_bucket = n_pairs // 4
        for bucket in ['close', 'medium', 'far', 'inf']:
            bucket_pairs = distance_buckets[bucket]
            if len(bucket_pairs) > 0:
                selected = min(per_bucket, len(bucket_pairs))
                pairs.extend(bucket_pairs[:selected])
    else:
        np.random.seed(42)
        while len(pairs) < n_pairs:
            x, y = np.random.choice(nodes, 2, replace=False)
            if not G.has_edge(x, y) and (x, y) not in pairs and (y, x) not in pairs:
                pairs.append((x, y))
    
    return pairs[:n_pairs]

def run_matching_experiment(G, pairs, network_name='unknown'):
    """运行匹配扰动实验"""
    results = []
    
    for i, (node_x, node_y) in enumerate(pairs):
        if (i + 1) % 20 == 0:
            print(f"  处理第 {i+1}/{len(pairs)} 对...")
        
        common_neighbors = compute_common_neighbors(G, node_x, node_y)
        social_distance = compute_social_distance(G, node_x, node_y)
        
        stress_result = compute_structural_stress(G, node_x, node_y)
        score = compute_compatibility_score(
            stress_result['I_stress'], 
            common_neighbors,
            method='exponential'
        )
        
        fiedler_before = compute_fiedler_value(stress_result['G_before'])
        fiedler_after = compute_fiedler_value(stress_result['G_after'])
        
        results.append({
            'node_x': node_x,
            'node_y': node_y,
            'common_neighbors': common_neighbors,
            'social_distance': social_distance,
            'I_stress': stress_result['I_stress'],
            'S_spec': stress_result['S_spec'],
            'delta_E': stress_result['delta_E'],
            'delta_L': stress_result['delta_L'],
            'delta_C': stress_result['delta_C'],
            'fiedler_before': fiedler_before,
            'fiedler_after': fiedler_after,
            'fiedler_change': fiedler_after - fiedler_before,
            'score': score,
            'network_name': network_name
        })
    
    return results

def analyze_results(results):
    """分析实验结果"""
    import pandas as pd
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("实验结果统计分析")
    print("="*60)
    
    for net_name in df['network_name'].unique():
        subset = df[df['network_name'] == net_name]
        print(f"\n【{net_name}】")
        print(f"  样本数: {len(subset)}")
        print(f"  平均相容性评分: {subset['score'].mean():.4f} ± {subset['score'].std():.4f}")
        print(f"  平均结构应力: {subset['I_stress'].mean():.4f} ± {subset['I_stress'].std():.4f}")
        print(f"  平均共同邻居数: {subset['common_neighbors'].mean():.2f}")
    
    print("\n【按社交距离分组】")
    df['distance_group'] = df['social_distance'].apply(
        lambda x: '2跳(有共同好友)' if x <= 2 else ('3-4跳' if x <= 4 else ('5+跳' if x < float('inf') else '不连通'))
    )
    
    for group in ['2跳(有共同好友)', '3-4跳', '5+跳', '不连通']:
        subset = df[df['distance_group'] == group]
        if len(subset) > 0:
            print(f"  {group}: 平均评分={subset['score'].mean():.4f}, 平均应力={subset['I_stress'].mean():.4f}, 样本数={len(subset)}")
    
    print("\n【相关性分析】")
    valid_df = df[df['social_distance'] < float('inf')]
    if len(valid_df) > 5:
        corr_cn_score = valid_df['common_neighbors'].corr(valid_df['score'])
        corr_dist_score = valid_df['social_distance'].corr(valid_df['score'])
        print(f"  共同邻居数 vs 评分 相关系数: {corr_cn_score:.4f}")
        print(f"  社交距离 vs 评分 相关系数: {corr_dist_score:.4f}")
    
    return df

def visualize_results(df, save_path='results'):
    """可视化实验结果"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 共同邻居数 vs 相容性评分
    ax1 = axes[0, 0]
    for net_name in df['network_name'].unique():
        subset = df[df['network_name'] == net_name]
        ax1.scatter(subset['common_neighbors'], subset['score'], alpha=0.6, label=net_name)
    ax1.set_xlabel('共同邻居数')
    ax1.set_ylabel('相容性评分')
    ax1.set_title('共同邻居数 vs 相容性评分')
    ax1.legend()
    
    # 2. 社交距离 vs 相容性评分
    ax2 = axes[0, 1]
    valid_df = df[df['social_distance'] < float('inf')]
    for net_name in valid_df['network_name'].unique():
        subset = valid_df[valid_df['network_name'] == net_name]
        ax2.scatter(subset['social_distance'], subset['score'], alpha=0.6, label=net_name)
    ax2.set_xlabel('社交距离')
    ax2.set_ylabel('相容性评分')
    ax2.set_title('社交距离 vs 相容性评分')
    ax2.legend()
    
    # 3. 结构应力 vs 相容性评分
    ax3 = axes[0, 2]
    for net_name in df['network_name'].unique():
        subset = df[df['network_name'] == net_name]
        ax3.scatter(subset['I_stress'], subset['score'], alpha=0.6, label=net_name)
    ax3.set_xlabel('结构应力 I_stress')
    ax3.set_ylabel('相容性评分')
    ax3.set_title('结构应力 vs 相容性评分')
    ax3.legend()
    
    # 4. 评分分布
    ax4 = axes[1, 0]
    network_names = df['network_name'].unique()
    scores_by_type = [df[df['network_name'] == t]['score'].values for t in network_names]
    ax4.boxplot(scores_by_type, labels=network_names)
    ax4.set_ylabel('相容性评分')
    ax4.set_title('评分分布')
    
    # 5. Fiedler值变化分布
    ax5 = axes[1, 1]
    for net_name in df['network_name'].unique():
        subset = df[df['network_name'] == net_name]
        ax5.hist(subset['fiedler_change'], alpha=0.5, label=net_name, bins=20)
    ax5.set_xlabel('Fiedler值变化')
    ax5.set_ylabel('频次')
    ax5.set_title('Fiedler值变化分布')
    ax5.legend()
    
    # 6. 按距离分组的平均评分
    ax6 = axes[1, 2]
    distance_groups = ['2跳(有共同好友)', '3-4跳', '5+跳', '不连通']
    avg_scores = []
    for group in distance_groups:
        subset = df[df['distance_group'] == group]
        avg_scores.append(subset['score'].mean() if len(subset) > 0 else 0)
    
    colors = ['green', 'yellow', 'orange', 'red']
    bars = ax6.bar(range(len(distance_groups)), avg_scores, color=colors)
    ax6.set_xticks(range(len(distance_groups)))
    ax6.set_xticklabels(distance_groups, rotation=15)
    ax6.set_ylabel('平均相容性评分')
    ax6.set_title('不同社交距离的平均评分')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/real_data_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存至 {save_path}/real_data_results.png")

def visualize_case_study(G, node_x, node_y, network_name, save_path='results'):
    """可视化案例分析"""
    os.makedirs(save_path, exist_ok=True)
    
    G_before, G_after = merge_ego_networks(G, node_x, node_y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ego_x = extract_ego_network(G, node_x)
    ego_y = extract_ego_network(G, node_y)
    
    def get_node_colors(graph):
        colors = []
        for node in graph.nodes():
            if node == node_x:
                colors.append('darkblue')
            elif node == node_y:
                colors.append('darkorange')
            elif node in ego_x.nodes() and node in ego_y.nodes():
                colors.append('purple')
            elif node in ego_x.nodes():
                colors.append('lightblue')
            else:
                colors.append('lightsalmon')
        return colors
    
    ax1 = axes[0]
    pos_before = nx.spring_layout(G_before, seed=42)
    colors_before = get_node_colors(G_before)
    nx.draw(G_before, pos_before, ax=ax1, node_color=colors_before, 
            node_size=300, with_labels=False, edge_color='gray', alpha=0.8)
    ax1.set_title(f'匹配前\n节点数: {G_before.number_of_nodes()}, 边数: {G_before.number_of_edges()}')
    
    ax2 = axes[1]
    pos_after = nx.spring_layout(G_after, seed=42)
    colors_after = get_node_colors(G_after)
    
    edges_before = set(G_before.edges())
    edges_after = set(G_after.edges())
    new_edge = edges_after - edges_before
    
    nx.draw_networkx_nodes(G_after, pos_after, ax=ax2, node_color=colors_after, node_size=300)
    nx.draw_networkx_edges(G_after, pos_after, ax=ax2, edgelist=list(edges_before), 
                           edge_color='gray', alpha=0.8)
    nx.draw_networkx_edges(G_after, pos_after, ax=ax2, edgelist=list(new_edge), 
                           edge_color='red', width=3, alpha=1.0)
    ax2.set_title(f'匹配后（红色为新增边）\n节点数: {G_after.number_of_nodes()}, 边数: {G_after.number_of_edges()}')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkblue', label=f'用户X (节点{node_x})'),
        Patch(facecolor='darkorange', label=f'用户Y (节点{node_y})'),
        Patch(facecolor='purple', label='共同好友'),
        Patch(facecolor='lightblue', label='X的好友'),
        Patch(facecolor='lightsalmon', label='Y的好友'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/case_{network_name}_{node_x}_{node_y}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    stress_result = compute_structural_stress(G, node_x, node_y)
    common_neighbors = compute_common_neighbors(G, node_x, node_y)
    social_distance = compute_social_distance(G, node_x, node_y)
    score = compute_compatibility_score(stress_result['I_stress'], common_neighbors)
    
    print(f"\n案例分析 [{network_name}]: 节点 {node_x} 与 节点 {node_y}")
    print(f"  共同邻居数: {common_neighbors}")
    print(f"  社交距离: {social_distance}")
    print(f"  结构应力 I_stress: {stress_result['I_stress']:.4f}")
    print(f"  谱变化 S_spec: {stress_result['S_spec']:.4f}")
    print(f"  能量变化 ΔE: {stress_result['delta_E']:.4f}")
    print(f"  相容性评分: {score:.4f} (满分1.0)")


# ============================================================
# 主程序
# ============================================================

def main():
    print("="*60)
    print("基于结构摄动理论的社会生态相容性匹配实验")
    print("使用真实数据集评估")
    print("="*60)
    
    # 配置
    DATA_DIR = 'C:/Users/timesouthli/CodeBuddy/网络科学/data'
    RESULTS_DIR = 'C:/Users/timesouthli/CodeBuddy/网络科学/results_real'
    N_PAIRS = 80  # 每个数据集采样的配对数
    
    loader = DatasetLoader(DATA_DIR)
    all_results = []
    networks = {}
    
    # ==================== 选择数据集 ====================
    print("\n请选择要使用的数据集:")
    print("  1. SNAP Facebook (推荐，4039节点)")
    print("  2. WeChat Data")
    print("  3. 两者都使用")
    print("  4. 使用本地文件")
    
    choice = input("\n请输入选项 (1/2/3/4，默认1): ").strip() or '1'
    
    if choice in ['1', '3']:
        print("\n" + "-"*40)
        print("[加载 SNAP Facebook 数据集]")
        G_fb = loader.load_facebook_network()
        if G_fb:
            networks['Facebook'] = G_fb
            stats = get_network_stats(G_fb)
            print(f"  平均度: {stats['avg_degree']:.2f}")
            print(f"  聚类系数: {stats['avg_clustering']:.4f}")
            print(f"  平均路径长度: {stats['avg_path_length']:.2f}")
    
    if choice in ['2', '3']:
        print("\n" + "-"*40)
        print("[加载 WeChat 数据集]")
        G_wc = loader.load_wechat_network()
        if G_wc:
            networks['WeChat'] = G_wc
            stats = get_network_stats(G_wc)
            print(f"  平均度: {stats['avg_degree']:.2f}")
            print(f"  聚类系数: {stats['avg_clustering']:.4f}")
    
    if choice == '4':
        print("\n请输入边列表文件的完整路径:")
        filepath = input().strip()
        if os.path.exists(filepath):
            G_local = loader.load_from_edgelist(filepath)
            if G_local:
                networks['LocalData'] = G_local
        else:
            print(f"文件不存在: {filepath}")
    
    if not networks:
        print("\n没有成功加载任何数据集，退出。")
        return
    
    # ==================== 运行实验 ====================
    print("\n" + "="*60)
    print("开始实验")
    print("="*60)
    
    for net_name, G in networks.items():
        print(f"\n[在 {net_name} 上运行实验]")
        
        # 如果网络太大，取最大连通分量
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            print(f"  使用最大连通分量: {G.number_of_nodes()} 节点")
        
        pairs = sample_candidate_pairs(G, N_PAIRS, ensure_diversity=True)
        print(f"  采样 {len(pairs)} 对候选用户")
        
        results = run_matching_experiment(G, pairs, net_name)
        all_results.extend(results)
        
        # 保存网络引用用于案例分析
        networks[net_name] = G
    
    # ==================== 分析与可视化 ====================
    print("\n" + "="*60)
    df = analyze_results(all_results)
    
    print("\n[生成可视化图表]")
    visualize_results(df, save_path=RESULTS_DIR)
    
    # 案例分析
    print("\n[案例分析]")
    for net_name, G in networks.items():
        # 有共同好友的案例
        close_cases = df[(df['network_name'] == net_name) & (df['common_neighbors'] > 0)]
        if len(close_cases) > 0:
            case = close_cases.iloc[0]
            print(f"\n案例: {net_name} - 有共同好友")
            visualize_case_study(G, case['node_x'], case['node_y'], net_name, RESULTS_DIR)
        
        # 无共同好友的案例
        far_cases = df[(df['network_name'] == net_name) & 
                       (df['common_neighbors'] == 0) & 
                       (df['social_distance'] < float('inf'))]
        if len(far_cases) > 0:
            case = far_cases.iloc[0]
            print(f"\n案例: {net_name} - 无共同好友")
            visualize_case_study(G, case['node_x'], case['node_y'], net_name + '_far', RESULTS_DIR)
    
    # 保存结果到CSV
    csv_path = os.path.join(RESULTS_DIR, 'experiment_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存至: {csv_path}")
    
    print("\n" + "="*60)
    print("实验完成!")
    print(f"结果保存在: {RESULTS_DIR}")
    print("="*60)
    
    return df


if __name__ == "__main__":
    df = main()
