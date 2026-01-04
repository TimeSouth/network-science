"""
基于结构摄动理论与反事实推演的社会生态相容性匹配方法实验
============================================================
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import linalg
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一部分：网络生成器
# ============================================================

def generate_hierarchical_network(n_nodes=1000, n_communities=10, p_in=0.3, p_out=0.01):
    """
    生成层级型网络：模拟存在明显层次结构的社交网络
    
    参数:
        n_nodes: 节点总数
        n_communities: 社区数量
        p_in: 社区内连边概率
        p_out: 社区间连边概率
    """
    # 使用随机块模型生成层级社区结构
    sizes = [n_nodes // n_communities] * n_communities
    sizes[-1] += n_nodes - sum(sizes)  # 处理余数
    
    # 构建概率矩阵
    p_matrix = np.full((n_communities, n_communities), p_out)
    np.fill_diagonal(p_matrix, p_in)
    
    G = nx.stochastic_block_model(sizes, p_matrix)
    
    # 添加社区标签
    community_labels = {}
    node_idx = 0
    for comm_id, size in enumerate(sizes):
        for _ in range(size):
            community_labels[node_idx] = comm_id
            node_idx += 1
    nx.set_node_attributes(G, community_labels, 'community')
    
    return G

def generate_clustered_network(n_nodes=1000, k=10, p=0.1):
    """
    生成紧密团簇型网络：使用Watts-Strogatz小世界模型
    
    参数:
        n_nodes: 节点数
        k: 每个节点连接的近邻数
        p: 重连概率
    """
    G = nx.watts_strogatz_graph(n_nodes, k, p)
    return G

def generate_sparse_network(n_nodes=1000, m=3):
    """
    生成稀疏弱连接型网络：使用Barabási-Albert无标度模型
    
    参数:
        n_nodes: 节点数
        m: 每个新节点连接的边数
    """
    G = nx.barabasi_albert_graph(n_nodes, m)
    return G

def get_network_stats(G):
    """计算网络基本统计特征"""
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
    }
    
    # 计算平均路径长度（仅对连通图）
    if nx.is_connected(G):
        stats['avg_path_length'] = nx.average_shortest_path_length(G)
        stats['diameter'] = nx.diameter(G)
    else:
        # 取最大连通分量
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        stats['avg_path_length'] = nx.average_shortest_path_length(subG)
        stats['diameter'] = nx.diameter(subG)
        stats['num_components'] = nx.number_connected_components(G)
    
    return stats

# ============================================================
# 第二部分：Ego网络提取与合并
# ============================================================

def extract_ego_network(G, node, radius=1):
    """
    提取节点的Ego网络
    
    参数:
        G: 原始网络
        node: 中心节点
        radius: 邻域半径（默认1跳）
    """
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
    """
    合并两个节点的Ego网络并添加强制连边
    
    返回:
        G_before: 合并前的网络（无X-Y边）
        G_after: 合并后的网络（有X-Y边）
    """
    # 提取两个ego网络
    ego_x = extract_ego_network(G, node_x, radius)
    ego_y = extract_ego_network(G, node_y, radius)
    
    # 合并网络
    merged_nodes = set(ego_x.nodes()) | set(ego_y.nodes())
    G_before = G.subgraph(merged_nodes).copy()
    
    # 添加强制连边
    G_after = G_before.copy()
    if not G_after.has_edge(node_x, node_y):
        G_after.add_edge(node_x, node_y)
    
    return G_before, G_after

# ============================================================
# 第三部分：结构应力评估指标
# ============================================================

def compute_laplacian_spectrum(G):
    """计算图的拉普拉斯特征值谱"""
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues = np.sort(linalg.eigvalsh(L))
    return eigenvalues

def compute_adjacency_spectrum(G):
    """计算图的邻接矩阵特征值谱"""
    A = nx.adjacency_matrix(G).toarray()
    eigenvalues = linalg.eigvalsh(A)
    return eigenvalues

def compute_graph_energy(G):
    """
    计算图能量：邻接矩阵特征值绝对值之和
    E(G) = Σ|μ_i|
    """
    eigenvalues = compute_adjacency_spectrum(G)
    return np.sum(np.abs(eigenvalues))

def compute_fiedler_value(G):
    """计算Fiedler值（拉普拉斯第二小特征值）"""
    if not nx.is_connected(G):
        return 0.0
    eigenvalues = compute_laplacian_spectrum(G)
    return eigenvalues[1] if len(eigenvalues) > 1 else 0.0

def compute_spectral_change(G_before, G_after):
    """
    计算谱变化指标
    S_spec = sqrt(Σ(λ'_i - λ_i)²)
    """
    spec_before = compute_laplacian_spectrum(G_before)
    spec_after = compute_laplacian_spectrum(G_after)
    
    # 对齐长度（如果节点数不同）
    min_len = min(len(spec_before), len(spec_after))
    spec_before = spec_before[:min_len]
    spec_after = spec_after[:min_len]
    
    diff = spec_after - spec_before
    S_spec = np.sqrt(np.sum(diff ** 2))
    
    return S_spec, spec_before, spec_after

def compute_energy_change(G_before, G_after):
    """计算图能量变化 ΔE"""
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
            # 计算各连通分量的加权平均
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

def compute_modularity_change(G_before, G_after):
    """计算模块度变化（使用Louvain社区检测）"""
    try:
        from networkx.algorithms.community import louvain_communities
        
        def get_modularity(G):
            if G.number_of_edges() == 0:
                return 0
            communities = louvain_communities(G, seed=42)
            return nx.community.modularity(G, communities)
        
        Q_before = get_modularity(G_before)
        Q_after = get_modularity(G_after)
        return Q_after - Q_before, Q_before, Q_after
    except:
        return 0, 0, 0

def compute_common_neighbors(G, node_x, node_y):
    """计算共同邻居数"""
    neighbors_x = set(G.neighbors(node_x))
    neighbors_y = set(G.neighbors(node_y))
    return len(neighbors_x & neighbors_y)

def compute_social_distance(G, node_x, node_y):
    """计算社交距离（最短路径长度）"""
    try:
        return nx.shortest_path_length(G, node_x, node_y)
    except nx.NetworkXNoPath:
        return float('inf')

# ============================================================
# 第四部分：综合应力指标与相容性评分
# ============================================================

def compute_structural_stress(G, node_x, node_y, weights=None):
    """
    计算综合结构应力指标 I_stress
    
    参数:
        G: 原始网络
        node_x, node_y: 待匹配的两个节点
        weights: 各指标权重 {'alpha': 谱变化, 'beta': 能量变化, 'gamma': 路径变化}
    """
    if weights is None:
        weights = {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.2, 'delta': 0.1}
    
    # 合并Ego网络
    G_before, G_after = merge_ego_networks(G, node_x, node_y)
    
    # 计算各项指标
    S_spec, _, _ = compute_spectral_change(G_before, G_after)
    delta_E, E_before, _ = compute_energy_change(G_before, G_after)
    delta_L, L_before, _ = compute_path_length_change(G_before, G_after)
    delta_C, _, _ = compute_clustering_change(G_before, G_after)
    
    # 归一化
    S_spec_norm = S_spec / (G_before.number_of_nodes() + 1)  # 归一化谱变化
    delta_E_norm = abs(delta_E) / (E_before + 1)  # 相对能量变化
    delta_L_norm = abs(delta_L) / (L_before + 0.1) if L_before != float('inf') else 1.0
    
    # 综合应力指标
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
    """
    计算相容性评分
    
    方法:
        'inverse': Score = 1 / (1 + I_stress)
        'exponential': Score = exp(-I_stress) * (1 + delta * ln(1 + N))
    """
    if method == 'inverse':
        score = 1.0 / (1.0 + I_stress)
    else:  # exponential
        base_score = np.exp(-I_stress)
        neighbor_bonus = 1 + delta * np.log(1 + common_neighbors)
        score = base_score * neighbor_bonus
    
    # 限制在0-1范围
    return min(max(score, 0), 1)

# ============================================================
# 第五部分：实验流程
# ============================================================

def sample_candidate_pairs(G, n_pairs=100, ensure_diversity=True):
    """
    从网络中采样候选用户对
    
    参数:
        G: 网络
        n_pairs: 采样对数
        ensure_diversity: 是否确保不同社交距离的多样性
    """
    nodes = list(G.nodes())
    pairs = []
    
    if ensure_diversity:
        # 按社交距离分层采样
        distance_buckets = defaultdict(list)
        
        # 随机采样一些节点对并计算距离
        np.random.seed(42)
        sampled = 0
        max_attempts = n_pairs * 50
        attempts = 0
        
        while sampled < n_pairs * 3 and attempts < max_attempts:
            attempts += 1
            x, y = np.random.choice(nodes, 2, replace=False)
            if G.has_edge(x, y):  # 跳过已连接的
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
        
        # 从各桶中均匀采样
        per_bucket = n_pairs // 4
        for bucket in ['close', 'medium', 'far', 'inf']:
            bucket_pairs = distance_buckets[bucket]
            if len(bucket_pairs) > 0:
                selected = min(per_bucket, len(bucket_pairs))
                pairs.extend(bucket_pairs[:selected])
    else:
        # 纯随机采样
        np.random.seed(42)
        while len(pairs) < n_pairs:
            x, y = np.random.choice(nodes, 2, replace=False)
            if not G.has_edge(x, y) and (x, y) not in pairs and (y, x) not in pairs:
                pairs.append((x, y))
    
    return pairs[:n_pairs]

def run_matching_experiment(G, pairs, network_type='unknown'):
    """
    运行匹配扰动实验
    
    返回实验结果列表
    """
    results = []
    
    for i, (node_x, node_y) in enumerate(pairs):
        if (i + 1) % 20 == 0:
            print(f"  处理第 {i+1}/{len(pairs)} 对...")
        
        # 计算基本信息
        common_neighbors = compute_common_neighbors(G, node_x, node_y)
        social_distance = compute_social_distance(G, node_x, node_y)
        
        # 计算结构应力
        stress_result = compute_structural_stress(G, node_x, node_y)
        
        # 计算相容性评分
        score = compute_compatibility_score(
            stress_result['I_stress'], 
            common_neighbors,
            method='exponential'
        )
        
        # Fiedler值变化
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
            'network_type': network_type
        })
    
    return results

# ============================================================
# 第六部分：结果分析与可视化
# ============================================================

def analyze_results(results):
    """分析实验结果"""
    import pandas as pd
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("实验结果统计分析")
    print("="*60)
    
    # 按网络类型分组统计
    for net_type in df['network_type'].unique():
        subset = df[df['network_type'] == net_type]
        print(f"\n【{net_type}网络】")
        print(f"  样本数: {len(subset)}")
        print(f"  平均相容性评分: {subset['score'].mean():.4f} ± {subset['score'].std():.4f}")
        print(f"  平均结构应力: {subset['I_stress'].mean():.4f} ± {subset['I_stress'].std():.4f}")
        print(f"  平均共同邻居数: {subset['common_neighbors'].mean():.2f}")
    
    # 按社交距离分组
    print("\n【按社交距离分组】")
    df['distance_group'] = df['social_distance'].apply(
        lambda x: '2跳(有共同好友)' if x <= 2 else ('3-4跳' if x <= 4 else ('5+跳' if x < float('inf') else '不连通'))
    )
    
    for group in ['2跳(有共同好友)', '3-4跳', '5+跳', '不连通']:
        subset = df[df['distance_group'] == group]
        if len(subset) > 0:
            print(f"  {group}: 平均评分={subset['score'].mean():.4f}, 平均应力={subset['I_stress'].mean():.4f}, 样本数={len(subset)}")
    
    # 相关性分析
    print("\n【相关性分析】")
    if len(df) > 5:
        # 共同邻居与评分的相关性
        valid_df = df[df['social_distance'] < float('inf')]
        if len(valid_df) > 5:
            corr_cn_score = valid_df['common_neighbors'].corr(valid_df['score'])
            corr_dist_score = valid_df['social_distance'].corr(valid_df['score'])
            print(f"  共同邻居数 vs 评分 相关系数: {corr_cn_score:.4f}")
            print(f"  社交距离 vs 评分 相关系数: {corr_dist_score:.4f}")
    
    return df

def visualize_results(df, save_path='results'):
    """可视化实验结果"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 共同邻居数 vs 相容性评分
    ax1 = axes[0, 0]
    for net_type in df['network_type'].unique():
        subset = df[df['network_type'] == net_type]
        ax1.scatter(subset['common_neighbors'], subset['score'], alpha=0.6, label=net_type)
    ax1.set_xlabel('共同邻居数')
    ax1.set_ylabel('相容性评分')
    ax1.set_title('共同邻居数 vs 相容性评分')
    ax1.legend()
    
    # 2. 社交距离 vs 相容性评分
    ax2 = axes[0, 1]
    valid_df = df[df['social_distance'] < float('inf')]
    for net_type in valid_df['network_type'].unique():
        subset = valid_df[valid_df['network_type'] == net_type]
        ax2.scatter(subset['social_distance'], subset['score'], alpha=0.6, label=net_type)
    ax2.set_xlabel('社交距离')
    ax2.set_ylabel('相容性评分')
    ax2.set_title('社交距离 vs 相容性评分')
    ax2.legend()
    
    # 3. 结构应力 vs 相容性评分
    ax3 = axes[0, 2]
    for net_type in df['network_type'].unique():
        subset = df[df['network_type'] == net_type]
        ax3.scatter(subset['I_stress'], subset['score'], alpha=0.6, label=net_type)
    ax3.set_xlabel('结构应力 I_stress')
    ax3.set_ylabel('相容性评分')
    ax3.set_title('结构应力 vs 相容性评分')
    ax3.legend()
    
    # 4. 不同网络类型的评分分布
    ax4 = axes[1, 0]
    network_types = df['network_type'].unique()
    scores_by_type = [df[df['network_type'] == t]['score'].values for t in network_types]
    ax4.boxplot(scores_by_type, labels=network_types)
    ax4.set_ylabel('相容性评分')
    ax4.set_title('不同网络类型的评分分布')
    
    # 5. Fiedler值变化分布
    ax5 = axes[1, 1]
    for net_type in df['network_type'].unique():
        subset = df[df['network_type'] == net_type]
        ax5.hist(subset['fiedler_change'], alpha=0.5, label=net_type, bins=20)
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
    
    bars = ax6.bar(range(len(distance_groups)), avg_scores, color=['green', 'yellow', 'orange', 'red'])
    ax6.set_xticks(range(len(distance_groups)))
    ax6.set_xticklabels(distance_groups, rotation=15)
    ax6.set_ylabel('平均相容性评分')
    ax6.set_title('不同社交距离的平均评分')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/experiment_results.png', dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图形，不阻塞
    print(f"\n图表已保存至 {save_path}/experiment_results.png")

def visualize_case_study(G, node_x, node_y, save_path='results'):
    """可视化案例分析：匹配前后的网络结构变化"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    G_before, G_after = merge_ego_networks(G, node_x, node_y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 节点颜色：区分X的朋友圈和Y的朋友圈
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
                colors.append('purple')  # 共同好友
            elif node in ego_x.nodes():
                colors.append('lightblue')
            else:
                colors.append('lightsalmon')
        return colors
    
    # 匹配前
    ax1 = axes[0]
    pos_before = nx.spring_layout(G_before, seed=42)
    colors_before = get_node_colors(G_before)
    nx.draw(G_before, pos_before, ax=ax1, node_color=colors_before, 
            node_size=300, with_labels=False, edge_color='gray', alpha=0.8)
    ax1.set_title(f'匹配前\n节点数: {G_before.number_of_nodes()}, 边数: {G_before.number_of_edges()}')
    
    # 匹配后
    ax2 = axes[1]
    pos_after = nx.spring_layout(G_after, seed=42)
    colors_after = get_node_colors(G_after)
    
    # 绘制普通边
    edges_before = set(G_before.edges())
    edges_after = set(G_after.edges())
    new_edge = edges_after - edges_before
    
    nx.draw_networkx_nodes(G_after, pos_after, ax=ax2, node_color=colors_after, node_size=300)
    nx.draw_networkx_edges(G_after, pos_after, ax=ax2, edgelist=list(edges_before), 
                           edge_color='gray', alpha=0.8)
    nx.draw_networkx_edges(G_after, pos_after, ax=ax2, edgelist=list(new_edge), 
                           edge_color='red', width=3, alpha=1.0)
    ax2.set_title(f'匹配后（红色为新增边）\n节点数: {G_after.number_of_nodes()}, 边数: {G_after.number_of_edges()}')
    
    # 添加图例
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
    plt.savefig(f'{save_path}/case_study_{node_x}_{node_y}.png', dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图形，不阻塞
    
    # 打印详细指标
    stress_result = compute_structural_stress(G, node_x, node_y)
    common_neighbors = compute_common_neighbors(G, node_x, node_y)
    social_distance = compute_social_distance(G, node_x, node_y)
    score = compute_compatibility_score(stress_result['I_stress'], common_neighbors)
    
    print(f"\n案例分析: 节点 {node_x} 与 节点 {node_y}")
    print(f"  共同邻居数: {common_neighbors}")
    print(f"  社交距离: {social_distance}")
    print(f"  结构应力 I_stress: {stress_result['I_stress']:.4f}")
    print(f"  谱变化 S_spec: {stress_result['S_spec']:.4f}")
    print(f"  能量变化 ΔE: {stress_result['delta_E']:.4f}")
    print(f"  路径变化 ΔL: {stress_result['delta_L']:.4f}")
    print(f"  聚类变化 ΔC: {stress_result['delta_C']:.4f}")
    print(f"  相容性评分: {score:.4f} (满分1.0)")

# ============================================================
# 主程序
# ============================================================

def main():
    print("="*60)
    print("基于结构摄动理论与反事实推演的社会生态相容性匹配实验")
    print("="*60)
    
    # 实验参数
    N_NODES = 500  # 节点数（可调整）
    N_PAIRS = 50   # 每种网络采样的配对数
    
    all_results = []
    
    # 1. 生成三种网络拓扑
    print("\n[步骤1] 生成网络拓扑...")
    
    print("\n  生成层级型网络...")
    G_hierarchical = generate_hierarchical_network(N_NODES, n_communities=10, p_in=0.2, p_out=0.005)
    stats_h = get_network_stats(G_hierarchical)
    print(f"    节点: {stats_h['nodes']}, 边: {stats_h['edges']}, 平均度: {stats_h['avg_degree']:.2f}")
    print(f"    聚类系数: {stats_h['avg_clustering']:.4f}, 平均路径: {stats_h['avg_path_length']:.2f}")
    
    print("\n  生成团簇型网络 (Watts-Strogatz)...")
    G_clustered = generate_clustered_network(N_NODES, k=8, p=0.1)
    stats_c = get_network_stats(G_clustered)
    print(f"    节点: {stats_c['nodes']}, 边: {stats_c['edges']}, 平均度: {stats_c['avg_degree']:.2f}")
    print(f"    聚类系数: {stats_c['avg_clustering']:.4f}, 平均路径: {stats_c['avg_path_length']:.2f}")
    
    print("\n  生成稀疏型网络 (Barabási-Albert)...")
    G_sparse = generate_sparse_network(N_NODES, m=2)
    stats_s = get_network_stats(G_sparse)
    print(f"    节点: {stats_s['nodes']}, 边: {stats_s['edges']}, 平均度: {stats_s['avg_degree']:.2f}")
    print(f"    聚类系数: {stats_s['avg_clustering']:.4f}, 平均路径: {stats_s['avg_path_length']:.2f}")
    
    # 2. 运行匹配实验
    print("\n[步骤2] 运行匹配扰动实验...")
    
    networks = [
        (G_hierarchical, '层级型'),
        (G_clustered, '团簇型'),
        (G_sparse, '稀疏型')
    ]
    
    for G, net_type in networks:
        print(f"\n  在{net_type}网络上实验...")
        pairs = sample_candidate_pairs(G, N_PAIRS, ensure_diversity=True)
        results = run_matching_experiment(G, pairs, net_type)
        all_results.extend(results)
    
    # 3. 分析结果
    print("\n[步骤3] 分析实验结果...")
    df = analyze_results(all_results)
    
    # 4. 可视化
    print("\n[步骤4] 生成可视化图表...")
    visualize_results(df, save_path='C:/Users/timesouthli/CodeBuddy/网络科学/results')
    
    # 5. 案例分析
    print("\n[步骤5] 案例分析...")
    
    # 选择一个有共同好友的案例
    close_pairs = df[df['common_neighbors'] > 0]
    if len(close_pairs) > 0:
        case = close_pairs.iloc[0]
        net_type = case['network_type']
        G_case = {'层级型': G_hierarchical, '团簇型': G_clustered, '稀疏型': G_sparse}[net_type]
        print(f"\n案例1: 有共同好友的匹配 ({net_type}网络)")
        visualize_case_study(G_case, case['node_x'], case['node_y'], 
                           save_path='C:/Users/timesouthli/CodeBuddy/网络科学/results')
    
    # 选择一个无共同好友的案例
    far_pairs = df[(df['common_neighbors'] == 0) & (df['social_distance'] < float('inf'))]
    if len(far_pairs) > 0:
        case = far_pairs.iloc[0]
        net_type = case['network_type']
        G_case = {'层级型': G_hierarchical, '团簇型': G_clustered, '稀疏型': G_sparse}[net_type]
        print(f"\n案例2: 无共同好友的匹配 ({net_type}网络)")
        visualize_case_study(G_case, case['node_x'], case['node_y'],
                           save_path='C:/Users/timesouthli/CodeBuddy/网络科学/results')
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)
    
    return df

if __name__ == "__main__":
    df = main()
