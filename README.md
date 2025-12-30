# 基于结构摄动理论与反事实推演的社会生态相容性匹配方法

## 项目简介

本项目实现了一种创新的社交/婚恋匹配方法，通过将待匹配两人的Ego网络（自我中心社交网络）进行"强制连边"合并，模拟建立关系的反事实场景，并评估融合后网络结构的应力变化（如拉普拉斯谱特征和图能量的波动），以量化匹配的**社会生态相容性**。

## 核心思想

传统匹配算法仅考虑用户兴趣或属性相似度，忽视了社会网络结构的影响。本方法回答一个关键问题：

> **如果将待匹配的两个人放在彼此的社交网络中，他们的连接会让整体网络结构发生多大变化？变化越小，意味着这段关系越容易被各自社交生态消化，从而更稳定。**

## 项目结构

```
网络科学/
├── experiment.py              # 合成网络实验（层级型、团簇型、稀疏型）
├── experiment_real_data.py    # 真实数据集实验（Facebook、WeChat）
├── data/                      # 数据集目录
│   ├── facebook_combined.txt  # SNAP Facebook数据集
│   └── wechat/                # 微信数据集
│       ├── friends.csv        # 好友关系
│       ├── users.csv          # 用户信息
│       ├── groups.csv         # 群组信息
│       └── group_members.csv  # 群组成员
├── results/                   # 合成网络实验结果
└── results_real/              # 真实数据实验结果
```

## 环境要求

- Python 3.9+
- 依赖包：
  ```bash
  pip install networkx numpy scipy matplotlib pandas
  ```

## 快速开始

### 1. 合成网络实验

使用生成的网络（层级型、团簇型、稀疏型）验证方法：

```bash
python experiment.py
```

### 2. 真实数据集实验

```bash
python experiment_real_data.py
```

运行后选择数据集：
- `1` - SNAP Facebook（推荐，自动下载）
- `2` - WeChat Data（需手动下载）
- `3` - 两者都使用
- `4` - 使用本地文件

## 数据集

### SNAP Facebook
- **来源**: https://snap.stanford.edu/data/egonets-Facebook.html
- **规模**: 4,039 节点, 88,234 边
- **特点**: 高聚类系数 (0.6055)，小世界特性
- **获取**: 程序自动下载

### WeChat Data
- **来源**: https://github.com/LiuTian821/WechatData
- **获取**: 手动下载后放入 `data/wechat/` 目录

## 核心算法

### 1. Ego网络合并与强制连边

```python
# 提取X和Y的Ego网络，合并后添加X-Y边
G_before, G_after = merge_ego_networks(G, node_x, node_y)
```

### 2. 结构应力评估指标

| 指标 | 含义 |
|------|------|
| **拉普拉斯谱变化 (S_spec)** | 网络连通性和社群结构变化 |
| **Fiedler值变化** | 代数连通度变化，跨社区检测 |
| **图能量变化 (ΔE)** | 网络整体耦合程度变化 |
| **平均路径长度变化 (ΔL)** | 网络距离结构变化 |
| **聚类系数变化 (ΔC)** | 局部紧密度变化 |

### 3. 综合应力指标

```
I_stress = α × S_spec_norm + β × |ΔE|/E + γ × ΔL_norm
```

### 4. 相容性评分

```python
Score = exp(-I_stress) × (1 + δ × ln(1 + N))
```
其中 N 为共同邻居数。

## 实验结果

### 预期发现

1. **共同邻居数与评分正相关** - 有共同好友的匹配评分更高
2. **社交距离与评分负相关** - 距离越远，评分越低
3. **Fiedler值变化** - 跨社区匹配时显著增加
4. **阈值现象** - 距离>4跳的匹配评分普遍<0.2

### 输出文件

- `experiment_results.png` - 6张分析图表
- `case_study_*.png` - 匹配前后网络对比图
- `experiment_results.csv` - 详细实验数据

## 理论基础

- **结构平衡理论** (Heider, 1946) - 社交网络中的认知一致性
- **结构摄动理论** (Lü et al., 2015) - 网络扰动与可预测性
- **Ego网络分析** - 社会嵌入性量化
- **小世界与无标度特性** - 社交网络拓扑特征

## 参考文献

1. Lü, L. et al. Toward link predictability of complex networks. PNAS, 2015.
2. Leskovec, J. & McAuley, J. Learning to Discover Social Circles in Ego Networks. NIPS, 2012.
3. Watts, D.J. & Strogatz, S.H. Collective dynamics of 'small-world' networks. Nature, 1998.
4. Barabási, A.L. & Albert, R. Emergence of scaling in random networks. Science, 1999.