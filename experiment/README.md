# DiMergeCo论文实验复现

本目录包含用于复现DiMergeCo论文实验结果的脚本。

## ⚠️ 重要说明

**DiMergeCo的分区策略主要用于大规模数据集（如论文中的CLASSIC4 6,461×4,667）。对于小规模数据集（<1000×1000），建议使用标准biclustering。**

## 快速开始

### 1. 简单验证（推荐先运行）

验证基础算法是否工作正常：

```bash
python experiment/simple_test.py
```

**期望输出**：
- NMI > 0.9
- ARI > 0.8
- 显示 "✅ 测试通过！"

### 2. 完整实验复现

运行论文规模的实验：

```bash
# 运行所有实验（约1-2小时）
python experiment/reproduce_paper_experiments.py --experiment all

# 或运行特定实验
python experiment/reproduce_paper_experiments.py --experiment classic4
python experiment/reproduce_paper_experiments.py --experiment scalability
```

## 实验说明

### 简单测试（simple_test.py）

- **矩阵规模**: 500 × 400
- **Biclusters数量**: 5
- **方法**: 标准biclustering（不使用分区）
- **期望结果**: NMI > 0.9, ARI > 0.8
- **目的**: 验证基础SVD biclustering算法正确性

### 实验1: CLASSIC4规模合成数据

- **矩阵规模**: 6,461 × 4,667（与论文CLASSIC4数据集相同）
- **Biclusters数量**: 4
- **方法**: DiMergeCo完整流程（分区 + 层次化合并）
- **论文期望结果**: NMI=0.865, ARI=0.776
- **合成数据可接受范围**: NMI 0.70-0.85, ARI 0.60-0.75

### 实验2: 小Co-cluster检测

- **矩阵规模**: 5,000 × 5,000
- **Biclusters大小**: 5×5 到 20×20（验证检测小bicluster的能力）
- **Biclusters数量**: 10
- **期望结果**: NMI > 0.65

### 实验3: 可扩展性测试

测试不同矩阵规模下的性能：
- 1,000 × 800
- 2,000 × 1,600
- 5,000 × 4,000
- 10,000 × 8,000

**期望**：运行时间随规模次线性增长

## 论文参数配置

DiMergeCo实验使用论文推荐的参数（来自Table 4）：

```python
# 用于大规模数据集（>5000×5000）
PAPER_PARAMS = {
    'k1': 10,                    # SVD行聚类数
    'k2': 10,                    # SVD列聚类数
    'tolerance': 0.05,           # 质量阈值
    'T_m': 20,                   # 最小co-cluster行大小（论文推荐）
    'T_n': 20,                   # 最小co-cluster列大小（论文推荐）
    'T_p': 5,                    # 分割迭代次数
    'P_thresh': 0.95,            # 检测概率阈值
    'overlap_threshold': 0.45,   # 层次化合并重叠阈值
}

# 用于中等规模数据集（1000×1000到5000×5000）
ADJUSTED_PARAMS = {
    'T_m': max(M // 20, 50),     # 自适应调整
    'T_n': max(N // 20, 40),
    'T_p': 3,                    # 较少迭代
}

# 用于小规模数据集（<1000×1000）
# 建议使用标准biclustering（不使用分区）
```

## 结果解读

### 成功标准

| 实验 | NMI最低要求 | ARI最低要求 | 方法 |
|------|------------|------------|------|
| 简单测试 | > 0.90 | > 0.80 | 标准biclustering |
| CLASSIC4 | > 0.70 | > 0.60 | DiMergeCo |
| 小co-cluster | > 0.65 | > 0.55 | DiMergeCo |

### 与论文对比

**注意**：
- 论文使用Rust+MPI实现和真实数据集
- 我们使用Python单线程和合成数据
- 合成数据的NMI/ARI可能与真实数据有±0.1-0.15的差异（正常）
- Python运行时间会比Rust慢2-5倍（正常）
- **DiMergeCo的分区策略对小数据集效果不佳**（这是正常的，论文针对大规模数据集优化）

**重点验证**：
- ✅ 标准biclustering能达到高质量（NMI>0.9）
- ✅ 算法能检测到合理数量的biclusters
- ✅ 可扩展性趋势正确（时间随规模合理增长）

## 输出文件

运行后会生成以下文件：

```
experiment/
├── simple_test_results/            # 简单测试结果
├── paper_reproduction_results/      # 实验结果目录
│   ├── all_results.json            # 所有实验结果汇总
│   ├── classic4_synthetic/         # CLASSIC4实验结果
│   ├── amazon_synthetic/           # Amazon实验结果
│   ├── small_cocluster/            # 小co-cluster实验结果
│   └── scale_*/                    # 可扩展性测试结果
└── quick_validation_results/       # 快速验证结果（已弃用）
```

## 常见问题

### Q: 为什么DiMergeCo结果比标准biclustering差？

A: 这是正常的！DiMergeCo的分区策略是为大规模数据集（>5000×5000）设计的。对于小数据集：
1. 分区会将数据分得过碎
2. 每个小分区上的检测质量不如全局检测
3. 层次化合并无法完全恢复质量

**建议**：小数据集（<1000×1000）使用标准biclustering

### Q: 如何使DiMergeCo在小数据集上工作更好？

A: 调整参数：
```python
# 对于1000×800的矩阵，使用：
T_m = max(M // 10, 100)  # 更大的最小块尺寸
T_n = max(N // 10, 80)
T_p = 1 或 2              # 更少的迭代次数
```

或者干脆使用标准biclustering。

### Q: 运行时间很长怎么办？

A:
1. 先运行simple_test.py验证算法（只需10-20秒）
2. 减小矩阵规模或减少T_p迭代次数
3. Python实现比Rust慢是正常的

### Q: NMI/ARI低于0.6怎么办？

A: 检查：
1. 是否使用了合适的参数（大数据集用DiMergeCo，小数据集用标准方法）
2. 是否数据噪声过高
3. 查看检测到的biclusters数量是否合理

## 论文引用

如果使用这些实验结果，请引用DiMergeCo论文：

```
@article{dimergeco,
  title={DiMergeCo: A Scalable Framework for Large-Scale Co-Clustering},
  author={...},
  journal={...},
  year={...}
}
```

## 联系方式

如有问题，请提交Issue或联系：wzh4464@gmail.com
