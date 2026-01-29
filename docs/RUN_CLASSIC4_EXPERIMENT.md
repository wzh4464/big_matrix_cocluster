# 在服务器上运行 CLASSIC4 实验

## 快速开始

### 1. 直接运行（前台）

```bash
cd /path/to/big_matrix_cocluster
python scripts/run_classic4_experiment.py
```

### 2. 后台运行（推荐用于服务器）

```bash
# 使用 nohup 在后台运行，输出重定向到日志文件
nohup python scripts/run_classic4_experiment.py > classic4.log 2>&1 &

# 查看进程
ps aux | grep run_classic4

# 实时查看日志
tail -f classic4.log

# 或者查看详细日志
tail -f classic4_experiment.log
```

### 3. 使用 screen（长时间运行推荐）

```bash
# 创建新 screen 会话
screen -S classic4

# 在 screen 中运行
python scripts/run_classic4_experiment.py

# 断开 screen（Ctrl+A, 然后按 D）
# 重新连接
screen -r classic4

# 查看所有 screen 会话
screen -ls
```

---

## 实验配置

### CLASSIC4 规模参数

```python
# 数据集
matrix_shape = (6461, 4667)      # 6,461 文档 × 4,667 特征
n_biclusters = 4                  # 4 个类别
bicluster_size = (800, 1600)     # 平均 ~1200，占比 18-25%
noise_level = 0.15                # 中等噪声

# DiMergeCo 参数（来自论文）
k1 = 10, k2 = 10                  # SVD 聚类数
tolerance = 0.05                  # 质量阈值
T_m = 20, T_n = 20                # 最小 Co-cluster 大小
T_p = 5                           # 分割迭代次数
P_thresh = 0.95                   # 检测概率阈值
overlap_threshold = 0.45          # 合并阈值（0.4-0.5）
```

### 优化配置（自动启用）

```python
# 聚合优化（默认启用）
use_optimized_aggregation = True  # O(n²) → O(n log n)
spatial_index = True              # 20×20 空间网格
parallel_workers = 4              # 4 个并行 worker
cache_enabled = True              # Jaccard 缓存
```

---

## 预期结果

### 运行时间

| 阶段 | 时间 | 说明 |
|------|------|------|
| 数据生成 | 30-60秒 | 生成 6461×4667 矩阵 |
| DiMergeCo 检测 | **2-5分钟** | **优化后** |
| 总时间 | 3-6分钟 | 完整实验 |

**注意**：未优化版本需要 **48+ 分钟**！

### 质量指标（合成数据）

| 指标 | 目标 | 论文值（真实数据）|
|------|------|------------------|
| NMI | > 0.75 | 0.865 |
| ARI | > 0.65 | 0.776 |

**说明**：合成数据与真实数据有差异，0.75-0.85 范围内即为优秀。

### 输出文件

```
classic4_results/
├── experiment_summary.json     # 实验汇总（JSON 格式）
├── biclusters.json            # 检测到的 biclusters
├── visualizations/            # 可视化结果
│   ├── heatmap.png
│   └── bicluster_distribution.png
└── performance_metrics.json   # 性能指标

classic4.log                   # nohup 输出日志
classic4_experiment.log        # 详细实验日志
```

---

## 监控运行状态

### 1. 查看日志

```bash
# 实时查看主日志
tail -f classic4.log

# 实时查看详细日志
tail -f classic4_experiment.log

# 搜索关键信息
grep "步骤\|完成\|NMI\|ARI" classic4_experiment.log
```

### 2. 检查进程

```bash
# 查看 Python 进程
ps aux | grep run_classic4

# 查看 CPU 和内存使用
top -p $(pgrep -f run_classic4)

# 或使用 htop（如果已安装）
htop -p $(pgrep -f run_classic4)
```

### 3. 查看中间结果

```bash
# 查看实验汇总（实验完成后）
cat classic4_results/experiment_summary.json

# 格式化查看 JSON
python -m json.tool classic4_results/experiment_summary.json
```

---

## 常见问题

### Q1: 内存不足

**症状**: `MemoryError` 或进程被 kill

**解决方案**:
```bash
# 检查可用内存
free -h

# 减少并行 worker（修改代码）
# 在 pipeline 配置中添加：
max_workers = 2  # 默认是 4
```

或使用更小的测试规模：
```python
# 修改 scripts/run_classic4_experiment.py 中的配置
matrix_shape = (3000, 2000)  # 减小矩阵大小
```

### Q2: 运行时间过长（>10分钟）

**可能原因**:
1. 服务器负载高（其他进程占用 CPU）
2. 优化未启用（检查代码）
3. 数据规模实际更大

**检查**:
```bash
# 查看 CPU 使用
top

# 确认优化启用（日志中应有）
grep "优化聚合：ENABLED" classic4_experiment.log
```

### Q3: NMI/ARI 指标过低

**可能原因**:
1. 使用合成数据（与真实数据有差异）- 正常
2. 参数配置不当
3. 随机种子影响

**建议**:
- 合成数据：NMI > 0.70, ARI > 0.60 即可接受
- 真实数据：应接近论文值（NMI 0.865, ARI 0.776）

### Q4: 如何停止运行

```bash
# 找到进程 ID
ps aux | grep run_classic4

# 优雅停止（发送 SIGTERM）
kill <PID>

# 强制停止（如果上面不行）
kill -9 <PID>

# 如果使用 screen
screen -r classic4
# 按 Ctrl+C 停止
```

---

## 高级用法

### 1. 修改参数

编辑 `scripts/run_classic4_experiment.py`，修改以下部分：

```python
# 第 60 行左右：数据生成配置
matrix, _, ground_truth, _ = create_synthetic_data_with_generator(
    n_biclusters=4,           # 修改 bicluster 数量
    matrix_shape=(6461, 4667),  # 修改矩阵大小
    bicluster_size_range=(800, 1600),  # 修改大小范围
    noise_level_spec=0.15,    # 修改噪声水平
    random_state=42           # 修改随机种子
)

# 第 90 行左右：DiMergeCo 参数
pipeline = create_dimergeco_pipeline(
    k1=10,                    # 修改聚类数
    k2=10,
    tolerance=0.05,           # 修改质量阈值
    T_m=20,                   # 修改分割参数
    T_n=20,
    T_p=5,
    P_thresh=0.95,
    overlap_threshold=0.45,   # 修改合并阈值
    random_state=42
)
```

### 2. 禁用优化（仅用于对比）

在 `create_dimergeco_pipeline` 后添加：

```python
# 禁用优化（不推荐，会慢 10-24 倍）
from detection import PartitionedBiclusterDetector

# 获取内部 detector 并禁用优化
detector = pipeline.analyzer.detector
if isinstance(detector, PartitionedBiclusterDetector):
    detector.use_optimized_aggregation = False
    detector.max_workers = 1
```

### 3. 调整 worker 数量

根据服务器 CPU 核心数调整：

```bash
# 查看 CPU 核心数
nproc

# 或
lscpu | grep "^CPU(s):"
```

然后修改代码（在 detector 配置部分）：

```python
import os

max_workers = min(os.cpu_count(), 8)  # 最多使用 8 个核心
```

---

## 检查点和恢复

当前脚本不支持自动检查点。如果需要长时间运行，建议：

1. **使用 screen 或 tmux**（推荐）
2. **监控日志输出**确保运行正常
3. **实验完成后保存结果**到其他位置

```bash
# 备份结果
cp -r classic4_results classic4_results_backup_$(date +%Y%m%d_%H%M%S)
```

---

## 性能基准

### 测试环境参考

| 配置 | 预期时间 |
|------|---------|
| 4 核 CPU, 16GB RAM | 3-5 分钟 |
| 8 核 CPU, 32GB RAM | 2-3 分钟 |
| 16 核 CPU, 64GB RAM | 1.5-2.5 分钟 |

**注意**: 实际时间受 CPU 型号、负载、I/O 等因素影响。

---

## 联系和支持

如果遇到问题：

1. 检查日志文件中的错误信息
2. 确认依赖包已正确安装（`pip install -r requirements.txt`）
3. 查看 GitHub Issues 或提交新 Issue

---

## 示例：完整运行流程

```bash
# 1. SSH 登录服务器
ssh user@server

# 2. 进入项目目录
cd /path/to/big_matrix_cocluster

# 3. 激活虚拟环境（如果有）
source venv/bin/activate

# 4. 创建 screen 会话
screen -S classic4

# 5. 运行实验
python scripts/run_classic4_experiment.py

# 6. 等待输出显示 "步骤 1", "步骤 2" 等信息

# 7. 断开 screen（Ctrl+A, 然后按 D）

# 8. 稍后重新连接查看结果
screen -r classic4

# 9. 实验完成后，查看结果
cat classic4_results/experiment_summary.json

# 10. 下载结果到本地（在本地机器上运行）
scp -r user@server:/path/to/big_matrix_cocluster/classic4_results ./
```

完成！
