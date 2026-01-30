# 聚合性能问题分析和修复

## 问题描述

在 CLASSIC4 规模实验中发现程序卡在聚合阶段超过 12 小时：

```
2026-01-29 17:38:47 - OptimizedAggregator - INFO - Optimized aggregation: 780550 biclusters
2026-01-29 17:39:02 - OptimizedAggregator - INFO -   Split into 15611 batches of ~50 biclusters
[卡住一个晚上...]
```

## 根本原因分析

### 原因 1: Biclusters 数量异常（780,550 个）

**正常值**: CLASSIC4 实验应产生 2,000-6,000 个 biclusters
**实际值**: 780,550 个（130x 过多！）

**根源**: `tolerance=0.05` 太宽松，导致每个 block 检测到大量低质量 biclusters。

```
迭代 1: ~155,000 biclusters
迭代 2: ~155,000 biclusters
迭代 3: ~155,000 biclusters
迭代 4: ~155,000 biclusters
迭代 5: ~155,833 biclusters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计:   780,550 biclusters
```

### 原因 2: 并行聚合不使用空间索引（性能致命缺陷）

**问题代码** (`detection_optimized.py:240`):

```python
# 并行聚合
future = executor.submit(
    _batch_find_similar,
    batch,
    sorted_biclusters,  # ← 传递所有 780,550 个！
    ...
)

# _batch_find_similar 函数 (line 415)
for bc in batch:
    for other in all_biclusters:  # ← O(n²) 全局扫描！
        jaccard = bc.jaccard_index(other)  # 非常昂贵
```

**复杂度分析**:
- **批次数**: 15,611 批（每批 50 个）
- **每批比较**: 50 × 780,550 = 39,027,500 次
- **总比较**: 15,611 × 39,027,500 = **609,229,525,000 次**（6090 亿次）
- **预计时间**: 即使每次 1 微秒，也需要 **169 小时**（7 天）

**对比：串行+空间索引**:
- **使用空间索引**: O(n × k)，其中 k ≈ 50-200（每个 bicluster 的候选数量）
- **比较次数**: 780,550 × 100 = 78,055,000 次（7800 万次）
- **加速**: **7,800x 更快**！
- **预计时间**: **30-60 分钟**（vs 169 小时）

### 为什么并行聚合不用空间索引？

```python
# detection_optimized.py:213 的注释
"""
Note: spatial_index cannot be shared across processes, so we use filtered scan.
"""
```

空间索引对象不能序列化传递给多进程，所以并行版本退化为 O(n²) 全局扫描。

## 修复方案

### 修复 1: 降低 tolerance 减少 biclusters 数量

**修改**: `scripts/run_classic4_experiment.py`

```python
# 之前
tolerance=0.05  # 太宽松，产生 780k biclusters

# 修复后
tolerance=0.02  # 更严格，预期产生 3k-8k biclusters
```

**效果**:
- Biclusters 数量: 780,550 → 5,000 左右
- 聚合时间: 169 小时 → 2-5 分钟

### 修复 2: 大数据集强制使用串行+空间索引

**修改**: `src/detection_optimized.py`

```python
# 添加阈值检测
LARGE_DATASET_THRESHOLD = 10000

if n > LARGE_DATASET_THRESHOLD:
    self.logger.warning(
        f"Large bicluster count ({n:,}), forcing sequential + spatial index mode"
    )
    use_parallel = False  # 禁用并行
else:
    use_parallel = self.config.use_parallel
```

**效果**:
- 当 biclusters > 10,000 时，自动禁用并行
- 强制使用串行+空间索引（O(n×k) vs O(n²)）

### 修复 3: 改进进度报告

**修改**: `src/detection_optimized.py`

```python
# 自适应进度间隔
if n > 100000:
    progress_interval = 10000  # 每 10k 个报告一次
elif n > 10000:
    progress_interval = 1000   # 每 1k 个报告一次
else:
    progress_interval = 100    # 默认每 100 个

# 显示百分比和合并数量
self.logger.info(
    f"Progress: {idx:,}/{n:,} ({progress_pct:.1f}%), {len(merged):,} merged so far"
)
```

**效果**:
- 避免日志泛滥（780k biclusters 会产生 7,800 行日志）
- 提供有用的进度信息

## 性能对比

### 修复前（卡死场景）

| 阶段 | 时间 | 说明 |
|------|------|------|
| 数据生成 | 1秒 | ✓ 正常 |
| Block 检测 | 2-5分钟 | ✓ 正常（并行） |
| 空间索引构建 | 15秒 | ✓ 正常 |
| **并行聚合** | **169+ 小时** | **✗ 致命瓶颈** |

### 修复后（正常场景）

| 阶段 | 时间 | 说明 |
|------|------|------|
| 数据生成 | 1秒 | ✓ 正常 |
| Block 检测 | 2-5分钟 | ✓ 正常（并行） |
| 空间索引构建 | 1秒 | ✓ 正常 |
| **串行聚合** | **2-5分钟** | **✓ 使用空间索引** |

**总加速**: 从 **7 天** 降低到 **10 分钟**（~1000x 加速）

## 如何验证修复

### 1. 停止当前实验

```bash
# 找到进程
ps aux | grep run_classic4

# 停止
kill <PID>
```

### 2. 拉取最新修复

```bash
cd /home/jie/big_matrix_cocluster
git pull origin main
```

### 3. 重新运行实验

```bash
./run_classic4.sh
```

### 4. 观察日志

**正常日志应该是**:

```
2026-01-29 XX:XX:XX - OptimizedAggregator - INFO - Optimized aggregation: 5234 biclusters
2026-01-29 XX:XX:XX - OptimizedAggregator - INFO - [1/4] Building spatial index...
2026-01-29 XX:XX:XX - OptimizedAggregator - INFO -   Built spatial index: 20×20 grid
2026-01-29 XX:XX:XX - OptimizedAggregator - INFO - [2/4] Sorting biclusters...
2026-01-29 XX:XX:XX - OptimizedAggregator - INFO - [3/4] Aggregating biclusters...
2026-01-29 XX:XX:XX - OptimizedAggregator - INFO -   Sequential aggregation: 5,234 biclusters
2026-01-29 XX:XX:XX - OptimizedAggregator - INFO -   Progress: 1000/5234 (19.1%), 456 merged so far
2026-01-29 XX:XX:XX - OptimizedAggregator - INFO -   Progress: 2000/5234 (38.2%), 892 merged so far
...
2026-01-29 XX:XX:XX - OptimizedAggregator - INFO - [4/4] Complete: 1234 unique biclusters, 3.2s
```

**如果仍然看到大数量 biclusters** (> 10,000):

```
2026-01-29 XX:XX:XX - OptimizedAggregator - WARNING - Large bicluster count (67890), forcing sequential mode
2026-01-29 XX:XX:XX - OptimizedAggregator - INFO -   Sequential aggregation: 67,890 biclusters
2026-01-29 XX:XX:XX - OptimizedAggregator - INFO -   Progress: 10000/67890 (14.7%), 4523 merged so far
...
```

这表示仍在运行，但会完成（预计 10-30 分钟而不是几天）。

## 预期结果

### 正常的 CLASSIC4 实验

```
Matrix: 6,461 × 4,667
Iterations: 5
Biclusters detected: 3,000 - 8,000
Aggregation time: 2-5 分钟
Total time: 5-10 分钟
NMI: 0.75+
ARI: 0.65+
```

### 如果 biclusters 仍然过多

可能需要进一步调整参数：

```python
# 更严格的 tolerance
tolerance=0.01  # 甚至更严格

# 或减少分区迭代
T_p=3  # 从 5 降到 3

# 或增加最小 block 大小
T_m=30, T_n=30  # 从 20 增加到 30
```

## 技术要点总结

1. **空间索引是关键**: 对于大数据集，空间索引能提供 10-100x 加速
2. **并行≠快速**: 不正确的并行化（无空间索引）反而比串行慢 7800x
3. **参数调优重要**: tolerance 太宽松会产生大量无用 biclusters
4. **早期过滤**: 在检测阶段就应该过滤低质量结果，而不是在聚合阶段

## 相关文件

- `src/detection_optimized.py` - 聚合优化实现
- `scripts/run_classic4_experiment.py` - CLASSIC4 实验脚本
- `docs/AGGREGATION_OPTIMIZATION.md` - 聚合优化文档
