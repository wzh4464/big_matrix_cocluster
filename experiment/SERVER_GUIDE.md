# 服务器运行指南

## 快速开始

### 1. 环境准备

```bash
# SSH登录到服务器
ssh user@your-server.com

# 克隆代码
git clone https://github.com/wzh4464/big_matrix_cocluster.git
cd big_matrix_cocluster

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
pip install tqdm psutil  # 额外的服务器监控工具
```

### 2. 运行大数据集实验

#### 快速测试（确认环境正常）

```bash
# 先运行简单测试确认安装正确
PYTHONPATH=. python experiment/simple_test.py
```

期望输出：NMI > 0.9, ARI > 0.8

#### 运行CLASSIC4规模实验（6,461×4,667）

```bash
# 单个实验（约30分钟-2小时，取决于服务器性能）
PYTHONPATH=. python experiment/run_large_scale_experiments.py \
    --dataset classic4 \
    --output results/classic4

# 后台运行（推荐）
nohup python -u experiment/run_large_scale_experiments.py \
    --dataset classic4 \
    --output results/classic4 \
    > classic4.log 2>&1 &

# 查看进度
tail -f classic4.log
```

#### 运行大规模实验（10,000×8,000）

```bash
PYTHONPATH=. python experiment/run_large_scale_experiments.py \
    --dataset large \
    --output results/large
```

#### 运行超大规模实验（20,000×15,000）

```bash
# 需要更多内存（建议16GB+）
PYTHONPATH=. python experiment/run_large_scale_experiments.py \
    --dataset xlarge \
    --output results/xlarge
```

#### 批量运行多个实验

```bash
# 运行多个预定义数据集
PYTHONPATH=. python experiment/run_large_scale_experiments.py \
    --batch \
    --batch-datasets classic4 large \
    --output results/batch

# 后台批量运行
nohup python -u experiment/run_large_scale_experiments.py \
    --batch \
    --batch-datasets classic4 large xlarge \
    --output results/batch_all \
    > batch.log 2>&1 &
```

#### 自定义规模实验

```bash
# 自定义矩阵大小和bicluster数量
PYTHONPATH=. python experiment/run_large_scale_experiments.py \
    --custom \
    --rows 15000 \
    --cols 12000 \
    --biclusters 8 \
    --output results/custom_15k
```

### 3. 查看可用数据集

```bash
PYTHONPATH=. python experiment/run_large_scale_experiments.py
```

输出示例：
```
可用的大规模数据集:

  classic4     - CLASSIC4规模（论文数据集）
               矩阵: 6,461 × 4,667 (约230.2 MB)
               Biclusters: 4

  large        - 大规模（10K×8K）
               矩阵: 10,000 × 8,000 (约610.4 MB)
               Biclusters: 6

  xlarge       - 超大规模（20K×15K）
               矩阵: 20,000 × 15,000 (约2288.8 MB)
               Biclusters: 8

  amazon       - Amazon规模（论文数据集完整版）
               矩阵: 123,321 × 23,379 (约21991.3 MB，需要32GB+内存）
               Biclusters: 24
```

## 服务器推荐配置

### 最低配置

| 数据集 | 内存 | CPU | 预计时间 |
|--------|------|-----|---------|
| classic4 (6K×4K) | 4GB | 2核 | 1-2小时 |
| large (10K×8K) | 8GB | 4核 | 2-4小时 |
| xlarge (20K×15K) | 16GB | 8核 | 4-8小时 |
| amazon (123K×23K) | 32GB | 16核 | 10-20小时 |

### 推荐配置

- **内存**: 数据集内存需求的2-3倍
- **CPU**: 多核心（DiMergeCo可以利用NumPy的多线程）
- **存储**: 至少10GB可用空间（用于结果和日志）

## 进阶使用

### 使用Screen/Tmux管理长时间运行的任务

```bash
# 使用screen
screen -S dimergeco
PYTHONPATH=. python experiment/run_large_scale_experiments.py --dataset classic4
# Ctrl+A, D 分离会话
# screen -r dimergeco 重新连接

# 使用tmux
tmux new -s dimergeco
PYTHONPATH=. python experiment/run_large_scale_experiments.py --dataset classic4
# Ctrl+B, D 分离会话
# tmux attach -t dimergeco 重新连接
```

### 监控资源使用

```bash
# 安装htop（如果没有）
sudo apt install htop  # Ubuntu/Debian
sudo yum install htop  # CentOS/RHEL

# 实时监控
htop

# 监控特定Python进程
watch -n 1 'ps aux | grep python'

# 监控内存使用
free -h
```

### 限制资源使用

```bash
# 限制最大内存使用（8GB）
ulimit -v 8388608  # 8GB in KB
PYTHONPATH=. python experiment/run_large_scale_experiments.py --dataset classic4

# 使用nice降低优先级（避免影响其他任务）
nice -n 19 python experiment/run_large_scale_experiments.py --dataset classic4
```

## 结果文件

实验完成后会生成以下文件结构：

```
results/
├── classic4/
│   ├── experiment_results.json      # 实验结果摘要
│   ├── biclusters.pkl               # 检测到的biclusters
│   ├── config.json                  # 实验配置
│   └── visualizations/              # 可视化图像
│       ├── matrix_with_biclusters.png
│       ├── bicluster_statistics.png
│       └── individual_biclusters_heatmaps.png
└── batch/
    ├── classic4/...
    ├── large/...
    └── batch_summary.json           # 批量实验摘要
```

### 下载结果到本地

```bash
# 使用scp
scp -r user@server:/path/to/results ./local_results

# 使用rsync（更快，支持断点续传）
rsync -avz --progress user@server:/path/to/results ./local_results
```

## 常见问题

### Q1: 内存不足怎么办？

A:
1. 使用更小的数据集（classic4 → large → xlarge）
2. 减少T_p参数（减少分区迭代次数）
3. 增加交换空间（不推荐，会很慢）

```bash
# 检查可用内存
free -h

# 如果必须增加swap（临时方案）
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Q2: 运行时间过长怎么办？

A:
1. 减少T_p参数（从5降到3）
2. 减少矩阵规模
3. 使用更快的CPU

### Q3: 如何中断运行中的实验？

A:
```bash
# 查找进程ID
ps aux | grep run_large_scale

# 温和终止
kill <PID>

# 强制终止（如果上面不行）
kill -9 <PID>
```

### Q4: 如何恢复中断的实验？

A: 目前脚本不支持断点续传。建议：
1. 使用screen/tmux避免SSH断开导致中断
2. 使用nohup后台运行
3. 定期保存中间结果（可以修改脚本添加checkpoint）

### Q5: 可视化文件太大无法下载？

A:
```bash
# 只下载JSON结果文件（几KB）
scp user@server:/path/to/results/*/experiment_results.json ./

# 在服务器上压缩后下载
tar -czf results.tar.gz results/
scp user@server:/path/to/results.tar.gz ./
```

## 性能优化技巧

### 1. 使用更快的NumPy后端

```bash
# 安装Intel MKL（如果服务器是Intel CPU）
pip install mkl mkl-service

# 或使用OpenBLAS
sudo apt install libopenblas-dev
```

### 2. 设置NumPy线程数

```bash
# 在脚本开头添加
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
```

### 3. 使用Python优化模式

```bash
python -O experiment/run_large_scale_experiments.py --dataset classic4
```

## 批处理脚本示例

创建 `run_all_experiments.sh`:

```bash
#!/bin/bash

# 设置环境变量
export PYTHONPATH=.
export OMP_NUM_THREADS=8

# 创建日志目录
mkdir -p logs

# 运行实验
datasets=("classic4" "large" "xlarge")

for dataset in "${datasets[@]}"; do
    echo "Starting $dataset experiment..."
    python experiment/run_large_scale_experiments.py \
        --dataset $dataset \
        --output results/$dataset \
        > logs/$dataset.log 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ $dataset completed successfully"
    else
        echo "✗ $dataset failed"
    fi
done

echo "All experiments completed!"
```

运行：
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

## 联系支持

如遇到问题：
1. 检查日志文件
2. 确认服务器资源充足
3. 提交Issue: https://github.com/wzh4464/big_matrix_cocluster/issues
4. 邮件: wzh4464@gmail.com
