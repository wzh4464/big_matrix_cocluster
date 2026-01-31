# CLAUDE.md - Development Notes

## Development Workflow

- **本地**：编辑代码，只能跑很小的测试（`python -m pytest test/ -v`）
- **服务器**：通过 `git pull` 拉取本地代码，运行大规模实验
- 代码修改流程：本地编辑 → git push → 服务器 git pull → 服务器运行

## 常用命令

### 本地
```bash
python -m pytest test/ -v          # 跑测试
black src/ test/                    # 格式化
```

### 服务器
```bash
git pull                            # 拉取最新代码
# CLASSIC4 实验
nohup python scripts/run_classic4_experiment.py > classic4.log 2>&1 &
tail -f classic4_experiment.log     # 查看进度

# 大规模实验
PYTHONPATH=. python experiment/run_large_scale_experiments.py --dataset classic4
```

## 日志文件位置

| 运行方式 | 日志位置 |
|---------|---------|
| `scripts/run_classic4_experiment.py` | `classic4_experiment.log`（项目根目录） |
| `experiment/run_server.sh` | `logs/<exp_type>_<timestamp>.log` |
| `nohup ... > file.log 2>&1 &` | 指定的 `file.log` |

## 已知性能问题

### 聚合阶段卡住（780k biclusters）

780,550 个 biclusters 在 `OptimizedAggregator._aggregate_sequential` 中聚合极慢。
根因：空间索引网格 20×20 = 400 格，每格平均 ~1950 个 biclusters，
`query_overlapping` 对每个候选计算 Jaccard（O(M+N)），总计算量巨大。
详见 `src/detection_optimized.py` 和 `src/hierarchical_merge.py`。
