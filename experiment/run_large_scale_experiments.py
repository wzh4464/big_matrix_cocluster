"""
大规模DiMergeCo实验脚本 - 服务器运行版本

适用于大规模数据集的DiMergeCo实验，包括：
- CLASSIC4规模（6,461×4,667）
- 更大规模数据集（可配置）
- 完整的进度监控和日志
- 资源使用跟踪
- 结果自动保存

运行方式：
    # 运行CLASSIC4规模实验
    python experiment/run_large_scale_experiments.py --dataset classic4 --output results/classic4

    # 运行超大规模实验
    python experiment/run_large_scale_experiments.py --dataset xlarge --output results/xlarge

    # 批量运行所有大数据集实验
    python experiment/run_large_scale_experiments.py --batch --output results/batch

    # 自定义规模
    python experiment/run_large_scale_experiments.py --custom --rows 10000 --cols 8000 --biclusters 10 --output results/custom
"""

import argparse
import numpy as np
import time
import json
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm

from src import (
    create_dimergeco_pipeline,
    create_synthetic_data_with_generator,
    Bicluster
)
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


# ==================== 大数据集配置 ====================

LARGE_SCALE_DATASETS = {
    'classic4': {
        'name': 'CLASSIC4规模（论文数据集）',
        'matrix_shape': (6461, 4667),
        'n_biclusters': 4,
        'bicluster_size_range': (800, 1600),
        'noise_level_spec': 0.15,
        'params': {
            'T_m': 20, 'T_n': 20, 'T_p': 5,
            'P_thresh': 0.95, 'k1': 10, 'k2': 10,
            'tolerance': 0.05, 'overlap_threshold': 0.45
        }
    },
    'large': {
        'name': '大规模（10K×8K）',
        'matrix_shape': (10000, 8000),
        'n_biclusters': 6,
        'bicluster_size_range': (800, 1500),
        'noise_level_spec': 0.15,
        'params': {
            'T_m': 30, 'T_n': 30, 'T_p': 5,
            'P_thresh': 0.95, 'k1': 10, 'k2': 10,
            'tolerance': 0.05, 'overlap_threshold': 0.45
        }
    },
    'xlarge': {
        'name': '超大规模（20K×15K）',
        'matrix_shape': (20000, 15000),
        'n_biclusters': 8,
        'bicluster_size_range': (1000, 2000),
        'noise_level_spec': 0.15,
        'params': {
            'T_m': 40, 'T_n': 40, 'T_p': 5,
            'P_thresh': 0.95, 'k1': 10, 'k2': 10,
            'tolerance': 0.05, 'overlap_threshold': 0.45
        }
    },
    'amazon': {
        'name': 'Amazon规模（论文数据集完整版）',
        'matrix_shape': (123321, 23379),
        'n_biclusters': 24,
        'bicluster_size_range': (3000, 8000),
        'noise_level_spec': 0.2,
        'params': {
            'T_m': 50, 'T_n': 50, 'T_p': 5,
            'P_thresh': 0.95, 'k1': 12, 'k2': 12,
            'tolerance': 0.05, 'overlap_threshold': 0.45
        }
    }
}


# ==================== 辅助函数 ====================

class ResourceMonitor:
    """资源使用监控器"""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None

    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def get_stats(self) -> Dict:
        """获取当前统计信息"""
        elapsed = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = current_memory - self.start_memory
        cpu_percent = self.process.cpu_percent()

        return {
            'elapsed_time': elapsed,
            'current_memory_mb': current_memory,
            'memory_delta_mb': memory_delta,
            'cpu_percent': cpu_percent
        }


def bicluster_to_labels(biclusters: List[Bicluster], n_elements: int) -> np.ndarray:
    """将biclusters转换为聚类标签"""
    labels = np.zeros(n_elements, dtype=int)
    for idx, bc in enumerate(biclusters, start=1):
        labels[bc.row_labels] = idx
    return labels


def evaluate_quality(detected: List[Bicluster], ground_truth: List[Bicluster],
                    matrix_shape: Tuple[int, int]) -> Dict:
    """计算聚类质量指标"""
    M, N = matrix_shape

    true_labels = bicluster_to_labels(ground_truth, M)
    detected_labels = bicluster_to_labels(detected, M)

    nmi = normalized_mutual_info_score(true_labels, detected_labels)
    ari = adjusted_rand_score(true_labels, detected_labels)

    return {
        'NMI': nmi,
        'ARI': ari,
        'n_detected': len(detected),
        'n_ground_truth': len(ground_truth)
    }


def format_size(size_bytes: float) -> str:
    """格式化字节大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.2f}h"


# ==================== 主实验函数 ====================

def run_large_scale_experiment(dataset_config: Dict, output_dir: Path,
                               dataset_name: str) -> Dict:
    """
    运行单个大规模实验

    Args:
        dataset_config: 数据集配置
        output_dir: 输出目录
        dataset_name: 数据集名称

    Returns:
        实验结果字典
    """
    print("\n" + "="*80)
    print(f"实验: {dataset_config['name']}")
    print(f"数据集: {dataset_name}")
    print("="*80)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化资源监控
    monitor = ResourceMonitor()
    monitor.start()

    # 步骤1: 生成合成数据
    print("\n[步骤 1/4] 生成大规模合成数据...")
    print(f"  矩阵形状: {dataset_config['matrix_shape']}")
    print(f"  Biclusters数量: {dataset_config['n_biclusters']}")

    data_start = time.time()

    matrix, _, ground_truth, _ = create_synthetic_data_with_generator(
        n_biclusters=dataset_config['n_biclusters'],
        matrix_shape=dataset_config['matrix_shape'],
        bicluster_size_range=dataset_config['bicluster_size_range'],
        noise_level_spec=dataset_config['noise_level_spec'],
        random_state=42
    )

    data_time = time.time() - data_start
    matrix_size_mb = matrix.nbytes / 1024 / 1024

    print(f"  ✓ 数据生成完成: {format_time(data_time)}")
    print(f"  矩阵大小: {format_size(matrix.nbytes)}")
    print(f"  Ground truth biclusters: {len(ground_truth)}")

    stats = monitor.get_stats()
    print(f"  内存使用: {stats['current_memory_mb']:.1f} MB "
          f"(增量: {stats['memory_delta_mb']:.1f} MB)")

    # 步骤2: 创建DiMergeCo pipeline
    print("\n[步骤 2/4] 创建DiMergeCo pipeline...")
    params = dataset_config['params']
    print(f"  参数: T_m={params['T_m']}, T_n={params['T_n']}, "
          f"T_p={params['T_p']}, P_thresh={params['P_thresh']}")

    pipeline = create_dimergeco_pipeline(
        k1=params['k1'],
        k2=params['k2'],
        tolerance=params['tolerance'],
        T_m=params['T_m'],
        T_n=params['T_n'],
        T_p=params['T_p'],
        P_thresh=params['P_thresh'],
        overlap_threshold=params['overlap_threshold'],
        use_spatial_indexing=True,
        output_directory=str(output_dir),
        random_state=42
    )

    print("  ✓ Pipeline创建完成")

    # 步骤3: 运行DiMergeCo检测
    print("\n[步骤 3/4] 运行DiMergeCo检测...")
    print("  这可能需要较长时间，请耐心等待...")

    pipeline.load_matrix(matrix)
    pipeline.ground_truth_biclusters = ground_truth

    detection_start = time.time()

    # 使用tqdm显示进度（如果可能）
    with tqdm(total=100, desc="  检测进度", ncols=80,
              bar_format='{l_bar}{bar}| {elapsed}<{remaining}') as pbar:
        # 这里运行检测
        pipeline.fit()
        pbar.update(100)

    detection_time = time.time() - detection_start

    print(f"  ✓ 检测完成: {format_time(detection_time)}")

    stats = monitor.get_stats()
    print(f"  总运行时间: {format_time(stats['elapsed_time'])}")
    print(f"  内存峰值: {stats['current_memory_mb']:.1f} MB")
    print(f"  CPU使用: {stats['cpu_percent']:.1f}%")

    # 步骤4: 评估结果
    print("\n[步骤 4/4] 评估结果...")

    results = pipeline.get_results()
    detected_biclusters = results.biclusters

    quality_metrics = evaluate_quality(
        detected_biclusters,
        ground_truth,
        matrix.shape
    )

    print(f"  检测到的biclusters: {len(detected_biclusters)}")
    print(f"  NMI: {quality_metrics['NMI']:.4f}")
    print(f"  ARI: {quality_metrics['ARI']:.4f}")

    # 与论文结果对比（如果是CLASSIC4）
    if 'classic4' in dataset_name.lower():
        paper_nmi, paper_ari = 0.865, 0.776
        print(f"\n  与论文CLASSIC4结果对比:")
        print(f"    NMI: {quality_metrics['NMI']:.4f} vs 论文 {paper_nmi:.4f} "
              f"(差异: {abs(quality_metrics['NMI']-paper_nmi):.4f})")
        print(f"    ARI: {quality_metrics['ARI']:.4f} vs 论文 {paper_ari:.4f} "
              f"(差异: {abs(quality_metrics['ARI']-paper_ari):.4f})")

    # 汇总结果
    final_stats = monitor.get_stats()

    experiment_results = {
        'dataset_name': dataset_name,
        'dataset_description': dataset_config['name'],
        'matrix_shape': dataset_config['matrix_shape'],
        'matrix_size_mb': matrix_size_mb,
        'n_biclusters_ground_truth': len(ground_truth),
        'n_biclusters_detected': len(detected_biclusters),
        'parameters': params,
        'timing': {
            'data_generation_time': data_time,
            'detection_time': detection_time,
            'total_time': final_stats['elapsed_time']
        },
        'resources': {
            'peak_memory_mb': final_stats['current_memory_mb'],
            'memory_delta_mb': final_stats['memory_delta_mb'],
            'cpu_percent': final_stats['cpu_percent']
        },
        'quality_metrics': quality_metrics,
        'timestamp': datetime.now().isoformat()
    }

    # 保存结果
    results_file = output_dir / 'experiment_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ 结果已保存到: {results_file}")

    # 清理内存
    del matrix
    gc.collect()

    return experiment_results


def run_batch_experiments(output_base_dir: Path, datasets: List[str] = None):
    """
    批量运行多个大数据集实验

    Args:
        output_base_dir: 输出基础目录
        datasets: 要运行的数据集列表，默认运行所有
    """
    if datasets is None:
        datasets = ['classic4', 'large']  # 默认运行这两个

    print("\n" + "="*80)
    print("批量运行大规模DiMergeCo实验")
    print("="*80)
    print(f"\n将运行以下数据集: {', '.join(datasets)}")
    print(f"输出目录: {output_base_dir}\n")

    all_results = {}

    for dataset_name in datasets:
        if dataset_name not in LARGE_SCALE_DATASETS:
            print(f"⚠️  警告: 数据集 '{dataset_name}' 不存在，跳过")
            continue

        dataset_config = LARGE_SCALE_DATASETS[dataset_name]
        output_dir = output_base_dir / dataset_name

        try:
            result = run_large_scale_experiment(
                dataset_config, output_dir, dataset_name
            )
            all_results[dataset_name] = result
            print(f"\n✓ {dataset_name} 实验完成")

        except Exception as e:
            print(f"\n✗ {dataset_name} 实验失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}

    # 保存批量结果摘要
    summary_file = output_base_dir / 'batch_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "="*80)
    print("批量实验完成摘要")
    print("="*80)

    for dataset_name, result in all_results.items():
        if 'error' in result:
            print(f"\n{dataset_name}: ✗ 失败")
            print(f"  错误: {result['error']}")
        else:
            print(f"\n{dataset_name}: ✓ 成功")
            print(f"  NMI: {result['quality_metrics']['NMI']:.4f}")
            print(f"  ARI: {result['quality_metrics']['ARI']:.4f}")
            print(f"  时间: {format_time(result['timing']['total_time'])}")
            print(f"  内存: {result['resources']['peak_memory_mb']:.1f} MB")

    print(f"\n详细结果已保存到: {summary_file}")


def run_custom_experiment(rows: int, cols: int, n_biclusters: int,
                         output_dir: Path):
    """运行自定义规模的实验"""

    # 根据矩阵大小自动调整参数
    T_m = max(rows // 100, 20)
    T_n = max(cols // 100, 20)

    custom_config = {
        'name': f'自定义规模（{rows}×{cols}）',
        'matrix_shape': (rows, cols),
        'n_biclusters': n_biclusters,
        'bicluster_size_range': (max(rows//20, 50), max(rows//10, 100)),
        'noise_level_spec': 0.15,
        'params': {
            'T_m': T_m, 'T_n': T_n, 'T_p': 5,
            'P_thresh': 0.95, 'k1': 10, 'k2': 10,
            'tolerance': 0.05, 'overlap_threshold': 0.45
        }
    }

    print(f"\n自动调整的参数: T_m={T_m}, T_n={T_n}")

    return run_large_scale_experiment(
        custom_config, output_dir, 'custom'
    )


# ==================== 命令行接口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='大规模DiMergeCo实验 - 服务器运行版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 运行CLASSIC4规模实验
  python experiment/run_large_scale_experiments.py --dataset classic4

  # 运行超大规模实验
  python experiment/run_large_scale_experiments.py --dataset xlarge

  # 批量运行
  python experiment/run_large_scale_experiments.py --batch

  # 自定义规模
  python experiment/run_large_scale_experiments.py --custom --rows 15000 --cols 12000 --biclusters 8
        """
    )

    parser.add_argument('--dataset', type=str,
                       choices=list(LARGE_SCALE_DATASETS.keys()),
                       help='选择预定义的数据集')

    parser.add_argument('--batch', action='store_true',
                       help='批量运行多个实验')

    parser.add_argument('--batch-datasets', type=str, nargs='+',
                       help='批量运行时指定数据集列表')

    parser.add_argument('--custom', action='store_true',
                       help='运行自定义规模实验')

    parser.add_argument('--rows', type=int, default=10000,
                       help='自定义实验的行数')

    parser.add_argument('--cols', type=int, default=8000,
                       help='自定义实验的列数')

    parser.add_argument('--biclusters', type=int, default=6,
                       help='自定义实验的bicluster数量')

    parser.add_argument('--output', type=str,
                       default='experiment/large_scale_results',
                       help='输出目录')

    args = parser.parse_args()

    output_base = Path(args.output)

    # 显示可用数据集信息
    if not any([args.dataset, args.batch, args.custom]):
        print("\n可用的大规模数据集:\n")
        for name, config in LARGE_SCALE_DATASETS.items():
            M, N = config['matrix_shape']
            size_mb = (M * N * 8) / 1024 / 1024  # float64
            print(f"  {name:12} - {config['name']}")
            print(f"               矩阵: {M:,} × {N:,} (约{size_mb:.1f} MB)")
            print(f"               Biclusters: {config['n_biclusters']}")
            print()

        print("使用 --help 查看详细用法\n")
        return

    # 运行实验
    if args.batch:
        datasets = args.batch_datasets if args.batch_datasets else ['classic4', 'large']
        run_batch_experiments(output_base, datasets)

    elif args.custom:
        run_custom_experiment(
            args.rows, args.cols, args.biclusters,
            output_base / 'custom'
        )

    elif args.dataset:
        config = LARGE_SCALE_DATASETS[args.dataset]
        run_large_scale_experiment(
            config, output_base / args.dataset, args.dataset
        )

    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
