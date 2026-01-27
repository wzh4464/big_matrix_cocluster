"""
复现DiMergeCo论文实验结果

使用合成数据模拟论文中的真实数据集，运行相同配置的实验，
验证Python实现的DiMergeCo算法的正确性和性能。

运行方式：
    python experiment/reproduce_paper_experiments.py --experiment all
    python experiment/reproduce_paper_experiments.py --experiment classic4
    python experiment/reproduce_paper_experiments.py --experiment small
    python experiment/reproduce_paper_experiments.py --experiment scalability
"""

import argparse
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

from src import (
    create_dimergeco_pipeline,
    create_synthetic_data_with_generator,
    Bicluster
)
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


# 论文推荐的配置（来自参数敏感性分析Table 4）
PAPER_PARAMS = {
    'k1': 10,
    'k2': 10,
    'tolerance': 0.05,
    'T_m': 20,
    'T_n': 20,
    'T_p': 5,
    'P_thresh': 0.95,
    'overlap_threshold': 0.45,  # τ ∈ [0.4, 0.5]的中点
    'use_spatial_indexing': True,
    'random_state': 42
}

# 论文数据集配置（使用合成数据模拟）
PAPER_DATASETS = {
    'classic4_synthetic': {
        'matrix_shape': (6461, 4667),
        'n_biclusters': 4,
        'bicluster_size_range': (800, 1600),
        'noise_level_spec': 0.15,
        'description': 'CLASSIC4规模合成数据（6,461文档×4,667特征，4个类别）'
    },
    'amazon_synthetic': {
        'matrix_shape': (10000, 5000),  # 简化版（完整版123,321×23,379过大）
        'n_biclusters': 24,
        'bicluster_size_range': (300, 800),
        'noise_level_spec': 0.2,
        'description': 'Amazon规模合成数据（简化版，原始为123,321×23,379）'
    },
    'small_cocluster': {
        'matrix_shape': (5000, 5000),
        'n_biclusters': 10,
        'bicluster_size_range': (5, 20),
        'noise_level_spec': 0.1,
        'description': '小Co-cluster检测测试（5×5到20×20，验证检测小bicluster能力）'
    }
}

# 论文期望结果（Table 1）
PAPER_EXPECTED_RESULTS = {
    'CLASSIC4': {'NMI': 0.865, 'ARI': 0.776},
    'Amazon': {'NMI': 0.768, 'ARI': 0.585},
    'RCV1-Large': {'NMI': 0.835, 'ARI': 0.758},
    'BCW': {'NMI': 0.824, 'ARI': 0.740}
}


def bicluster_to_labels(biclusters: List[Bicluster], n_elements: int) -> np.ndarray:
    """
    将biclusters转换为聚类标签数组。

    Args:
        biclusters: 检测到的biclusters列表
        n_elements: 总元素数量（行数）

    Returns:
        标签数组（未分配的元素标记为0）
    """
    labels = np.zeros(n_elements, dtype=int)

    for idx, bc in enumerate(biclusters, start=1):
        # 使用row_labels（行索引）作为聚类标签
        labels[bc.row_labels] = idx

    return labels


def evaluate_quality(detected: List[Bicluster], ground_truth: List[Bicluster],
                    matrix_shape: Tuple[int, int]) -> Dict:
    """
    计算聚类质量指标（NMI和ARI）。

    Args:
        detected: 检测到的biclusters
        ground_truth: 真实biclusters
        matrix_shape: 矩阵形状

    Returns:
        包含NMI和ARI的字典
    """
    M, N = matrix_shape

    # 转换为标签
    true_labels = bicluster_to_labels(ground_truth, M)
    detected_labels = bicluster_to_labels(detected, M)

    # 计算指标
    nmi = normalized_mutual_info_score(true_labels, detected_labels)
    ari = adjusted_rand_score(true_labels, detected_labels)

    return {
        'NMI': nmi,
        'ARI': ari,
        'n_detected': len(detected),
        'n_ground_truth': len(ground_truth)
    }


def run_experiment(dataset_name: str, dataset_config: Dict,
                  output_dir: Path) -> Dict:
    """
    运行单个实验。

    Args:
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        output_dir: 输出目录

    Returns:
        实验结果字典
    """
    print(f"\n{'='*60}")
    print(f"实验: {dataset_name}")
    print(f"描述: {dataset_config['description']}")
    print(f"{'='*60}")

    # 生成合成数据
    print("\n[1/4] 生成合成数据...")
    matrix, _, ground_truth, _ = create_synthetic_data_with_generator(
        n_biclusters=dataset_config['n_biclusters'],
        matrix_shape=dataset_config['matrix_shape'],
        bicluster_size_range=dataset_config['bicluster_size_range'],
        noise_level_spec=dataset_config['noise_level_spec'],
        random_state=PAPER_PARAMS['random_state']
    )
    print(f"  矩阵形状: {matrix.shape}")
    print(f"  Ground truth biclusters: {len(ground_truth)}")

    # 创建DiMergeCo pipeline
    print("\n[2/4] 创建DiMergeCo pipeline...")
    pipeline = create_dimergeco_pipeline(
        k1=PAPER_PARAMS['k1'],
        k2=PAPER_PARAMS['k2'],
        tolerance=PAPER_PARAMS['tolerance'],
        T_m=PAPER_PARAMS['T_m'],
        T_n=PAPER_PARAMS['T_n'],
        T_p=PAPER_PARAMS['T_p'],
        P_thresh=PAPER_PARAMS['P_thresh'],
        overlap_threshold=PAPER_PARAMS['overlap_threshold'],
        use_spatial_indexing=PAPER_PARAMS['use_spatial_indexing'],
        output_directory=str(output_dir / dataset_name),
        random_state=PAPER_PARAMS['random_state']
    )

    # 运行检测
    print("\n[3/4] 运行DiMergeCo检测...")
    pipeline.load_matrix(matrix)
    pipeline.ground_truth_biclusters = ground_truth

    start_time = time.time()
    pipeline.fit()
    elapsed_time = time.time() - start_time

    print(f"  运行时间: {elapsed_time:.2f} 秒")

    # 评估结果
    print("\n[4/4] 评估结果...")
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

    # 与论文结果对比（如果有）
    if 'classic4' in dataset_name.lower():
        expected = PAPER_EXPECTED_RESULTS['CLASSIC4']
        print(f"\n  与论文CLASSIC4结果对比:")
        print(f"    NMI: {quality_metrics['NMI']:.4f} vs 论文 {expected['NMI']:.4f} "
              f"(差异: {abs(quality_metrics['NMI']-expected['NMI']):.4f})")
        print(f"    ARI: {quality_metrics['ARI']:.4f} vs 论文 {expected['ARI']:.4f} "
              f"(差异: {abs(quality_metrics['ARI']-expected['ARI']):.4f})")

        # 判断是否接近
        nmi_close = abs(quality_metrics['NMI'] - expected['NMI']) < 0.15
        ari_close = abs(quality_metrics['ARI'] - expected['ARI']) < 0.15

        if nmi_close and ari_close:
            print(f"    ✅ 结果与论文接近（合成数据允许±0.15差异）")
        else:
            print(f"    ⚠️  结果有偏差，可能原因：合成数据 vs 真实数据")

    # 汇总结果
    experiment_results = {
        'dataset_name': dataset_name,
        'matrix_shape': dataset_config['matrix_shape'],
        'n_biclusters_ground_truth': len(ground_truth),
        'n_biclusters_detected': len(detected_biclusters),
        'execution_time': elapsed_time,
        'quality_metrics': quality_metrics,
        'parameters': PAPER_PARAMS
    }

    return experiment_results


def run_scalability_test(output_dir: Path) -> Dict:
    """
    运行可扩展性测试。

    测试不同矩阵规模下的性能，验证算法的可扩展性。
    """
    print(f"\n{'='*60}")
    print("可扩展性测试")
    print(f"{'='*60}")

    sizes = [
        (1000, 800, 4),
        (2000, 1600, 4),
        (5000, 4000, 4),
        (10000, 8000, 4),
    ]

    results = []

    for M, N, K in sizes:
        print(f"\n测试规模: {M}×{N}，{K}个biclusters")

        # 生成数据
        matrix, _, ground_truth, _ = create_synthetic_data_with_generator(
            n_biclusters=K,
            matrix_shape=(M, N),
            bicluster_size_range=(int(M*0.1), int(M*0.2)),
            noise_level_spec=0.15,
            random_state=42
        )

        # 运行DiMergeCo
        pipeline = create_dimergeco_pipeline(
            **PAPER_PARAMS,
            output_directory=str(output_dir / f"scale_{M}x{N}")
        )
        pipeline.load_matrix(matrix)

        start_time = time.time()
        pipeline.fit()
        elapsed_time = time.time() - start_time

        detected = pipeline.get_results().biclusters
        quality = evaluate_quality(detected, ground_truth, matrix.shape)

        results.append({
            'size': f"{M}×{N}",
            'n_elements': M * N,
            'time': elapsed_time,
            'NMI': quality['NMI'],
            'ARI': quality['ARI']
        })

        print(f"  时间: {elapsed_time:.2f}s, NMI: {quality['NMI']:.4f}, ARI: {quality['ARI']:.4f}")

    # 分析趋势
    print("\n可扩展性分析:")
    print(f"  {'规模':<15} {'元素数':<12} {'时间(s)':<10} {'NMI':<8} {'ARI':<8}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['size']:<15} {r['n_elements']:<12} {r['time']:<10.2f} {r['NMI']:<8.4f} {r['ARI']:<8.4f}")

    return {'scalability_results': results}


def main():
    parser = argparse.ArgumentParser(description='复现DiMergeCo论文实验')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'classic4', 'amazon', 'small', 'scalability'],
                       help='要运行的实验')
    parser.add_argument('--output', type=str, default='experiment/paper_reproduction_results',
                       help='输出目录')

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 运行实验
    if args.experiment == 'all' or args.experiment == 'classic4':
        all_results['classic4'] = run_experiment(
            'classic4_synthetic',
            PAPER_DATASETS['classic4_synthetic'],
            output_dir
        )

    if args.experiment == 'all' or args.experiment == 'amazon':
        all_results['amazon'] = run_experiment(
            'amazon_synthetic',
            PAPER_DATASETS['amazon_synthetic'],
            output_dir
        )

    if args.experiment == 'all' or args.experiment == 'small':
        all_results['small_cocluster'] = run_experiment(
            'small_cocluster',
            PAPER_DATASETS['small_cocluster'],
            output_dir
        )

    if args.experiment == 'all' or args.experiment == 'scalability':
        all_results['scalability'] = run_scalability_test(output_dir)

    # 保存所有结果
    results_file = output_dir / 'all_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"所有实验完成！结果已保存到: {results_file}")
    print(f"{'='*60}")

    # 打印总结
    print("\n实验总结:")
    for exp_name, exp_result in all_results.items():
        if exp_name == 'scalability':
            print(f"\n  {exp_name}: 已完成 {len(exp_result['scalability_results'])} 个规模测试")
        else:
            metrics = exp_result.get('quality_metrics', {})
            print(f"\n  {exp_name}:")
            print(f"    NMI: {metrics.get('NMI', 0):.4f}")
            print(f"    ARI: {metrics.get('ARI', 0):.4f}")
            print(f"    时间: {exp_result.get('execution_time', 0):.2f}秒")


if __name__ == '__main__':
    main()
