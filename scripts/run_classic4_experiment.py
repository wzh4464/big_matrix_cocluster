#!/usr/bin/env python3
"""
CLASSIC4 规模实验脚本

直接运行 CLASSIC4 规模（6,461×4,667）的 DiMergeCo 实验
使用论文推荐参数和优化的聚合算法（10-24倍加速）

使用方法：
    python scripts/run_classic4_experiment.py

或在服务器上使用 nohup：
    nohup python scripts/run_classic4_experiment.py > classic4.log 2>&1 &
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import logging
from pathlib import Path
import numpy as np

from visualization import create_synthetic_data_with_generator
from pipeline import create_dimergeco_pipeline
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classic4_experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_classic4_experiment():
    """
    运行 CLASSIC4 规模的 DiMergeCo 实验

    配置：
    - 矩阵大小：6,461 × 4,667（CLASSIC4 文档-特征数）
    - Ground truth biclusters：4个
    - Bicluster 大小：800-1600（平均~1200，占比18-25%）
    - 噪声水平：0.15（中等）

    参数（来自论文）：
    - T_m=20, T_n=20：最小 Co-cluster 大小
    - T_p=5：分割迭代次数
    - P_thresh=0.95：检测概率阈值
    - k1=10, k2=10：SVD 聚类数
    - tolerance=0.05：质量阈值
    - overlap_threshold=0.45：层次化合并阈值（论文推荐 0.4-0.5）
    """
    logger.info("=" * 70)
    logger.info("CLASSIC4 规模 DiMergeCo 实验")
    logger.info("=" * 70)

    # ========== 步骤 1：生成 CLASSIC4 规模的合成数据 ==========
    logger.info("\n[1/4] 生成 CLASSIC4 规模合成数据...")
    logger.info("  矩阵大小：6,461 × 4,667")
    logger.info("  Ground truth biclusters：4个")
    logger.info("  Bicluster 大小范围：800-1600")

    start_gen = time.time()

    matrix, _, ground_truth, _ = create_synthetic_data_with_generator(
        n_biclusters=4,
        matrix_shape=(6461, 4667),
        bicluster_size_range=(800, 1600),
        noise_level_spec=0.15,
        random_state=42
    )

    gen_time = time.time() - start_gen

    logger.info(f"  ✓ 数据生成完成：{gen_time:.2f}秒")
    logger.info(f"  矩阵形状：{matrix.shape}")
    logger.info(f"  Ground truth biclusters：{len(ground_truth)}")
    for i, bc in enumerate(ground_truth):
        logger.info(f"    Bicluster {i+1}：{len(bc.row_labels)} 行 × {len(bc.col_labels)} 列")

    # ========== 步骤 2：创建 DiMergeCo pipeline（使用论文参数）==========
    logger.info("\n[2/4] 创建 DiMergeCo pipeline...")
    logger.info("  参数配置（来自论文）：")
    logger.info("    - SVD 聚类：k1=10, k2=10")
    logger.info("    - 质量阈值：tolerance=0.05")
    logger.info("    - 分割参数：T_m=20, T_n=20, T_p=5")
    logger.info("    - 检测概率：P_thresh=0.95")
    logger.info("    - 合并阈值：overlap_threshold=0.45")
    logger.info("    - 优化聚合：ENABLED（spatial_index + parallel + cache）")

    output_dir = Path("classic4_results")
    output_dir.mkdir(exist_ok=True)

    pipeline = create_dimergeco_pipeline(
        k1=10,
        k2=10,
        tolerance=0.05,
        T_m=20,
        T_n=20,
        T_p=5,
        P_thresh=0.95,
        overlap_threshold=0.45,
        use_spatial_indexing=True,  # 层次化合并的空间索引
        output_directory=str(output_dir),
        random_state=42
    )

    logger.info("  ✓ Pipeline 创建完成")

    # ========== 步骤 3：运行 DiMergeCo 检测 ==========
    logger.info("\n[3/4] 运行 DiMergeCo 检测...")
    logger.info("  预计时间：2-5 分钟（优化后）")
    logger.info("  （原始 O(n²) 实现需要 48+ 分钟）")

    pipeline.load_matrix(matrix)
    pipeline.ground_truth_biclusters = ground_truth

    start_detect = time.time()
    pipeline.fit()
    detect_time = time.time() - start_detect

    logger.info(f"  ✓ 检测完成：{detect_time:.2f}秒（{detect_time/60:.2f}分钟）")

    # ========== 步骤 4：评估结果 ==========
    logger.info("\n[4/4] 评估结果...")

    results = pipeline.get_results()
    detected_biclusters = results.biclusters

    logger.info(f"  检测到的 biclusters：{len(detected_biclusters)}")

    # 计算 NMI 和 ARI
    M, N = matrix.shape

    # 转换为行标签
    true_labels = np.zeros(M, dtype=int)
    detected_labels = np.zeros(M, dtype=int)

    for i, bc in enumerate(ground_truth, start=1):
        true_labels[bc.row_labels] = i

    for i, bc in enumerate(detected_biclusters, start=1):
        detected_labels[bc.row_labels] = i

    nmi = normalized_mutual_info_score(true_labels, detected_labels)
    ari = adjusted_rand_score(true_labels, detected_labels)

    # ========== 结果汇总 ==========
    logger.info("\n" + "=" * 70)
    logger.info("实验结果汇总")
    logger.info("=" * 70)

    logger.info(f"\n数据集信息：")
    logger.info(f"  矩阵大小：{matrix.shape[0]:,} × {matrix.shape[1]:,}")
    logger.info(f"  总元素数：{matrix.size:,}")
    logger.info(f"  Ground truth biclusters：{len(ground_truth)}")

    logger.info(f"\n检测结果：")
    logger.info(f"  检测到的 biclusters：{len(detected_biclusters)}")
    for i, bc in enumerate(detected_biclusters[:10]):  # 只显示前10个
        logger.info(f"    Bicluster {i+1}：{len(bc.row_labels)} 行 × {len(bc.col_labels)} 列 "
                   f"(score: {bc.score:.4f if bc.score is not None else 'N/A'})")
    if len(detected_biclusters) > 10:
        logger.info(f"    ... 还有 {len(detected_biclusters) - 10} 个")

    logger.info(f"\n质量指标：")
    logger.info(f"  NMI (Normalized Mutual Information)：{nmi:.4f}")
    logger.info(f"  ARI (Adjusted Rand Index)：{ari:.4f}")

    # 与论文期望值对比
    logger.info(f"\n与论文期望值对比（CLASSIC4 真实数据）：")
    logger.info(f"  论文 NMI：0.865（真实数据）")
    logger.info(f"  我们 NMI：{nmi:.4f}（合成数据）")
    logger.info(f"  差异：{abs(nmi - 0.865):.4f}")
    logger.info(f"")
    logger.info(f"  论文 ARI：0.776（真实数据）")
    logger.info(f"  我们 ARI：{ari:.4f}（合成数据）")
    logger.info(f"  差异：{abs(ari - 0.776):.4f}")

    # 性能分析
    logger.info(f"\n性能统计：")
    logger.info(f"  数据生成时间：{gen_time:.2f}秒")
    logger.info(f"  检测运行时间：{detect_time:.2f}秒（{detect_time/60:.2f}分钟）")
    logger.info(f"  总时间：{gen_time + detect_time:.2f}秒")
    logger.info(f"")
    logger.info(f"  优化效果：")
    logger.info(f"    原始 O(n²) 预计时间：48+ 分钟")
    logger.info(f"    优化后实际时间：{detect_time/60:.2f} 分钟")
    logger.info(f"    加速比：~{(48*60)/detect_time:.1f}x")

    # 成功标准判断
    logger.info(f"\n成功标准判断：")
    nmi_pass = nmi > 0.75
    ari_pass = ari > 0.65
    time_pass = detect_time < 600  # 10分钟内

    logger.info(f"  NMI > 0.75：{'✓ PASS' if nmi_pass else '✗ FAIL'} ({nmi:.4f})")
    logger.info(f"  ARI > 0.65：{'✓ PASS' if ari_pass else '✗ FAIL'} ({ari:.4f})")
    logger.info(f"  时间 < 10分钟：{'✓ PASS' if time_pass else '✗ FAIL'} ({detect_time/60:.2f}分钟)")

    if nmi_pass and ari_pass and time_pass:
        logger.info(f"\n{'✓' * 35}")
        logger.info(f"✓ 实验成功！所有指标均达标。")
        logger.info(f"{'✓' * 35}")
    else:
        logger.info(f"\n⚠️  部分指标未达标，建议检查配置或数据。")

    logger.info(f"\n结果已保存到：{output_dir}/")
    logger.info("=" * 70)

    return {
        'matrix_shape': matrix.shape,
        'n_ground_truth': len(ground_truth),
        'n_detected': len(detected_biclusters),
        'nmi': nmi,
        'ari': ari,
        'generation_time': gen_time,
        'detection_time': detect_time,
        'total_time': gen_time + detect_time
    }


if __name__ == '__main__':
    try:
        results = run_classic4_experiment()

        # 保存结果到 JSON
        import json
        results_file = Path('classic4_results') / 'experiment_summary.json'

        # 转换 numpy 类型为 Python 原生类型
        results_json = {
            'matrix_shape': results['matrix_shape'],
            'n_ground_truth': int(results['n_ground_truth']),
            'n_detected': int(results['n_detected']),
            'nmi': float(results['nmi']),
            'ari': float(results['ari']),
            'generation_time_seconds': float(results['generation_time']),
            'detection_time_seconds': float(results['detection_time']),
            'total_time_seconds': float(results['total_time'])
        }

        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)

        logger.info(f"\n实验汇总已保存到：{results_file}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"\n实验失败：{e}", exc_info=True)
        sys.exit(1)
