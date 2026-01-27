"""
快速验证DiMergeCo实现是否正确

运行一个小规模实验，验证：
1. 代码能运行
2. 检测质量合理（NMI>0.7, ARI>0.6）
3. 各组件正常工作

运行方式：
    python experiment/quick_paper_validation.py
"""

from src import create_dimergeco_pipeline, create_synthetic_data_with_generator
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import time


def quick_validation():
    print("DiMergeCo快速验证测试\n")
    print("="*50)

    # 生成小规模测试数据
    print("\n[1/3] 生成测试数据...")
    matrix, _, ground_truth, _ = create_synthetic_data_with_generator(
        n_biclusters=5,
        matrix_shape=(1000, 800),
        bicluster_size_range=(80, 150),
        noise_level_spec=0.1,
        random_state=42
    )
    print(f"  矩阵: {matrix.shape}")
    print(f"  Ground truth: {len(ground_truth)} biclusters")

    # 使用调整后的参数运行DiMergeCo（对于1000×800矩阵，使用较大的T_m/T_n避免过多分区）
    print("\n[2/3] 运行DiMergeCo (使用调整后的参数)...")
    pipeline = create_dimergeco_pipeline(
        k1=8, k2=8,
        tolerance=0.05,
        T_m=50, T_n=40, T_p=3,  # 增大T_m/T_n，减少迭代次数
        P_thresh=0.95,
        overlap_threshold=0.45,
        output_directory="experiment/quick_validation_results",
        random_state=42
    )

    pipeline.load_matrix(matrix)
    pipeline.ground_truth_biclusters = ground_truth

    start = time.time()
    pipeline.fit()
    elapsed = time.time() - start

    print(f"  运行时间: {elapsed:.2f}秒")

    # 评估结果
    print("\n[3/3] 评估结果...")
    results = pipeline.get_results()
    detected = results.biclusters

    # 转换为标签
    true_labels = np.zeros(matrix.shape[0])
    detected_labels = np.zeros(matrix.shape[0])

    for i, bc in enumerate(ground_truth, 1):
        true_labels[bc.row_labels] = i
    for i, bc in enumerate(detected, 1):
        detected_labels[bc.row_labels] = i

    nmi = normalized_mutual_info_score(true_labels, detected_labels)
    ari = adjusted_rand_score(true_labels, detected_labels)

    print(f"  检测到: {len(detected)} biclusters")
    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")

    # 判断
    print("\n" + "="*50)
    if nmi > 0.7 and ari > 0.6:
        print("✅ 验证通过！DiMergeCo工作正常。")
        print("   质量指标达到预期水平（NMI>0.7, ARI>0.6）")
    elif nmi > 0.5 and ari > 0.4:
        print("⚠️  部分通过。质量指标可接受但低于预期。")
        print("   可能需要调整参数或检查数据生成。")
    else:
        print("❌ 验证失败！质量指标过低。")
        print("   需要检查实现是否有问题。")

    print("="*50)


if __name__ == '__main__':
    quick_validation()
