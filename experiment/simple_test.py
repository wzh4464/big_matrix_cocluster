"""
简单测试 - 使用标准biclustering（不使用DiMergeCo分区）
"""

from src import create_pipeline, create_synthetic_data_with_generator
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import time


print("简单Biclustering测试（不使用DiMergeCo分区）\n")
print("="*50)

# 生成测试数据
print("\n[1/3] 生成测试数据...")
matrix, _, ground_truth, _ = create_synthetic_data_with_generator(
    n_biclusters=5,
    matrix_shape=(500, 400),  # 更小的矩阵
    bicluster_size_range=(40, 80),
    noise_level_spec=0.1,
    random_state=42
)
print(f"  矩阵: {matrix.shape}")
print(f"  Ground truth: {len(ground_truth)} biclusters")

# 使用标准biclustering（不分区）
print("\n[2/3] 运行标准biclustering...")
pipeline = create_pipeline(
    k1=8, k2=8,
    tolerance=0.05,
    scoring_method="svr_normalized",
    output_directory="experiment/simple_test_results",
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
    print("✅ 测试通过！基础算法工作正常。")
elif nmi > 0.5 and ari > 0.4:
    print("⚠️  部分通过。质量可接受但低于预期。")
else:
    print("❌ 测试失败！")

print("="*50)
