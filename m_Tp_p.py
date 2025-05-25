###
# File: /m_Tp_p.py
# Created Date: Thursday, July 4th 2024
# Author: Zihan
# -----
# Last Modified: Thursday, 4th July 2024 10:08:14 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date            By      Comments
# ----------      ------  ---------------------------------------------------------
###

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List

# 定义基本参数
p = 0.1  # 每次找到一个物体的概率
N = 100  # 总物体数
M = 10   # 每次找到的物体数
T_max = 150  # 最大搜索次数
num_simulations = 1000  # 模拟次数
n_jobs = 30  # 使用的CPU核心数

# 单次模拟搜索过程
def single_search_simulation(T: int, p: float, N: int, M: int) -> int:
    found = sum(np.random.binomial(M, p) for _ in range(T))
    return min(found, N)  # 确保不超过总物体数

# 模拟搜索过程
def simulate_search(p: float, N: int, M: int, T_max: int, num_simulations: int, n_jobs: int) -> List[float]:
    expected_results = []
    for T in tqdm(range(70, T_max + 1), desc="Simulating search process"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(single_search_simulation)(T, p, N, M) for _ in range(num_simulations)
        )
        results = [r for r in results if isinstance(r, (int, float))]  # 过滤结果中的非数值
        expected_results.append(np.mean(results))
    return expected_results

# 运行模拟
expected_results = simulate_search(p, N, M, T_max, num_simulations, n_jobs)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(range(70, T_max + 1), expected_results, label='Expected number of found co-clusters')
plt.xlabel('Search times (Tp)')
plt.ylabel('Number of found co-clusters (m)')
plt.title('Relationship between search times and number of found co-clusters')

# 设置x轴显示的刻度
plt.xticks([70, 80, 90, 100, 110, 120])

# 添加红色网格线
plt.axhline(y=95, color='red', linestyle='--', label='y=95')
plt.axvline(x=99, color='red', linestyle='--', label='x=99')

# 添加图例和网格
plt.legend()
plt.grid(True)

# 保存和显示图表
plt.savefig('m_Tp.png')
plt.show()
