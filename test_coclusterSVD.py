###
 # File: /test_coclusterSVD.py
 # Created Date: Wednesday August 30th 2023
 # Author: Zihan
 # -----
 # Last Modified: Wednesday, 17th January 2024 6:20:52 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import sys

sys.path.append("..")
from coclusterSVD import score

import numpy as np
from numpy.linalg import svd

def test_score():
    # X = [a, a, a]
    # a = rand(3, 1)
    a = np.random.rand(3, 1)
    X = np.concatenate((a, a, a), axis=1)
    
    I: list[bool] = [True, False, True]
    J: list[bool] = [True, True, True]
    
    # I = [1, 3]
    # J = [1, 2, 3]
    
    I = np.array(I)
    J = np.array(J)
    
    assert abs(score(X, I, J)) < 1e-10
    
def test_speed_svd(min=0,max=1,sizex=1000,sizey=1000):
    # X, X_min = min, X_max = max, X.shape = (sizex, sizey)
    X = np.random.rand(sizex, sizey)*(max-min)+min
    svd(X)


    
def test_svd():
    from functools import partial
    test = partial(test_speed_svd)
    
    import multiprocessing as mp
    import time
    
    pool = mp.Pool(processes=48)
    start = time.time()
    
    pool.map(test, range(48))
    pool.close()
    pool.join()
    
    end = time.time()
    print(end - start)
    
def test_copy_matrix_helper(sizex=100,sizey=100,X = None):
    if X is None:
        X = np.random.rand(sizex, sizey)
    indexx = np.random.randint(0, sizex, sizex)
    indexy = np.random.randint(0, sizey, sizey)
    X[indexx, :] = X[indexx, :]
    X[:, indexy] = X[:, indexy]
    return X
    

def test_copy_matrix(min=0,max=1,sizex=1000,sizey=1000):
    X = np.random.rand(sizex, sizey)*(max-min)+min
    from functools import partial
    test = partial(test_copy_matrix_helper, X=X)
    
    import multiprocessing as mp
    import time
    
    pool = mp.Pool(processes=48)
    start = time.time()
    
    pool.map(test, range(48))
    pool.close()
    pool.join()
    
    end = time.time()
    print(end - start)
    
import matplotlib.pyplot as plt

# higher dpi
plt.rcParams['savefig.dpi'] = 600

# thinner margin
# plt.rcParams['savefig.pad_inches'] = 0.001

# Data
percentages = [5, 20, 50, 80, 100]  # x-axis: dataset sizes as percentages
times_cocc = [85, 1007]  # y-axis values for CoCC
times_our_method = [41, 80, 315, 788, 1750]  # y-axis values for Our Method
times_our_method_svd = [9, 17, 56, 120, 266]  # y-axis values for Our Method (svd refactored to rust)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(percentages[:len(times_cocc)], times_cocc, marker='o', label='CoCC')
plt.plot(percentages, times_our_method, marker='s', label='Our Method')
plt.plot(percentages, times_our_method_svd, marker='^', label='Our Method (svd refactored to rust)')

# Adding labels and title
plt.xlabel('Dataset Size (%)')
plt.ylabel('Time (min)')
plt.title('Time Curve on Different Sizes of Reuters-21578 Dataset')
plt.legend()

# Show the plot
plt.grid(True)
plt.savefig('time_curve.png', bbox_inches='tight')