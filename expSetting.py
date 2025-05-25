'''
File: /expSetting.py
Created Date: Wednesday September 20th 2023
Author: Zihan
-----
Last Modified: Wednesday, 20th September 2023 12:45:22 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

import numpy as np
# import matplotlib.pyplot as plt
import time

np.set_printoptions(precision=3, suppress=True, linewidth=100)

def generate(num_bicluster, M, N, m, n, seed=42):
    '''
    Generate a matrix with biclusters
    
    m: list of row sizes of biclusters
    n: list of column sizes of biclusters
    '''
    time_start = time.time()

    A = np.zeros((M, N))

    # check: len(num_bicluster) == len(m) == len(n)
    assert num_bicluster == len(m) == len(n)
    
    # insert biclusters into A disjointly
    startx = starty = 0
    for i in range(num_bicluster):
        A[startx:startx+m[i], starty:starty+n[i]] = i+1
        startx += m[i]
        starty += n[i]

    timeA = time.time() - time_start
    print('timeA: ', timeA)

    # permutation of M and N
    permx = np.random.RandomState(seed).permutation(M)
    permy = np.random.RandomState(seed+1).permutation(N)

    # print permx and permy
    # print('permx: ', permx)
    # print('permy: ', permy)

    # permute A
    B = A[permx, :]
    B = B[:, permy]
    
    timeB = time.time() - time_start - timeA
    print('timeB: ', timeB)

    return B, permx, permy, A