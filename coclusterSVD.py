'''
File: /coclusterSVD.py
Created Date: Wednesday August 30th 2023
Author: Zihan Wu
-----
Last Modified: Wednesday, 30th August 2023 3:54:20 pm
Modified By: the developer formerly known as Zihan Wu at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

# define class coclusterSVD with only methods (score, scoreHelper)
from numpy import sum, abs, corrcoef, eye, min
from numpy.linalg import svd
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

def scoreHelper(length, C):
    return 1 - 1 / (length - 1) * sum(C, axis=1)

def score(X, subrowI : list, subcolJ : list):
    # Compute the compatibility for submatrix X_IJ
    #
    #   X: the data matrix
    #   I: the row cluster assignment; I is boolean array.
    #   J: the column cluster assignment; J is boolean array.

    lenI = sum(subrowI)
    lenJ = sum(subcolJ)

    S1 = abs(corrcoef(X[subrowI, :][:, subcolJ]) - eye(lenJ))
    S2 = abs(corrcoef(X[subrowI, :][:, subcolJ].T) - eye(lenI))

    return min([scoreHelper(lenJ, S1), scoreHelper(lenI, S2)])

def cocluster(X, scale, k):
    # [U, S, V] = svds(X, scale);
    # s is a vector saving the singular values of X
    
    U, S, Vh = svd(X, full_matrices=False)
    
    # [row_idx, row_cluster, row_dist, row_sumd] = kmeans(U, k);
    # [col_idx, col_cluster, col_dist, col_sumd] = kmeans(V, k); 
    kmeans_U = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(U)
    kmeans_V = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(Vh.T)
    row_idx = kmeans_U.labels_
    col_idx = kmeans_V.labels_
    
    print('row_idx', row_idx)
    print('col_idx', col_idx)
    
    return