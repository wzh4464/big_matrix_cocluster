"""
File: /coclusterSVD.py
Created Date: Wednesday August 30th 2023
Author: Zihan Wu
-----
Last Modified: Wednesday, 30th August 2023 7:21:10 pm
Modified By: the developer formerly known as Zihan Wu at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
"""

# define class coclusterSVD with only methods (score, scoreHelper)
from numpy import NaN, ndarray, sum, abs, corrcoef, eye, min, zeros
import numpy as np
from numpy.linalg import svd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def scoreHelper(length, C) -> ndarray:
    # check if there is nan in C
    if np.isnan(C).any():
        # throw exception
        raise ValueError("C contains NaN, probably because of constant rows/columns")
    return 1 - 1 / (length - 1) * sum(a=C, axis=1)


def score(X: np.ndarray, subrowI: np.ndarray, subcolJ: np.ndarray) -> float:
    # Compute the compatibility for submatrix X_IJ
    #
    #   X: the data matrix
    #   I: the row cluster assignment; I is boolean array.
    #   J: the column cluster assignment; J is boolean array.

    lenI: np.ndarray = sum(a=subrowI)
    lenJ: np.ndarray = sum(a=subcolJ)

    S1: np.ndarray = abs(
        corrcoef(X[subrowI, :][:, subcolJ], rowvar=False) - eye(lenJ)
    )
    S2: np.ndarray = abs(
        corrcoef(X[subrowI, :][:, subcolJ].T, rowvar=False) - eye(lenI)
    )

    s1 = scoreHelper(length=lenJ, C=S1)
    s2 = scoreHelper(length=lenI, C=S2)

    # cat s1 and s2 into a vector
    s = np.concatenate((s1, s2), axis=0)

    return min(s)


def cocluster(X, scale, k):
    # [U, S, V] = svds(X, scale);
    # s is a vector saving the singular values of X

    U, S, Vh = svd(X, full_matrices=False)

    # [row_idx, row_cluster, row_dist, row_sumd] = kmeans(U, k);
    # [col_idx, col_cluster, col_dist, col_sumd] = kmeans(V, k);
    kmeans_U = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(U)
    kmeans_V = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(Vh.T)
    row_idx = kmeans_U.labels_
    col_idx = kmeans_V.labels_

    # print('row_idx', row_idx)
    # print('col_idx', col_idx)
    scoreMat = compute_scoreMat(k=k, X=X, row_idx=row_idx, col_idx=col_idx)

    return

def compute_scoreMat(k, X, row_idx, col_idx):
    scoreMat = zeros(shape=(k, k)) * NaN
    for i in range(k):
        for j in range(k):
            # if either row_idx == i has less than one element
            # or col_idx == j has less than one element, then skip
            if sum(a=row_idx == i) < 2 or sum(a=col_idx == j) < 2:
                continue
            scoreMat[i, j] = score(X=X, subrowI=row_idx == i, subcolJ=col_idx == j)

    # show the score matrix
    plt.imshow(scoreMat, cmap="hot", interpolation="nearest")
    plt.show()
    
    return scoreMat