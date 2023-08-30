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
30-08-2023		Zihan	Documented
30-08-2023	    Zihan   isBiclusterIntersect
30-08-2023	    Zihan   biclusterList
"""

# define class coclusterSVD with only methods (score, scoreHelper)
from numpy import NaN, ndarray, sum, abs, corrcoef, eye, min, zeros
import numpy as np
from numpy.linalg import svd
from sklearn.cluster import KMeans
import bicluster as bc


def scoreHelper(length, C) -> ndarray:
    '''
    helper function for score
    $$ s_i = 1 - \frac{1}{n-1} \sum_{j=1}^n |c_{ij}| $$
    '''
    # check if there is nan in C
    if np.isnan(C).any():
        # throw exception
        raise ValueError("C contains NaN, probably because of constant rows/columns")
    return 1 - 1 / (length - 1) * sum(a=C, axis=1)


def score(X: np.ndarray, subrowI: np.ndarray, subcolJ: np.ndarray) -> float:
    '''
    Compute the compatibility for submatrix X_IJ
    input:
        X: the data matrix
        I: the row cluster assignment; I is boolean array.
        J: the column cluster assignment; J is boolean array.
    output:
        s: the compatibility score
    '''
    
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


def coclusterAtom(X, tor, k, M, N) -> list:
    '''
    cocluster the data matrix X, especially for the case when X is a `atom` matrix
    input:
        X: object submatrix
        tor: threshold
        k: number of clusters
    output:
        biclusterList: list of biclusters
    '''

    U, S, Vh = svd(X.matrix, full_matrices=False)

    # [row_idx, row_cluster, row_dist, row_sumd] = kmeans(U, k);
    # [col_idx, col_cluster, col_dist, col_sumd] = kmeans(V, k);
    kmeans_U = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(U)
    kmeans_V = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(Vh.T)
    row_idx = kmeans_U.labels_
    col_idx = kmeans_V.labels_

    # print('row_idx', row_idx)
    # print('col_idx', col_idx)
    scoreMat = compute_scoreMat(k=k, X=X.matrix, row_idx=row_idx, col_idx=col_idx)

    biclusterList = []
    for i in range(k):
        for j in range(k):
            if scoreMat[i, j] < tor:
                # initialize rowIdx to be a boolean array with all False, length = M
                rowIdx = np.zeros(shape=(M,), dtype=bool)
                colIdx = np.zeros(shape=(N,), dtype=bool)
                
                rowIdx[X.startx : X.startx + X.matrix.shape[0]] = row_idx == i
                colIdx[X.starty : X.starty + X.matrix.shape[1]] = col_idx == j
                
                bicluster = bc.bicluster(
                    row_idx=rowIdx,
                    col_idx=colIdx,
                    score=scoreMat[i, j]
                )
                biclusterList.append(bicluster)
    return biclusterList

def compute_scoreMat(k, X, row_idx, col_idx):
    '''
    compute the score matrix
    input:
        k: number of clusters
        X: data matrix
        row_idx: row cluster assignment
        col_idx: column cluster assignment
    output:
        scoreMat: score matrix
    '''
    scoreMat = zeros(shape=(k, k)) * NaN
    for i in range(k):
        for j in range(k):
            # if either row_idx == i has less than one element
            # or col_idx == j has less than one element, then skip
            if sum(a=row_idx == i) < 2 or sum(a=col_idx == j) < 2:
                continue
            scoreMat[i, j] = score(X=X, subrowI=row_idx == i, subcolJ=col_idx == j)

    # show the score matrix
    # plt.imshow(scoreMat, cmap="hot", interpolation="nearest")
    # plt.show()
    
    # print number of each cluster
    # print('number of x clusters')
    # for i in range(k):
    #     print('cluster', i, ':', sum(a=row_idx == i))
    # print('number of y clusters')
    # for i in range(k):
    #     print('cluster', i, ':', sum(a=col_idx == i))
    
    return scoreMat

def isBiclusterIntersectGeneral(bc1 : bc.bicluster, bc2 : bc.bicluster) -> bool:
    '''
    check if two biclusters intersect
    well, in generalized meaning, two biclusters intersect if and only if
    their row_idx and col_idx has at least one common element
    '''
    return (bc1.row_idx & bc2.row_idx).any() and (bc1.col_idx & bc2.col_idx).any()
