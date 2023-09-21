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

19-09-2023		Zihan	Add Tp to calculate the number of times of re-partitioning
12-09-2023		Zihan	Added tailPar
30-08-2023		Zihan	Documented
30-08-2023	    Zihan   isBiclusterIntersect
30-08-2023	    Zihan   biclusterList
"""

# define class coclusterSVD with only methods (score, scoreHelper)
from numpy import NaN, ndarray, sum, min, zeros
import numpy as np
from numpy.linalg import svd
from sklearn.cluster import KMeans
import bicluster as bc
# import matplotlib.pyplot as plt
from scipy.stats import hypergeom

def scoreHelper(length, C) -> ndarray:
    """
    helper function for score
    $$ s_i = 1 - \frac{1}{n-1} \sum_{j=1}^n |c_{ij}| $$
    """
    # check if there is nan in C
    if np.isnan(C).any():
        # throw exception
        raise ValueError("C contains NaN, probably because of constant rows/columns")
    return 1 - 1 / (length - 1) * sum(a=C, axis=1)


def score(X: np.ndarray, subrowI: np.ndarray, subcolJ: np.ndarray) -> float:
    """
    Compute the compatibility for submatrix X_IJ
    input:
        X: the data matrix
        I: the row cluster assignment; I is boolean array.
        J: the column cluster assignment; J is boolean array.
    output:
        s: the compatibility score
    """
    lenI = sum(a=subrowI)
    if not isinstance(lenI, np.integer):
        raise TypeError("Expected an integer value for lenI")
    lenJ = sum(a=subcolJ)
    if not isinstance(lenJ, np.integer):
        raise TypeError("Expected an integer value for lenJ")

    # See each column as a vector,
    # and compute the correlation between each pair of columns
    subX = X[np.ix_(subrowI, subcolJ)]

    # Adding a small noise to constant columns
    std_devs = np.std(subX, axis=0)
    std_devs_y = np.std(subX.T, axis=0)
    constant_columns = std_devs == 0
    constant_rows = std_devs_y == 0
    subX[:, constant_columns] += 1e-15 * np.random.rand(
        subX.shape[0], np.sum(constant_columns)
    )
    subX[constant_rows, :] += 1e-15 * np.random.rand(
        np.sum(constant_rows), subX.shape[1]
    )

    # SS1: np.ndarray = abs(corrcoef(subX, rowvar=False) - eye(lenJ))
    # SS2: np.ndarray = abs(corrcoef(subX.T, rowvar=False) - eye(lenI))
    SS1 = zeros(shape=(subX.shape[1], subX.shape[1]))
    for i in range(subX.shape[1]):
        for j in range(subX.shape[1]):
            if i == j:
                SS1[i, j] = 0
            else:
                x1 = subX[:, i]
                x2 = subX[:, j]
                SS1[i, j] = np.exp(
                    -np.linalg.norm(x1 - x2) ** 2
                    / (2 * np.linalg.norm(x1) * np.linalg.norm(x2))
                )

    SS2 = zeros(shape=(subX.shape[0], subX.shape[0]))
    for i in range(subX.shape[0]):
        for j in range(subX.shape[0]):
            if i == j:
                SS2[i, j] = 0
            else:
                x1 = subX[i, :]
                x2 = subX[j, :]
                SS2[i, j] = np.exp(
                    -np.linalg.norm(x1 - x2) ** 2
                    / (2 * np.linalg.norm(x1) * np.linalg.norm(x2))
                )
    # Pearson correlation fails when there is constant vectors
    # Thus check the NaN and redo the computation

    S1 = corHelper(SS1, subX)
    S2 = corHelper(SS2, subX.T)

    s1 = scoreHelper(length=lenJ, C=S1)
    s2 = scoreHelper(length=lenI, C=S2)

    # cat s1 and s2 into a vector
    s = np.concatenate((s1, s2), axis=0)

    return min(s)


def corHelper(SS, X):
    """
    helper function for correlation matrix
    """
    isNaNMat = np.isnan(SS)
    S = SS
    # recomputer the correlation matrix element if it is NaN
    for i in range(isNaNMat.shape[0]):
        for j in range(isNaNMat.shape[1]):
            if isNaNMat[i, j]:
                # let flag1 & flag2 tells if the ith column and jth column are constant
                flag1 = np.all(X[:, i] == X[:, i][0])
                flag2 = np.all(X[:, j] == X[:, j][0])
                if flag1 and flag2:
                    if i == j:
                        S[i, j] = 0
                    else:
                        S[i, j] = 1
                elif flag1 or flag2:
                    S[i, j] = 0
                else:
                    raise Exception("Error: NaN in correlation matrix (corHelper)")

    return S


def coclusterAtom(X, tor, k, M, N) -> list:
    """
    cocluster the data matrix X, especially for the case when X is a `atom` matrix
    input:
        X: object submatrix
        tor: threshold
        k: number of clusters
    output:
        biclusterList: list of biclusters
    """

    U, S, Vh = svd(X.matrix, full_matrices=False)

    # [row_idx, row_cluster, row_dist, row_sumd] = kmeans(U, k);
    # [col_idx, col_cluster, col_dist, col_sumd] = kmeans(V, k);
    kmeans_U = KMeans(n_clusters=k, random_state=0, n_init="auto").fit((U @ np.diag(S)))
    kmeans_V = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(
        (np.diag(S) @ Vh).T
    )
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
                    row_idx=rowIdx, col_idx=colIdx, score=scoreMat[i, j]
                )
                biclusterList.append(bicluster)
    return biclusterList


def compute_scoreMat(k, X, row_idx, col_idx):
    """
    compute the score matrix
    input:
        k: number of clusters
        X: data matrix
        row_idx: row cluster assignment
        col_idx: column cluster assignment
    output:
        scoreMat: score matrix
    """
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


def isBiclusterIntersectGeneral(bc1: bc.bicluster, bc2: bc.bicluster):
    """
    check if two biclusters intersect
    well, in generalized meaning, two biclusters intersect if and only if
    their row_idx and col_idx has at least one common element
    """
    # sameRow = (bc1.row_idx & bc2.row_idx).any()
    # sameCol = (bc1.col_idx & bc2.col_idx).any()
    # print("sameRow", sameRow)
    # print("sameCol", sameCol)
    return (bc1.row_idx & bc2.row_idx).any() or (bc1.col_idx & bc2.col_idx).any()

def tailParHelper(partNum, totalNum, threshold, blockSize):
    '''
    Compute the tail probability estimation Helper
    input:
        partNum: number of partitions           $M^{(k)}$
        totalNum: total number of elements      $M$
        threshold: threshold                    $T_m$
        blockSize: block size                   $\phi$
        
    par:
        s:                                      $s_i^{(k)} 
                                                    = \cfrac{M^{(k)}}{M}
                                                        -\cfrac{T_m-1}{\phi_i}$
    
    output:
        tailProb: tail probability estimation   $\exp(-2 (s^{(k)} )^2 \phi)$
    '''
    
    s = partNum / totalNum - (threshold - 1) / blockSize
    return np.exp(-2 * (s ** 2) * blockSize)

def tailPar(partNum, totalNum, threshold, blockSize):
    '''
    Compute the tail probability estimation
        if blocksize is a array, 
        then compute the tail probability estimation for each blocksize
        if blocksize is a integer,
        then compute directly using the Helper function
    
    input:
        partNum: number of partitions           $M^{(k)}$
        totalNum: total number of elements      $M$
        threshold: threshold                    $T_m$
        blockSize: block size                   $\phi$
            
    output:
        tailProb: tail probability estimation
    '''
    
    if isinstance(blockSize, np.ndarray):
        tailProb = np.zeros(shape=(blockSize.shape[0],))
        for i in range(blockSize.shape[0]):
            tailProb[i] = tailParHelper(partNum, totalNum, threshold, blockSize[i])
    elif isinstance(blockSize, np.integer) or isinstance(blockSize, int):
        tailProb = tailParHelper(partNum, totalNum, threshold, blockSize)
    else:
        raise TypeError("blockSize should be either a integer or a numpy array")
    return tailProb

def isBiclusterFoundConst(A, Tp, Tm, Tn, phi, psi, label):
    '''
    This function tells after Tp times of re-partitioning, 
    will the bicluster labeled by `label` be found
    
    input:
        A: data matrix
        Tp: number of times of re-partitioning
        Tm: threshold for row
        Tn: threshold for column
        phi: block size x
        psi: block size y
        label: label of the bicluster
        
    output:
        number of times of re-partitioning that the bicluster is found
    '''
    M, N = A.shape
    
    # if phi and psi are integers, then convert them to numpy array
    if isinstance(phi, np.integer) or isinstance(phi, int):
        numBlockx = M // phi 
        numBlocky = N // psi 
        phi = np.ones(shape=(numBlockx,)) * phi
        psi = np.ones(shape=(numBlocky,)) * psi
        
        # intize phi and psi
        phi = phi.astype(int)
        psi = psi.astype(int)
    
    
    
    for i in range(Tp):
        count = 0
        # re-partitioning
        permx = np.random.permutation(M)
        permy = np.random.permutation(N)
        # print i and --- to show the progress
        # print(i, end="---")
        for j in range(phi.shape[0]):
            for k in range(psi.shape[0]):
                # partition the matrix
                # subA = A[permx[j*phi[j]:(j+1)*phi[j]], permy[k*psi[k]:(k+1)*psi[k]]]
                subA = A[np.ix_(permx[j*phi[j]:(j+1)*phi[j]], 
                                permy[k*psi[k]:(k+1)*psi[k]])]
                # compute the number of partitions
                # plt.imshow(subA, cmap="hot", interpolation="nearest")
                # plt.show()
                
                # MATLAB: ijCollection = find(subA == label);
                ijCollection = np.where(subA == label)
                # print(ijCollection)
                # if no element in subA is label `label`, then skip
                if ijCollection[0].shape[0] == 0:
                    continue
                
                count += ijCollection[0].shape[0]
                
                # TmTest is number of rows that subA has label `label`
                # MATLAB TmTest = size(unique(ijCollection(:, 1)), 1);
                TmTest = np.unique(ijCollection[0]).shape[0]
                TnTest = np.unique(ijCollection[1]).shape[0]
                
                # print("TmTest", TmTest)
                # print("TnTest", TnTest)
                
                if TmTest >= Tm and TnTest >= Tn:
                    # print("found at i =", i, "j =", j, "k =", k)
                    # print("M =", M, "N =", N)
                    # print("phi =", phi, "psi =", psi)
                    # print("Tm =", Tm, "Tn =", Tn)
                    # print("Tp =", Tp)
                    return i + 1
        # print("count", count)
        # print("i", i)
    return -1

def Tp(ranges: range, phi=100, Tm=4, M=1000):
    '''
    compute the number of times of re-partitioning theoretically
    input:
        ranges: range of block size
        phi: block size (assume phi = psi)
        Tm: threshold for row (assume Tm = Tn)
        M: number of rows/columns (assume M = N)
    output:
        number of times of re-partitioning
    '''
    m = M / phi
    
    def q(Mk):
        return hypergeom.cdf(Tm - 1, M, phi, Mk)
    
    def qq(Mk):
        return (1 - (1 - q(Mk)) ** 2) ** (m ** 2)
    
    def Tp_Mk(Mk):
        return np.log(0.01) / np.log(qq(Mk))

    # ceil and convert to int
    return np.ceil([Tp_Mk(Mk) for Mk in ranges]).astype(int)

def find_bicluster_count(A, Tp, Tm, Tn, sizex, sizey, num_iter=100):
    '''
    compute the number of times of re-partitioning experimentally
    input:
        A: data matrix
        Tp: number of times of re-partitioning
        Tm: threshold for row
        Tn: threshold for column
        sizex: block size x
        sizey: block size y
        num_iter: number of iterations
    output:
        ratio: ratio of times of re-partitioning that the bicluster is found
        number of times of re-partitioning that the bicluster is found
    '''
    count = 0
    result = []
    for i in range(num_iter):
        ite = isBiclusterFoundConst(A, Tp, Tm, Tn, sizex, sizey, 10)
        result.append(ite)
        if ite != -1:
            count += 1
    return count/num_iter, result

class TpPair_List:
    def __init__(self, ranges, Tp_list):
        self.TpList = []
        if len(ranges) != len(Tp_list):
            raise ValueError("len(ranges) != len(Tp_list)")
        if len(ranges) == 0:
            raise ValueError("len(ranges) == 0")
        for i in range(len(ranges)):
            self.TpList.append(TpPair(ranges[i], Tp_list[i]))
            
    def __getitem__(self, index):
        return self.TpList[index]
    
    def __len__(self):
        return len(self.TpList)

class TpPair:
    def __init__(self, Mk, Tp):
        self.Mk = Mk
        self.Tp = Tp

    def __str__(self):
        return "Mk: " + str(self.Mk) + " Tp: " + str(self.Tp)

    def __repr__(self):
        return "Mk: " + str(self.Mk) + " Tp: " + str(self.Tp)

    def __getitem__(self, index):
        return (self.Mk[index], self.Tp[index])

    def __len__(self):
        return len(self.Mk)

if __name__ == "__main__":
    Tp_list = Tp(range(10, 24, 2), phi=100, Tm=7, M=1000)
    print(Tp_list)