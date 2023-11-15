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
15-11-2023		Zihan	add cocluster_List
1-11-2023		Zihan	use svd to compute the score
1-11-2023		Zihan	esitmateRank
30-10-2023		Zihan	add imageShowBicluster
30-10-2023		Zihan	Add saveNewMat
30-10-2023		Zihan	Add scoreInd
19-09-2023		Zihan	Add Tp to calculate the number of times of re-partitioning
12-09-2023		Zihan	Added tailPar
30-08-2023		Zihan	Documented
30-08-2023	    Zihan   isBiclusterIntersect
30-08-2023	    Zihan   biclusterList
2023-10-10      Zihan	packaged into a module
"""

# define class coclusterSVD with only methods (score, scoreHelper)
from scipy.stats import hypergeom
import big_matrix_cocluster.submatrix as sm
import big_matrix_cocluster.bicluster as bc
from numpy import NaN, ndarray, sum, min, zeros
import numpy as np
from numpy.linalg import svd
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
# from line_profiler import LineProfiler
import time
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

DEBUG = False

# from . import bicluster as bc

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    # solve the relative import problem


class coclusterer:
    matrix: np.ndarray
    M: int
    N: int
    biclusterList: list
    newMat: np.ndarray
    debug: bool = False
    U: np.ndarray
    S: np.ndarray
    Vh: np.ndarray
    scoreMat: np.ndarray

    def __init__(self, matrix: np.ndarray, M: int, N: int, debug: bool = False):
        """
        input:
            matrix: data matrix
            M: number of rows
            N: number of columns
        """
        self.matrix = matrix
        self.M = M
        self.N = N
        self.biclusterList = []
        # set self.newMat not callable yet
        self.newMat = None
        self.debug = debug
        self.calSVD()
        self.scoreMat = None

    def calSVD(self):
        self.U, self.S, self.Vh = svd(self.matrix, full_matrices=False)

    def cocluster_List(self, args):
        tor, k1, k2, atomOrNot = args
        if atomOrNot:
            self._extracted_from_cocluster_5(k1, k2, tor)
        return self

    def cocluster(self, tor: float, k1: int, k2: int, atomOrNot: bool = False):
        if atomOrNot:
            self._extracted_from_cocluster_5(k1, k2, tor)
        return self

    # TODO Rename this here and in `cocluster_List` and `cocluster`
    def _extracted_from_cocluster_5(self, k1, k2, tor):
        startx = 0
        starty = 0
        self.scoreMat = np.zeros(shape=(k1, k2), dtype=float)
        X = sm.submatrix(matrix=self.matrix, startx=startx, starty=starty)
        self.biclusterList = self.coclusterAtom(tor=tor, k1=k1, k2=k2, X=X)

    def printBiclusterList(self, *args, **kwargs):
        # if save:
        if kwargs.get("save", True):
            # save to result/biclusterList.txt
            path = kwargs.get("path", "result/biclusterList.txt")
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                for i in range(len(self.biclusterList)):
                    f.write("bicluster " + str(i) + "\n")
                    f.write("row members " +
                            str(self.biclusterList[i].row_bi_labels) + "\n")
                    f.write("col members " +
                            str(self.biclusterList[i].col_bi_labels) + "\n")
                    f.write("score " + str(self.biclusterList[i].score) + "\n")
                    # print ------
                    f.write("------" + "\n")
        else:
            for i in range(len(self.biclusterList)):
                print("bicluster", i)
                print("row members", self.biclusterList[i].row_bi_labels)
                print("col members", self.biclusterList[i].col_bi_labels)
                print("score", self.biclusterList[i].score)
                # print ------
                print("------")

        return self.biclusterList

    def saveNewMat(self, filename):
        # np.save(filename, self.newMat)
        # if no such path, then create it
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        np.save(filename, self.newMat)

    def isSubmatrixBicluster(self, subrowI: np.ndarray, subcolJ: np.ndarray, i: int, j: int, tor: float = 0.02):
        """
        check if the submatrix X_IJ is a bicluster

        Args:
            X (np.ndarray): the data matrix
            subrowI (np.ndarray): row index of the submatrix
            subcolJ (np.ndarray): column index of the submatrix
            tor (float): ratio of the second largest singular value to the largest singular value

        Returns:
            bool: is the submatrix a bicluster or not
        """

        subX = self.matrix[np.ix_(subrowI, subcolJ)]
        svd = TruncatedSVD(n_components=np.min(subX.shape), random_state=42)
        svd.fit(subX)
        ratio = svd.singular_values_[1]/svd.singular_values_[0]
        # if second largest singular value is small, then return True
        if ratio < tor:
            self.scoreMat[i, j] = ratio
            return True
        else:
            self.scoreMat[i, j] = NaN
            return False

    def coclusterAtom(self, tor, k1, k2, X, PARALLEL=False):
        """
        cocluster the data matrix X, especially for the case when X is a `atom` matrix
        input:
            X: object submatrix
            tor: threshold
            k: number of clusters
        output:
            biclusterList: list of biclusters
        """

        if self.debug:
            # print S
            print("S", self.S)

        # [row_idx, row_cluster, row_dist, row_sumd] = kmeans(U, k);
        # [col_idx, col_cluster, col_dist, col_sumd] = kmeans(V, k);
        kmeans_U = KMeans(n_clusters=k1, random_state=0,
                          n_init="auto").fit((self.U @ np.diag(self.S)))
        kmeans_V = KMeans(n_clusters=k2, random_state=0, n_init="auto").fit(
            (np.diag(self.S) @ self.Vh).T
        )
        row_idx = kmeans_U.labels_
        col_idx = kmeans_V.labels_

        # re-order the row_idx and col_idx
        self.newMat = self.matrix.copy()
        self.newMat = self.newMat[np.ix_(row_idx, col_idx)]

        biclusterList = []
        for i in range(k1):
            for j in range(k2):
                rowIdx = row_idx == i
                colIdx = col_idx == j
                rowTrueNum = sum(a=rowIdx)
                colTrueNum = sum(a=colIdx)
                if rowTrueNum < 2 or colTrueNum < 2:
                    continue
                if self.isSubmatrixBicluster(subrowI=rowIdx, subcolJ=colIdx, tor=tor, i=i, j=j):
                    biclusterList.append(bc.bicluster(
                        row_idx=row_idx == i, col_idx=col_idx == j, score=self.scoreMat[i, j]
                    ))

        return biclusterList

    def imageShowBicluster(self, *args, **kwargs):
        """
        show/save every bicluster in a matrix using plt
        e.g imageShowBicluster(save = True, filename = "result/nturgb.png")
        or  imageShowBicluster(save = False)
        """

        # if no bicluster, then return
        if len(self.biclusterList) == 0:
            return None

        # Set the font size and DPI
        plt.rcParams['font.size'] = 6  # very small font
        plt.rcParams['figure.dpi'] = 300  # very high dpi

        # plt
        canvas = np.zeros(shape=(self.M, self.N))
        for i in range(len(self.biclusterList)):
            canvas[np.ix_(self.biclusterList[i].row_idx,
                          self.biclusterList[i].col_idx)] = i + 1

        # use subplot to show the original matrix and the new matrix
        fig, (ax1, ax2) = plt.subplots(1, 2)
        im1 = ax1.imshow(self.matrix, cmap="hot", interpolation="nearest")
        im2 = ax2.imshow(canvas, cmap="hot", interpolation="nearest")

        # add colorbars
        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)

        plt.show()

        # if save is True, then save the image
        if kwargs.get("save", True):
            filename = kwargs.get("filename", "result/nturgb.png")
            # if no such path, then create it
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            plt.savefig(filename)

        # if save is False, then show the image
        else:
            plt.show()

        return canvas


def scoreHelper(length, C) -> ndarray:
    """
    helper function for score
    $$ s_i = 1 - \frac{1}{n-1} \sum_{j=1}^n |c_{ij}| $$
    """
    # check if there is nan in C
    if np.isnan(C).any():
        # throw exception
        raise ValueError(
            "C contains NaN, probably because of constant rows/columns")
    return 1 - 1 / (length - 1) * sum(a=C, axis=1)


def scoreInd(X: np.ndarray, subrowIind: np.ndarray, subcolJind: np.ndarray) -> float:
    """
    Use subindex to compute the score
    Args:
        X (np.ndarray): the data matrix
        subrowIind (np.ndarray): index version of subrowI 
        subcolJind (np.ndarray): index version of subcolJ 

    Returns:
        float: score

    for example:
    score = scoreInd(X, np.arange(0, 10), np.arange(0, 10))
    """
    subrowI = np.zeros(shape=(X.shape[0],), dtype=bool)
    subcolJ = np.zeros(shape=(X.shape[1],), dtype=bool)
    subrowI[subrowIind] = True
    subcolJ[subcolJind] = True

    return score(X, subrowI, subcolJ)


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

    if DEBUG:
        start = time.time()

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
    subX[:, constant_columns] += 1e-8 * np.random.rand(
        subX.shape[0], np.sum(constant_columns)
    )
    subX[constant_rows, :] += 1e-8 * np.random.rand(
        np.sum(constant_rows), subX.shape[1]
    )

    PEARSON = False

    if PEARSON:
        SS1: np.ndarray = abs(np.corrcoef(subX, rowvar=False) - np.eye(lenJ))
        SS2: np.ndarray = abs(np.corrcoef(subX.T, rowvar=False) - np.eye(lenI))
    else:
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
    score = min(s)

    if DEBUG:
        end = time.time()
        score_time = end - start
        path = "result/score_.txt"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        if score_time > 100:
            submatrix = X[np.ix_(subrowI, subcolJ)]
            # np.save("result/submatrix_0.npy", submatrix)
            # save, if submatrix_0.npy is already exist, then save to submatrix_1.npy
            i = 0
            while os.path.exists("result/submatrix_" + str(i) + ".npy"):
                i += 1
            np.save("result/submatrix_" + str(i) + ".npy", submatrix)
            # save score_time with txt file
            path = "result/score_time" + str(i) + ".txt"

        with open(path, "a") as f:
            f.write("score time: " + str(score_time) + "\n")
            f.write("X.shape: " + str(X.shape) + "\n")
            f.write("subrowI.shape: " + str(np.sum(subrowI)) + "\n")
            f.write("subcolJ.shape: " + str(np.sum(subcolJ)) + "\n")
            f.write("score: " + str(score) + "\n")
            if score_time > 100:
                f.write("submatrix_" + str(i) + ".npy\n")
            f.write("-----\n")

    return score


def estimateRank(X: np.ndarray, subrowI: np.ndarray, subcolJ: np.ndarray, tor1=0.95, tor2=0.99, DEBUG=True) -> tuple:
    """
    Estimate the rank of the submatrix X_IJ
    input:
        X: the data matrix
        I: the row cluster assignment; I is boolean array.
        J: the column cluster assignment; J is boolean array.
    output:
        r: rank of the submatrix
    """
    subX = X[np.ix_(subrowI, subcolJ)]
    svd = TruncatedSVD(n_components=np.min(subX.shape), random_state=42)
    svd.fit(subX)
    acc = np.cumsum(svd.explained_variance_ratio_)
    r1 = np.where(acc > tor1)[0][0]
    r2 = np.where(acc > tor2)[0][0]
    if DEBUG:
        print("r1", r1)
        print("r2", r2)
        print("acc", acc)

    return r1, r2


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
                    raise Exception(
                        "Error: NaN in correlation matrix (corHelper)")

    return S


def compute_scoreMat(k1, k2, X, row_idx, col_idx, PARALLEL=False):
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
    scoreMat = np.zeros(shape=(k1, k2)) * NaN

    def update_counter(counter):
        # with counter.get_lock():
        counter.value += 1

    if PARALLEL:
        with mp.Manager() as manager:
            counter = manager.Value('i', 0)  # Shared counter
            pool = mp.Pool(k1 * k2)

            result = []
            jumped = []

            # Submit tasks
            for i in range(k1 * k2):
                if sum(a=row_idx == i // k2) < 2 or sum(a=col_idx == i % k2) < 2:
                    jumped.append(i)
                    update_counter(counter)  # Update the counter
                    continue
                result.append([
                    pool.apply_async(score, args=(
                        X, row_idx == i // k2, col_idx == i % k2), callback=lambda x: update_counter(counter)),
                    i // k2, i % k2
                ])

            pool.close()

            # Update loop
            with tqdm(total=k1 * k2, desc="Processing", position=0) as pbar:
                while True:
                    completed = counter.value  # Get the current counter value
                    pbar.n = completed  # Update the progress bar
                    pbar.refresh()  # Refresh the progress bar
                    if completed >= k1 * k2:
                        break
                    # Wait for a short time before checking again
                    time.sleep(0.1)

            pool.join()

            for i in range(len(result)):
                scoreMat[result[i][1], result[i][2]] = result[i][0].get()

    else:

        for i in range(k1):
            for j in range(k2):
                # if either row_idx == i has less than one element
                # or col_idx == j has less than one element, then skip
                if sum(a=row_idx == i) < 2 or sum(a=col_idx == j) < 2:
                    continue
                scoreMat[i, j] = score(
                    X=X, subrowI=row_idx == i, subcolJ=col_idx == j)

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
            tailProb[i] = tailParHelper(
                partNum, totalNum, threshold, blockSize[i])
    elif isinstance(blockSize, np.integer) or isinstance(blockSize, int):
        tailProb = tailParHelper(partNum, totalNum, threshold, blockSize)
    else:
        raise TypeError(
            "blockSize should be either a integer or a numpy array")
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


if __name__ == '__main__':
    A = bc.bicluster(row_idx=np.array([True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]), col_idx=np.array(
        [True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]), score=0)
    B = bc.bicluster(row_idx=np.array([True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]), col_idx=np.array(
        [True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]), score=1)
    print(A == B)
