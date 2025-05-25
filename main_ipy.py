###
# File: ./big_matrix_cocluster/main_ipy.py
# Created Date: Sunday, May 25th 2025
# Author: Zihan
# -----
# Last Modified: Sunday, 25th May 2025 11:44:24 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

# %% [markdown]
# # Main notebook for simulation experiments

# %% [markdown]
# ## Imports

# %%
import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt

import sys

sys.path.append("../")

import coclusterSVD as ccSVD
import bicluster as bc
import importlib
import multiprocessing
from expSetting import generate

import expSetting
import coclusterSVD
import bicluster

# multiprocessing.set_start_method('fork')

np.set_printoptions(precision=3, suppress=True, linewidth=100)
# plot math font
plt.rcParams["mathtext.fontset"] = "cm"
# plot dpi
plt.rcParams["figure.dpi"] = 300
# plot width notebook
plt.rcParams["figure.figsize"] = [6.0, 4.0]


# # list findfont
# import matplotlib.font_manager as fm
# fm.findSystemFonts(fontpaths=None, fontext='otf')


# %% [markdown]
# ## Generate Big Matrix ($M \times N$)

# %%
seed = 42
K = 10  # number of biclusters

# random n(0) n(1) n(2) ... n(K-1) as the length of each bicluster base vector
# n = np.random.RandomState(seed=seed).randint(
# num_pool/5, num_pool, K)
# m is the height of biclusters
# m = np.random.RandomState(
# seed=seed+1).randint(num_pool/5, num_pool, K)
phi = 100
psi = 100

m = np.ones(K, dtype=int) * phi
n = np.ones(K, dtype=int) * psi

M = 10000
N = 10000

B, permx, permy, A = generate(seed=seed, num_bicluster=K, M=M, N=N, m=m, n=n)

# print('permx: ', permx)
# print('permy: ', permy)

# plt.imshow(A, cmap='hot', interpolation='nearest')
# plt.show()

# plt.imshow(B, cmap='hot', interpolation='nearest')
# plt.show()


# %% [markdown]
# ## Generate submatrix list (and labelMatList)

# %%
# import SciPy
# from scipy import

partition: int = 10
ranges = range(50, 200, 10)

Tp_list = ccSVD.Tp(ranges=ranges, phi=phi, Tm=4, M=M)


# fit a line
from scipy import stats

logrange = 6

slope, intercept, r_value, p_value, std_err = stats.linregress(
    ranges[0:logrange], np.log(Tp_list[0:logrange])
)
print("slope: ", slope)
print("intercept: ", intercept)
print("r_value: ", r_value)
print("p_value: ", p_value)
print("std_err: ", std_err)

# print x: ranges, y1: log(Tp_list), y2: slope * ranges + intercept ([0:5])
plt.plot(
    ranges,
    np.log(Tp_list),
    "o",
    ranges[0:logrange],
    slope * ranges[0:logrange] + intercept,
    "r",
)
# log scale
# plt.yscale('log')
# x label "size of bicluster"
plt.xlabel("size of bicluster")
# y label "Tp"
plt.ylabel("log(Tp)")

# num on top of each point
for i, txt in enumerate(Tp_list):
    if i > 6:
        continue
    # place higher
    # label = str(x,y)
    label = str("(" + str(ranges[i]) + "," + str(Tp_list[i]) + ")")
    # grid on
    plt.grid(True)
    plt.annotate(
        label,
        (ranges[i], np.log(Tp_list[i])),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=6,
    )

plt.show()

# %%
Tm = 4
Tn = 4
Tp = 300

# combine ranges and Tp_list


TpList = ccSVD.TpPair_List(ranges, Tp_list)

print("TpList: ", TpList)
for x in TpList:
    print("Mk: ", x.Mk, "Tp: ", x.Tp)
    print([x.Mk] * K)
# how to call Tp_cal
# results = [ccSVD.find_bicluster_count(generate(seed=seed, num_bicluster=K, M=M, N=N, m=[x.Mk]*K, n=[x.Mk]*K)[3],
#                                       x.Tp, Tm, Tn, phi, psi, 1000) for x in TpList]

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
results = pool.starmap(
    ccSVD.find_bicluster_count,
    [
        (
            generate(seed=seed, num_bicluster=K, M=M, N=N, m=[x.Mk] * K, n=[x.Mk] * K)[
                3
            ],
            x.Tp,
            Tm,
            Tn,
            phi,
            psi,
            1000,
        )
        for x in TpList
    ],
)
# results = pool.map(Tp_cal, Tp_list)
# results = pool.map(find_bicluster_count, [A]*len(Tp_list), Tp_list, [Tm]*len(Tp_list), [Tn]*len(Tp_list), [sizex]*len(Tp_list), [sizey]*len(Tp_list), [1000]*len(Tp_list))

# %%
for result in results:
    print(result)

# %% [markdown]
# ## Coclustering on each submatrix

# %%
importlib.reload(ccSVD)
importlib.reload(bc)
from joblib import Parallel, delayed

# ans = ccSVD.coclusterAtom(testB, tor=10e-2, k = 15)

# biclusterList = []
# for item in subMatList:
# for item in labelMatList:
# concat the biclusterList
#     biclusterList += ccSVD.coclusterAtom(X=item, tor=10e-5, k = 15, M=M, N=N)

result = Parallel(n_jobs=-1)(
    delayed(ccSVD.coclusterAtom)(X=item, tor=10e-5, k=15, M=M, N=N)
    for item in labelMatList
)
biclusterList = []
for item in result:
    biclusterList += item


# %%
# show all item in result (<class 'list'>)
print("len(result): ", len(result))
for item in result:
    # print('item type: ', type(item))
    print("item len: ", len(item))

# %% [markdown]
# ## research on `result[0]`

# %%
testResult = result[0]
print("testResult type: ", type(testResult))

for i in testResult:
    print("i type: ", type(i))
    print("i", i)

# %%
plt.imshow(labelMatList[0].matrix, cmap="hot", interpolation="nearest")
plt.show()

print("labelZero Matrix: ", labelMatList[0].matrix)

# %% [markdown]
# ## Merge coclustering results

# %%
# while biclusterList not empty
from coclusterSVD import isBiclusterIntersectGeneral

importlib.reload(ccSVD)
importlib.reload(bc)


count = 0
flag = True  # if flag is True, then the item is not merged
while len(biclusterList) > 0:
    count += 1
    item: bc.bicluster = biclusterList.pop(0)
    for item_c in biclusterList:
        # concat the biclusterList
        if ccSVD.isBiclusterIntersectGeneral(bc1=item, bc2=item_c):
            # if True:
            # newRowIdx is OR(rowIdx, rowIdx_c)
            # e.g. rowIdx = [True, False, True], rowIdx_c = [False, True, True], newRowIdx = [True, True, True]
            newRowIdx = np.logical_or(item.row_idx, item_c.row_idx)
            newColIdx = np.logical_or(item.col_idx, item_c.col_idx)
            newScore = ccSVD.score(B, newRowIdx, newColIdx)
            print("newScore: ", newScore)
            # if newScore < item.score + item_c.score or newScore < 0.01:
            print("delta = ", newScore - item.score - item_c.score)
            print("item.score: ", item.score)
            print("item_c.score: ", item_c.score)
            print("-" * 20)
            if newScore < item.score + item_c.score or newScore < 0.1:
                # replace item_c with new bicluster
                biclusterList.remove(item_c)
                biclusterList.append(bc.bicluster(newRowIdx, newColIdx, newScore))
                count = 0
                flag = False
                print("MERGED!")
                break
    if flag:
        biclusterList.append(item)
        flag = True
    if count > 2 * len(biclusterList):
        break

print("biclusterList.length: ", len(biclusterList))


# %%
from coclusterSVD import isBiclusterIntersectGeneral

importlib.reload(ccSVD)
importlib.reload(bc)

# count number of intersection
count = 0
for i in range(len(biclusterList)):
    for j in range(i + 1, len(biclusterList)):
        if ccSVD.isBiclusterIntersectGeneral(
            bc1=biclusterList[i], bc2=biclusterList[j]
        ):
            count += 1

print("count: ", count)


# %%
# print all length of biclusters
ite = 0
print("len(biclusterList): ", len(biclusterList))
resultOneZeroMat = np.zeros((M, N))
for item in biclusterList:
    # print('ite: ', ite)
    print("row length: ", np.sum(item.row_idx))
    # # print('row_size: ', item.row_idx.size)
    print("col length: ", np.sum(item.col_idx))
    # # print('col_size: ', item.col_idx.size)
    print("score: ", item.score)
    # print('row_idx: ', item.row_idx)
    # print('col_idx: ', item.col_idx)
    print("------------------")

    resultOneZeroMat[np.ix_(item.row_idx, item.col_idx)] = 1
    ite += 1

plt.imshow(resultOneZeroMat, cmap="hot", interpolation="nearest")
plt.show()

labelMatrixCut = labelMatrix > 0.5
plt.imshow(labelMatrixCut, cmap="hot", interpolation="nearest")
plt.show()

# print abs(showMat - labelMatrixCut)
# print('diff: ', np.sum(showMat - labelMatrixCut)/M/N)
print("diff: ", np.sum(np.abs(resultOneZeroMat - labelMatrixCut)) / M / N)

# %% [markdown]
# ## TestA

# %%
# cluster1index = permx
testA = A[0 : n[0], 0 : m[0]]
plt.imshow(testA, cmap="hot", interpolation="nearest")
plt.show()
# testI is all true with shape (n[0],)
# testJ is all true with shape (m[0],)
testI = np.ones(n[0], dtype=bool)
testJ = np.ones(m[0], dtype=bool)

testScore = ccSVD.score(testA, testI, testJ)
print("testScore: ", testScore)

# print 10x10 of testA
print("testA: ", testA[0:10, 0:10])


# %%
# m[0]
# n[0]
print("m[0]: ", m[0])
print("n[0]: ", n[0])

# find range(0, n[0]) from permx and save it to IDX_I_one
# find range(0, m[0]) from permy and save it to IDX_J_one

IDX_I_one = permx[0 : n[0]]
IDX_J_one = permy[0 : m[0]]

# show limited to M // partition * N // partition submatrix
IDX_I_one = IDX_I_one[IDX_I_one < M // partition]
IDX_J_one = IDX_J_one[IDX_J_one < N // partition]

print("IDX_I_one: ", IDX_I_one)
print("IDX_J_one: ", IDX_J_one)

# %% [markdown]
# ## Work on one submatrix

# %% [markdown]
# ### Get the small matrix

# %%
testB = subMatList[0]
testB_label = labelMatList[0]

plt.imshow(testB.matrix, cmap="hot", interpolation="nearest")
plt.show()

plt.imshow(testB_label.matrix, cmap="hot", interpolation="nearest")
plt.show()

# %%
ans = ccSVD.coclusterAtom(testB, tor=10e-2, k=15, M=M, N=N)
print("ans: ", ans)

# %% [markdown]
# ## Work on labelMatList[0]

# %%
importlib.reload(ccSVD)
importlib.reload(bc)

# imshow B
plt.imshow(labelMatList[0].matrix, cmap="hot", interpolation="nearest")

U, S, Vh = np.linalg.svd(labelMatList[0].matrix, full_matrices=False)
C = np.dot(U, np.diag(S))

# %%
testLabelMat = labelMatList[0]
ansBiList = ccSVD.coclusterAtom(testLabelMat, tor=10e-2, k=15, M=M, N=N)

# show testLabelMat.matrix == i
for i in range(num_bicluster):
    plt.imshow(testLabelMat.matrix == i, cmap="hot", interpolation="nearest")
    plt.show()


for i in ansBiList:
    print("i: ", i)
    print("row length: ", np.sum(i.row_idx))
    print("col length: ", np.sum(i.col_idx))
    print("i.score: ", i.score)

    showMat = np.zeros(testLabelMat.matrix.shape)
    showMat[np.ix_(i.row_idx, i.col_idx)] = 1
    plt.imshow(showMat, cmap="hot", interpolation="nearest")
    plt.show()


# %%
import pickle

pickle.dump({"result": result, "counts": counts}, open("result.pkl", "wb"))
