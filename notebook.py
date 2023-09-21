'''
File: /notebook.py
Created Date: Wednesday September 20th 2023
Author: Zihan
-----
Last Modified: Wednesday, 20th September 2023 3:18:09 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

# %% [markdown]
# # Main notebook for simulation experiments

# %% [markdown]
# ## Imports

# %%
import numpy as np
import coclusterSVD as ccSVD
import multiprocessing
from expSetting import generate


# multiprocessing.set_start_method('fork')

np.set_printoptions(precision=3, suppress=True, linewidth=100)

seed = 42
K = 10 # number of biclusters

phi = 100
psi = 100

m = np.ones(K, dtype=int) * phi
n = np.ones(K, dtype=int) * psi

M = 10000
N = 10000

partition : int = 10
ranges = range(50, 200 ,10)

Tp_list = ccSVD.Tp(ranges=ranges, phi=phi, Tm=4, M=M)


# %%
Tm = 4
Tn = 4
Tp = 300

# combine ranges and Tp_list


TpList = ccSVD.TpPair_List(ranges, Tp_list)

print('TpList: ', TpList)
for x in TpList:
    print('Mk: ', x.Mk, 'Tp: ', x.Tp)
    print([x.Mk]*K)
# how to call Tp_cal
# results = [ccSVD.find_bicluster_count(generate(seed=seed, num_bicluster=K, 
#                                       M=M, N=N, m=[x.Mk]*K, n=[x.Mk]*K)[3], 
#                                       x.Tp, Tm, Tn, phi, psi, 1000) for x in TpList]

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
results = pool.starmap(ccSVD.find_bicluster_count, 
                       [(generate(seed=seed, num_bicluster=K, 
                                  M=M, N=N, m=[x.Mk]*K, n=[x.Mk]*K)[3],
                                    x.Tp, Tm, Tn, phi, psi, 100) for x in TpList])
# results = pool.map(Tp_cal, Tp_list)
# results = pool.map(find_bicluster_count, 
#                       [A]*len(Tp_list), Tp_list, [Tm]*len(Tp_list), [Tn]*len(Tp_list),
#                       [sizex]*len(Tp_list), [sizey]*len(Tp_list), [1000]*len(Tp_list))

# %%
for result in results:
    print(result)
