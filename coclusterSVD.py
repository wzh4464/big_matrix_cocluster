'''
File: /coclusterSVD.py
Created Date: Wednesday August 30th 2023
Author: Zihan Wu
-----
Last Modified: Wednesday, 30th August 2023 3:31:42 pm
Modified By: the developer formerly known as Zihan Wu at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

# define class coclusterSVD with only methods (score, scoreHelper)
from numpy import sum, abs, corrcoef, eye, min
from numpy.linalg import svd

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
    
    U, S, V = svd(X, full_matrices=False)
    print('changed')
    return S