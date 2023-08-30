'''
File: /test_coclusterSVD.py
Created Date: Wednesday August 30th 2023
Author: Zihan Wu
-----
Last Modified: Wednesday, 30th August 2023 10:33:30 am
Modified By: the developer formerly known as Zihan Wu at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from coclusterSVD import score

import numpy as np

def test_score():
    # X = [a, a, a]
    # a = rand(3, 1)
    a = np.random.rand(3, 1)
    X = np.concatenate((a, 0.9 * a, 0.8 * a), axis=1)
    
    I: list[bool] = [True, True, True]
    J: list[bool] = [True, True, True]
    
    assert score(X, I, J) == 0.0