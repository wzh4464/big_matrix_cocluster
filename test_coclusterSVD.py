'''
File: /test_coclusterSVD.py
Created Date: Wednesday August 30th 2023
Author: Zihan Wu
-----
Last Modified: Tuesday, 5th September 2023 3:50:04 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
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
    X = np.concatenate((a, a, a), axis=1)
    
    I: list[bool] = [True, False, True]
    J: list[bool] = [True, True, True]
    
    assert abs(score(X, I, J)) < 1e-10
    
if __name__ == "__main__":
    test_score()