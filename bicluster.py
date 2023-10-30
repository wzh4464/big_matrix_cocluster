'''
File: /bicluster.py
Created Date: Wednesday August 30th 2023
Author: Zihan Wu
-----
Last Modified: Monday, 30th October 2023 4:55:30 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------

30-10-2023		Zihan	Add idxtoLabel
'''

from attr import dataclass
import numpy as np


@dataclass(eq=False, init=False)
class bicluster:
    '''
    row_idx: np.ndarray (bool)
    col_idx: np.ndarray (bool)
    score: float
    row_bi_labels: np.ndarray (int)
    col_bi_labels: np.ndarray (int)
    '''
    row_idx: np.ndarray
    col_idx: np.ndarray
    score: float
    # data: np.ndarray

    def __init__(self, row_idx, col_idx, score):
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.score = score
        if not self.idxtoLabel():
            raise ValueError("row_idx and col_idx must be np.ndarray")

    def __eq__(self, other) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            np.all(self.row_idx == other.row_idx)
            and
            np.all(self.col_idx == other.col_idx)
        )

    def idxtoLabel(self):
        # if self has no attribute row_idx
        if not hasattr(self, 'row_idx'):
            return False
        if not hasattr(self, 'col_idx'):
            return False

        self.row_bi_labels = []
        self.col_bi_labels = []
        for i in range(self.row_idx.shape[0]):
            if self.row_idx[i]:
                self.row_bi_labels.append(i)
        for i in range(self.col_idx.shape[0]):
            if self.col_idx[i]:
                self.col_bi_labels.append(i)

        return True


if __name__ == "__main__":
    # test bicluster
    bi = bicluster(np.array([True, False, True]),
                   np.array([True, False, False]), 1.0)
    bi.idxtoLabel()
    print(bi.row_bi_labels)
    print(bi.col_bi_labels)
