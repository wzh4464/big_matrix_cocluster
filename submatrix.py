'''
File: /submatrix.py
Created Date: Wednesday August 30th 2023
Author: Zihan
-----
Last Modified: Wednesday, 30th August 2023 10:10:57 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from attr import dataclass
import numpy as np

@dataclass
class submatrix:
    matrix : np.ndarray
    startx : int
    starty : int