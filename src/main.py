"""
File: /main.py
Created Date: Monday August 28th 2023
Author: Hance Ng
-----
Last Modified: Monday, 28th August 2023 10:18:41 pm
Modified By: the developer formerly known as Hance Ng at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------

29-08-232023	ABC	    Permutaion of 0:49
"""

import numpy as np

## psedo random vector with fixed seed, ori(3x50)
ori = np.random.RandomState(42).rand(3, 50)
a = ori[0]
b = ori[1]
c = ori[2]

# print("a: ", a)
# print("b: ", b)
# print("c: ", c)

n = np.random.RandomState(42).randint(0, 10, 1)[0]
print("n: ", n)

# p1 and p2 are two random permutation of 0:49
p1 = np.random.RandomState(42).permutation(50)
p2 = np.random.RandomState(42).permutation(50)

print("p1: ", p1)
print("p2: ", p2)
