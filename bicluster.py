'''
File: /bicluster.py
Created Date: Wednesday August 30th 2023
Author: Zihan Wu
-----
Last Modified: Wednesday, 30th August 2023 9:01:09 pm
Modified By: the developer formerly known as Zihan Wu at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from attr import dataclass
import numpy as np

@dataclass
class bicluster:
    row_idx: np.ndarray
    col_idx: np.ndarray
    score: float
    data: np.ndarray

    

# from dataclasses import dataclass
 
# # A class for holding an employees content
# @dataclass
# class employee:
 
#     # Attributes Declaration
#     # using Type Hints
#     name: str
#     emp_id: str
#     age: int
#     city: str
 
 
# emp1 = employee("Satyam", "ksatyam858", 21, 'Patna')
# emp2 = employee("Anurag", "au23", 28, 'Delhi')
# emp3 = employee("Satyam", "ksatyam858", 21, 'Patna')
 
# print("employee object are :")
# print(emp1)
# print(emp2)
# print(emp3)
 
# # printing new line
# print()
 
# # referring two object to check equality
# print("Data in emp1 and emp2 are same? ", emp1 == emp2)
# print("Data in emp1 and emp3 are same? ", emp1 == emp3)