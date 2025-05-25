'''
File: /bicluster.py
Created Date: Wednesday August 30th 2023
Author: Zihan Wu
-----
Last Modified: Wednesday, 6th September 2023 10:41:25 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from attr import dataclass
import numpy as np

@dataclass(eq=False)
class bicluster:
    '''
    row_idx: np.ndarray
    col_idx: np.ndarray
    score: float
    '''
    row_idx: np.ndarray
    col_idx: np.ndarray
    score: float
    # data: np.ndarray
    def __eq__(self, other) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            np.all(self.row_idx == other.row_idx) 
            and
            np.all(self.col_idx == other.col_idx) 
            # and
            # self.score == other.score
        )

    

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