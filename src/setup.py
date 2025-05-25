"""
File: /setup.py
Created Date: Wednesday August 30th 2023
Author: Zihan Wu
-----
Last Modified: Wednesday, 30th August 2023 12:46:16 pm
Modified By: the developer formerly known as Zihan Wu at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize([]))
