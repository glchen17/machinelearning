# !/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *

mat = tile([2, 2], (2, 1)) - [[0, 0], [1, 0]]
print(shape(mat))
print(mat)
sqMat = mat**2
print(sqMat)
sqDis = sqMat.sum(axis=1)
print(sqDis)
dis = sqDis**0.5
print(dis)
sortedDis = dis.argsort()
print(sortedDis)

# numpy的矩阵，总是用元组的最后一位表示列
print(zeros((2, 2)))
