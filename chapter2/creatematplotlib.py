# !/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from chapter2.KNN import *

datingDataMat, datingLabels = fileToMatrix("datingTestSet2.txt")


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()