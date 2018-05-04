# !/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 标准线性回归
def standRegress(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # linalg:线性代数库；linalg.det：计算行列式
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


# 局部加权线性回归
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 创建对角矩阵
    weights = mat(eye((m)))
    # 权重大小以指数级衰减
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    # linalg:线性代数库；linalg.det：计算行列式，若行列式为0，不可逆
    if linalg.det(xTx) == 0.0:
        print("This Matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


# 局部加权线性回归测试
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


# 绘制标准线性回归结果图
def plotStandRegress(xArr, yArr, ws):
    import matplotlib.pyplot as plt
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    # 按行排序
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


# 绘制局部加权线性回归结果图
def plotlwlr(xArr, yArr):
    import matplotlib.pyplot as plt
    xMat = mat(xArr)
    yMat = mat(yArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    # 平滑参数k=1.0
    ax1 = fig.add_subplot(311)
    yHat = lwlrTest(xArr, xArr, yArr, 1.0)
    ax1.plot(xSort[:, 1], yHat[srtInd])
    ax1.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s=2, c='red')
    # 平滑参数k=0.01
    ax2 = fig.add_subplot(312)
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    ax2.plot(xSort[:, 1], yHat[srtInd])
    ax2.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s=2, c='red')
    # 平滑参数k=0.003
    ax3 = fig.add_subplot(313)
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    ax3.plot(xSort[:, 1], yHat[srtInd])
    ax3.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s=2, c='red')

    plt.show()


# test
xArr, yArr = loadDataSet('ex0.txt')
ws = standRegress(xArr, yArr)
plotStandRegress(xArr, yArr, ws)
plotlwlr(xArr, yArr)











