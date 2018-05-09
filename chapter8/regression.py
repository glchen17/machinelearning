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
    """
    普通最小二乘法，每个数据点的权重相同
    :param xArr: 数据集
    :param yArr: 与数据集对应的样例真实值
    :return: 回归系数
    """
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
    """
    对每个数据点赋予权重，对附近的点赋予更高的权重，类似于支持向量机
    :param testPoint: 测试样例
    :param xArr: 数据集
    :param yArr: 与数据集对应的样例真实值
    :param k: 高斯核平滑参数（附近点的权重，k=1表示每一个点所占权重相同，k越小，用于训练模型的数据越少）
    :return: 测试样例的预测值
    """
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
    # print(type(yHat))
    return yHat


def rssError(yArr, yHatArr):
    """
    求总误差和
    :param yArr: 与数据集对应的样例真实值
    :param yHatArr: 与数据集对应的样例预测值
    :return: 所有样例真实值和预测值之间的差异和
    """
    # sum((y_i - yh_i)^2)，平方：1、误差取正；2、放大误差
    return ((yArr - yHatArr) ** 2).sum()


# 岭回归，lambda(Python保留关键字)缺省值0.2
def ridgeRegress(xMat, yMat, lam=0.2):
    """
    用于处理特征数多于样本数，和在估计中加入偏差
    :param xMat: 数据集（矩阵）
    :param yMat: 与数据集对应的样例真实值（矩阵）
    :param lam: 惩罚项，限制w之和，减少不重要的参数（缩减）
    :return: 回归系数
    """
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    # 如果lambda设为0，可能就不可逆（奇异矩阵）
    if linalg.det(denom) == 0.0:
        print("This Matrix is singular, cannot do inverse")
        return
    # 计算回归系数
    ws = denom.I * (xMat.T * yMat)
    return ws


# 岭回归测试
def ridgeTest(xArr, yArr):
    """
    :param xArr:
    :param yArr:
    :return:
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 数据标准化，1、所有特征减去各自均值
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    # 2、除以各自方差
    xMat = (xMat - xMeans) / xVar
    # 在30个不同的lambda下调用岭回归，lambda以指数级变化
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat, yMat, exp(i-10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):
    """
    数据集数据标准化（归一化）
    :param xMat: 要进行标准化处理的矩阵
    :return: 标准化后的矩阵
    """
    inMat = xMat.copy()
    # mean()计算均值，0压缩行（1压缩列），对各列求均值，返回1*n矩阵
    inMeans = mean(inMat, 0)
    # var()等价于mean(abs(x - x.mean())**2)
    # 计算方差，0压缩行（1压缩列），对各列求方差，返回1*n矩阵
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return inMat

# 前向逐步回归
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """
    :param xArr:
    :param yArr:
    :param eps:
    :param numIt:
    :return: Wbest（转置）的集合
    数据标准化，使其分布满足0均值和单位方差
    在每轮迭代过程中：
        设置当前最小误差lowestError为正无穷
        对每个特点：
            增大或缩小：
                改变一个系数得到一个新的W
                如果误差Error小于当前最小误差lowestError：
                    设置Wbest等于当前的W
            将W设置为新的Wbest
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 数据标准化
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat





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


# 绘制岭回归缩减效果
def plotRidge(ridgeWeights):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.xlabel('log(lambda)')
    plt.show()

# test
"""
xArr, yArr = loadDataSet('ex0.txt')
ws = standRegress(xArr, yArr)
plotStandRegress(xArr, yArr, ws)
plotlwlr(xArr, yArr)

abX, abY = loadDataSet('abalone.txt')
ridgeWeights = ridgeTest(abX, abY)
plotRidge(ridgeWeights)
xArr, yArr = loadDataSet('abalone.txt')
stageWise(xArr, yArr, 0.001, 5000)
"""











