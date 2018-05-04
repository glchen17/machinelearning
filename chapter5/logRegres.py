# !/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *

# 加载数据
def loadData():
    dataMat = []
    labelMat = []
    fr = open("testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 为方便计算， X0设置为1
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 文件中的第三列为类别
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 梯度上升优化算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    # transpose()矩阵转置
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    # 步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # 回归系数
    weights = ones((n, 1))
    # 重复maxCycles次
    for k in range(maxCycles):
        # 计算整个数据集的梯度，h和error是都是向量
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        # 更新回归系数向量
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# 随机梯度上升算法，每次只用数据集中的一个样本点来更新系数，顺序遍历整个数据集
def stocGradAscent0(dataMatrix, classLabels):
    # 强制类型转换，避免array和list混用
    dataMatrix = array(dataMatrix)
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        # h和error是数值
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进的随机梯度上升算法, 多加一个迭代次数控制函数
# 在每次迭代中随机地用单个样本点更新数据集，每次迭代，遍历整个数据集
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 强制类型转换，避免array和list混用
    dataMatrix = array(dataMatrix)
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights


# 画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadData()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# Logistic回归分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainingWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    # 可视化
    # plotBestFit(trainingWeights)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainingWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print('the erroe rate of this test is : %f' % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is : %f" % (numTests, errorSum / float(numTests)))





# test
multiTest()
"""
dataArr, labelMat = loadData()
# weights = gradAscent(dataArr, labelMat)
weights = stocGradAscent1(dataArr, labelMat)
#
plotBestFit(weights)
"""



