# !/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *

# 加载数据
def loadData(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# i是alpha的下标，m是alpha的数目
def selectJrand(i, m):
    j = i
    # 随机选择一个j，不同于i
    while(j == i):
        j = int(random.uniform(0, m))
    return j


# 调整过大或过小的alpha（alpha需要满足一定的约束条件；0<=alpha<=C sum(alpha*y)=0）
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj



# 简化版的SMO算法
# 参数：数据集，类别标签，常数C，容错率，最大循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 数据处理，转换成mat
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    # 没有任何alpha优化的情况下遍历数据集的次数
    while(iter < maxIter):
        # 记录alpha优化次数
        alphaPairsChanged = 0
        # 遍历整个数据集
        for i in range(m):
            # 预测第i个实例的类别（multiply: 对应元素相乘，因为是矩阵，*表示dot(.*)）
            fxi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 预测值与真实值之间的误差
            Ei = fxi - float(labelMat[i])
            # 如果误差很大（toler容错率），且alpha可优化，则应该对该实例对应的alpha值优化
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 选择另一个要改变的alpha
                j = selectJrand(i, m)
                # 预测第j个实例的类别（multiply: 对应元素相乘，因为是矩阵，*表示dot(.*)）
                fxj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                # 预测值与真实值之间的误差
                Ej = fxj - float(labelMat[j])
                # 保存i和j之前的alpha值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 保证alpha在0和C之间
                # 第i个实例的类别和第j个实例的类别不同
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                # 若相同
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                # 最优修改量
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                # 计算新的alphas[j]
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 如果alphas[j]有轻微的改变
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # 修改alphas[i]，修改两与alphas[j]相同，但方向相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 给这两个值设置一个常数项
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[j] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter : %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
        # 如果所有的alpha都没有改变
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number : %d " % iter)
    return b, alphas


# test
dataArr, labelArr = loadData('testSet.txt')
b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)