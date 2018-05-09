# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def loadData(fileName):
    """
    加载数据
    :param fileName:
    :return:
    """
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = [float(word) for word in curLine]
        dataSet.append(fltLine)
    return dataSet


def distEclud(vecA, vecB):
    """
    计算两个数据样本间的距离
    :param vecA:
    :param vecB:
    :return:
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """
    随机选择k个质心
    :param dataSet:
    :param k:
    :return:
    """
    dataMat = np.mat(dataSet)
    n = np.shape(dataMat)[1]
    # 初始化k个质心向量
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        # 对每一个特征，保证生成的随机数在原数据集范围内
        minJ = min(dataMat[:, j])
        rangeJ = float(max(dataMat[:, j]) - minJ)
        # rand生成k个0~1之间的数
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    """
    聚类操作，通过计算每一个样本到k个簇质心的距离按就近原则把每一个样本分类到k个簇
    每个簇的质心通过随机选择初始化，每一次分类后更新质心
    :param dataSet: 数据集
    :param k: 划分为k个簇
    :param distMeans: 求距离的函数
    :param createCent: 生成质心的函数
    :return: k个簇的质心组成的集合
    过程：
    创建k个点作为其实质心（经常是随机选择）
    当任意一个点的簇分配结果发生改变时
        对数据及中的每个数据点
            对每个质心
                计算质心与数据点之间的距离
            将数据点分配到距其最近的簇
        对每一个簇，计算簇中所有点的均值并将均值作为其质心
    """
    dataMat = np.mat(dataSet)
    m = np.shape(dataMat)[0]
    # 簇分配结果矩阵，第一列存储索引，第二列存储误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataMat, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            # 对第I个数据样本
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                # 对第I个数据样本求到第J个质心的距离
                distJI = distMeans(centroids[j, :], dataMat[i, :])
                # print("distJI", distJI)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        # 更新每个簇的质心
        for cent in range(k):
            # 求属于第cent簇的所有样本集合
            ptsInClust = dataMat[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 用所有样本的平均值，表示当前簇的质心
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


# test
dataSet = np.mat(loadData('testSet.txt'))
myCentroids, clustAssing = kMeans(dataSet, 4)

