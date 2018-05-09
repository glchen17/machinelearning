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
    # 簇分配结果矩阵，第一列存储簇索引，第二列存储误差
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
        # print(centroids)
        # 更新每个簇的质心
        for cent in range(k):
            # 求属于第cent簇的所有样本集合
            ptsInClust = dataMat[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 用所有样本的平均值，表示当前簇的质心
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def binKMeans(dataSet, k, distMeans=distEclud):
    """
    二分k-均值聚类算法
    :param dataSet:
    :param k:
    :param distMeans:
    :return:
    过程：
    将所有点看成一个簇
    当簇数目小于k时
        对于每一个簇
            计算总误差
            在给定的簇上面进行k-均值聚类（k=2）
            计算将该簇一分为二之后的总误差
        选择使得误差最小的哪个簇进行划分操作
    """
    dataMat = np.mat(dataSet)
    m = np.shape(dataMat)[0]
    # 簇分配结果矩阵，第一列存储簇索引，第二列存储误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 创建一个初始簇
    centroid0 = np.mean(dataMat, axis=0).tolist()[0]
    centList = []
    centList.append(centroid0)
    for i in range(m):
        clusterAssment[i, 1] = distMeans(np.mat(centroid0), dataMat[i, :]) ** 2
    while len(centList) < k:
        # 当簇数目小于设定值k时，尝试对当前的每一个进行二分k-均值划分（贪心策略，只考虑当前）
        # 最终选择误差最小的那个簇进行划分
        lowestSSE = np.inf
        for i in range(len(centList)):
            # 求属于第i簇的所有样本集合
            ptsInCurrCluster = dataMat[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 计算将该簇一分为二之后的总误差
            # 质心矩阵、分配结果矩阵（第一列存储簇索引，第二列存储误差）
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeans)
            # 当前簇划分后总误差
            sseSplit = np.sum(splitClustAss[:, 1])
            # 其他簇的误差
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], :])
            print("sseSplit, and sseNotSplit", sseSplit, sseNotSplit)
            # 若本次划分的总误差 < lowestSSE，则本次划分被保存
            if sseSplit + sseNotSplit < lowestSSE:
                # 更新最佳划分簇索引
                bestCenToSplit = i
                # 更新最佳划分簇质心
                bestNewCents = centroidMat
                # 更新最佳分配策略和误差
                bestClustAss = splitClustAss.copy()
                # 更新最小误差
                lowestSSE = sseSplit + sseNotSplit
        # 因为采用的时二分法，所以分出来的簇只有0或1
        # 我们把新分配的第0簇还设为之前的簇索引，而把新分配的第1簇索引设为len(centList)，放在列表末尾
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCenToSplit
        print("the bestCenToSplit is : ", bestCenToSplit)
        print("the len of bestClustAss is : ", len(bestClustAss))
        # 更新簇列表中被划分的簇（一分为二）中的第0簇的质心
        # bestNewCents[0, :]是一个二维矩阵，虽然只有一行，故需转成list，取第一行数据加入到centList
        centList[bestCenToSplit] = bestNewCents[0, :].tolist()[0]
        # 第1簇的索引为len(centList)，故质心数据放在列表末尾
        centList.append(bestNewCents[1, :].tolist()[0])
        # 把所有属于被划分簇的样本的分配结果更新为现在的bestClustAss
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCenToSplit)[0], :] = bestClustAss
    return np.mat(centList), clusterAssment



import urllib
import json

def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params  # print url_params
    print
    yahooApi
    c = urllib.request.urlopen(yahooApi)
    return json.loads(c.read())


from time import sleep


def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print("error fetching")

        sleep(1)
    fw.close()


def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(vecB[0, 1] * np.pi / 180) * \
        np.cos(np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return np.arccos(a + b) * 6371.0  # pi is imported with numpy


import matplotlib.pyplot as plt


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = binKMeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


"""
# test
dataSet = np.mat(loadData('testSet2.txt'))
myCentroids, clustAssing = binKMeans(dataSet, 3)
print(myCentroids)

"""
print(geoGrab('1 VA center', 'Augusta, ME'))