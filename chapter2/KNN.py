# !/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import operator


# 构造分类器，用于分类的inX, 训练的样本集dataSet, 标签向量labels, 最近邻居数目k
def classfiy0(inX, dataSet, labels, k):
    # shape[0]返回行数, shape[1]返回列数
    dataSetSize = dataSet.shape[0]
    """1、把当前数据复制成训练集大小，以便同训练集中每一个数据比较"""
    # tile(A, n)将A数组重复n次, 这里是列数不变，行数变dataSetSize行
    # 跟dataset做差，即与每一个训练数据做差（求距离）
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 分别对每一个数据平方
    sqdiffMat = diffMat**2
    # 将矩阵的每一行向量相加
    sqDistance = sqdiffMat.sum(axis=1)
    # 平方根
    distances = sqDistance**0.5
    """2、将比较结果排序"""
    # 返回从小到大排序后的索引值
    sortedDistIndicies = distances.argsort()
    """3、统计最近k个值的类别"""
    # 新建字典，保存最近的K个值分别是什么类别
    classCount = {}
    for i in range(k):
        # 获取第i个值的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        # 统计每个得数目
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    """4、对统计结果拍序"""
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    """5、返回在这K个值中，出现次数最多的类别"""
    return sortedClassCount[0][0]


# 将文本转换成Numpy矩阵
def fileToMatrix(filename):
    # 打开文件
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    # 构建一个全零矩阵用来存储特征信息
    returnMat = zeros((numberOfLines, 3))
    # 构建一个标签数组用来存储特征对应的类别标签
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        # 复制特征信息
        returnMat[index, :] = listFromLine[0:3]
        # 复制类别标签
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化数据
def autoNorm(dataSet):
    # min(0)每一列中的最小值, min(1)每一行中的最小值
    minValues = dataSet.min(0)
    # max(0)每一列中的最大值
    maxValues = dataSet.max(0)
    # 取值范围
    ranges = maxValues - minValues
    # 初始化矩阵
    normDataSet = mat(zeros(shape(dataSet)))
    # print(normDataSet)
    # 返回dataset的行数
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minValues, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minValues


