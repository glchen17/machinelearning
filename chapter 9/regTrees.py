# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


# 构造树节点
class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left


# 加载数据
def loadDataSet(fileName):
    """
    从文件读取数据集
    :param fileName: 文件名
    :return: 数据集
    """
    dataMat = []
    fr  = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将每行映射成浮点数
        fltLine = [float(word) for word in curLine]
        dataMat.append(fltLine)
    return dataMat


# 二分数据集
def binSplitDataSet(dataSet, feature, value):
    """
    根据提供的value值把数据集dataSet中的feature特征分成两部分
    :param dataSet:
    :param feature: 要划分的特征
    :param value: 划分阈值
    :return: 划分后的而两部分数据集
    """
    # nonzero返回的索引值数组是一个2维tuple数组，如果a是一个二维数组，则索引值数组有两个array
    # 第一个array从行维度来描述索引值
    # 第二个array从列维度来描述索引值。
    # 用nonzero获取feature值大于value的列的行索引,[0]表示行索引
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    """
    :param dataSet:
    :return: 叶节点的模型（目标变量的均值）
    """
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """
    :param dataSet:
    :return: 目标变量的平方误差
    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    选择最优二元切分策略
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops:
    :return: 最佳切分的特征和阈值
    对每个特征：
        对每个特征值：
            将数据及切分成两份
            计算切分的误差
            如果当前误差小于最小误差，那么将当前切分设定为最佳切分并更新最小误差
    返回最佳切分的特征和阈值
    """
    # 容许的误差下降至
    tolS = ops[0]
    # 切分的最小样本数
    tolN = ops[1]
    # 如果特征值的数目为1，则不再切分，返回None和叶子节点
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    # 初始化最佳切分变量
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    # 对每一个特征
    for featIndex in range(n - 1):
        # 原文的代码会出现TypeError: unhashable type: 'matrix'
        # 需要转成list[0]
        # 对每一个特征值（去重复）
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            # 尝试切分
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果切分后的子集过小，放弃切分
            if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
                continue
            # 保存当前特征值的切分结果
            newS = errType(mat0) + errType(mat1)
            # 如果当前的切分效果比之前的最佳切分策略的效果更好，更新最佳切分策略
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果切分数据集后效果提升不够大，放弃切分，返回None和叶子节点
    if S - bestS < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分后的子集过小，放弃切分，返回None和叶子节点
    if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 递归创建树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 选择最优划分特征和划分阈值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 如果不可分，直接返回阈值当作叶节点，递归返回（退出条件）
    if feat == None:
        return val
    retTree = {}
    # 划分特征
    retTree['spInd'] = feat
    # 划分阈值
    retTree['spVal'] = val
    # 左子树的数据集、右子树的数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 递归生成左子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    # 递归生成右子树
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# test
myData = loadDataSet('ex2.txt')
myMat = np.mat(myData)
print(createTree(myMat, ops=(10000, 4)))