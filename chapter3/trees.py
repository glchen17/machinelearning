# !/usr/bin/env python
# -*- coding: utf-8 -*-
import operator
from math import log
import pickle


# 计算香农熵
def calcShannonEnt(dataSet):
    """1、计算每个类别的频数"""
    numEntries = len(dataSet)
    # 类别字典，保存不同类别的频数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # 如果当前类别不在字典中，将其加入
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 当前类别数量+1
        labelCounts[currentLabel] += 1
    """2、用香农熵公式计算香农熵"""
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 划分数据集，以axis索引位的特征为根节点
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


# 选择最好的数据集划分形式
def choseBestFeatureToSplit(dataSet):
    # 特征个数， 有一个是类别（去掉）
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 计算以第i个特征作为划分节点时的信息增益，选择信息增益最大的特征作为划分节点
    for i in range(numFeature):
        # 取当前数据集的第i个特征（第i列的所有值）
        featList = [example[i] for example in dataSet]
        # 当前特征的可能取值范围（去重复）
        uniqueValues = set(featList)
        newEntropy = 0.0
        # 计算当前特征的信息增益
        for value in uniqueValues:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 修正最大信息增益，最优划分节点
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 多数表决确定叶子节点的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 递归构建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时，返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = choseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    featVlues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featVlues)
    for value in uniqueValues:
        # 注意分号，复制labels到subLabels，单独开辟了一块内存空间
        # 如果没有分号的则是subLabels指向labels指向的内存
        # 会因修改labels内容而出错
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, fileName):
    try:
        with open(fileName, 'wb') as fw:
            pickle.dump(inputTree, fw)
    except IOError as e:
        print("File Error : " + str(e))


def grabTree(fileName):
    fr = open(fileName, 'rb')
    return pickle.load(fr)


"""
myTree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}, 3: 'maybe'}}
storeTree(myTree, 'classifierStorage.txt')
print(grabTree('classifierStorage.txt'))
"""

