# !/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *

from chapter4.dataAnalysis import lineplot


# 创建实验样本
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 0代标非侮辱性言论， 1代标侮辱性言论
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 创建一个包含在所有文档中出现的不重复的词表
def createVocabList(dataSet):
    # 创建一个空集
    vocabSet = set([])
    # 将每篇文档返回的新词集合添加到该集合中
    for document in dataSet:
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


""" 词集模型，每个词只能出现一次，以每个词的出现与否作为一个特征"""
# vocabList 词汇表, inputList 文档， 返回值 文档向量
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个其中所含元素都为0的向量，表示词汇表中的单词是否在文档中出现
    returnVec = [0] * len(vocabList)
    # 遍历文档中的所有单词，检查是否出现在词汇表中
    for word in inputSet:
        if word in vocabList:
            # 在词汇表中出现，标记为1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    # 返回文档向量，
    return returnVec


# 训练朴素贝叶斯算法，trainMatrix 文档矩阵；trainCategory 文档类别标签所构成的向量
def trainNB0(trainMatrix, trainCategory):
    # 计算文档数目和第一个文档中词条数
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 求在训练集中任取一个文档是侮辱性（trainCategory=1）的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化，没有用p0Num = zeros(numWords)，是为了避免某一个概率值为0，
    # 使得最后的乘积也是0，即使变成log()，log(0)也是不对的
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 相应地 p0Denom = 0.0 修改为 p0Denom = 2.0
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 如果是侮辱性言论，
        if trainCategory[i] == 1:
            # 把当前文档的词条向量加到p1Num上，p1Num：侮辱性言论中每个单词出现次数
            p1Num += trainMatrix[i]
            # p1Denom：侮辱性言论中总单词数
            p1Denom += sum(trainMatrix[i])
        # 非侮辱性
        else:
            # 把当前文档的词条向量加到p0Num上，p0Num：非侮辱性言论中每个单词出现次数
            p0Num += trainMatrix[i]
            # p0Denom：非侮辱性言论中总单词数
            p0Denom += sum(trainMatrix[i])
    # 对每个元素做除法（log把乘变成加避免下溢出）
    p1Vect = log(p1Num / p1Denom)
    # print(p1Vect)
    p0Vect = log(p0Num / p0Denom)
    # print(p0Vect)
    # lineplot(p0Num, p0Vect)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # p1：是侮辱性文档的概率，对每一个单词累加log()概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    # p0：是非侮辱性文档的概率
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    # 如果p1 > p0，侮辱性文档，反之，非
    if p1 > p0:
        return 1
    else:
        return 0


"""如果一个词在文档中出现不止一次，这可能意味着包含盖茨是否出现在文档中所不能表达的信息"""
# 朴素贝叶斯词袋模型，每个词可以出现多次
def bagOfWord2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec










