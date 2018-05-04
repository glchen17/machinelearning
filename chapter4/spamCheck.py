# !/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import re
from chapter4 import bayes
from numpy import *



# 字符串拆分、小写、去除长度小于3的
def textParse(bigString):
    listOfTokens = re.split('\w+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# spamTest():
def spamTest():
    docList = []
    classList = []
    fullList = []
    for i in range(1, 26):
        # 把spam（垃圾邮件）文件夹下的文本加入到docList、fullList中
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullList.extend(wordList)
        # 垃圾邮件类别为1，calssList加入1
        classList.append(1)
        # 把ham（非垃圾邮件）文件夹下的文本加入到docList、fullList中
        wordList = textParse(open('email/ham/%d.txt' % i, encoding='gb18030', errors='ignore').read())
        docList.append(wordList)
        fullList.extend(wordList)
        # 垃圾邮件类别为0，calssList加入0
        classList.append(0)
    # 根据输入的文档生成包含文档中所有单词的词汇表
    vocabList = bayes.createVocabList(docList)
    # 生成长度为50的列表，元素值为0-49，用作docList列表的索引
    trainingSet = list(range(50))
    # 声明testSet列表
    testSet = []

    # 在trainingSet中任取10个不重复数据的索引加入到测试及
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 把索引对应的trainingSet中的值加入到testSet中
        testSet.append(trainingSet[randIndex])
        # 删除加入到testSet中的索引
        del trainingSet[randIndex]
    # 声明trainMat(训练数据集)、trainClasses(训练数据集的分类列表)
    trainMat = []
    trainClasses = []
    # 给训练数据集、trainClasses添加数据
    for docIndex in trainingSet:
        trainMat.append(bayes.setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 调用训练算法进行训练
    p0V, p1V, pSpam = bayes.trainNB0(array(trainMat), array(trainClasses))
    # 使用测试数据集测试训练后的算法的错误率
    errorCount = 0
    for docIndex in testSet:
        # 对于测试数据，求每一个文档的词条向量
        wordVector = bayes.setOfWords2Vec(vocabList, docList[docIndex])
        # 对每一个词条向量分类并与真实分类进行比较计算错误率
        if bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount) / len(testSet))


# spamTest()
# , encoding='gb18030', errors='ignore'
