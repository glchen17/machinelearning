# !/usr/bin/env python
# -*- coding: utf-8 -*-
import operator

from chapter4.bayes import *
from chapter4.spamCheck import *
import feedparser

# 返回出现频率最高的30个词
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


# 加载数据，计算贝叶斯的错误率
def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        # NY is class 1
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # 获取出现频率最高的30个词
    top30Words = calcMostFreq(vocabList, fullText)
    # 去掉出现次数最高的30个词，语言中大部分都是冗余和结构辅助性内容，
    # 即出现次数多的中有大量的停用词
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    # 任取20条数据的索引加入到测试数据集中
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    # 构造训练数据集
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        # 词袋模型
        trainMat.append(bagOfWord2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    # 训练算法
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 初始化错误率，并计算算法错误率
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is : ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topSF.append((vocabList[i], p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*")
    for item in sortedNY:
        print(item[0])


# 测试
# y = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
ny = feedparser.parse('https://dalian.craigslist.com.cn/')

print("ny", ny)
# sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
sf = feedparser.parse('https://xian.craigslist.com.cn/')
print("sf", sf)
vocablist, psf, pny = localWords(ny, sf)
