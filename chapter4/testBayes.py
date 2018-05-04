# !/usr/bin/env python
# -*- coding: utf-8 -*-
import re

from numpy import *
from chapter4 import bayes

def testingNB():
    listOPosts, listClasses = bayes.loadDataSet()
    # 创建一个包含所有词的列表
    myVocabList = bayes.createVocabList(listOPosts)
    # print(myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(bayes.setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', bayes.classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['garbage', 'stupid']
    thisDoc = array(bayes.setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', bayes.classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()

"""
mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
# listOfTokens = mySent.split()
listOfTokens = re.split('\W+', mySent)
print(listOfTokens)
"""
