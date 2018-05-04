# !/usr/bin/env python
# -*- coding: utf-8 -*-

from chapter2.KNN import *


# 分类器针对约会网站分类
def datingClass():
    hoRatio = 0.10
    datingDataMat, datingLabels = fileToMatrix("datingTestSet2.txt")
    normMat, ranges, minValues = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classfiy0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is : %f" % (errorCount / float(numTestVecs)))

datingClass()


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingMat, datingLabels = fileToMatrix("datingTestSet2.txt")
    normMat, ranges, minValues = autoNorm(datingMat)
    inArry = [ffMiles, percentTats, iceCream]
    classifierResult = classfiy0(inArry, datingMat, datingLabels, 3)
    print("You will probably like this person: " + resultList[classifierResult - 1])


classifyPerson()




