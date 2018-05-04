# !/usr/bin/env python
# -*- coding: utf-8 -*-
from chapter7.adaboosting import *


# 加载数据
def loadDataSet(fileName):
    # 计算特征个数（如果用下面的fr来读，那么训练集和测试集中就少了一个样本数据）
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            # 不能split()后直接append,应该先用float格式化数据，统一数据类型
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# test
dataMat, classLabels = loadDataSet("horseColicTraining2.txt")
testMat, testLabels = loadDataSet('horseColicTest2.txt')
classifierArr = adaBoostTrainDS(dataMat, classLabels, 50)
predicted = adaClassify(testMat, classifierArr)
errArr = mat(ones((67, 1)))
errCount = errArr[predicted != mat(testLabels).T].sum()
print("错误率： ", errCount / 67)