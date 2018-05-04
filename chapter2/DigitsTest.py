# !/usr/bin/env python
# -*- coding: utf-8 -*-
from os import listdir

from chapter2.KNN import *

# 将二进制图像矩阵转换成一维数组
def imgToVector(filename):
    returnVect = zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(linestr[j])
    return returnVect


# 手写数字识别
def handwritingClassTest():
    hwLabels = []
    # 训练数据集
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 获取文件名
        fileNameStr = trainingFileList[i]
        filestr = fileNameStr.split('.')[0]
        classNumStr = int(filestr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = imgToVector("trainingDigits/%s" % fileNameStr)

    # 测试数据集
    testFileList = listdir("testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        filestr = fileNameStr.split('.')[0]
        classNumStr = int(filestr.split('_')[0])
        vectorUnderTest = imgToVector("trainingDigits/%s" % fileNameStr)
        classifierResult = classfiy0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("the total number of errors is : %d" % errorCount)
    print("the total error rate is : %f" % (errorCount / float(mTest)))


handwritingClassTest()