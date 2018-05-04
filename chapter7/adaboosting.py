# !/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *


# 加载一个简单的数据集
def loadSimpleData():
    dataMat = matrix([[1.0, 2.1],
                      [2.0, 1.1],
                      [1.3, 1.0],
                      [1.0, 1.0],
                      [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


# 利用单层决策树（树桩）分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    # 创建一个与数据集样本数量相同的列向量，初始化为1
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        # <= threshVal的归为-1类， 其余为1类
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        # 》 threshVal的归为-1类， 其余为1类
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 构建单层决策树（数据集，类别标签，迭代次数）
def buildStump(dataArr, classLabels, D):
    """
    将最小错误率minError设为正无穷
    对数据集中的每一个特征（第一层循环）：
        对每个步长（第二层循环）：
            对每个不等号（第三层循环）：
                建立一棵单层决策树并加以利用加权数据集对他进行测试
                如果错误率低于minError，则将当前单层决策树设为最佳单层决策
    返回最佳单层决策树
    """
    dataMat = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMat)
    # 步数（所有可能值范围）
    numSteps = 10.0
    # 保存给定权重向量D所得到的最佳单层决策树
    bestStump = {}
    # 最佳分类获得的类别向量（列向量）
    bestClassEst = mat(zeros((m, 1)))
    # 初始化最小错误率为正无穷
    minError = inf
    # 对数据集中的每一个特征，按列循环
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        # 步长
        stepSize = (rangeMax - rangeMin) / numSteps
        # 对每个步长
        for j in range(-1, int(numSteps) + 1):
            # 对每个不等号，lt: less than; gt: great than
            for inequal in ['lt', 'gt']:
                # 计算阈值，初值 + 步数 * 步长
                threshVal = (rangeMin + float(j) * stepSize)
                # 利用单层决策树返回预测分类结果
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
                # 分类结果的正误，1分类错误，0正确，初始化为全1
                errArr = mat(ones((m, 1)))
                # 把正确分类的置为0
                errArr[predictedVals == labelMat] = 0
                # 计算加权错误率，错误向量 dot* 权重向量，weightedError是一个值:[[ 0.57142857]]
                weightedError = D.T * errArr
                # print("weightedError : ", weightedError)
                # 更新最小加权错误率
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


# 基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    对每次迭代：
        利用buildStump()函数找到最佳的单层决策树
        将最佳单层决策树加入到单层决策树组
        计算alpha
        计算新的权重向量D
        更新累计类别估计值
        如果错误率等于0.0，则退出循环
    """
    # 声明单层决策树组保存每一次迭代的最佳DS（decision stump）
    weakClassArr = []
    m = shape(dataArr)[0]
    # 初始化权重向量（列），保存每个数据的权重，初始时每个数据点的权重相同
    D = mat(ones((m, 1)) / m)
    # 初始化aggClassEst（列向量），保存每个数据点的类别累计估计值
    aggClassEst = mat(zeros((m, 1)))
    # 训练numIt次或者直到错误率为0
    for i in range(numIt):
        """利用buildStump()函数找到最佳的单层决策树"""
        # 用buildStump获得最小错误率的单层决策树，最小错误率以及估计的类别向量
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D : ", D.T)
        """计算alpha，表示本次单层决策树输出结果的权重"""
        # max(error, 1e-16)确保在没有错误时不会发生除零溢出
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        # 把alpha值加入到bestStump字典中
        bestStump['alpha'] = alpha
        """将最佳单层决策树加入到单层决策树组"""
        weakClassArr.append(bestStump)
        print("classEst : ", classEst.T)
        """计算新的权重向量D"""
        # 如果当前样本被正确分类((1 * 1)or(-1 * -1)) = 1，乘参数-1 = -alpha，权重下降
        # 如果当前样本被错分((1 * -1)or(-1 * 1)) = -1，乘以参数-1 = alpha，权重升高
        expon = multiply(-1 * mat(classLabels).T, classEst) * alpha
        # 其实是一个迭代的过程D_i+1 = D_i * exp(expon)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # 运行时类别估计值
        aggClassEst += alpha * classEst
        print("aggClassEst : ", aggClassEst.T)
        # 使用sign进行二分类
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error : ", errorRate)
        """在训练错误率达到0，提前结束循环"""
        if errorRate == 0.0:
            break
    return weakClassArr


# adaBOost分类函数
def adaClassify(dataToClass, classifierArr):
    dataMat = mat(dataToClass)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)


# test
dataMat, classLabels = loadSimpleData()
classifierArr = adaBoostTrainDS(dataMat, classLabels, 30)
print(adaClassify([[5, 5], [0, 0]], classifierArr))




