# !/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *


class opstruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 误差缓存，第一列是表示eCache是否有效的标志位，第二列时实际的E值
        self.eCache = mat(zeros((self.m, 2)))


# 加载数据
def loadData(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


# 计算给定alpha的时候的误差 k用来指定计算哪个样本的误差
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 调整过大或过小的alpha（alpha需要满足一定的约束条件；0<=alpha<=C sum(alpha*y)=0）
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 计算权重W
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


# i是alpha的下标，m是alpha的数目
def selectJrand(i, m):
    j = i
    # 随机选择一个j，不同于i
    while(j == i):
        j = int(random.uniform(0, m))
    return j


# 选择合适的第二个alphas值 使得每次有优化的时候的步长都最大
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 先设置为有效的
    oS.eCache[i] = [1, Ei]
    # 返回eCache矩阵中非零元素的索引值数组，访问有效（计算好）的误差缓存
    validEcacheList = nonzero(oS.eCache[:, 0])[0]
    # 循环选择其中使得变化量最大的值
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            # 增量
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 如果是第一次 就随机选择
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


# 更新缓存
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# SMO内部函数
def innerL(i, oS):
    # 计算alpha_i的误差
    Ei = calcEk(oS, i)
    # 如果误差很大（toler容错率），且alpha可优化，则应该对该实例对应的alpha值优化
    # alpha要满足约束条件：0<=alpha<=C sum(alpha*y)=0
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or (oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        # 选择具有最大步长的alpha_j
        j, Ej = selectJ(i, oS, Ei)
        # 修改前保存alpha_i和alpha_j的值
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        # 判断是否修改了
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0
        # alpha_j修改了，同时修改alpha_i（修改量相同，方向相反）
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 修改缓冲区里alpha_i的误差
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if 0 < oS.alphas[i] and oS.C > oS.alphas[i]:
            oS.b = b1
        elif 0 < oS.alphas[j] and oS.C > oS.alphas[j]:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        # alpha_i和alpha_j修改了，返回1
        return 1
    else:
        # alpha_i和alpha_j未修改，返回0
        return 0


# 完整的Platt SMO算法
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = opstruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 循环退出条件：迭代次数超过了指定的最大值，或者遍历整个集合都未对任意alpha对进行修改时
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            # 遍历任意可能的alpha
            for i in range(oS.m):
                # alphaPairsChanged加上alpha对是否修改，如果有任意一对alpha发生改变，innerL返回1
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter : %d i : %d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            # 非边界alpha值，不在边界0或C上的
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i : %d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number : %d" % iter)
    return oS.b, oS.alphas


# 可视化
def plotSVM(dataArr, labelArr, b, ws):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    for i in range(len(labelArr)):
        if labelArr[i] == -1:
            xcord0.append(dataArr[i][0])
            ycord0.append(dataArr[i][1])
        else:
            xcord1.append(dataArr[i][0])
            ycord1.append(dataArr[i][1])
    ax.scatter(xcord0, ycord0, s=40, alpha=0.7)
    ax.scatter(xcord1, ycord1, s=40, alpha=0.7)
    plt.title('Support Vectors Circled')

    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataArr[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    # dataMat = mat(dataArr)
    x1 = max(dataArr)[0]
    x2 = min(dataArr)[0]
    a1, a2 = ws
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # x = arange(-2.0, 12.0, 0.1)
    # y = (-ws[0] * x - b) / ws[1]
    # ax.plot(x, y)
    # ax.axis([-2, 12, -8, 6])
    plt.show()




# test
dataArr, labelArr = loadData('testSet.txt')
b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
print("b : ", b)
ws = calcWs(alphas, dataArr, labelArr)
print("ws : ", ws)
plotSVM(dataArr, labelArr, float(b), ws)

