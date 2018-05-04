# !/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
# from chapter3.trees import *

# boxstyle文本框样式， fc(face color)背景透明度
decisionNode = dict(boxstyle="round4, pad=0.5", fc="0.8")
leafNode = dict(boxstyle="circle", fc="0.8")
# 箭头样式
arrow_args = dict(arrowstyle="<-")


# 绘制节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # 被注释的地方xy(x, y)和插入文本的地方xytext(x, y)
    # xycoords和textcoords指定xy和xytext的坐标系。此处是左下角(0.0,0.0)，右上角(1.0,1.0)
    # 文本在文本框中的va(纵向),ha(横向)居中
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction",
                            xytext=centerPt, textcoords="axes fraction", va="center",
                            ha="center", bbox=nodeType, arrowprops=arrow_args)


# 获取叶节点数目
def getNumLeafs(myTree):
    numLeafs = 0
    # Python3与Python2的区别，先转换成list，再按索引取值
    # firstStr = myTree.keys()[0]
    firstStr = list(myTree.keys())[0]
    # 子树
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            # 如果是decisionNode，递归
            numLeafs += getNumLeafs(secondDict[key])
        else:
            # leafNode
            numLeafs += 1
    return numLeafs


# 获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    # 当前树的根节点
    firstStr = list(myTree.keys())[0]
    # 子树
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            # 如果是decisionNode（有子节点），递归
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            # leafNode，叶子节点
            thisDepth = 1
        # 修正maxDepth，保证maxDepth是最大值
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 在父子节点之间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


# 绘制决策树
def plotTree(myTree, parentPt, nodeTxt):
    # 当前树的叶子节点数和深度
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    # 当前根节点
    firstStr = list(myTree.keys())[0]
    # 修正当前位置，xOff + 当前树的叶子节点数 / 2W + 1 / 2W
    # 加1/2W 是因为初始位置是-1/2W，修正这个位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / (2.0 * plotTree.totalW), plotTree.yOff)
    # 在父子节点间填充文本信息
    plotMidText(cntrPt, parentPt, nodeTxt)
    # decisionNode，绘制
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 当前树的子节点
    secondDict = myTree[firstStr]
    # 深度加1，修正plotTree.yOff - 1/D
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    # 遍历绘制子节点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ ==  'dict':
            # decisionNode，调用plotTree绘制
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # 遇到leafNode，修正xOff + 1/W，调用plotNode绘制
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 树的宽度
    plotTree.totalW = float(getNumLeafs(inTree))
    # 树的深度
    plotTree.totalD = float(getTreeDepth(inTree))
    # 初始偏移量-1/2W，每遇到一个叶节点加1/W，使画出来的树尽可能居中
    # 如3个叶子（1/6, 1/2, 5/6）,4个叶子（1/8, 3/8, 5/8, 7/8）
    plotTree.xOff = -0.5 / plotTree.totalW
    # 初始深度0，第一层
    plotTree.yOff = 1.0
    # 绘制图形
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# myTree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}, 3: 'maybe'}}
# createPlot(myTree)
