# !/usr/bin/env python
# -*- coding: utf-8 -*-

def loadData():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5, ], [2, 5]]


def createC1(dataSet):
    """
    构建大小为1的所有候选项集的集合
    :param dataSet:
    :return:
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    """
    Ck中的元素经过最小支持度过滤得到Lk，和每个元素的支持度
    :param D: 数据集
    :param Ck:
    :param minSupport: 最小支持度（过滤）
    :return:
    过程：
    对数据集中的每条交易记录tran
    对每个候选项集can：
        检查一下can是否是tran的子集：
            如果是：则增加can的计数值
            如果不是：把can的值设为1
    对每个候选项集：
        如果其支持度不低于最小值，则保留该项集
    返回所有频繁项集列表
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    """
    合并频繁项集为k个元素
    :param Lk: 频繁项集列表
    :param k: 项集元素个数
    :return:
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # 每一个项集同其后面的项集做比较
            # 如果前k-2个元素相同（取不到k-2），则合并为k个元素的项集，添加到retList中
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def Apriori(dataSet, minSupport=0.5):
    """

    :param dataSet: 数据集
    :param minSupport: 最小支持度
    :return:
    过程：
    C1->L1->C2->L2->C3->L3-> ... ->Ck->Lk
    """
    # 单个物品项组成的集合
    C1 = createC1(dataSet)
    # 集合表示的数据集D
    D = list(map(set, dataSet))
    #
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def calcConf(freqSet, H, supportData, bigRuleList, minConf=0.7):
    """
    从频繁项集freqSet，计算关联规则(freqSet - H) --> H的置信度
    并过滤置信度低于最小置信度阈值的规则
    :param freqSet: 频繁项集
    :param H: 出现在规则右部
    :param supportData: 支持度字典
    :param bigRuleList: 关联规则列表
    :param minConf: 最小置信度阈值
    :return: 满足最小可信度要求的规则列表
    """
    # 初始化满足最小可信度要求的规则列表
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet-conseq, "-->", conseq, "conf : ", conf)
            bigRuleList.append((freqSet - conseq, conseq, conf))
            # 与bigRuleList对应的置信度值列表
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, bigRuleList, minConf=0.7):
    """
    由于freqSet中的元素个数大于等于3，计算规则的右部为多个元素的情形
    :param freqSet: 频繁项集
    :param H: 出现在规则右部的元素列表
    :param supportData: 支持度列表
    :param bigRuleList: 关联规则列表
    :param minConf: 最小置信度阈值
    :return:
    """
    m = len(H[0])
    # m + 1，其实示一个递归变量，递归地增加右部元素个数
    if len(freqSet) > (m + 1):
        # 合并元素，任意组合成m+1个元素的项集
        Hmp1 = aprioriGen(H, m + 1)
        # 过滤关联规则(freqSet - Hmp1) --> Hmp1的置信度小于最小置信度阈值的规则
        # 规则右部有m + 1个元素
        Hmp1 = calcConf(freqSet, Hmp1, supportData, bigRuleList, minConf)
        # 递归累加右部元素个数
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, bigRuleList, minConf)
    else:
        # 当右部元素个数和频繁项集freqSet的长度一样时，不再划分，返回
        return


def generateRules(L, supportData, minConf=0.7):
    """
    基于给定的频繁项集L和支持度supportData，计算满足最小支持度阈值的规则
    :param L: 频繁项集列表
    :param supportData: 包含那些频繁项集支持数据的字典
    :param minConf: 最小置信度阈值
    :return: 包含可信度的规则列表
    """
    # 初始化规则存放列表
    bigRuleList = []
    # 遍历频繁项集
    for i in range(1, len(L)):
        #
        for freqSet in L[i]:
            # 只包含单个元素集合的列表H1
            H1 = [frozenset([item]) for item in freqSet]
            # 过滤关联规则(freqSet - H1) --> H1的置信度小于最小置信度阈值的规则
            # H1是右部为单个元素的情形
            H1 = calcConf(freqSet, H1, supportData, bigRuleList, minConf)
            if i > 1:
                # 如果频繁项集中元素个数大于等于3，就需要考虑右部为多个元素的情形
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


dataSet = loadData()
L, supportData = Apriori(dataSet)
rules = generateRules(L, supportData, minConf=0.7)
print(rules)