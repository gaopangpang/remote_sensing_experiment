import numpy as np
def creatDataSet():
    dataSet = [[1,1,'yes'],#构建数据集，1代表yes,0代表no，前两列属性特征是labels = ['no sufacing','flippers']，当两者都是1时，判断是鱼
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']
    ]
    labels = ['no sufacing','flippers']
    return dataSet,labels

#计算给定数据集的香农熵的函数
def calcShannonEnt(dataSet):
    # 求list的长度，表示计算参与训练的数据量
    numEntries = len(dataSet)
    # 计算分类标签label出现的次数
    labelCounts = {} #注意这里是字典
    #独特元素的数量及其出现次数
    for featVec in dataSet:
        # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = featVec[-1]
        # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1

    # 对于 label 标签的占比，求出 label 标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率。
        prob = float(labelCounts[key])/numEntries
        # 计算香农熵，以 2 为底求对数
        shannonEnt -= prob * np.log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, index, value):
    """splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
    Args:
        dataSet 数据集                 待划分的数据集
        index 表示每一行的index列        划分数据集的特征
        value 表示index列对应的value值   需要返回的特征的值。
    Returns:
        index列为value的数据集【该数据集需要排除index列】
    """
    retDataSet = []
    for featVec in dataSet:
        # index列为value的数据集【该数据集需要排除index列】
        # 判断index列的值是否为value
        if featVec[index] == value:
            # chop out index used for splitting
            # [:index]表示前index行，即若 index 为2，就是取 featVec 的前 index 行
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index + 1:])
            # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据
            # 收集结果值 index列为value的行【该行需要排除index列】
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
     """chooseBestFeatureToSplit(选择最好的特征)
 
         Args:
             dataSet 数据集
         Returns:
             bestFeature 最优的特征列
    """
     # 求第一行有多少列的 Feature, 最后一列是label列嘛
     numFeatures = len(dataSet[0])-1
     # 数据集的原始信息熵
     baseEntropy = calcShannonEnt(dataSet)
     # 最优的信息增益值, 和最优的Featurn编号
     bestInfoGain, bestFeature = 0.0, -1
     # iterate over all the features
     for i in range(numFeatures):
         # create a list of all the examples of this feature
         # 获取对应的feature下的所有数据
         featList = [example[i] for example in dataSet]
         # get a set of unique values
         # 获取剔重后的集合，使用set对list数据进行去重
         uniqueVals = set(featList)
         # 创建一个临时的信息熵
         newEntropy = 0.0
         # 遍历某一列的value集合，计算该列的信息熵
         # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
         for value in uniqueVals:
             subDataSet = splitDataSet(dataSet, i, value)
             # 计算概率
             prob = len(subDataSet) / float(len(dataSet))
             # 计算信息熵
             newEntropy += prob * calcShannonEnt(subDataSet)
         # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
         # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
         infoGain = baseEntropy - newEntropy
         print
         ('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
         if (infoGain > bestInfoGain):
             bestInfoGain = infoGain
             bestFeature = i
     return bestFeature

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[bestFeat])
    # 取出最优列，然后它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print 'myTree', value, myTree
    return myTree