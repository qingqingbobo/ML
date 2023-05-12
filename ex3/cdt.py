import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target


# 计算信息熵
def entroy(y):
    # 计算各类样本所占的比例
    # pd.value_counts找出y中不同类的个数
    # y.shape[0]是y总的个数
    p = pd.value_counts(y) / y.shape[0]
    # 代入公式
    ent = np.sum(-p * np.log2(p))
    return ent


# 计算某一属性的信息增益
def infoGainCalc(xfeature, y, entD, is_continuous):
    
    # 数据集合总数量
    m = y.shape[0]
    # 提取属性值，如果是离散属性，则需要去重
    uniqueValue = pd.unique(xfeature)

    # # 属性值为连续值
    if is_continuous:
        # 属性值升序排序
        uniqueValue.sort()
        # 分割点集合
        splitPointSet = [(uniqueValue[i] + uniqueValue[i + 1]) / 2 for i in range(len(uniqueValue) - 1)]
        # 初始化最小的信息熵及其分割点
        minEnt = float('inf')
        minEntPoint = None

        for splitPoint in splitPointSet:
            # 把属性值小于等于分割点和大于分割点的样本分为两个集合
            Dv1 = y[xfeature <= splitPoint]
            Dv2 = y[xfeature > splitPoint]
            featureEnt = Dv1.shape[0] / m * entroy(Dv1) + Dv2.shape[0] / m * entroy(Dv2)

            if featureEnt < minEnt:
                minEnt = featureEnt
                minEntPoint = splitPoint
        
        gain = entD - minEnt

        return [gain, minEntPoint]
        
    # 属性值为离散值
    else:
        featureEnt = 0
        # 遍历不同的属性值
        for value in uniqueValue:
            # 取出当前属性值，计算权重和其信息熵
            Dv = y[xfeature == value]
            featureEnt += Dv.shape[0] / m * entroy(Dv)
        
        gain  = entD - featureEnt
        return [gain]


# 选择最优属性
def chooseBestFeatureInfoGain(X, y):
    # 列出所有备选属性
    features = X.columns
    # 初始化最优属性名和最大信息增益
    bestFeatureName = None
    bestInfoGain = [float('-inf')]
    # 算出当前不划分时的信息熵
    entD = entroy(y)

    # 对每一个属性，计算信息增益，并得到最优属性名和最大信息增益
    for featName in features:
        # 判断属性取值是否为连续值
        is_continuous = type_of_target(X[featName]) == 'continuous'
        info_gain = infoGainCalc(X[featName], y, entD, is_continuous)
        # 更新最优属性
        if info_gain[0] > bestInfoGain[0]:
            bestFeatureName = featName
            bestInfoGain = info_gain

    return bestFeatureName, bestInfoGain

    
def createTree(X, y):
    # 空间极纯
    if y.nunique() == 1:
        return y.values[0]
    
    # 属性极致
    if X.empty:
        return pd.value_counts(y).index[0]
    
    # 寻找最优属性
    bestFeatureName, bestSplitPonit = chooseBestFeatureInfoGain(X, y)
    # 创建分支节点
    myTree = {bestFeatureName:{}}
    
    # 最优属性的取值
    if bestFeatureName is None:
        return y.value_counts().index[0]
    
    featureValues = X.loc[:, bestFeatureName]
    # 离散值属性
    if len(bestSplitPonit) == 1:
        uniqueVals = pd.unique(featureValues)
        # 下一分支，此属性去掉
        subX = X.drop(bestFeatureName, axis = 1)

        for value in uniqueVals:
            myTree[bestFeatureName][value] = createTree(subX[featureValues == value], y[featureValues == value])  
    
    # 连续值属性
    elif len(bestSplitPonit) == 2:
        value = bestSplitPonit[1]
        upPart = '>= {:.3f}'.format(value)
        downPart = '< {:.3f}'.format(value)
        subX = X.drop(bestFeatureName, axis = 1)
        myTree[bestFeatureName][upPart] = createTree(subX[featureValues >= value], y[featureValues >= value])
        myTree[bestFeatureName][downPart] = createTree(subX[featureValues < value], y[featureValues < value])

    return myTree


if __name__ == '__main__':
    # watermelon3.0
    # 读入数据
    dataPath = r'D:\\vscode\\Markdown\\ML\watermelon3.0.txt'
    dataSet = pd.read_csv(dataPath, index_col=0)
    # 去掉编号行
    dataSet.reset_index(inplace=True, drop=True)
    # 分出属性和标签
    X = dataSet.iloc[:, :8]
    y = dataSet.iloc[:, 8]
    # print(X)
    # print(y)

    myTree = createTree(X, y)
    print(myTree)