import numpy as np
import pandas as pd

# 读入数据
dataPath = r'D:\\vscode\\Markdown\\ML\watermelon2.0.txt'
dataSet = pd.read_csv(dataPath, index_col=0)
# 去掉编号行
dataSet.reset_index(inplace=True, drop=True)

# 分出训练集和测试集
X = dataSet.iloc[:, :6]
y = dataSet.iloc[:, 6]

train = [1, 2, 3, 6, 7, 10, 14, 15, 16, 17]
train = [i - 1 for i in train]
X_train = dataSet.iloc[train, :6]
y_train = dataSet.iloc[train, 6]

test = [4, 5, 8, 9, 11, 12, 13]
test = [i - 1 for i in test]
X_test = dataSet.iloc[test, :6]
y_test = dataSet.iloc[test, 6]

# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)
# print(pd.value_counts(y_test).index[0])
# print(np.sum(y_test.values == '是'))
# print(len(y_test))


# 计算基尼值
def giniValueCal(y):
    p = pd.value_counts(y) / y.shape[0]
    giniValue = 1 - np.sum(p ** 2)
    return giniValue

# 计算某一属性的基尼指数
def giniIndexCal(feature, y):
    
    # 数据集合总数量
    m = y.shape[0]
    # 提取属性值，如果是离散属性，则需要去重
    uniqueValue = pd.unique(feature)

    giniIndex = 0
    # 遍历不同的属性值
    for value in uniqueValue:
        # 取出当前属性值，计算权重和其信息熵
        Dv = y[feature == value]
        giniIndex += Dv.shape[0] / m * giniValueCal(Dv)
    
    return giniIndex

# 选择最优属性
def chooseBestFeatureGiniIndex(X, y):

    # 列出所有备选属性
    features = X.columns
    # 初始化最优属性名和最小基尼指数
    bestFeatureName = None
    bestGiniIndex = [float('inf')]

    # 对每一个属性，计算基尼指数，并得到最优属性名和最小基尼指数
    for featureName in features:
        giniIndex = giniIndexCal(X[featureName], y)
        # 更新最优属性
        if giniIndex < bestGiniIndex:
            bestFeatureName = featureName
            bestGiniIndex = giniIndex

    return bestFeatureName

def createTree(X, y):
    # 空间极纯
    if y.nunique() == 1:
        return y.values[0]
    
    # 属性极致
    if X.empty:
        return pd.value_counts(y).index[0]
    
    # 寻找最优属性
    bestFeatureName = chooseBestFeatureGiniIndex(X, y)

    # 创建分支节点
    myTree = {bestFeatureName:{}}

    # 划分子树
    featureValues = X.loc[:, bestFeatureName]
    uniqueVals = pd.unique(featureValues)
    # 更新数据集
    subX = X.drop(bestFeatureName, axis = 1)

    for value in uniqueVals:
        # 属性空虚
        # if subX[featureValues == value].shape[0] == 0:
        #     myTree[bestFeatureName][value] = pd.value_counts(y).index[0]
        # else:
        myTree[bestFeatureName][value] = createTree(subX[featureValues == value], y[featureValues == value])
    
    return myTree


def createTreePrePruning(X_train, y_train, X_test, y_test):
    
    # 空间极纯
    if y_train.nunique() == 1:
        return y_train.values[0]
    
    # 属性极致
    if X_train.empty:
        return pd.value_counts(y_train).index[0]
    
    # 寻找最优属性
    bestFeatureName = chooseBestFeatureGiniIndex(X_train, y_train)
    
    # 更新训练数据集
    subX_train = X_train.drop(bestFeatureName, axis = 1)
    
    # 预剪枝评估
    if not X_test.empty:
        # print(1)
        subX_test = X_test.drop(bestFeatureName, axis = 1)
        
        # 划分前测试集正确率
        labelPre = pd.value_counts(y_train).index[0]
        testRatioPre = np.sum(y_test.values == labelPre) / len(y_test)
        # print(testRatioPre)

        # 划分后测试集正确率
        testRatioPost = 0.0
        featureValuesTrain = X_train.loc[:, bestFeatureName]
        featureValuesTest = X_test.loc[:, bestFeatureName]
        uniqueVals = pd.unique(featureValuesTrain)
        for value in uniqueVals:
            labelPost = pd.value_counts(y_train[featureValuesTrain == value]).index[0]
            testRatioPost += np.sum(y_test[featureValuesTest == value].values == labelPost) / len(y_test)
        # print(testRatioPost)

    if X_test.empty:
        # 创建分支节点
        myTree = {bestFeatureName:{}}
        featureValuesTest = X_test.loc[:, bestFeatureName]
        for value in uniqueVals:
            # 属性空虚
            # if subX[featureValues == value].shape[0] == 0:
            #     myTree[bestFeatureName][value] = pd.value_counts(y).index[0]
            # else:
            myTree[bestFeatureName][value] = createTreePrePruning(subX_train[featureValuesTrain == value], y_train[featureValuesTrain == value], subX_test[featureValuesTest  == value], y_test[featureValuesTest  == value])

    elif testRatioPre > testRatioPost:
        return y_train.values[0]

    else:
        # 创建分支节点
        myTree = {bestFeatureName:{}}
        featureValuesTest = X_test.loc[:, bestFeatureName]
        for value in uniqueVals:
            # 属性空虚
            # if subX[featureValues == value].shape[0] == 0:
            #     myTree[bestFeatureName][value] = pd.value_counts(y).index[0]
            # else:
            myTree[bestFeatureName][value] = createTreePrePruning(subX_train[featureValuesTrain == value], y_train[featureValuesTrain == value], subX_test[featureValuesTest  == value], y_test[featureValuesTest  == value])
    
    return myTree


def createTreeWithLabel(X, y):
    # 空间极纯
    if y.nunique() == 1:
        return y.values[0]
    
    # 属性极致
    if X.empty:
        return pd.value_counts(y).index[0]
    
    # 寻找最优属性
    bestFeatureName = chooseBestFeatureGiniIndex(X, y)

    # 创建分支节点
    myTree = {bestFeatureName: {'labelPre': pd.value_counts(y).index[0]}}

    # 划分子树
    featureValues = X.loc[:, bestFeatureName]
    uniqueVals = pd.unique(featureValues)
    # 更新数据集
    subX = X.drop(bestFeatureName, axis = 1)

    for value in uniqueVals:
        # 属性空虚
        # if subX[featureValues == value].shape[0] == 0:
        #     myTree[bestFeatureName][value] = pd.value_counts(y).index[0]
        # else:
        myTree[bestFeatureName][value] = createTreeWithLabel(subX[featureValues == value], y[featureValues == value])
    
    return myTree

def convertTree(labeledTree):
    treeNew = labeledTree.copy()
    nodeName = list(labeledTree.keys())[0]
    treeNew[nodeName] = labeledTree[nodeName].copy()
    for val in list(labeledTree[nodeName].keys()):
        if val == 'labelPre':
            treeNew[nodeName].pop(val)
        elif type(labeledTree[nodeName][val]) == dict:
            treeNew[nodeName][val] = convertTree(labeledTree[nodeName][val])
    return treeNew

def treePostPruning(labeledTree, X_test, y_test):

    treeNew = labeledTree.copy()
    # 取当前决策节点的名称，即属性名称
    featureName = list(labeledTree.keys())[0]
    # 该属性下所有值的字典
    treeNew[featureName] = labeledTree[featureName].copy()
    featureValueDict =  treeNew[featureName]
    labelPre = featureValueDict.pop('labelPre')

    subTreeFlag = 0
    
    if not X_test.empty:
        subX_test = X_test.drop(featureName, axis = 1)

    for featureValue in featureValueDict.keys():
        if not X_test.empty and type(featureValueDict[featureValue]) == dict:
            subTreeFlag = 1
            featureValuesTest = X_test.loc[:, featureName]
            treeNew[featureName][featureValue] = treePostPruning(featureValueDict[featureValue], subX_test[featureValuesTest == featureValue], y_test[featureValuesTest == featureValue])

        if X_test.empty and type(featureValueDict[featureValue]) == dict:
            subTreeFlag = 1
            treeNew[featureName][featureValue] = convertTree(featureValueDict[featureValue])

    # dict = {'纹理': {"prun_label": 1, '稍糊': 1, '清晰': 0, '模糊': 1}}
    # print(type(dict))
    # key = dict.keys()
    # print(dict['纹理']['稍糊'])

    if subTreeFlag == 0:
        print(1)
        # 划分前测试集正确率
        print(y_test.values)
        print(labelPre)
        testRatioPre = np.sum(y_test.values == labelPre) / len(y_test)
        print(testRatioPre)
        # 划分后测试集正确率
        testRatioPost = 0.0
        featureValuesTest = X_test.loc[:, featureName]
        uniqueVals = pd.unique(featureValuesTest)
        for value in uniqueVals:
            labelPost = featureValueDict[value]
            testRatioPost += np.sum(y_test[featureValuesTest == value].values == labelPost) / len(y_test)
        print(testRatioPost)

        if testRatioPost < testRatioPre:
            treeNew = labelPre
        
    return treeNew


myTree1 = createTree(X_train, y_train)
print(myTree1)
myTree2 = createTreePrePruning(X_train, y_train, X_test, y_test)
print(myTree2)
# myTree3 = createTreeWithLabel(X, y)
# print(myTree3)
# myTree4 = convertTree(myTree3)
# print(myTree4)
# myTree5 = treePostPruning(myTree3, X_test, y_test)
# print(myTree5)


