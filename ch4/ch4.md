###第四章作业
####课后习题
#####4.1
对每个属性都进行决策，可以得到一颗包含所有情况的决策树。即列出假设空间，再得到符合训练集的版本空间，训练集每个数据在叶节点中都有对应。因为没有冲突数据，所以不存在分类错误。
综上，**存在这样一颗决策树**。

#####4.3
代码：
```
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

# 读入数据
dataPath = r'D:\\vscode\\Markdown\\ML\\ch4\watermelon3.0.txt'
dataSet = pd.read_csv(dataPath, index_col=0)
# 去掉编号行
dataSet.reset_index(inplace=True, drop=True)

# 分出属性和标签
X = dataSet.iloc[:, :8]
y = dataSet.iloc[:, 8]
# print(X)
# print(y)

# 计算信息熵
def entroy(y):
    # 计算各类样本所占的比例
    p = pd.value_counts(y) / y.shape[0]
    # 代入公式
    ent = np.sum(-p * np.log2(p))
    return ent

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
        info_gain = infoGain(X[featName], y, entD, is_continuous)
        # 更新最优属性
        if info_gain[0] > bestInfoGain[0]:
            bestFeatureName = featName
            bestInfoGain = info_gain

    return bestFeatureName, bestInfoGain

# 计算某一属性的信息增益
def infoGain(feature, y, entD, is_continuous = False):
    
    # 数据集合总数量
    m = y.shape[0]
    # 提取属性值，如果是离散属性，则需要去重
    uniqueValue = pd.unique(feature)

    # 属性值为连续值
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
            Dv1 = y[feature <= splitPoint]
            Dv2 = y[feature > splitPoint]
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
            Dv = y[feature == value]
            featureEnt += Dv.shape[0] / m * entroy(Dv)
        
        gain  = entD - featureEnt
        return [gain]
    
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
    featureValues = X.loc[:, bestFeatureName]
    # 离散值属性
    if len(bestSplitPonit) == 1:
        uniqueVals = pd.unique(featureValues)
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

myTree = createTree(X, y)
print(myTree)
```
得到决策树为：
**{'纹理': 
$\qquad${'清晰': 
$\qquad\qquad${'密度': 
$\qquad\qquad\qquad${'>= 0.382': '是', 
$\qquad\qquad\qquad$'< 0.382': '否'}}, 
$\qquad$'稍糊': 
$\qquad\qquad${'触感': 
$\qquad\qquad\qquad${' 软粘': '是', 
$\qquad\qquad\qquad$'硬滑': '否}}, 
$\qquad$'模糊': '否'}}**
与书本结果一致

#####4.4
**预剪枝**
```
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


myTree1 = createTree(X_train, y_train)
print(myTree1)
myTree2 = createTreePrePruning(X_train, y_train, X_test, y_test)
print(myTree2)
```
结果：
未剪枝：
{'色泽': {'青绿': {'敲声': {'浊响': '是', '清脆': '否', '沉闷': '否'}}, '乌黑': {'根
蒂': {'蜷缩': '是', '稍蜷': {'纹理': {'稍糊': '是', '清晰': '否'}}}}, '浅白': '否'}}
预剪枝：
{'色泽': {'青绿': '是', '乌黑': {'根蒂': {'蜷缩': '是', '稍蜷': '是'}}, '浅白': '否'}}
**有效果但与教材的出入较大**

**后剪枝**
```
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
```
**无效果:)**

#####4.7
递归是深度优先建立节点，用队列是广度优先建立节点
主要思想是在建立树节点的时候，先**把这个节点的所有子节点建立完，把非叶子节点放入队列，再建立下一层节点的子节点**
```
从A中选择一个最优划分属性a*;
    for a* 的每一个值 a*v do:
            新建一个节点，并将节点连接到父节上;
        令 Dv表示为D中在a*上取值为a*v的样本子集;
            if Dv为空:
                将当前标记为叶节点，其标记类别为D中样本最多的类;
                continue;
            end if
            if A\a*为空 or Dv上样本都为同一类别 or depth == maxDepth:
                将当前节点标记为叶节点，其标记类别为Dv中样本最多的类;
                continue;
            end if 			
```

#####4.9
初始化每个样本的权重为1，每次计算的时候按样本当前权重来计算
对当前判断属性选择无缺失值的样本来计算，计算每个属性值划分出样本的Gini指数，按照划分样本占无缺失值总样本的加权比例求和之后，乘以**无缺失值样本占总样本的比例的倒数**(因为基尼指数越小越容易被选择，而缺失值越多数据越不准确，应该越不被选上，所以乘以比例的倒数)，得出当前属性的基尼指数
$Gini(D) = 1- \sum_{k = 1} ^ {|γ|} {\widetilde{p}_k ^ 2}$ 
$Gini\_index(D, a) = \frac{1}{\widetilde{p}} \times \sum_{v = 1} ^V \widetilde{v}Gini(D^v)$ 