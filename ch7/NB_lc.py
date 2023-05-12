import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from collections import namedtuple

def trainNB(X, y):
    
    # m: 样例数，n: 属性个数
    m, n = X.shape
    # 求先验概率，拉普拉斯平滑
    p1 = (len(y[y == '是']) + 1) / (m + 2)
    # print(y[y == '是'])

    # 正例下各属性的条件概率
    p1_list = []
    # 反例下各属性的条件概率
    p0_list = []

    # 抽出不同类别的集合，并计算其个数
    X1 = X[y == '是']
    m1, _ = X1.shape
    X0 = X[y == '否']
    m0, _ = X0.shape

    for i in range(n):
        # 第i个属性所有样本
        xi = X.iloc[:, i]
        # 第i个属性的情况
        p_xi = namedtuple(X.columns[i], ['is_continuous', 'conditional_pro'])
        # 判断第i个属性是否为连续值
        is_continuous = type_of_target(xi) == 'continuous'

        x1i = X1.iloc[:, i]
        x0i = X0.iloc[:, i]
        
        # 连续性属性，conditional_pro：均值和方差
        if is_continuous:
            x1i_mean = np.mean(x1i)
            x1i_var = np.var(x1i)
            x0i_mean = np.mean(x0i)
            x0i_var = np.var(x0i)

            p1_list.append(p_xi(is_continuous, [x1i_mean, x1i_var]))
            p0_list.append(p_xi(is_continuous, [x0i_mean, x0i_var]))

        # 连续性属性，conditional_pro：各属性值的条件概率
        else:
            # 不同属性值
            uniqueVal = xi.unique()
            # 不同属性值的个数
            uniqueValNum = len(uniqueVal)

            # 正例中不同属性值的个数，拉普拉斯平滑
            # X与x1i中共同包含的标签
            common_labels = np.intersect1d(uniqueVal, x1i.unique())
            # x1i 序列中包含的标签进行计数
            x1i_value_count = pd.value_counts(x1i)[common_labels].fillna(0)
            # 包含 uniqueVal 中所有的标签，并将 x1i 中不包含的标签的值设置为 0 
            x1i_value_count = x1i_value_count.reindex(uniqueVal, fill_value=0) + 1 

            # 反例中不同属性值的个数，拉普拉斯平滑
            common_labels = np.intersect1d(uniqueVal, x0i.unique()) 
            x0i_value_count = pd.value_counts(x0i)[common_labels].fillna(0)
            x0i_value_count = x0i_value_count.reindex(uniqueVal, fill_value=0) + 1 

            # 不用条件概率相乘，取对数将连乘转化为连加防止下溢
            p1_list.append(p_xi(is_continuous, np.log(x1i_value_count / (m1 + uniqueValNum))))
            p0_list.append(p_xi(is_continuous, np.log(x0i_value_count / (m0 + uniqueValNum))))
    
    return p1, p1_list, p0_list

def predictNB(x, p1, p1_list, p0_list):

    n = len(x)
    # 初始化后验概率
    x_p1 = np.log(p1)
    x_p0 = np.log(1 - p1)

    for i in range(n):
        p1_xi = p1_list[i]
        p0_xi = p0_list[i]
        # 连续值算概率密度（正态分布）
        if p1_xi.is_continuous:
            mean1, var1 = p1_xi.conditional_pro
            mean0, var0 = p0_xi.conditional_pro
            x_p1 += np.log(1 / (np.sqrt(2 * np.pi) * var1) * np.exp(- (x[i] - mean1) ** 2 / 2 * var1 ** 2))
            x_p0 += np.log(1 / (np.sqrt(2 * np.pi) * var0) * np.exp(- (x[i] - mean0) ** 2 / 2 * var0 ** 2))

        # 离散值加上条件概率的log值
        else:
            x_p1 += p1_xi.conditional_pro[x[i]]
            x_p0 += p0_xi.conditional_pro[x[i]]

    if x_p1 > x_p0:
        return '是'
    else:
        return '否'

if __name__ == '__main__':
    # 读入数据
    dataPath = r'D:\\vscode\\Markdown\\ML\watermelon3.0.txt'
    dataSet = pd.read_csv(dataPath, index_col=0)
    # 去掉编号行
    dataSet.reset_index(inplace=True, drop=True)

    X = dataSet.iloc[:, :-1]
    # print(X)
    y = dataSet.iloc[:, -1]
    # print(y)
    
    p1, p1_list, p0_list = trainNB(X, y)
    # 测1其实就是第一个数据
    x_test = X.iloc[0, :]

    print(predictNB(x_test, p1, p1_list, p0_list))