import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

# sigmoid函数
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

# 损失函数
def J_cost(X, y, beta):
    # X加一列，用于矩阵相乘时带上偏置项
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    # 课本上的损失函数
    Lbeta = -y * np.dot(X_hat, beta) + np.log(1 + np.exp(np.dot(X_hat, beta)))

    return Lbeta.sum()

def gradient(X, y, beta):
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    p1 = sigmoid(np.dot(X_hat, beta))
    # 课本上损失函数的一阶导数
    gra = (-X_hat * (y - p1)).sum(0)

    return gra.reshape(-1, 1)

def update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost):
    # 梯度下降法更新梯度
    for i in range(num_iterations):

        grad = gradient(X, y, beta)
        beta = beta - learning_rate * grad

        if (i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))
        
    return beta

# 初始化系数
def initialize_beta(n):
    beta = np.random.randn(n + 1, 1) * 0.5 + 1
    return beta

def logistic_model(X, y, num_iterations=100, learning_rate=1.2, print_cost=False):
    m, n = X.shape
    beta = initialize_beta(n)

    return update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost)


def predict(X, beta):
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    p1 = sigmoid(np.dot(X_hat, beta))

    p1[p1 >= 0.5] = 1
    p1[p1 < 0.5] = 0

    return p1

# 读入数据
dataPath = r'D:\\vscode\\Markdown\\ML\watermelon3.0.txt'
dataSet = pd.read_csv(dataPath, index_col=0)
# 去掉编号行
dataSet.reset_index(inplace=True, drop=True)
# print(dataSet)
# 提取属性空间
X = dataSet.iloc[:, :-1]
# print(X)
# 提取标签，并转为0，1标签
y = dataSet.iloc[:, -1]
# print(y)
y[y == '是'] = 1
y[y == '否'] = 0
y = y.astype(int)
# print(y)

# 找出所有的离散属性
cat_cols = [col for col in X.columns if X[col].dtype == 'object']
# print(cat_cols)
# 对离散属性进行 One-Hot 编码
onehot_X = pd.get_dummies(X[cat_cols])
# print(onehot_X)
# 将原始 DataFrame 和 One-Hot 编码 DataFrame 进行合并
X = pd.concat([X, onehot_X], axis=1)
# print(X)
# 删除原始 DataFrame 中的离散属性列
X.drop(columns=cat_cols, inplace=True)
# print(X)

from sklearn import model_selection
from sklearn import metrics

# generalization of test and train set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
# print(X_train)
# print(y_train)
# model training
beta = logistic_model(X_train, y_train)
print(len(beta))
print(beta)
# model testing
y_pred = predict(X_test, beta)

# 输出测试集上的结果
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))