import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def softmax(scores):
    # 计算总和
    sum_exp = np.sum(np.exp(scores), axis = 1,keepdims = True)
    softmax = np.exp(scores) / sum_exp
    return softmax

def one_hot(label_arr, n_samples, n_classes):
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), label_arr.T] = 1
    return one_hot

def train(data_arr, label_arr, n_class, iters = 1000, alpha = 0.1, lam = 0.01):
    n_samples, n_features = data_arr.shape
    n_classes = n_class
    # 随机初始化权重矩阵
    weights = np.random.rand(n_class, n_features)
    # 定义损失结果
    all_loss = list()
    # 计算 one-hot 矩阵
    y_one_hot = one_hot(label_arr, n_samples, n_classes)
    for i in range(iters):
        # 计算 m * k 的分数矩阵
        scores = np.dot(data_arr, weights.T)
        # 计算 softmax 的值
        probs = softmax(scores)
        # 计算损失函数值
        loss = - (1.0 / n_samples) * np.sum(y_one_hot * np.log(probs))
        all_loss.append(loss)
        # 求解梯度
        dw = -(1.0 / n_samples) * np.dot((y_one_hot - probs).T, data_arr) + lam * weights
        dw[:,0] = dw[:,0] - lam * weights[:,0]
        # 更新权重矩阵
        weights  = weights - alpha * dw
    return weights, all_loss

def predict(test_dataset, label_arr, weights):
    scores = np.dot(test_dataset, weights.T)
    probs = softmax(scores)
    return np.argmax(probs, axis=1).reshape((-1,1))

if __name__ == "__main__":

    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # print(X_train)
    # print(y_train)
    y_train = np.array(y_train).reshape((-1,1))
    # print(y_train)
    y_val = np.array(y_val).reshape((-1,1))

    weights, all_loss = train(X_train, y_train, n_class = 4)
    print(weights)
    # 计算预测的准确率
    n_test_samples = X_val.shape[0]
    y_predict = predict(X_val, y_val, weights)
    accuray = np.sum(y_predict == y_val) / n_test_samples
    print(accuray)

    # 绘制损失函数
    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1000), all_loss)
    plt.title("Development of loss during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()