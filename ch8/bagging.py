import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import resample


def stumpClassify(X, dim, thresh_val, thresh_inequal):
    
    # 返回的是2D数组
    ret_array = np.ones((X.shape[0], 1))

    if thresh_inequal == 'lt':
        ret_array[X[:, dim] <= thresh_val] = -1
    else:
        ret_array[X[:, dim] > thresh_val] = -1
    
    # print(ret_array)
    return ret_array


def buildStump(X, y):
    m, n = X.shape
    # 决策树桩，只有一个划分属性
    best_stump = {}

    min_error = 1

    # 对每一维都判断是否为最优属性
    for dim in range(n):

        x_min = np.min(X[:, dim])
        x_max = np.max(X[:, dim])
        # 连续型变量的划分点
        split_points = [(x_max - x_min) / 20 * i + x_min for i in range(20)]
        
        # sorted_x = np.sort(X[:, dim])
        # split_points = [(sorted_x[i] + sorted_x[i + 1]) / 2 for i in range(m - 1)]
        
        # 划分点以上为正，还是一下为正
        for inequal in ['lt', 'gt']:
            for thresh_val in split_points:

                # 计算错误率
                ret_array = stumpClassify(X, dim, thresh_val, inequal)
                error = np.mean(ret_array != y)

                if error < min_error:
                    best_stump['dim'] = dim
                    best_stump['thresh'] = thresh_val
                    best_stump['inequal'] = inequal
                    best_stump['error'] = error
                    min_error = error
    
    # print(best_stump)
    return best_stump


def stumpBagging(X, y, nums=20):
    stumps = []
    seed = 16
    for _ in range(nums):
        # sklearn 中自带的实现自助采样的方法
        X_, y_ = resample(X, y, random_state=seed)  
        seed += 1
        # 添加基学习器
        stumps.append(buildStump(X_, y_))
    
    return stumps


def stumpPredict(X, stumps):
    ret_arrays = np.ones((X.shape[0], len(stumps)))

    for i, stump in enumerate(stumps):
        # 因为stumpClassify返回的是2D数组，所以用ret_arrays[:, [i]]
        # ret_arrays[:, i]返回的是1D数组
        ret_arrays[:, [i]] = stumpClassify(X, stump['dim'], stump['thresh'], stump['inequal'])

    return np.sign(np.sum(ret_arrays, axis=1))


def pltStumpBaggingDecisionBound(X_, y_, stumps):
    pos = y_ == 1
    neg = y_ == -1
    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(-0.1, 0.7, 600)

    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
    Z_ = stumpPredict(np.c_[X_tmp.ravel(), Y_tmp.ravel()], stumps).reshape(X_tmp.shape)

    plt.contour(X_tmp, Y_tmp, Z_, [0], colors='orange', linewidths=1)

    plt.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')
    plt.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_path = r'D:\\vscode\\Markdown\\ML\watermelon3.0α.txt'

    data = pd.read_table(data_path, delimiter=' ')

    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values

    y[y == 0] = -1

    stumps = stumpBagging(X, y, 21)

    print(np.mean(stumpPredict(X, stumps) == y))
    pltStumpBaggingDecisionBound(X, y, stumps)