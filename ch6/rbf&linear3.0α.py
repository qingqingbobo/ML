from sklearn import svm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

dataPath = r'D:\\vscode\\Markdown\\ML\watermelon3.0α.txt'
data = pd.read_table(dataPath, delimiter=' ', dtype = float)

X = data.iloc[:, [0, 1]].values
y = data.iloc[:, 2].values
# print(X)
# print(y)

y[y == 0] = -1
# print(y)

# 画图函数
def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.grid(axis='y', linestyle='-.')


def plt_support_(clf, X_, y_, kernel, c):
    pos = y_ == 1
    neg = y_ == -1
    ax = plt.subplot()

    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(0, 0.8, 600)

    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)

    Z_rbf = clf.predict(np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)

    # ax.contourf(X_, Y_, Z_rbf, alpha=0.75)
    cs = ax.contour(X_tmp, Y_tmp, Z_rbf, [0], colors='orange', linewidths=1)
    ax.clabel(cs, fmt={cs.levels[0]: 'decision boundary'})

    set_ax_gray(ax)

    ax.scatter(X_[pos, 0], X_[pos, 1], label='1', color='C0')
    ax.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')

    ax.scatter(X_[clf.support_, 0], X_[clf.support_, 1], marker='o', c='none', edgecolors='g', s=150,
               label='support_vectors')

    ax.legend()
    ax.set_title('{} kernel, C={}'.format(kernel, c))
    plt.show()

# C = 1，kernel='linear': 线性核函数
clf_linear = svm.SVC(C=1, kernel='linear')
clf_linear.fit(X, y.astype(int))
print('线性核SVM：')
# print('预测值：', clf_linear.predict(X))
# print('真实值：', y.astype(int))
print('支持向量索引值：', len(clf_linear.support_))

# C = 1，kernel='rbf': 高斯核函数
clf_rbf = svm.SVC(C=1, kernel='rbf', degree=3, gamma='scale')
clf_rbf.fit(X, y.astype(int))
print('高斯核SVM：')
# print('预测值：', clf_rbf.predict(X))
# print('真实值：', y.astype(int))
print('支持向量数目：', len(clf_rbf.support_))

# C = 100：经验误差小，kernel='linear': 线性核函数
clf_linear = svm.SVC(C=100, kernel='linear')
clf_linear.fit(X, y.astype(int))
print('线性核SVM：')
# print('预测值：', clf_linear.predict(X))
# print('真实值：', y.astype(int))
print('支持向量索引值：', len(clf_linear.support_))

# C = 100：经验误差小，kernel='rbf': 高斯核函数
clf_rbf = svm.SVC(C=100, kernel='rbf', degree=3, gamma='scale')
clf_rbf.fit(X, y.astype(int))
print('高斯核SVM：')
# print('预测值：', clf_rbf.predict(X))
# print('真实值：', y.astype(int))
print('支持向量索引值：', len(clf_rbf.support_))

C = 100
plt_support_(clf_rbf, X, y, 'rbf', C)
plt_support_(clf_linear, X, y, 'linear', C)