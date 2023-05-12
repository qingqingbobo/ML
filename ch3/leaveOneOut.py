import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut

data = np.loadtxt('D:\\vscode\Markdown\ML\Transfusion.txt', delimiter=',').astype(int)

X = data[:, : 4]
y = data[:, 4]
m, n = X.shape

# normalization
X = (X - X.mean(0)) / X.std(0)

# shuffle
index = np.arange(m)
np.random.shuffle(index)

X = X[index]
y = y[index]

# LOO
lr = linear_model.LogisticRegression(C = 2)

loo = LeaveOneOut()
accuracy = 0
for train, test in loo.split(X, y):
    lr_ = linear_model.LogisticRegression(C = 2)
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
    lr_.fit(X_train, y_train)

    accuracy += lr_.score(X_test, y_test)

print(1 - accuracy / m)

# LOO self
score_myLoo = []
for i in range(m):
    lr_ = linear_model.LogisticRegression(C = 2)
    X_test = X[i, :]
    y_test = y[i]

    X_train = np.delete(X, i, axis = 0)
    y_train = np.delete(y, i, axis = 0)

    lr_.fit(X_train, y_train)

    score_myLoo.append(int(lr_.predict(X_test.reshape(1, -1))) == y_test)

print(1 - np.mean(score_myLoo))