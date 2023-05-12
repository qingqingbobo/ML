import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

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

# 10 cross validation
lr = linear_model.LogisticRegression(C = 2)
score = cross_val_score(lr, X, y, cv = 10)
print(1 - score.mean())


# 10 cross self
num_split = int(m / 10)
myScore = []
for i in range(10):
    lr = linear_model.LogisticRegression(C = 2)

    test_index = range(i * num_split, (i + 1) * num_split)
    X_test_ = X[test_index]
    y_test_ = y[test_index]

    X_train_ = np.delete(X, test_index, axis=0)
    y_train_ = np.delete(y, test_index, axis=0)

    lr.fit(X_train_, y_train_)
    myScore.append(lr.score(X_test_, y_test_))

print(1 - np.mean(myScore))
