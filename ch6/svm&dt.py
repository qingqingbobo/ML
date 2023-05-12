import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate
from sklearn import svm, tree

# 导入数据
iris = datasets.load_iris()
# print(iris)
# 导入属性值，并标注属性名
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
# print(X.head())
# 导入标签
y = pd.Series(iris['target_names'][iris['target']])
# print(y.head())

# 线性核SVM的正确率
linear_svm = svm.SVC(C=1, kernel='linear')
linear_scores = cross_validate(linear_svm, X, y, cv=5, scoring='accuracy')
print('线性核SVM的正确率:', linear_scores['test_score'].mean())

# 高斯核SVM的正确率
rbf_svm = svm.SVC(C=1)
rbf_scores = cross_validate(rbf_svm, X, y, cv=5, scoring='accuracy')
print('高斯核SVM的正确率:', rbf_scores['test_score'].mean())

# 决策树正确率
clf = tree.DecisionTreeClassifier(random_state=42)

# Train the classifier on the training set
clf.fit(X, y)

# Compute the accuracy of the classifier
tree_scores = cross_validate(clf, X, y, cv=5, scoring='accuracy')
print('决策树的正确率:', tree_scores['test_score'].mean())