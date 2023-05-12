import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import model_selection
from sklearn import metrics

# load the CSV file as a numpy matrix
dataSet = np.loadtxt('D:\\vscode\Markdown\ML\watermelon_3a.csv', delimiter = ",")

# separate the data from the target attributes
X = dataSet[:, 1:3]
y = dataSet[:, 3]

# generalization of test and train set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)

# model training
clf = LDA()
clf.fit(X_train, y_train)

# model testing
y_pred = clf.predict(X_test)
print(y_pred)
print(1 - clf.score(X_train, y_train))

# summarize the accuracy of fitting
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
