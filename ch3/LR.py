"""
data importation
"""

import numpy as np
import matplotlib.pyplot as plt

# load the CSV file as a numpy matrix
dataSet = np.loadtxt('D:\\vscode\Markdown\ML\watermelon_3a.csv', delimiter = ",")

# separate the data from the target attributes
X = dataSet[:, 1:3]
y = dataSet[:, 3]

# draw scatter diagram to show the raw data
f1 = plt.figure(1)
plt.title('watermelon_3a')  
plt.xlabel('density')  
plt.ylabel('ratio_sugar')

plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'k', s=100, label = 'bad')

plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=100, label = 'good')
plt.legend(loc = 'upper right')  
plt.show()


''' 
using sklearn lib for logistic regression
'''
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# generalization of test and train set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)

# model training
log_model = LogisticRegression() 
log_model.fit(X_train, y_train) 

# model testing
y_pred = log_model.predict(X_test)

# summarize the accuracy of fitting
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))