import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics

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
# 标签编码
le = LabelEncoder()
y = le.fit_transform(y)
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

# generalization of test and train set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)

# model training
log_model = LogisticRegression() 
log_model.fit(X_train, y_train) 
print(log_model.coef_)
# model testing
y_pred = log_model.predict(X_test)

# summarize the accuracy of fitting
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


