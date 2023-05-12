import numpy as np

class DecisionTreeRegression:
    def __init__(self, max_depth=5, min_samples_split=2):
        # 树的最大深度
        self.max_depth = max_depth
        # 分支最少需要节点
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_row(row, self.root) for _, row in X.iterrows()])

    def _build_tree(self, X, y, depth=0):
        # 达到最大深度或者样本数过少
        if depth == self.max_depth or len(X) < self.min_samples_split:
            return np.mean(y)

        # 选出最优属性和最优划分点
        best_feature, best_value = self._choose_best_split(X, y)

        # 没有最优属性
        if best_feature is None:
            return np.mean(y)

        # 分支，左子树和右子树索引
        left_idx = X[best_feature] < best_value
        right_idx = X[best_feature] >= best_value

        left = self._build_tree(X[left_idx], y[left_idx], depth+1)
        right = self._build_tree(X[right_idx], y[right_idx], depth+1)

        return {'feature': best_feature, 'value': best_value, 'left': left, 'right': right}

    def _choose_best_split(self, X, y):
        # 初始化最优特征，最优划分点，最小均方误差
        best_feature, best_value, best_mse = None, None, float('inf')
        # 遍历每一个特征
        for feature in X.columns:
            for value in X[feature].unique():
                left_idx = X[feature] < value
                right_idx = X[feature] >= value

                if len(y[left_idx]) > 0 and len(y[right_idx]) > 0:
                    left_mse = np.mean((y[left_idx] - np.mean(y[left_idx]))**2)
                    right_mse = np.mean((y[right_idx] - np.mean(y[right_idx]))**2)
                    mse = left_mse + right_mse

                    if mse < best_mse:
                        best_feature, best_value, best_mse = feature, value, mse

        return best_feature, best_value

    def _predict_row(self, row, node):
        if isinstance(node, np.float64):
            return node

        if row[node['feature']] < node['value']:
            return self._predict_row(row, node['left'])
        else:
            return self._predict_row(row, node['right'])



if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    import pandas as pd

    housing_data = fetch_california_housing()
    X = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
    y = housing_data.target

    # print(X)
    # print(len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=0)

    # Set the number of samples to be used
    n_train = 100
    n_test = 30

    # Select a subset of samples from X_train
    X_train_subset = X_train.iloc[:n_train, :]

    # Select a subset of corresponding samples from y_train
    y_train_subset = y_train[:n_train]
    print(X_train_subset)
    print(y_train_subset)
    # Select a subset of samples from X_test
    X_test_subset = X_test.iloc[:n_test, :]

    # Select a subset of corresponding samples from y_test
    y_test_subset = y_test[:n_test]

    # Fit your decision tree model on the subset of training data
    tree = DecisionTreeRegression(max_depth=5)
    tree.fit(X_train_subset, y_train_subset)

    # Predict on the subset of test data
    y_pred_subset = tree.predict(X_test_subset)

    mse = np.mean((y_pred_subset - y_test_subset)**2)
    print(mse)

    

