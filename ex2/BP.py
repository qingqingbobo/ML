import numpy as np
import copy
import pandas as pd
import bpnnUtil

class BpNet(object):
    # 初始化
    def __init__(self,  layer_dims_, learning_rate=0.1, seed=16, initializer='he', optimizer='gd'):
        # 每一层的神经元数
        self.layer_dims_ = layer_dims_
        self.learning_rate = learning_rate
        self.seed = seed
        self.initializer = initializer
        # 优化方法
        self.optimizer = optimizer
    
    # 调试参数
    def fit(self, X_, y_, num_epochs=100):
        # m样本数，n属性数
        m, n = X_.shape
        layer_dims_ = copy.deepcopy(self.layer_dims_)
        # 增加输入层
        layer_dims_.insert(0, n)

        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)

        # 初始化每一层的参数
        self.parameters_ = bpnnUtil.he_initializer(layer_dims_, self.seed)

                # 选择优化方案
        assert self.optimizer in ('gd', 'sgd')
        if self.optimizer == 'gd':
            parameters_, costs = self.optimizer_gd(X_, y_, self.parameters_, num_epochs, self.learning_rate)
        else:
            parameters_, costs = self.optimizer_sgd(X_, y_, self.parameters_, num_epochs, self.learning_rate, self.seed)

        self.parameters_ = parameters_
        self.costs = costs

        return self
    
    def predict(self, X_):
            if not hasattr(self, "parameters_"):
                raise Exception('you have to fit first before predict.')

            a_last, _ = self.forward_L_layer(X_, self.parameters_)
            if a_last.shape[1] == 1:
                predict_ = np.zeros(a_last.shape)
                predict_[a_last>=0.5] = 1
            else:
                predict_ = np.argmax(a_last, axis=1)
            return predict_

    def compute_cost(self, y_hat_, y_):
        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)
        if y_.shape[1] == 1:
            cost = bpnnUtil.cross_entry_sigmoid(y_hat_, y_)
        else:
            cost = bpnnUtil.cross_entry_softmax(y_hat_, y_)
        return cost

    def forward_one_layer(self, a_pre_, w_, b_, activation_):
        # 计算激活前的输出值，矩阵点乘，结果的维度与下一层的维度相同
        z_ = np.dot(a_pre_, w_.T) + b_
        assert activation_ in ('sigmoid', 'relu', 'softmax')

        # 激活
        if activation_ == 'sigmoid':
            a_ = bpnnUtil.sigmoid(z_)
        elif activation_ == 'relu':
            a_ = bpnnUtil.relu(z_)
        else:
            a_ = bpnnUtil.softmax(z_)

        # 存入本层参数
        cache_ = (a_pre_, w_, b_, z_)  # 将向前传播过程中产生的数据保存下来，在向后传播过程计算梯度的时候要用上的。
        return a_, cache_
        

    def forward_L_layer(self, X_, parameters_):
        # 需要前传的层数
        L_ = int(len(parameters_) / 2)
        # 初始化每层的参数
        caches = []
        # 输入值开始
        a_ = X_
        
        for i in range(1, L_):
            w_ = parameters_['W' + str(i)]
            b_ = parameters_['b' + str(i)]
            a_pre_ = a_
            a_, cache_ = self.forward_one_layer(a_pre_, w_, b_, 'relu')
            caches.append(cache_)
        
        w_last = parameters_['W' + str(L_)]
        b_last = parameters_['b' + str(L_)]

        # 最后是一个节点说明是二分类，用sigmoid
        # 多个节点是多分类，用softmax
        if w_last.shape[0] == 1:
            a_last, cache_ = self.forward_one_layer(a_, w_last, b_last, 'sigmoid')
        else:
            a_last, cache_ = self.forward_one_layer(a_, w_last, b_last, 'softmax')

        caches.append(cache_)
        return a_last, caches

    def backward_one_layer(self, da_, cache_, activation_):
        # 当前层参数值
        (a_pre_, w_, b_, z_) = cache_
        m = da_.shape[0]

        assert activation_ in ('sigmoid', 'relu', 'softmax')

        # 算出本层未激活前函数的导数值
        if activation_ == 'sigmoid':
            dz_ = bpnnUtil.sigmoid_backward(da_, z_)
        elif activation_ == 'relu':
            dz_ = bpnnUtil.relu_backward(da_, z_)
        else:
            dz_ = bpnnUtil.softmax_backward(da_, z_)

        # 计算本层其他参数的导数值
        # z = w*x + b
        # dw = dz * x
        # dx = dz * w
        dw = np.dot(dz_.T, a_pre_) / m
        db = np.sum(dz_, axis=0, keepdims=True) / m
        da_pre = np.dot(dz_, w_)

        assert dw.shape == w_.shape
        assert db.shape == b_.shape
        assert da_pre.shape == a_pre_.shape

        return da_pre, dw, db

    # 向后传播
    def backward_L_layer(self, a_last, y_, caches):
        grads = {}
        L = len(caches)

        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)

        if y_.shape[1] == 1:  # 目标值只有一列表示为二分类
            # 计算输出错误率对输出值的导数
            da_last = -(y_ / a_last - (1 - y_) / (1 - a_last))
            # 计算输出层参数的导数和隐层输出值的导数
            da_pre_L_1, dwL_, dbL_ = self.backward_one_layer(da_last, caches[L - 1], 'sigmoid')

        else:  # 经过one hot，表示为多分类
            # 在计算softmax的梯度时，可以直接用 dz = a - y可计算出交叉熵损失函数对z的偏导， 所以这里第一个参数输入直接为y_
            da_pre_L_1, dwL_, dbL_ = self.backward_one_layer(y_, caches[L - 1], 'softmax')

        # 把Δ存入梯度
        grads['da' + str(L)] = da_pre_L_1
        grads['dW' + str(L)] = dwL_
        grads['db' + str(L)] = dbL_

        # 往后传播，并存入梯度
        for i in range(L - 1, 0, -1):
            da_pre_, dw, db = self.backward_one_layer(grads['da' + str(i + 1)], caches[i - 1], 'relu')

            grads['da' + str(i)] = da_pre_
            grads['dW' + str(i)] = dw
            grads['db' + str(i)] = db

        return grads

    def optimizer_gd(self, X_, y_, parameters_, num_epochs, learning_rate):
        costs = []
        # 训练epochs次
        for i in range(num_epochs):
            # 前传，并缓存参数
            a_last, caches = self.forward_L_layer(X_, parameters_)
            # 后传，返回梯度
            grads = self.backward_L_layer(a_last, y_, caches)
            # 更新参数
            parameters_ = bpnnUtil.update_parameters_with_gd(parameters_, grads, learning_rate)
            
            # 计算损失函数
            cost = self.compute_cost(a_last, y_)
            costs.append(cost)

        return parameters_, costs

    def optimizer_sgd(self, X_, y_, parameters_, num_epochs, learning_rate, seed):
        '''
        sgd中，更新参数步骤和gd是一致的，只不过在计算梯度的时候是用一个样本而已。
        '''
        np.random.seed(seed)
        costs = []
        m_ = X_.shape[0]
        for _ in range(num_epochs):

            # 每次随机选择一些样本来训练
            random_index = np.random.randint(0, m_)

            a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_)
            grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)

            parameters_ = bpnnUtil.update_parameters_with_sgd(parameters_, grads, learning_rate)

            a_last_cost, _ = self.forward_L_layer(X_, parameters_)

            cost = self.compute_cost(a_last_cost, y_)

            costs.append(cost)

        return parameters_, costs

if __name__ == '__main__':
    # from sklearn import datasets
    # from sklearn.model_selection import train_test_split
    # from sklearn.metrics import accuracy_score

    # iris = datasets.load_iris()
    # X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    # X = (X - np.mean(X, axis=0)) / np.var(X, axis=0)

    # y = pd.Series(iris['target_names'][iris['target']])
    # y = pd.get_dummies(y)
    # # print(y)
    # # print(y.values)
    # # generalization of test and train set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # bp = BpNet([10, 3], learning_rate=0.003, optimizer='gd')
    # bp.fit(X_train.values, y_train.values, num_epochs=2000)
    # y_pred = bp.predict(X_test)
    # # print(y_pred.shape)
    # # print(y_pred)
    # # print(y_test.shape)
    # # print(y_test)
    # y_test_max = np.argmax(y_test.values, axis=1)
    # print(y_test_max)
    # accuracy = accuracy_score(y_test_max, y_pred)
    # print("Accuracy of GD:", accuracy)

    # bp1 = BpNet([10, 3], learning_rate=0.003, optimizer='sgd')
    # bp1.fit(X.values, y.values, num_epochs=2000)
    # y_pred1 = bp1.predict(X_test)
    # accuracy1 = accuracy_score(y_test_max, y_pred1)
    # print("Accuracy of SGD:", accuracy1)
    
    # bpnnUtil.plot_costs([bp.costs, bp1.costs], ['gd_cost', 'sgd_cost'])

    data = np.loadtxt('D:\\vscode\Markdown\ML\Transfusion.txt', delimiter=',').astype(int)

    X = data[:, : 4]
    y = data[:, 4]

    # normalization
    X = (X - X.mean(0)) / X.std(0)
    # 隐层太窄，cost有0.5
    bp = BpNet([10, 1], learning_rate=0.003, optimizer='gd')
    bp.fit(X, y, num_epochs=2000)
    
    bp1 = BpNet([10, 1], learning_rate=0.003, optimizer='sgd')
    bp1.fit(X, y, num_epochs=2000)
    
    bpnnUtil.plot_costs([bp.costs, bp1.costs], ['gd_cost', 'sgd_cost'])