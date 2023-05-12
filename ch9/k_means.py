import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 初始化k个聚类中心点
def KMeansInitCentroids(X, k):
    # 随机选择k个聚类中心的索引
    kCentroidsIndex = np.random.randint(0, X.shape[0], size = k)
    # 返回k个聚类中心
    return X[kCentroidsIndex]

def findClosestCentroids(X, centroids):
    cluster = np.zeros(len(X)) #用于存储每个样本所属的聚类中心
    for i in range(len(X)):
        min_cluster = -1 #最近的聚类中心的编号
        min_dist = float('inf') #最近的距离

        for j in range(len(centroids)):
            dist = np.sum(np.power(X[i] - centroids[j], 2)) #计算样本到聚类中心的距离
            if dist < min_dist:
                min_cluster = j
                min_dist = dist

        cluster[i] = min_cluster #记录样本所属的聚类中心

    return cluster

def computeCentroids(X, cluster):
    k = set(np.ravel(cluster).tolist()) #聚类中心的编号
    k = list(k)

    centroids = np.ndarray((len(k), X.shape[1])) #用于存储每个聚类中心的坐标

    for i in range(len(k)):
        cluster_data = X[np.where(cluster == k[i])[0]] #当前聚类中心所包含的样本
        centroids[i] = np.sum(cluster_data, axis= 0) / len(cluster_data) #计算新的聚类中心坐标

    return centroids

def kMeans(X, num_cluster = 3, max_iters = 20):
    initCentroids = KMeansInitCentroids(X, num_cluster) #初始化聚类中心
    #迭代
    for i in range(max_iters):
        
        if(i == 0):
            centroids = initCentroids
            #print(centroids)
            
        #计算样本到聚类中心的距离，并返回每个样本所属的聚类中心
        cluster = findClosestCentroids(X, centroids)
        #重新计算聚类中心
        centroids = computeCentroids(X, cluster)

        print(f"Iteration {i+1}/{max_iters}: Cluster={cluster}, Centroids={centroids}")

    return cluster, centroids


if __name__ == "__main__":
    X = np.loadtxt('D:\\vscode\Markdown\ML\watermelon4.0.txt', delimiter=',')
    #执行聚类算法，返回聚类结果和聚类中心
    cluster, centroids = kMeans(X, 2, 20)
    #绘图
    plt.subplot(221)
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b', 'y'])
    #绘制样本点
    plt.scatter(X[:, 0], X[:, 1], c=np.ravel(cluster), cmap=cm_dark, s=20)
    #绘制中心点
    plt.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(len(centroids)), cmap=cm_dark, marker='*', s=500)

    cluster, centroids = kMeans(X, 3, 20)
    #绘图
    plt.subplot(222)
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b', 'y'])
    #绘制样本点
    plt.scatter(X[:, 0], X[:, 1], c=np.ravel(cluster), cmap=cm_dark, s=20)
    #绘制中心点
    plt.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(len(centroids)), cmap=cm_dark, marker='*', s=500)

    cluster, centroids = kMeans(X, 4, 20)
    #绘图
    plt.subplot(223)
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b', 'y'])
    #绘制样本点
    plt.scatter(X[:, 0], X[:, 1], c=np.ravel(cluster), cmap=cm_dark, s=20)
    #绘制中心点
    plt.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(len(centroids)), cmap=cm_dark, marker='*', s=500)
    plt.show()