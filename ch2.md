###第二章作业
####课后习题
#####2.1
$\because$ 留出法使用分层采样
$\therefore$ 训练集中有350个正例，350个反例
**$C_{500}^{350} * C_{500}^{350}$种划分方式**
#####2.2
**10折交叉验证法错误率为50%**
$\because$采用分层采样，每一折正反例各一半，训练集样本数相同
$\therefore$ 随机猜测，错误率为50%
**留一法错误率为100%**
$\because$ 训练集99个，少了一个测试集的样例，训练集中与测试集相反的样本占多数
$\therefore$ 总是预测为与测试集相反的结果，错误率为100%

#####2.3
**不一定**
$\because$ F1的值与PR图中选择的点有关，不确定。而BEP是P = R时的点

#####2.4
TPR = TP / (TP + FN)
R = TP / (TP + FN)
**$\therefore$ TPR = R**
FPR = FP / (FP + TN)
P = TP / (TP + FP)
**FPR与P、R无明显关系**

#####2.6
错误率 = 1 - 查准率
ROC曲线是对样本取正例的概率排序，每次取其中一个样本的概率作为临界值，算得预测正例和反例，与样本真正值比较，可以得出错误率
**$\therefore$ ROC曲线上的每一个点，都对应着一个错误率**

#####2.8
**Min-max 规范化**
**优点**
1、计算相对简单
2、当新样本进来时，只有在新样本大于原最大值或者小于原最小值时，才需要重新计算规范化之后的值
**缺点**
1、容易受异常值影响

**z-score 规范化**
**优点**
1、受异常值影响小
**缺点**
1、计算相对复杂
2、每次新样本进来都需要重新计算规范化

####思考题
对chatGPT：
第一印象很厉害。南方周末、澎湃新闻有文章说chatGPT很厉害，可以续写《红楼梦》。
用过觉得不错。用百度搜索有关指令集的叙述题，查准率低；但GPT能给出合理说法。
现在不想用GPT，不能把握住把它当成助手还是依赖它。
如果不学点真本事出来，估计要被淘汰了:)