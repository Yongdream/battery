import numpy as np
import matplotlib.pyplot as plt
from utlis import utils
from prettytable import PrettyTable


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list, title=''):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签
        self.title = title

    def update(self, preds, labels):
        for p, t in zip(preds, labels):     # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1          # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self, show=True):
        acc, summary_str, recall = utils.calculate_summary(self.matrix, self.labels)
        if show:
            print(summary_str)
        return str(acc), summary_str, recall

    def plot(self):  # 绘制混淆矩阵
        # 创建一个图形对象
        fig = plt.figure(figsize=(8, 6))
        # 在图形对象中创建一个Axes对象并获取它
        ax = fig.add_subplot(111)  # 注意这里的参数是子图的编号

        matrix = self.matrix
        # print(matrix)
        # 计算每一类的比率
        class_ratios = matrix / matrix.sum(axis=0, keepdims=True)
        acc, summary, recall_isc = self.summary()
        summary += "confusion_matrix:" + "\n" + str(matrix)

        ax.imshow(class_ratios, cmap=plt.cm.Purples)

        # 设置x轴坐标label
        ax.set_xticks(range(self.num_classes))
        ax.set_xticklabels(self.labels, rotation=45)
        # 设置y轴坐标label
        ax.set_yticks(range(self.num_classes))
        ax.set_yticklabels(self.labels)
        # 显示colorbar
        fig.colorbar(ax.imshow(class_ratios, cmap=plt.cm.Purples))
        ax.set_xlabel('True Labels')
        ax.set_ylabel('Predicted Labels')
        ax.set_title(self.title + ' Confusion matrix (acc=' + acc + ')')

        # 在图中标注数量/概率信息
        thresh = class_ratios.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                ratio = class_ratios[y, x]
                if ratio > 0:
                    ax.texts.append(ax.text(x, y, f"{info}\n({ratio:.2%})",
                             verticalalignment='center',
                             horizontalalignment='center',
                             color="white" if ratio > thresh else "black",
                             fontsize=8))  # 标注文字大小为8
                else:
                    ax.texts.append(ax.text(x, y, f"{info}",
                             verticalalignment='center',
                             horizontalalignment='center',
                             color="white" if ratio > thresh else "black",
                             fontsize=8))  # 标注文字大小为8
        fig.tight_layout()
        fig.show()
        return summary, recall_isc, fig
