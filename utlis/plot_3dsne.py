import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
import warnings
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_embedding(data, label, classes, alpha=1.0, ax=None):
    """
    data为n * 3矩阵
    label为n * 1向量，对应着data的标签
    classes为类别列表，用于图例显示
    alpha为散点的透明度，用于调整目标域颜色的深浅
    ax为绘图的Axes对象
    """

    colors = plt.cm.Set3(np.unique(label))

    for i, c in zip(np.unique(label), colors):
        indices = np.where(label == i)
        ax.scatter(data[indices, 0], data[indices, 1], data[indices, 2], s=10, c=[c], alpha=alpha)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_3D(source_data, source_label, target_data, target_label, classes):
    print('Computing t-SNE embedding for source domain')
    tsne = TSNE(n_components=3, init='pca', random_state=0)

    source_result = tsne.fit_transform(source_data.copy())
    source_label = source_label.copy()

    print('Computing t-SNE embedding for target domain')
    target_result = tsne.fit_transform(target_data.copy())
    target_label = target_label.copy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Source and Target Domains')

    # 绘制源域数据
    plot_embedding(source_result, source_label, classes, ax=ax)
    source_legend = ax.legend(classes, loc='center left', bbox_to_anchor=(0.9, 1))

    # 绘制目标域数据，并将颜色相对于源域的颜色淡一些
    plot_embedding(target_result, target_label, classes, alpha=0.3, ax=ax)
    target_legend = ax.legend(classes, loc='center left', bbox_to_anchor=(0.9, 0.7))

    ax.add_artist(source_legend)
    ax.add_artist(target_legend)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

    return fig


# 生成源域数据
# np.random.seed(10)
# num_samples = 128
# feature = 128
# source_data, source_label = make_blobs(n_samples=num_samples, centers=5, n_features=256, random_state=0)
#
# # 生成目标域数据
# # np.random.seed(1)
# num_samples = 128
# # target_data = np.random.randn(num_samples, feature)
# # target_label = np.random.randint(0, 5, size=num_samples)
# target_data, target_label = make_blobs(n_samples=num_samples, centers=5, n_features=256, random_state=1)
#
# # 定义类别列表
# classes = ['Cor', 'Isc', 'Noi', 'Nor', 'Sti']
#
# # 调用绘图函数
# plot_3D(source_data, source_label, target_data, target_label, classes)
