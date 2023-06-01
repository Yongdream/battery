import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_embedding(data, label, classes, alpha=1.0, ax=None, target_marker='o', target_size=80, cmap='Set3'):
    """
    data为n * 2矩阵
    label为n * 1向量，对应着data的标签
    classes为类别列表，用于图例显示
    alpha为散点的透明度，用于调整目标域颜色的深浅
    ax为绘图的Axes对象
    target_marker为目标域数据的标记类型，默认为圆形（'o'）
    target_size为目标域数据的标记大小，默认为10
    cmap为颜色映射名称，默认为'Set3'
    """

    cmap = plt.cm.get_cmap(cmap)
    unique_labels = np.unique(label)
    num_labels = len(unique_labels)
    colors = cmap(np.linspace(0.3, 0.5, num_labels))

    for i, c in zip(unique_labels, colors):
        indices = np.where(label == i)
        if ax is not None:
            if target_marker != 'o':
                # 绘制目标域数据为星星
                ax.scatter(data[indices, 0], data[indices, 1], marker=target_marker, s=target_size, c=[c], alpha=alpha)
            else:
                # 绘制源域数据为散点
                ax.scatter(data[indices, 0], data[indices, 1], s=target_size, c=[c], alpha=alpha)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_2D(source_data, source_label, target_data, target_label, classes):
    print('Computing t-SNE embedding for source domain')
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    source_result = tsne.fit_transform(source_data.copy())
    source_label = source_label.copy()

    print('Computing t-SNE embedding for target domain')
    target_result = tsne.fit_transform(target_data.copy())
    target_label = target_label.copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Source and Target Domains')

    plot_embedding(source_result, source_label, classes, ax=ax)
    source_legend = ax.legend(classes, loc='center left', bbox_to_anchor=(0.95, 1))

    plot_embedding(target_result, target_label, classes, alpha=0.3, ax=ax, target_marker='*')
    # target_legend = ax.legend(classes, loc='center left', bbox_to_anchor=(0.95, 0.7))

    ax.add_artist(source_legend)
    # ax.add_artist(target_legend)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

    return fig


# data = (b, feature)       label = (b, )
# num_samples = 128
# feature = 128

# # 生成源域数据和标签
# source_data = np.random.randn(num_samples, 128)
# source_label = np.random.randint(0, 5, size=num_samples)
# # source_data, source_label = make_blobs(n_samples=num_samples, centers=5, n_features=256, random_state=0)
#
# # 生成目标域数据和标签
# target_data = np.random.randn(num_samples, 128)
# target_label = np.random.randint(0, 5, size=num_samples)
# # target_data, target_label = make_blobs(n_samples=num_samples, centers=5, n_features=256, random_state=1)
#
# # 定义类别列表
# classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
#
# # 绘制源域和目标域的二维表示
# fig = plot_2D(source_data, source_label, target_data, target_label, classes)
# # plt.show()
