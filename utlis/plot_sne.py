import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_embedding(data, label, classes, alpha=1.0, ax=None):
    """
    data为n * 2矩阵
    label为n * 1向量，对应着data的标签
    classes为类别列表，用于图例显示
    alpha为散点的透明度，用于调整目标域颜色的深浅
    ax为绘图的Axes对象
    """

    colors = plt.cm.Set3(np.unique(label))

    for i, c in zip(np.unique(label), colors):
        indices = np.where(label == i)
        ax.scatter(data[indices, 0], data[indices, 1], s=10, c=[c], alpha=alpha)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_2D(source_data, source_label, target_data, target_label, classes):
    print('Computing t-SNE embedding for source domain')
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    source_result = tsne.fit_transform(source_data.detach().cpu().numpy())
    source_label = source_label.detach().cpu().numpy()

    print('Computing t-SNE embedding for target domain')
    target_result = tsne.fit_transform(target_data.detach().cpu().numpy())
    target_label = target_label.detach().cpu().numpy()

    fig, ax = plt.subplots()
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


# # data = (b, feature)       label = (b, )
# # 生成源域数据和标签
# source_data = np.random.randn(200, 128)
# source_label = np.random.randint(0, 5, size=200)
#
# # 生成目标域数据和标签
# target_data = np.random.randn(150, 128)
# target_label = np.random.randint(0, 5, size=150)
#
# # 定义类别列表
# classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
#
# # 绘制源域和目标域的二维表示
# fig = plot_2D(source_data, source_label, target_data, target_label, classes)
# # plt.show()
