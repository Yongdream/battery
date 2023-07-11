import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch import nn
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_embedding(data, label, classes, alpha=1.0, ax=None, target_marker='o', target_size=40, cmap='tab20'):
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
    colors = cmap(np.linspace(0, 0.5, num_labels))

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


def plot_label_2D(source_data, source_label, target_data, target_label, classes):
    tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)
    plot_only = 3000

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title('Source and Target Label')

    print('Computing t-SNE embedding for source domain')
    source_result = tsne.fit_transform(source_data[:plot_only, :].cpu().detach().numpy())
    source_label = source_label[:plot_only].cpu().detach().numpy()
    plot_embedding(source_result, source_label, classes, ax=ax, alpha=0.7)

    source_legend = ax.legend(classes, loc='center left', bbox_to_anchor=(0.95, 1))
    ax.add_artist(source_legend)

    print('Computing t-SNE embedding for target domain')
    target_result = tsne.fit_transform(target_data[:plot_only, :].cpu().detach().numpy())
    target_label = target_label[:plot_only].cpu().detach().numpy()
    plot_embedding(target_result, target_label, classes, ax=ax, alpha=0.7)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

    return fig


def plot_domain_2D(source_data, source_label, target_data, target_label, classes):
    tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)
    plot_only = 3000

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title('Source and Target Domain')

    print('Computing t-SNE embedding for source domain')
    source_domain_result = tsne.fit_transform(source_data[:plot_only, :].cpu().detach().numpy())
    source_domain_label = source_label[:plot_only].cpu().detach().numpy()
    plot_embedding(source_domain_result, source_domain_label, classes, ax=ax, cmap='Accent')

    print('Computing t-SNE embedding for target domain')
    target_result = tsne.fit_transform(target_data[:plot_only, :].cpu().detach().numpy())
    target_label = target_label[:plot_only].cpu().detach().numpy()
    plot_embedding(target_result, target_label, classes, ax=ax, cmap='Dark2')

    target_legend = ax.legend(classes, loc='center left', bbox_to_anchor=(0.95, 1))
    ax.add_artist(target_legend)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

    return fig

# def plot_label_2D(source_data, source_label, target_data, target_label, classes):
#     tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)
#     plot_only = 3000
#
#     fig = plt.figure(figsize=(15, 5))
#     gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
#
#     ax0 = plt.subplot(gs[0])
#     ax1 = plt.subplot(gs[1])
#     ax2 = plt.subplot(gs[2])
#
#     ax0.set_title('Source Label')
#     ax1.set_title('Target Label')
#     ax2.set_title('Combined Label')
#
#     print('Computing t-SNE embedding for source domain')
#     source_result = tsne.fit_transform(source_data[:plot_only, :].cpu().detach().numpy())
#     source_label = source_label[:plot_only].cpu().detach().numpy()
#     plot_embedding(source_result, source_label, classes, ax=ax0)
#
#     print('Computing t-SNE embedding for target domain')
#     target_result = tsne.fit_transform(target_data[:plot_only, :].cpu().detach().numpy())
#     target_label = target_label[:plot_only].cpu().detach().numpy()
#     plot_embedding(target_result, target_label, classes, ax=ax1)
#
#     combined_result = np.vstack((source_result, target_result))
#     combined_label = np.hstack((source_label, target_label))
#     plot_embedding(combined_result, combined_label, classes, ax=ax2)
#
#     source_legend = ax2.legend(classes, loc='center left', bbox_to_anchor=(0.95, 1))
#     ax2.add_artist(source_legend)
#
#     for ax in [ax0, ax1, ax2]:
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         ax.set_aspect('equal')
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     plt.subplots_adjust(wspace=0.3)
#     plt.show()
#
#     return fig



