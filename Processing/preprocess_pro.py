import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *

datasets = ['udds', 'us06', 'fuds']


def sliding_window(data, window_size, stride):
    """
    将时间序列数据切分成固定大小的时间窗口。
    滑窗从后往前滑
    :param data: 时间序列数据。
    :param window_size: 时间窗口的大小。
    :param stride: 窗口滑动的步长。
    :return: 分割后的时间窗口数据。
    """
    num_windows = (data.shape[0] - window_size) // stride + 1
    windows = np.zeros((num_windows, window_size, data.shape[1]))

    for i in range(num_windows - 1, -1, -1):
        start = i * stride
        end = start + window_size
        windows[i] = data[start:end]
    return windows


def extract_features(data):
    """
    提取时间序列数据的均值、方差、最大变化率。
    :param data: 时间序列数据。
    :return: 提取的特征矩阵。
    """
    mean = np.mean(data, axis=1, keepdims=True)
    var = np.var(data, axis=1, keepdims=True)

    # 计算累积变化率
    diff = np.diff(data, axis=0)
    cum_change_rate = np.cumsum(diff, axis=0)
    abs_cum_change_rate = np.abs(cum_change_rate)
    max_abs_cum_change_rate = np.max(abs_cum_change_rate, axis=1, keepdims=True)
    max_abs_cum_change_rate = np.insert(max_abs_cum_change_rate, 0, 0, axis=0)

    features = np.hstack((data, mean, var, max_abs_cum_change_rate))
    return features


def normalize_matrix(a):
    max_value = np.max(a)
    min_value = np.min(a)
    # 将每个元素除以矩阵的最大值和最小值之差，使得矩阵的数值范围变为 [0, 1] 之间
    a_normalized = (a - min_value) / (max_value - min_value)
    return a_normalized


def preprocess_dataset(dataset_folder, Classification, folder):
    file_list = os.listdir(dataset_folder)
    data_list = []
    isc_count, cor_count, nor_count, noi_count, sti_count = 0, 0, 0, 0, 0
    fault_count = 0

    for filename in file_list:
        if not filename.endswith('.csv'):
            continue
        file_path = os.path.join(dataset_folder, filename)
        values = filename.split('_')[1]
        data = pd.read_csv(file_path, header=0)

        win_data_list = sliding_window(data, 256, 5)

        for i, win_data in enumerate(win_data_list, start=1):
            features = normalize_matrix(win_data).astype('float')
            data_list.append(features)

            # 提取特征
            # features = extract_features(features)

            if Classification == 'multiple':
                if values == 'Isc':
                    i += isc_count
                elif values == 'Cor':
                    i += cor_count
                elif values == 'Nor':
                    i += nor_count
                elif values == 'noi':
                    i += noi_count
                elif values == 'sti':
                    i += sti_count
                save_folder = os.path.join(folder, values)
            elif Classification == 'single':
                if values == 'Isc' or values == 'Cor':
                    i += fault_count
                    savename = 'fault'
                elif values == 'Nor':
                    i += nor_count
                    savename = values
                save_folder = os.path.join(folder, savename)

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_filename = os.path.join(save_folder, f'{values}_{i}.npy')
            np.save(save_filename, features)

        if Classification == 'multiple':
            if values == 'Isc':
                isc_count += len(win_data_list)
            elif values == 'Cor':
                cor_count += len(win_data_list)
            elif values == 'Nor':
                nor_count += len(win_data_list)
            elif values == 'noi':
                noi_count += len(win_data_list)
            elif values == 'sti':
                sti_count += len(win_data_list)
        elif Classification == 'single':
            if values == 'Isc' or values == 'Cor':
                fault_count += len(win_data_list)
            elif values == 'Nor':
                nor_count += len(win_data_list)

    print(f'{dataset_folder} {Classification} preprocess ok！')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("Please provide dataset and Classification arguments.")

    dataset = sys.argv[1]
    Classification = sys.argv[2]

    if dataset == 'fuds':
        dataset_folder = '../data/fuds'
    elif dataset == 'udds':
        dataset_folder = '../data/udds'
    elif dataset == 'us06':
        dataset_folder = '../data/us06'
    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')

    folder = os.path.join(os.pardir, output_folder, dataset)
    os.makedirs(folder, exist_ok=True)

    preprocess_dataset(dataset_folder, Classification, folder)

