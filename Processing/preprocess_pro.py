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

        win_data_list = sliding_window(data, 300, 15)    # 滑窗获得数据

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


def swap_channel_data(dataset_folder):
    file_list = os.listdir(dataset_folder)
    for filename in file_list:
        if not filename.endswith('.csv'):
            continue
        file_path = os.path.join(dataset_folder, filename)
        values = filename.split('_')[1]

        if values != 'Nor':
            original_ch_num = file_path.split('_')[-1].replace('.csv', '')
            original_ch = 'CH500' + original_ch_num

            df = pd.read_csv(file_path)

            # 对每个非故障通道，生成一个新的CSV文件
            other_columns = [col for col in df.columns if col != original_ch]
            for new_fault_channel in other_columns:
                columns_order = [col for col in df.columns if col != new_fault_channel] + [new_fault_channel]
                new_df = df[columns_order]

                # 构建新的文件名
                new_fault_channel = new_fault_channel.replace('CH500', '', 1)
                new_filename = file_path.replace(original_ch_num, new_fault_channel)

                # 保存处理后的csv文件
                new_df.to_csv(new_filename, index=False)
                print(f"Generated file: {new_filename}")


if __name__ == '__main__':
    Classification = 'multiple'
    datasets = ['fuds', 'udds', 'us06']
    output_folders = ['../processed/fuds', '../processed/udds', '../processed/us06']

    for dataset, folder in zip(datasets, output_folders):
        if dataset == 'fuds':
            dataset_folder = '../data/fuds'
        elif dataset == 'udds':
            dataset_folder = '../data/udds'
        elif dataset == 'us06':
            dataset_folder = '../data/us06'
        else:
            raise Exception(f'Not Implemented. Check one of {datasets}')

        os.makedirs(folder, exist_ok=True)

        # swap_channel_data(dataset_folder)

        preprocess_dataset(dataset_folder, Classification, folder)

