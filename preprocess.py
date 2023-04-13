import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile
from sklearn.model_selection import train_test_split

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']


def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape


def load_and_save2(category, filename, dataset, dataset_folder, shape):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(':')[0], line.split(':')[1].split(',')
        start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i) - 1 for i in values]
        temp[start - 1:end - 1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)


def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return (a / 2 + 0.5)


def normalize2(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a


def normalize3(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


# 避免出现除数为零的情况，如果训练集中的某个特征的最大值和最小值相等，那么在进行归一化处理时，
# 分母 max_a - min_a 就会为零，导致除数为零错误。加上一个极小的值 0.0001 可以避免这种错误的发生，保证程序的稳定性


def noemalize_matrix(a):
    max_value = np.max(a)
    min_value = np.min(a)
    # 将每个元素除以矩阵的最大值和最小值之差，使得矩阵的数值范围变为 [0, 1] 之间
    a_normalized = (a - min_value) / (max_value - min_value)
    return a_normalized


def convertNumpy(df):
    x = df[df.columns[3:]].values[::10, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)


def sliding_window(data, window_size, stride):
    """
    将时间序列数据切分成固定大小的时间窗口。
    :param data: 时间序列数据。
    :param window_size: 时间窗口的大小。
    :param stride: 窗口滑动的步长。
    :return: 分割后的时间窗口数据。
    """
    num_windows = (data.shape[0] - window_size) // stride + 1
    windows = np.zeros((num_windows, window_size, data.shape[1]))

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windows[i] = data[start:end]
    return windows


def load_data(dataset, Classification):
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    i = 1
    if dataset == 'synthetic':
        train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
        test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
        dat = pd.read_csv(train_file, header=None)
        split = 10000
        train = normalize(dat.values[:, :split].reshape(split, -1))
        test = normalize(dat.values[:, split:].reshape(split, -1))
        lab = pd.read_csv(test_labels, header=None)
        lab[0] -= split
        labels = np.zeros(test.shape)
        for i in range(lab.shape[0]):
            point = lab.values[i][0]
            labels[point - 30:point + 30, lab.values[i][1:]] = 1
        test += labels * np.random.normal(0.75, 0.1, test.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'SMD':
        dataset_folder = 'data/SMD'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                # filename.strip('.txt') 是去掉文件扩展名后的文件名，
                s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
    elif dataset == 'UCR':
        dataset_folder = 'data/UCR'
        file_list = os.listdir(dataset_folder)
        for filename in file_list:
            if not filename.endswith('.txt'): continue
            vals = filename.split('.')[0].split('_')
            dnum, vals = int(vals[0]), vals[-3:]
            vals = [int(i) for i in vals]
            temp = np.genfromtxt(os.path.join(dataset_folder, filename),
                                 dtype=np.float64,
                                 delimiter=',')
            min_temp, max_temp = np.min(temp), np.max(temp)
            temp = (temp - min_temp) / (max_temp - min_temp)
            train, test = temp[:vals[0]], temp[vals[0]:]
            labels = np.zeros_like(test)
            labels[vals[1] - vals[0]:vals[2] - vals[0]] = 1
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
    elif dataset == 'NAB':
        dataset_folder = 'data/NAB'
        file_list = os.listdir(dataset_folder)
        with open(dataset_folder + '/labels.json') as f:
            labeldict = json.load(f)
        for filename in file_list:
            if not filename.endswith('.csv'): continue
            df = pd.read_csv(dataset_folder + '/' + filename)
            vals = df.values[:, 1]
            labels = np.zeros_like(vals, dtype=np.float64)
            for timestamp in labeldict['realKnownCause/' + filename]:
                tstamp = timestamp.replace('.000000', '')
                index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
                labels[index - 4:index + 4] = 1
            min_temp, max_temp = np.min(vals), np.max(vals)
            vals = (vals - min_temp) / (max_temp - min_temp)
            train, test = vals.astype(float), vals.astype(float)
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            fn = filename.replace('.csv', '')
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
    elif dataset == 'MSDS':
        dataset_folder = 'data/MSDS'
        df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
        df_test = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
        df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
        _, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
        train, _, _ = normalize3(df_train, min_a, max_a)
        test, _, _ = normalize3(df_test, min_a, max_a)
        labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
        labels = labels.values[::1, 1:]
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
    elif dataset == 'SWaT':
        dataset_folder = 'data/SWaT'
        file = os.path.join(dataset_folder, 'series.json')
        df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
        df_test = pd.read_json(file, lines=True)[['val']][7000:12000]
        train, min_a, max_a = normalize2(df_train.values)
        test, _, _ = normalize2(df_test.values, min_a, max_a)
        labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset in ['SMAP', 'MSL']:
        dataset_folder = 'data/SMAP_MSL'
        file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
        values = pd.read_csv(file)
        values = values[values['spacecraft'] == dataset]  # 截取‘spacecraft’列等于dataset的数据块
        filenames = values['chan_id'].values.tolist()  # 将 chan_id 列的值转换为列表,唯一编号
        for fn in filenames:
            train = np.load(f'{dataset_folder}/train/{fn}.npy')
            test = np.load(f'{dataset_folder}/test/{fn}.npy')
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test, min_a, max_a)  # 根据训练数据集的最小最大进行归一化
            np.save(f'{folder}/{fn}_train.npy', train)
            np.save(f'{folder}/{fn}_test.npy', test)
            labels = np.zeros(test.shape)
            indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
            indices = indices.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i + 1], :] = 1
            np.save(f'{folder}/{fn}_labels.npy', labels)
    elif dataset == 'WADI':
        dataset_folder = 'data/WADI'
        ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
        train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
        # skiprows=1000：跳过 CSV 文件中的前 1000 行数据，不读取它们。nrows=2e5：只读取 CSV 文件中的 200000 行数据
        test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
        train.dropna(how='all', inplace=True);
        test.dropna(how='all', inplace=True)
        train.fillna(0, inplace=True);
        test.fillna(0, inplace=True)
        test['Time'] = test['Time'].astype(str)
        test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
        labels = test.copy(deep=True)
        for i in test.columns.tolist()[3:]: labels[i] = 0
        for i in ['Start Time', 'End Time']:
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
        for index, row in ls.iterrows():
            to_match = row['Affected'].split(', ')
            matched = []
            for i in test.columns.tolist()[3:]:
                for tm in to_match:
                    if tm in i:
                        matched.append(i);
                        break
            st, et = str(row['Start Time']), str(row['End Time'])
            labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
        train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
        print(train.shape, test.shape, labels.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'MBA':
        dataset_folder = 'data/MBA'
        ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
        train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
        test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
        train, test = train.values[1:, 1:].astype(float), test.values[1:, 1:].astype(float)
        train, min_a, max_a = normalize3(train)
        test, _, _ = normalize3(test, min_a, max_a)
        ls = ls.values[:, 1].astype(int)
        labels = np.zeros_like(test)
        for i in range(-20, 20):
            labels[ls + i, :] = 1
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))

    # fuds preprocess
    elif dataset == 'fuds':
        dataset_folder = 'data/fuds'
        file_list = os.listdir(dataset_folder)
        data_list = []
        if Classification == 'multiple':
            isc_count, cor_count, nor_count, noi_count, sti_cont = 0, 0, 0, 0, 0
            for filename in file_list:
                if not filename.endswith('.csv'): continue
                file_path = os.path.join(dataset_folder, filename)
                values = filename.split('_')[1]
                data = pd.read_csv(file_path, header=0)

                win_data_list = sliding_window(data, 225, 10)
                for i, win_data in enumerate(win_data_list, start=1):
                    train = noemalize_matrix(win_data).astype('float')
                    data_list.append(train)
                    if values == 'Isc':
                        i = i + isc_count
                    if values == 'Cor':
                        i = i + cor_count
                    if values == 'Nor':
                        i = i + nor_count
                    if values == 'noi':
                        i = i + noi_count
                    if values == 'sti':
                        i = i + sti_cont
                    save_folder = os.path.join(folder, values)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_filename = os.path.join(save_folder, f'{values}_{i}.npy')
                    np.save(save_filename, train)

                if values == 'Isc':
                    isc_count += len(win_data_list)
                if values == 'Cor':
                    cor_count += len(win_data_list)
                elif values == 'Nor':
                    nor_count += len(win_data_list)
        elif Classification == 'single':
            fault_count, nor_count = 0, 0
            for filename in file_list:
                if not filename.endswith('.csv'): continue
                file_path = os.path.join(dataset_folder, filename)
                values = filename.split('_')[1]
                data = pd.read_csv(file_path, header=0)
                # data = data.iloc[1:, 2:-5]

                win_data_list = sliding_window(data, 225, 10)
                for i, win_data in enumerate(win_data_list, start=1):
                    train = noemalize_matrix(win_data).astype('float')
                    data_list.append(train)
                    if values == 'Isc' or 'Cor':
                        i = i + fault_count
                        savename = 'fault'
                    if values == 'Nor':
                        i = i + nor_count
                        savename = values

                    save_folder = os.path.join(folder, savename)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_filename = os.path.join(save_folder, f'{savename}_{i}.npy')
                    np.save(save_filename, train)

                if values == 'Isc' or 'Cor':
                    fault_count += len(win_data_list)
                elif values == 'Nor':
                    nor_count += len(win_data_list)
        print('preprocess ok！')

    # udds preprocess
    elif dataset == 'udds':
        dataset_folder = 'data/udds'
        file_list = os.listdir(dataset_folder)
        data_list = []
        if Classification == 'multiple':
            isc_count, cor_count, nor_count, noi_count, sti_cont  = 0, 0, 0, 0, 0
            for filename in file_list:
                if not filename.endswith('.csv'): continue
                file_path = os.path.join(dataset_folder, filename)
                values = filename.split('_')[1]
                data = pd.read_csv(file_path, header=0)

                win_data_list = sliding_window(data, 225, 10)

                for i, win_data in enumerate(win_data_list, start=1):
                    train = noemalize_matrix(win_data).astype('float')
                    data_list.append(train)
                    if values == 'Isc':
                        i = i + isc_count
                    if values == 'Cor':
                        i = i + cor_count
                    if values == 'Nor':
                        i = i + nor_count
                    if values == 'noi':
                        i = i + noi_count
                    if values == 'sti':
                        i = i + sti_cont
                    save_folder = os.path.join(folder, values)

                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_filename = os.path.join(save_folder, f'{values}_{i}.npy')
                    np.save(save_filename, train)
                if values == 'Isc':
                    isc_count += len(win_data_list)
                if values == 'Cor':
                    cor_count += len(win_data_list)
                elif values == 'Nor':
                    nor_count += len(win_data_list)
        elif Classification == 'single':
            fault_count, nor_count = 0, 0
            for filename in file_list:
                if not filename.endswith('.csv'): continue
                file_path = os.path.join(dataset_folder, filename)
                values = filename.split('_')[1]
                data = pd.read_csv(file_path, header=0)

                win_data_list = sliding_window(data, 225, 10)

                for i, win_data in enumerate(win_data_list, start=1):
                    # train = noemalize_matrix(win_data).astype('float')
                    train = win_data.astype('float')
                    data_list.append(train)
                    if values == 'Isc' or 'Cor':
                        i = i + fault_count
                        savename = 'fault'
                    if values == 'Nor':
                        i = i + nor_count
                        savename = values
                    save_folder = os.path.join(folder, savename)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_filename = os.path.join(save_folder, f'{savename}_{i}.npy')
                    np.save(save_filename, train)
                if values == 'Isc' or 'Cor':
                    fault_count += len(win_data_list)
                elif values == 'Nor':
                    nor_count += len(win_data_list)

        print('preprocess ok！')

    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')


if __name__ == '__main__':
    commands = sys.argv[1]
    Classification = sys.argv[2]
    load = []
    if len(commands) > 0:
        load_data(commands, Classification)

    # commands = sys.argv[1:]
    # load = []
    # if len(commands) > 0:
    # 	for d in commands:
    # 		load_data(d)
    else:
        print("Usage: python preprocess.py <datasets>")
        print(f"where <datasets> is space separated list of {datasets}")
