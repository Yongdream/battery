import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import os
import re
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler



class WaveletTransformDataset:
    def __init__(self, data_df, wavelet='sym4', level=2):
        self.data_df = data_df
        self.wavelet = wavelet
        self.level = level

    def get_transformed_data(self):
        wavelet = pywt.Wavelet(self.wavelet)
        level = self.level
        X_transformed = np.zeros((self.data_df.shape[0], self.data_df.shape[1]))

        for i in range(0, self.data_df.shape[1]):
            temp = self.data_df.iloc[:, i]
            coeffs = pywt.wavedec(self.data_df.iloc[:, i], wavelet, level)
            x_reconstructed = pywt.waverec(coeffs, wavelet)
            if x_reconstructed.shape[0] == X_transformed.shape[0]:
                X_transformed[:, i] = x_reconstructed
            else:
                X_transformed[:, i] = x_reconstructed[:-1]

        return pd.DataFrame(X_transformed, columns=self.data_df.columns)


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


# 定义文件夹路径
folder_path = '../data/udds/'
win_data = []
onehot_data = np.empty((0, 1, 3))
value_to_code = {1: [1, 0, 0], 5: [0, 1, 0], 10: [0, 0, 1]}
data_size = 0
# 遍历文件夹中的CSV文件
for file_path in glob.glob(os.path.join(folder_path, '*.csv')):
    data = pd.read_csv(file_path, header=1)
    filename = os.path.basename(file_path)
    # 使用切片删除第一列和最后一列
    data = data.iloc[:, 1:-1]
    data = data.values

    # 标准化数据
    scaler = StandardScaler()
    data_df = pd.DataFrame(data)
    data_df.iloc[:, :] = scaler.fit_transform(data_df.iloc[:, :])

    dwt_data = WaveletTransformDataset(data_df).get_transformed_data()
    print('Refactoring complete!')

    win_data_temp = sliding_window(dwt_data, 225, 10)   # (507, 225, 20)
    win_data_temp = win_data_temp.astype(np.float32)
    # win_data_temp = win_data_temp.transpose(0, 2, 1)    # (507, 20, 225)
    # win_data_temp = win_data_temp.reshape(win_data_temp.shape[0],
    #                                       win_data_temp.shape[1], 15, 15)   # (507, 20, 15, 15)
    # win_data_temp = win_data_temp.transpose(0, 2, 3, 1)
    print('sliding window complete!')

    num_str = re.search(r'\d+', filename).group()  # 使用正则表达式从文件名中提取num的值
    num = int(num_str)  # 将num_str转换为整数类型

    labels = num

    data_size_temp = win_data_temp.shape[0]
    data_size += data_size_temp

    onehot_data_temp = np.array(value_to_code[num])
    onehot_data_temp2 = np.tile(onehot_data_temp, (data_size_temp, 1, 1))
    print(onehot_data_temp2.shape)

    onehot_data = np.concatenate((onehot_data, onehot_data_temp2), axis=0)

    print('one hot')
    # for i in range(win_data_temp.shape[0]):
    #
    #     onehot_data.extend(onehot_data_temp)
    # labeled_list_temp = [(win_data_temp[i], labels) for i in range(win_data_temp.shape[0])]

    # # 进行Onehot编码
    # onehot_data = torch.zeros(len(data), len(label_codes))
    # for i, d in enumerate(data):
    #     onehot_data[i, label_codes[d]] = 1

    win_data.extend(win_data_temp)  # (476, 15, 15, 20)

    print(1)
    print(win_data_temp.shape)
    print(f'{filename} data is labeled!')
    value_to_code = {1: [1, 0, 0], 5: [0, 1, 0], 10: [0, 0, 1]}



# # 标准化数据
# scaler = StandardScaler()
# data_df = pd.DataFrame(data)
# data_df.iloc[:, 1:-1] = scaler.fit_transform(data_df.iloc[:, 1:-1])
#
# dwt_data = WaveletTransformDataset(data_df).get_transformed_data()
# print('Refactoring complete!')
#
# win_data = sliding_window(dwt_data, 225, 10)
# print('sliding window complete!')
#
# labels = torch.zeros(win_data.shape[0], dtype=torch.long)
#
# folder_path = '../data/udds'
# csv_files = []
# num_list = []
# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.csv'):  # 如果文件名以'.csv'结尾
#         num_str = re.search(r'\d+', file_name).group()  # 使用正则表达式从文件名中提取num的值
#         num = int(num_str)  # 将num_str转换为整数类型
#         num_list.append((file_name, num))
#
# print(num_list)  # 输出所有CSV文件名
#
# labels.fill_(int(num))
# labeled_data = list(zip(win_data, labels))







# # 划分训练集和测试集
# X_train = dwt_data[:800]
# y_train = np.zeros(800)
# y_train[400:] = 1
#
# X_test = dwt_data[800:]
# y_test = np.zeros(200)
# y_test[100:] = 1
#
# # 创建数据集对象和数据加载器
# train_dataset = WaveletTransformDataset(X_train, y_train)
# test_dataset = WaveletTransformDataset(X_test, y_test)
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# # 创建模型对象
# model = TransformerModel(input_size = dwt_data.shape[1], output_size=2)
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# for epoch in range(10):
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         output = model(batch_X.float())
#         loss = criterion(output, batch_y.long())
#         loss.backward()
#         optimizer.step()
#     print("Epoch {}, Loss: {:.4f}".format(epoch+1, loss.item()))
#
# # 在测试集上评估模型
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for batch_X, batch_y in test_loader:
#         output = model(batch_X.float())
#         predicted = torch.argmax(output, dim=1)
#         total += batch_y.size(0)
#         correct += (predicted == batch_y).sum().item()
#
# print("Accuracy: {:.2f}%".format(100 * correct / total))
