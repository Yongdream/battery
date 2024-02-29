import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def calculate_channel_change_rate(data):
    # 直接计算每个通道的变化率
    channel_change_rate = np.diff(data, axis=1)

    return channel_change_rate


def plot_channels(data, channel_change_rate):
    channels, seq = data.shape
    time_steps = np.arange(seq - 1)

    plt.figure(figsize=(10, 6))
    plt.title('Channel Change Rates')
    plt.xlabel('Time')
    plt.ylabel('Change Rate')

    # 绘制每个通道的变化率线段
    for i in range(channels):
        plt.plot(time_steps, channel_change_rate[i, :], label=f'Channel {i + 1}')

    # plt.legend()

    plt.xlim(-50, 10000)
    plt.ylim(-0.52, 0.52)

    plt.show()


def save_channel_change_rate_to_csv(channel_change_rate, output_file):
    _, seq = channel_change_rate.shape
    time_steps = np.arange(seq)

    # 转置并添加时间序列
    transposed_data = np.vstack([time_steps, channel_change_rate]).T

    # Save the transposed array with time steps to a CSV file
    np.savetxt(output_file, transposed_data, delimiter=',', fmt='%1.6f')


# 设定文件路径
directory = '../data/udds/'
output_directory = '../result/diffFig/'
start_index = 800
end_index = 1050

# file_info = [
#     {'name': 'FUDS_Isc_1ohm_0718_3.csv'},
#     {'name': 'FUDS_Isc_5ohm_0714_3.csv'},
#     {'name': 'FUDS_Isc_10ohm_0715_3.csv'},
#     {'name': 'FUDS_Nor_0714.csv'},
#     {'name': 'FUDS_Cor_0630_2.csv'},
#     {'name': 'FUDS_Cor_0718_3.csv'}
# ]

file_info = [
    {'name': 'UDDS_Cor_0627_2.csv'},
    {'name': 'UDDS_Cor_0628_2.csv'},
    {'name': 'UDDS_Cor_0629_2.csv'},
    {'name': 'UDDS_Isc_1ohm_0718_3.csv'},
    {'name': 'UDDS_Isc_5ohm_0629_2.csv'},
    {'name': 'UDDS_Isc_10ohm_0630_2.csv'}
]

for file in file_info:
    # 构建文件路径和保存路径
    file_path = os.path.join(directory, file['name'])
    save_csv_directory = os.path.join(output_directory, 'csv', file['name'])

    # 使用 Pandas 加载数据
    df = pd.read_csv(file_path, skiprows=1)

    # 转换为 NumPy 数组并转置
    data_np = df.values.T

    # 计算每个通道的变化率
    channel_change_rate_np = calculate_channel_change_rate(data_np)

    # 保存转置的变化率和通道序号
    # save_channel_change_rate_to_csv(channel_change_rate_np, save_csv_directory)

    # 绘制变化率图
    plot_channels(data_np, channel_change_rate_np)
