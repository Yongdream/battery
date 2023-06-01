import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def signal_noise(signal, snr=10):
    # SNR越高，说明信号的质量越好，可靠性越高.以dB为单位
    noise = np.random.randn(signal.shape[0], 1)  # 产生N(0,1)噪声数据，第二维度为1
    noise = noise - np.mean(noise)  # 均值为0

    signal_power = np.linalg.norm(signal - np.mean(signal)) ** 2 / signal.shape[0]
    # 第3列信号的std**2
    noise_variance = signal_power / np.power(10, (snr / 10))  # 此处是噪声的std**2
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise  # 此处是噪声的std**2
    signal = signal.reshape(-1, 1)
    signal_noiset = signal + noise

    ps = np.linalg.norm(signal - np.mean(signal)) ** 2  # 信号与信号均值的差的平方和的平方根的平方除以信号长度的结果
    pn = np.linalg.norm(noise) ** 2                     # 噪声的标准差的平方乘以噪声长度的结果
    snr_t = 10 * np.log10(ps / pn)
    return signal_noiset


def signal_stick(data, sti_step=200, con_step=20, start=1):
    step = sti_step + con_step
    rounds = len(data) // step + 1
    for j in range(1, rounds):
        data[start:start + sti_step] = np.repeat(data[start - 1], sti_step)
        start += step
        # for i in range(start, start + sti_step):
        #     data[i] = data[start - 1]
        # start += step
    return np.array(data)


def simulate_noise(source_folder, target_folder, plot_image=False):
    pattern = f"{source_folder}/*_Nor*"
    file_paths = glob.glob(pattern)

    # 对每个文件进行操作并保存到目标文件夹中
    for file_path in file_paths:
        df = pd.read_csv(file_path, header=0)

        col_name1, col_name3 = df.columns[0], df.columns[2]
        x1 = torch.tensor(df[col_name1].values)
        x3 = torch.tensor(df[col_name3].values)

        col_name = df.columns[2]
        x = torch.tensor(df[col_name].values).numpy()
        # 添加高斯白噪声
        x_noisy = signal_noise(x)
        # 将添加噪声后的张量插入到新的DataFrame中
        df_noisy = df.copy()
        df_noisy[col_name] = x_noisy

        if plot_image:
            # 将张量转换为NumPy数组并绘制曲线
            plt.figure(figsize=(10, 6))
            start_idx, end_idx = 1000, 1300
            # plt.plot(x1[start_idx:end_idx].numpy(), label='Channel 1')
            plt.plot(x3[start_idx:end_idx].numpy(), label='Raw Channel 3')
            plt.plot(x_noisy[start_idx:end_idx], label='Noisy Channel 3')
            plt.legend()
            plt.show()

        # 拼接新文件名并保存到目标文件夹中
        file_name = os.path.basename(file_path)
        file_name_gk = file_name.split('_')[0]
        file_name_t = file_name.split('_')[2]
        file_name = f' {file_name_gk}_noi_{file_name_t}'
        new_file_path = os.path.join(target_folder, file_name)
        df_noisy.to_csv(new_file_path, index=False)

    print('Noise fault simulation completed')


def simulate_stick(source_folder, target_folder, plot_image=False):
    pattern = f"{source_folder}/*_Nor*"
    file_paths = glob.glob(pattern)

    for file_path in file_paths:
        df = pd.read_csv(file_path, header=0)

        col_name = df.columns[2]
        origininf = df[df.columns[3]].values
        origin = df[col_name].values
        x_stick = signal_stick(df[col_name].values)
        df_stick = df.copy()
        df_stick[col_name] = x_stick

        if plot_image:
            plt.figure(figsize=(10, 6))
            start_idx, end_idx = 1000, 1600
            plt.plot(origininf[start_idx:end_idx], label='Raw')
            plt.plot(x_stick[start_idx:end_idx], label='keep_last')
            plt.legend()
            # plt.legend(labels=['keep_last'])
            plt.show()

        file_name = os.path.basename(file_path)
        file_name_gk = file_name.split('_')[0]
        file_name_t = file_name.split('_')[2]
        file_name = f' {file_name_gk}_sti_{file_name_t}'
        new_file_path = os.path.join(target_folder, file_name)
        df_stick.to_csv(new_file_path, index=False)

    print('Stick fault simulation completed')


# 定义源文件夹路径和目标文件夹路径
run_work = 'us06'
source_folder = f'../data/{run_work}'
target_folder = f'../data/{run_work}'

simulate_noise(source_folder, target_folder)
simulate_stick(source_folder, target_folder)

