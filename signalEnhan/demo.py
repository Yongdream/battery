import numpy as np
import matplotlib.pyplot as plt
import os


def extract_features_v2(U_values):
    U_ave = np.mean(U_values, axis=0)
    # 计算每个通道与其平均值的差值
    diffs = (U_values - U_ave[np.newaxis, :]) / U_ave[np.newaxis, :]
    # 计算 xi^2
    xi_squared = np.mean(diffs ** 2, axis=0)

    # 计算累积变化率
    diff = np.diff(U_values, axis=0)
    cum_change_rate = np.cumsum(diff, axis=0)
    abs_cum_change_rate = np.abs(cum_change_rate)
    max_abs_cum_change_rate = np.max(abs_cum_change_rate, axis=0, keepdims=True)
    max_abs_cum_change_rate = max_abs_cum_change_rate.squeeze()

    return U_ave, xi_squared, max_abs_cum_change_rate



# 定义文件路径的前缀和后缀
prefix = 'processed/fuds/Isc/Isc_'
suffix = '.npy'
save_folder = 'result/squar'

# 创建一个存储所有xi_squared值的列表
all_xi_squared = []

# 循环前100个文件
for i in range(1, 300):
    filepath = f"{prefix}{i}{suffix}"

    if os.path.exists(filepath):
        # 从文件加载数据并转置
        U_values = np.load(filepath).T

        # 确保现在的形状是 (16, 300)
        assert U_values.shape == (16, 300)

        # 计算 xi^2 并存储
        _, e, xi_squared = extract_features_v2(U_values)
        all_xi_squared.append(xi_squared)


        plt.figure(figsize=(10, 6))
        plt.plot(xi_squared)
        plt.title(r'$\xi^2$ Variation Over Time Points')
        plt.xlabel('Time Point')
        plt.ylabel(r'$\xi^2$ Value')
        plt.grid(True)
        # 保存图像
        save_path = os.path.join(save_folder, f'Isc_{i}_plot.png')
        plt.savefig(save_path)
        plt.close()  # 关闭图像，避免打开过多图窗
