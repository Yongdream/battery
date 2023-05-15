import torch
import torch.nn as nn
from utils import LabelSmoothing

# 创建LabelSmoothing实例
smoothing_factor = 0.1
label_smoothing = LabelSmoothing(smoothing=smoothing_factor)

# 生成随机输入和目标标签
batch_size = 32
num_classes = 10
input_size = 100
x = torch.randn(batch_size, input_size)
target = torch.randint(0, num_classes, (batch_size,))

# 计算损失
loss = label_smoothing(x, target)

print("Loss:", loss.item())

# import numpy as np
# import matplotlib.pyplot as plt
#
# file_path = "processed/udds/Isc/Isc_1230.npy"
# data_o = np.load(file_path)
#
#
# # 创建包含20个子图的图像
# fig, axes = plt.subplots(5, 4, figsize=(12, 10))
#
# # 遍历前20列，并在每个子图中输出图像
# for i, ax in enumerate(axes.flat):
#     data = data_o[:, i]
#
#     # fft_data = np.fft.fft(data)
#     fft_data = data
#
#     # 将频域数据 reshape 成二维图像
#     m = 15  # 图像的行数
#     n = len(data) // m  # 图像的列数，假设每行有 len(data) // m 列
#     fft_data_2d = np.reshape(fft_data, (m, n))
#
#     # 对二维图像进行整形操作
#     processed_data = np.log(fft_data_2d+0.001)
#     # processed_data = fft_data_2d
#
#     # 在当前子图中绘制预处理后的二维图像
#     ax.imshow(processed_data, cmap='hot', aspect='auto')
#     ax.set_title(f'Column {i+1}')
#
# # 调整子图间距和整体布局
# plt.tight_layout()
#
# # 显示图像
# plt.show()




# folder_path = r'E:/Sail/421'
# sub_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
#
# all_dataframes = []
# for sub_folder in sub_folders:
#     sub_folder_path = os.path.join(folder_path, sub_folder)
#     csv_files = [f for f in os.listdir(sub_folder_path) if f.endswith('.csv')]
#
#     dfs = []
#     for csv_file in csv_files:
#         csv_file_path = os.path.join(sub_folder_path, csv_file)
#         df = pd.read_csv(csv_file_path)



