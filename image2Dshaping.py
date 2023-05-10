import os
import numpy as np
import matplotlib.pyplot as plt

# 定义文件路径的起始和结束索引
start_index = 1
end_index = 100

de_ta = end_index - start_index

# 遍历指定范围的文件路径
for j in range(de_ta):

    # 构建当前索引对应的文件路径
    file_index = start_index + j
    file_path = f"processed/udds/Cor/Cor_{file_index}.npy"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        continue

    fig, axes = plt.subplots(5, 4, figsize=(12, 10))
    # 加载数据
    data_o = np.load(file_path)

    for i, ax in enumerate(axes.flat):
        # 获取列数据
        data = data_o[:, i]

        # fft_data = np.fft.fft(data)
        fft_data = data

        # 将频域数据 reshape 成二维图像
        m = 15  # 图像的行数
        n = len(data) // m  # 图像的列数，假设每行有 len(data) // m 列
        fft_data_2d = np.reshape(fft_data, (m, n))

        # 对二维图像进行整形操作
        processed_data = np.log(fft_data_2d + 0.001)
        # processed_data = fft_data_2d

        # 在当前子图中绘制预处理后的二维图像
        ax.imshow(processed_data, cmap='hot', aspect='auto')
        ax.set_title(f'File {file_index}, Column {i+1}')

    plt.tight_layout()
    # plt.show()
    # 保存图像
    plt.savefig(f'result/2Dsave/Cor/img_{file_index}.png')
    plt.clf()  # 清除图像，准备下一轮循环

    # 关闭图像
    plt.close(fig)
