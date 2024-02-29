import os
import numpy as np
import matplotlib.pyplot as plt


def process_and_save_images(start_index, end_index):
    de_ta = end_index - start_index

    # 遍历指定范围的文件路径
    for j in range(de_ta):
        # 构建当前索引对应的文件路径
        file_index = start_index + j
        file_path = f"processed/udds00/Nor/Nor_{file_index}.npy"

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
        plt.savefig(f'result/2Dsave/Nor/img_{file_index}.png')
        plt.clf()  # 清除图像，准备下一轮循环

        # 关闭图像
        plt.close(fig)


def count_files_in_subfolders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    folder_names = []
    file_counts = []

    for subfolder in subfolders:
        folder_name = os.path.basename(subfolder)
        folder_names.append(folder_name)

        file_count = sum(1 for _ in os.scandir(subfolder) if _.is_file())
        file_counts.append(file_count)

    return folder_names, file_counts

def plot_file_counts(folder_names, file_counts):
    plt.bar(folder_names, file_counts)
    plt.xlabel('Subfolder')
    plt.ylabel('File Count')
    plt.title('Number of Files in Each Subfolder')
    plt.show()


def delete_files_subfolders(folder_path, delete_ratio):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in subfolders:
        folder_name = os.path.basename(subfolder)

        if folder_name not in ['Cor', 'Nor', 'Isc']:
            files = [f.path for f in os.scandir(subfolder) if f.is_file()]
            num_files_to_delete = int(delete_ratio * len(files))

            for file_path in files[:num_files_to_delete]:
                os.remove(file_path)


def count_files_by_category(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    categories = []
    file_counts = []

    for subfolder in subfolders:
        folder_name = os.path.basename(subfolder)

        files = [f.path for f in os.scandir(subfolder) if f.is_file()]
        category = folder_name if folder_name == 'Nor' else 'Non-Nor'

        categories.append(category)
        file_counts.append(len(files))

    return categories, file_counts


folder_path = '../processed/udds'
folder_names, file_counts = count_files_in_subfolders(folder_path)

# 绘制柱形图显示每个子文件夹中文件的数量
plot_file_counts(folder_names, file_counts)

# # 按指定比率删除指定子文件夹内的文件
# delete_ratio = 0.6
# delete_files_subfolders(folder_path, delete_ratio)
#
# # 统计每个类别中文件的数量
# folder_names, file_counts = count_files_in_subfolders(folder_path)
#
# #  绘制柱形图显示每个类别中文件的数量
# plot_file_counts(folder_names, file_counts)






