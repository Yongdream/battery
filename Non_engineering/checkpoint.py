import os

# 指定文件夹路径和目标文件夹路径
folder_path = '../checkpoint/'
destination_path = 'checkpoint/[0]-[1]'

# 遍历文件夹中的子文件夹
for root, dirs, files in os.walk(folder_path):
    for dir_name in dirs:
        # 检查子文件夹名是否以 "BiGruAdFeatures" 开头
        if dir_name.startswith("BiGruAdFeatures"):
            # 构造 train.log 文件的路径
            log_file_path = os.path.join(root, dir_name, "train.log")
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='GBK') as file:
                    # 读取第四行的内容
                    lines = file.readlines()
                    if len(lines) >= 4:
                        transfer_task_line = lines[3].strip()
                        # 提取末尾部分进行比较
                        if transfer_task_line.endswith('[[0], [1]]'):
                            # 移动子文件夹到目标文件夹
                            source_dir = os.path.join(root, dir_name)
                            destination_dir = os.path.join(destination_path, dir_name)
                            os.rename(source_dir, destination_dir)
                            print(f"Moved {dir_name} to {destination_path}")
