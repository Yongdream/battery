import os
import pandas as pd


dataset_folder = 'dataorigin/udds'
target_folder = 'data/udds'
file_list = os.listdir(dataset_folder)
for filename in file_list:
    if not filename.endswith('.csv'): continue

    file_path = os.path.join(dataset_folder, filename)
    data = pd.read_csv(file_path, header=0)
    if '10ohm' in filename:
        data = data.iloc[1000:, 2:-5]  # 删除前1000行
    else:
        data = data.iloc[1:, 2:-5]

    file_name = os.path.basename(file_path)
    new_file_path = os.path.join(target_folder, file_name)
    data.to_csv(new_file_path, index=False)
    print(filename + "完成...")
