import os
import pandas as pd

run_work = 'udds'
dataset_folder = f'../dataorigin/{run_work}'
target_folder = f'../data/{run_work}'
file_list = os.listdir(dataset_folder)
for filename in file_list:
    if not filename.endswith('.csv'): continue

    file_path = os.path.join(dataset_folder, filename)
    data = pd.read_csv(file_path, header=0)
    if '10ohm' in filename:
        data = data.iloc[800:, 2:-9]
    else:
        data = data.iloc[1:, 2:-9]

    file_name = os.path.basename(file_path)
    new_file_path = os.path.join(target_folder, file_name)
    data.to_csv(new_file_path, index=False)
    print(filename + "完成...")
