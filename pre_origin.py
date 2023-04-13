import os
import pandas as pd


dataset_folder = 'dataorigin/fuds'
target_folder = 'data/fuds'
file_list = os.listdir(dataset_folder)
for filename in file_list:
    if not filename.endswith('.csv'): continue
    file_path = os.path.join(dataset_folder, filename)
    data = pd.read_csv(file_path, header=0)
    data = data.iloc[1:, 2:-5]

    file_name = os.path.basename(file_path)
    new_file_path = os.path.join(target_folder, file_name)
    data.to_csv(new_file_path, index=False)