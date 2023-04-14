# Define the path to the folder containing the subfolders with .npy files
import os

import numpy as np
import torch


data_folder = '/path/to/folder'

# Define the number of subfolders and the number of one-hot encoding categories
num_subfolders = 5
num_categories = 5

# Create a dictionary to map subfolder names to category indices
category_dict = {'subfolder1': 0, 'subfolder2': 1, 'subfolder3': 2, 'subfolder4': 3, 'subfolder5': 4}

# Initialize empty lists to store the data and labels
x_all = []
y_all = []

# Loop through the subfolders
for subfolder_name in os.listdir(data_folder):
    if not os.path.isdir(os.path.join(data_folder, subfolder_name)):
        continue
    subfolder_path = os.path.join(data_folder, subfolder_name)

    # Create a one-hot encoding vector for the category
    category_index = category_dict[subfolder_name]
    category_one_hot = np.zeros(num_categories)
    category_one_hot[category_index] = 1

    # Loop through the .npy files in the subfolder
    for npy_filename in os.listdir(subfolder_path):
        if not npy_filename.endswith('.npy'):
            continue
        npy_filepath = os.path.join(subfolder_path, npy_filename)

        # Load the .npy file and append it to x_all
        npy_data = np.load(npy_filepath)
        x_all.append(npy_data)

        # Append the category label to y_all
        y_all.append(category_one_hot)

# Convert the data and labels to numpy arrays
x_all = np.array(x_all)
y_all = np.array(y_all)

# Convert the numpy arrays to PyTorch tensors
x_all = torch.from_numpy(x_all)
y_all = torch.from_numpy(y_all)


