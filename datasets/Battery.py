import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

import glob

# BAT_state = ['Cor', 'Isc', 'noi', 'Nor', 'sti']
BAT_state = ['Isc', 'noi', 'Nor', 'sti']
label_state = [i for i in range(len(BAT_state))]

WS = ["udds", "fuds", "us06"]


def get_files(n):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for w in range(len(n)):
        state = WS[n[w]]
        for i in tqdm(range(len(BAT_state))):
            # 构建搜索路径
            search_path = os.path.join(".", "processed", state, BAT_state[i], "*")

            # 使用glob模块搜索匹配的文件
            file_list = [file_path for file_path in glob.glob(search_path) if file_path.endswith(".npy")]
            for file_path in file_list:
                data1 = np.load(file_path)      # (225, 20)
                lab1 = label_state[i]

                data.append(data1)
                lab.append(lab1)
    return [data, lab]


class Battery(object):
    num_classes = len(BAT_state)
    inputchannel = 1

    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Retype(),
            ]),
            'val': Compose([
                Reshape(),
                Retype(),
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.source_N)  # 传入了源域和目标域
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.3, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.5, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:
            #get source train and val
            list_data = get_files(self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val


# data_dir = 'E:/Galaxy/yang7hi_battery/processed'
# battery = Battery(data_dir, transfer_task=[[0], [1]])
# battery.data_split()
