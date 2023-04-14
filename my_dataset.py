import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
from sklearn.model_selection import train_test_split

from tqdm import tqdm


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, timeWin_path: list, timeWin_class: list, transform=None ):
        super().__init__()
        self.timeWin_path = timeWin_path
        self.timeWin_class = timeWin_class
        self.transform = transform
    
    def __len__(self):
        return len(self.timeWin_path)
    
    def __getitem__(self, index):
        timeWin = torch.from_numpy(np.load(self.timeWin_path[index]))
        label = self.timeWin_class[index]
        return timeWin, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def prepro(dir=r'processed/fuds/',  number=1000, normal=True,
           rate=[0.7, 0.2, 0.1], enc=True, enc_step=28):
    """
    对数据进行预处理
    Args:
        dir: 元数据地址
        number: 每种信号个数,总共5类5
        normal: 是否标准化5.默认True
        rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
        enc: 训练集、验证集是否采用数据增强.Bool,默认True
        enc_step: 增强数据集采样顺延间隔

    Returns:Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

    """
    def slice_enc(data, label):
        data = np.stack(x, axis=0)
        label = np.stack(y, axis=0)
        n = data.shape[0]
        # 计算每个集合的大小
        train_size = int(rate[0] * n)
        valid_size = int(rate[1] * n)
        test_size = int(rate[2] * n)

        Train_X, Valid_X, Test_X = np.split(data, [train_size, train_size + valid_size])
        Train_Y, Valid_Y, Test_Y = np.split(label, [train_size, train_size + valid_size])
        return Train_X, Valid_X, Test_X, Train_Y, Valid_Y, Test_Y

    print("data_preprocess_begin")
    # Initialize empty lists to store the data and labels
    Train_Xt = []
    Train_Yt = []
    Valid_Xt = []
    Valid_Yt = []
    Test_Xt = []
    Test_Yt = []

    error_list = os.listdir(dir)
    err_num = len(error_list)
    category_dict = {'Cor': 0, 'Isc': 1, 'noi': 2, 'Nor': 3, 'sti': 4}
    for i in tqdm(error_list):
        x = []
        y = []
        error_index = i
        # Create a one-hot encoding vector for the category
        category_index = category_dict[str(error_index)]
        category_one_hot = np.zeros(err_num)
        category_one_hot[category_index] = 1

        # 获得该文件夹下所有.npy文件名
        d_path = os.path.join(dir + str(i))
        for npy_filename in os.listdir(d_path):
            if not npy_filename.endswith('.npy'):
                continue
            npy_filepath = os.path.join(d_path, npy_filename)

            npy_data = np.load(npy_filepath)
            x.append(npy_data)
            y.append(category_one_hot)

        xs = np.stack(x, axis=0)
        ys = np.stack(y, axis=0)
        train_x, valid_x, test_x, train_y, valid_y, test_y = slice_enc(xs, ys)
        Train_Xt.append(train_x)
        Train_Yt.append(train_y)
        Valid_Xt.append(valid_x)
        Valid_Yt.append(valid_y)
        Test_Xt.append(test_x)
        Test_Yt.append(test_y)
    Train_X = np.concatenate(Train_Xt, axis=0)
    Train_Y = np.concatenate(Train_Yt, axis=0)
    Valid_X = np.concatenate(Valid_Xt, axis=0)
    Valid_Y = np.concatenate(Valid_Yt, axis=0)
    Test_X = np.concatenate(Test_Xt, axis=0)
    Test_Y = np.concatenate(Test_Yt, axis=0)

    print("data_preprocess_end")

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


class DANNDataLoader():
    def __init__(self):
        # 训练参数
        batch_size = 128
        epochs = 20
        num_classes = 5
        length = 300
        BatchNorm = True  # 是否批量归一化MaxPooling2D
        number = 1000  # 每类样本的数量
        normal = True  # 是否标准化
        rate = [0.7, 0.28, 0.02]  # 测试集验证集划分比例

        self.x_train, self.x_train_rel, self.lab_train, self.y_train, self.x_valid, self.x_valid_rel, self.lab_valid, self.y_valid, self.x_test, self.x_test_rel, self.lab_test, self.y_test = prepro(
            length=length,
            number=number,
            number_set=number,
            normal=normal,
            rate=rate,
            enc=True, enc_step=100)
        self.sample_true = np.ones(self.x_train.shape[0])
        self.x_target, self.x_target_rel, self.lab_target, self.y_target, self.x_valid_target, self.x_valid_rel_target, self.lab_valid_target, self.y_valid_target, self.x_test_target, self.x_test_rel_target, self.lab_test_target, self.y_test_target = prepro(
            dirA=r'data/Fault_data_with_lianjie_sensor_2600/',
            length=length,
            number=number,
            number_set=number,
            normal=normal,
            rate=rate,
            enc=True, enc_step=100)
        self.sample_false = np.zeros(self.x_target.shape[0])

        self.train_x = np.concatenate((self.x_train, self.x_target), axis=0)
        self.train_rel = np.concatenate((self.x_train_rel, self.x_target_rel), axis=0)
        self.train_lab = np.concatenate((self.lab_train, self.lab_target), axis=0)
        self.train_y = np.concatenate((self.y_train, self.y_target), axis=0)
        self.train_true = np.concatenate((self.sample_true, self.sample_false), axis=0)

        # 将验证集和测试机合并
        self.test_x = np.concatenate((self.x_valid_target, self.x_test_target), axis=0)
        self.test_rel = np.concatenate((self.x_valid_rel_target, self.x_test_rel_target), axis=0)
        self.test_lab = np.concatenate((self.lab_valid_target, self.lab_test_target), axis=0)
        self.test_y = np.concatenate((self.y_valid_target, self.y_test_target), axis=0)
        self.test_true = np.zeros(self.x_test_target.shape[0] + self.x_valid_target.shape[0])

        self.num_train_data, self.num_valid_data, self.num_test_data = self.train_x.shape[0], self.x_valid.shape[0], \
                                                                       self.test_x.shape[0]

    def get_train_batch(self, batch_size):
        # 从训练集里随机取出 batch_size 个样本
        idx = np.random.randint(0, self.num_train_data, batch_size)
        # train_lab为故障电池的索引,
        return self.train_x[idx, :, :], self.train_lab[idx, :], self.train_rel[idx, :, :], self.train_y[idx, :], \
               self.train_true[idx]

    def get_valid_batch(self, batch_size):
        # 从训练集里随机取出 batch_size 个样本
        idx = np.random.randint(0, self.num_test_data, batch_size)
        return self.x_valid[idx, :, :], self.lab_valid[idx, :]

    def get_test_batch(self, batch_size):
        # 从训练集里随机取出 batch_size 个样本
        idx = np.random.randint(0, self.num_valid_data, batch_size)
        return self.test_x[idx, :, :], self.test_lab[idx, :], self.test_rel[idx, :, :], self.test_y[idx, :], \
               self.test_true[idx]


prepro()

