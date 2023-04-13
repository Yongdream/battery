import numpy as np
import torch
from torch.utils.data import Dataset

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