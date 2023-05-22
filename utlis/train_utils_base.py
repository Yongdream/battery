import logging
import os
import time
import warnings

import torch
from torch import nn
from torch import optim
import numpy as np

import datasets


class TrainUtils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Load the datasets
        # getattr()函数是Python内置的一个函数，它用于获取对象的属性值
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}


