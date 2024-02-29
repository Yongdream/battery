import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class WDCNNModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        self.__in_features = 128

        if pretrained:
            warnings.warn("Pretrained model is not available")

        self.Conv1D_1 = nn.Conv1d(12, 16, kernel_size=64, stride=16, padding=0)
        self.BN_1 = nn.BatchNorm1d(16)
        self.ReLU_1 = nn.ReLU()
        self.MaxPooling1D_1 = nn.MaxPool1d(kernel_size=2)

        self.Conv1D_2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm1d(32)
        self.ReLU_2 = nn.ReLU()
        self.MaxPooling1D_2 = nn.MaxPool1d(kernel_size=2)

        self.Conv1D_3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.BN_3 = nn.BatchNorm1d(64)
        self.ReLU_3 = nn.ReLU()
        self.MaxPooling1D_3 = nn.MaxPool1d(kernel_size=2)

        self.Conv1D_4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.BN_4 = nn.BatchNorm1d(64)
        self.ReLU_4 = nn.ReLU()
        self.MaxPooling1D_4 = nn.MaxPool1d(kernel_size=2, padding=1)

        self.Conv1D_5 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.BN_5 = nn.BatchNorm1d(64)
        self.ReLU_5 = nn.ReLU()
        self.MaxPooling1D_5 = nn.MaxPool1d(kernel_size=2, padding=1)

        self.Flatten_1 = nn.Flatten()
        self.Dense_layer1 = nn.Linear(64, 100)
        self.pred_loss = nn.Linear(100, 5)

        self.fc1 = torch.nn.Linear(64, 128)

    def feature_extractor_model(self, input_tensor):
        # (b, 12, 300)
        Conv1D_1 = self.Conv1D_1(input_tensor)  # torch.Size([128, 16, 15])
        BN_1 = self.BN_1(Conv1D_1)
        ReLU_1 = self.ReLU_1(BN_1)
        MaxPooling1D_1 = self.MaxPooling1D_1(ReLU_1)    # torch.Size([128, 16, 7])

        Conv1D_2 = self.Conv1D_2(MaxPooling1D_1)    # torch.Size([128, 32, 15])
        BN_2 = self.BN_2(Conv1D_2)
        ReLU_2 = self.ReLU_2(BN_2)
        MaxPooling1D_2 = self.MaxPooling1D_2(ReLU_2)    # torch.Size([128, 32, 3])

        Conv1D_3 = self.Conv1D_3(MaxPooling1D_2)        # torch.Size([128, 64, 3])
        BN_3 = self.BN_3(Conv1D_3)
        ReLU_3 = self.ReLU_3(BN_3)
        MaxPooling1D_3 = self.MaxPooling1D_3(ReLU_3)    # torch.Size([128, 64, 1])

        Conv1D_4 = self.Conv1D_4(MaxPooling1D_3)        # torch.Size([128, 64, 1])
        BN_4 = self.BN_4(Conv1D_4)
        ReLU_4 = self.ReLU_4(BN_4)
        MaxPooling1D_4 = self.MaxPooling1D_4(ReLU_4)

        Conv1D_5 = self.Conv1D_5(MaxPooling1D_4)
        BN_5 = self.BN_5(Conv1D_5)
        ReLU_5 = self.ReLU_5(BN_5)
        MaxPooling1D_5 = self.MaxPooling1D_5(ReLU_5)    # torch.Size([128, 64, 1])

        Flatten_1 = self.Flatten_1(MaxPooling1D_5)      # torch.Size([128, 64])

        return Flatten_1

    def forward(self, input_tensor):
        feature_extractor_model = self.feature_extractor_model(input_tensor)
        # Dense_layer1 = self.Dense_layer1(feature_extractor_model)
        # pred = self.pred_loss(Dense_layer1)

        out = self.fc1(feature_extractor_model)
        return out

    def output_num(self):
        return self.__in_features


# batch_size = 128
# input_dim = 300
# sequence_length = 12
# input_tensor = torch.randn(batch_size, sequence_length, input_dim)
# print('input_tensor:', input_tensor.shape)
#
#
# # 创建模型实例
# model = WDCNNModel()
# predicted_output = model(input_tensor)
# # 打印输出张量的形状
# print("Output shape:", predicted_output.shape)


