import torch.nn.functional as F

import warnings
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, pretrained=False, in_channel=225, out_channel=10):
        super(Network, self).__init__()

        if pretrained:
            warnings.warn("Pretrained model is not available")

        self.patch_embedding = nn.Linear(in_channel, 256)
        self.position_embedding = nn.Parameter(torch.FloatTensor(torch.randn([1, 20, 256])))
        # Position embedding

        self.conv1 = nn.Conv1d(20, 32, kernel_size=64, stride=16, padding=16)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 16, kernel_size=3, stride=1, padding=1)

        self.BN_1 = nn.BatchNorm1d(32, track_running_stats=False)
        self.BN_2 = nn.BatchNorm1d(64, track_running_stats=False)

        self.MaxPooling1D_1 = nn.MaxPool1d(kernel_size=2)
        self.MaxPooling1D_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(p=0.3)  # Dropout layer 1
        self.dropout2 = nn.Dropout(p=0.3)  # Dropout layer 2

        self.fc = nn.Linear(64, out_channel)  # Classification layer

    def forward(self, x):
        # x = x.permute(0, 2, 1).float()      # (B, 20, 225)
        x = self.patch_embedding(x)         # (B, 20, 256)
        x = x + self.position_embedding     # (B, 20, 256)

        x = self.conv1(x)                   # (B, 64, 15)
        # x = self.BN_1(x)
        x = F.relu(x)
        x = self.MaxPooling1D_1(x)          # (B, 64, 7)
        # x = self.dropout1(x)

        x = self.conv2(x)                   # (B, 32, 7)
        x = self.BN_2(x)
        x = F.relu(x)
        x = self.MaxPooling1D_2(x)          # (B, 32, 3)
        # x = self.dropout2(x)
        # x = F.relu(self.conv3(x))

        x = torch.mean(x, dim=2)    # Global average pooling  (B, 32)
        x = self.fc(x)              # Classification layer    (B, 5)

        return x


# convnet without the last layer
class NetworkFeatures(nn.Module):
    def __init__(self, pretrained=False):
        super(NetworkFeatures, self).__init__()
        self.model_cnn = Network(pretrained)
        self.__in_features = 256

    def forward(self, x):
        x = self.model_cnn(x)
        return x

    def output_num(self):
        return self.__in_features