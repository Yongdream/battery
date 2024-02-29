import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class CBDANModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        self.__in_features = 128

        if pretrained:
            warnings.warn("Pretrained model is not available")

        # Other layers
        self.Conv1D_1 = nn.Conv1d(12, 64, kernel_size=9, stride=1, padding=4)
        self.ReLU_1 = nn.ReLU()
        self.MaxPooling1D_1 = nn.MaxPool1d(kernel_size=2)
        self.Dropout_1 = nn.Dropout(0.5)

        self.Conv1D_2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm1d(32)
        self.ReLU_2 = nn.ReLU()
        self.MaxPooling1D_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Dropout_2 = nn.Dropout(0.5)

        self.GlobalAveragePooling1D = nn.AdaptiveAvgPool1d(1)
        self.GlobalMaxPool1D = nn.AdaptiveMaxPool1d(1)

        self.Dense_layer1 = nn.Linear(32, 8)
        self.Dense_layer2 = nn.Linear(8, 32)

        self.Concatenate = lambda x: torch.cat(x, dim=1)
        self.spatial_conv1d = nn.Conv1d(2, 1, kernel_size=7, stride=1, padding=3)

        self.Flatten_1 = nn.Flatten()

        self.Dense_1 = nn.Linear(32, 32)
        self.ReLU_3 = nn.ReLU()

        self.pred_loss = nn.Linear(32, 5)

        self.Concatenate_pred = lambda x: torch.cat(x, dim=1)
        self.Dense_2 = nn.Linear(32, 32)
        self.ReLU_5 = nn.ReLU()
        self.Dense_3 = nn.Linear(32, 5)
        self.domain_loss = nn.Linear(1, 1)

        self.fc1 = torch.nn.Linear(2400, 128)

    def CBAM(self, input_tensor):
        # torch.Size([128, 32, 75])
        # Channel Attention
        avgpool = self.GlobalAveragePooling1D(input_tensor)
        avgpool = torch.squeeze(avgpool, dim=2)

        maxpool = self.GlobalMaxPool1D(input_tensor)
        maxpool = torch.squeeze(maxpool, dim=2)

        avg_out = self.Dense_layer2(self.Dense_layer1(avgpool))
        max_out = self.Dense_layer2(self.Dense_layer1(maxpool))

        channel1 = avg_out + max_out
        # torch.Size([128, 32])
        channel2 = torch.sigmoid(channel1)
        channel3 = channel2.view(channel2.size(0), 32, 1)
        # torch.Size([128, 1, 32])
        channel_out = torch.mul(input_tensor, channel3)

        # Spatial Attention
        avgpool_spatial = torch.mean(channel_out, dim=1, keepdim=True)
        maxpool_spatial = torch.max(channel_out, dim=1, keepdim=True)[0]
        spatial = self.Concatenate([avgpool_spatial, maxpool_spatial])

        spatial1 = self.spatial_conv1d(spatial)
        spatial_out = torch.sigmoid(spatial1)

        CBAM_out = channel_out * spatial_out
        return CBAM_out

    def feature_extractor_model(self, input_tensor):
        Conv1D_1 = self.Conv1D_1(input_tensor)
        # print(Conv1D_1.shape)
        ReLU_1 = self.ReLU_1(Conv1D_1)
        MaxPooling1D_1 = self.MaxPooling1D_1(ReLU_1)
        # print(MaxPooling1D_1.shape)
        Conv1D_2 = self.Conv1D_2(MaxPooling1D_1)
        # print(Conv1D_2.shape)
        BN_2 = self.BN_2(Conv1D_2)
        ReLU_2 = self.ReLU_2(BN_2)
        MaxPooling1D_2 = self.MaxPooling1D_2(ReLU_2)    # torch.Size([128, 32, 75])
        # print(MaxPooling1D_2.shape)
        CBAM_layer = self.CBAM(MaxPooling1D_2)
        feature_fla = self.Flatten_1(CBAM_layer)
        feature = self.fc1(feature_fla)
        return feature

    def forward(self, input_tensor):
        feature_extractor_model = self.feature_extractor_model(input_tensor)
        return feature_extractor_model

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
# model = CBDANModel()
# predicted_output = model(input_tensor)
# # 打印输出张量的形状
# print("Output shape:", predicted_output.shape)