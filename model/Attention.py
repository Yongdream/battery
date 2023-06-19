import torch
import torch.nn as nn
import warnings
from torch.nn import init


class AttAd(nn.Module):
    def __init__(self, pretrained=False):
        super(AttAd, self).__init__()

        if pretrained:
            warnings.warn("Pretrained model is not available")

    def forward(self, inputs):
        # (b, 16, 256)
        output = inputs


        return output


class AttFeatures(nn.Module):
    def __init__(self, pretrained=False):
        super(AttFeatures, self).__init__()
        self.model_net = AttAd(pretrained=pretrained)
        self.__in_features = 128

    def forward(self, x):
        x = self.model_net(x)
        return x

    def output_num(self):
        return self.__in_features


batch_size = 128
input_dim = 256
sequence_length = 16
input_tensor = torch.randn(batch_size, sequence_length, input_dim)

# 创建模型实例
model = AttFeatures()

# 进行前向传播
output = model(input_tensor)

# 打印输出张量的形状
print("Output shape:", output.shape)

