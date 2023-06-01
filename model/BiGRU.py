import torch
import torch.nn as nn
import warnings
from torch.nn import init


class BiGruAd(nn.Module):
    def __init__(self, hidden_size=32, out_size=128, pretrained=False, n_layers=5, batch_size=1):
        super(BiGruAd, self).__init__()

        if pretrained:
            warnings.warn("Pretrained model is not available")

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        self.conv1 = nn.Conv1d(16, 128, kernel_size=10, stride=2, padding=4)
        self.act1 = nn.ReLU()
        self.AvgPooling_1 = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()

        # self.conv3 = nn.Conv1d(256, 64, kernel_size=3, stride=1, padding=1)
        # self.act3 = nn.ReLU()

        self.gru_1 = torch.nn.GRU(64, hidden_size, 5, batch_first=True, bidirectional=True)
        # outputs:(b, 128, 64=hidden_dim*2)    hidden:(n_layers*2, b, 32=hidden_dim)
        # output是所有隐藏层的状态，hidden是最后一层隐藏层的状态
        # hidden 就是上下文输出，output 就是 RNN 输出

        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        # num_directions 是GRU模型中的方向数（单向或双向)
        self.gru_2 = torch.nn.GRU(64, 16, 2, batch_first=True, bidirectional=True)

        self.fc1 = torch.nn.Linear(128, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)

    def forward(self, inputs):
        # (b, 16, 256)

        outputs = self.conv1(inputs)     # (b, 64, 256)
        outputs = self.act1(outputs)
        outputs = self.AvgPooling_1(outputs)    # (b, 64, 256)
        outputs = self.conv2(outputs)
        outputs = self.act2(outputs)
        # outputs = self.conv3(outputs)
        # outputs = self.act3(outputs)

        outputs = outputs.permute(0, 2, 1)

        outputs, _ = self.gru_1(outputs)
        outputs, _ = self.gru_2(outputs)

        outputs = outputs.permute(0, 2, 1)

        flatten_outputs = outputs.flatten(start_dim=1)

        outputs = self.fc1(outputs)           # (b, 128, 256)
        outputs = self.fc2(outputs)           # (b, 128, 512)
        outputs = self.fc3(outputs)
        outputs = outputs[:, -1, :]           # (b, 128)

        return outputs

    def init_hidden(self):
        # 这个函数写在这里，有一定迷惑性，这个不是模型的一部分，是每次第一个向量没有上下文，在这里捞一个上下文，仅此而已。
        hidden = torch.autograd.Variable(
            torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_size, device='cuda'))
        return hidden


class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv2 = nn.Conv1d(32, 256, kernel_size=3, padding=1, stride=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1, stride=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv1d(128, 32, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        y = self.conv2(x)
        y = self.act2(y)
        y = self.conv3(y)
        y = self.act3(y)
        y = self.conv4(y)
        return x+y


class BiGruAdFeatures(nn.Module):
    def __init__(self, pretrained=False):
        super(BiGruAdFeatures, self).__init__()
        self.model_net = BiGruAd(pretrained=pretrained)
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
model = BiGruAdFeatures()

# 进行前向传播
output = model(input_tensor)

# 打印输出张量的形状
print("Output shape:", output.shape)

