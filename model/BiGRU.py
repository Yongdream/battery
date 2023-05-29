import torch
import torch.nn as nn
import warnings


class BiGruAd(nn.Module):
    def __init__(self, input_dim=23, hidden_size=32, out_size=128, pretrained=False, n_layers=5, batch_size=1):
        super(BiGruAd, self).__init__()

        if pretrained:
            warnings.warn("Pretrained model is not available")

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        # num_directions 是GRU模型中的方向数（单向或双向)
        self.gru = torch.nn.GRU(input_dim, hidden_size, n_layers, batch_first=True, bidirectional=True)

        # 加了一个线性层，全连接
        self.fc1 = torch.nn.Linear(hidden_size * 2, 128)
        # 加入了第二个全连接层
        self.fc2 = torch.nn.Linear(128, 256)
        # 加入了第三个全连接层
        self.fc3 = torch.nn.Linear(256, 128)

    def forward(self, inputs):
        # hidden 就是上下文输出，output 就是 RNN 输出
        # (b, 23, 225)
        inputs = inputs.permute(0, 2, 1)    # (b, 225, 23)

        output, hidden = self.gru(inputs)        # (b, 225, 64=hidden_dim*2)    (n_layers*2, b, 32)
        # output是所有隐藏层的状态，hidden是最后一层隐藏层的状态
        temp = torch.cat((hidden[:self.n_layers, :, :], hidden[self.n_layers:, :, :]), dim=0)
        output = self.fc1(output)           # (b, 225, 128)
        output = self.fc2(output)           # (b, 225, 256)
        output = self.fc3(output)           # (b, 225, 128)
        # 仅仅获取 time seq 维度中的最后一个向量
        # the last of time_seq
        output = output[:, -1, :]           # (b, 128)

        return output

    def init_hidden(self):
        # 这个函数写在这里，有一定迷惑性，这个不是模型的一部分，是每次第一个向量没有上下文，在这里捞一个上下文，仅此而已。
        hidden = torch.autograd.Variable(
            torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_size, device='cuda'))
        return hidden


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


batch_size = 200
input_dim = 225
sequence_length = 23
input_tensor = torch.randn(batch_size, sequence_length, input_dim)

# 创建模型实例
model = BiGruAdFeatures()

# 进行前向传播
output = model(input_tensor)

# 打印输出张量的形状
print("Output shape:", output.shape)

