import torch
import torch.nn as nn
import warnings

from torch.nn import init

class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=2, stride=2, dilation=2)
        self.act = nn.ReLU()
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=1, stride=2)

    def forward(self, inputs):
        # [128, 64, 64]
        x = self.conv2(inputs) # [128, 64, 32]
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        y = self.conv4(inputs)
        y = self.act(y)
        return x + y



class Res_Altsm(nn.Module):
    def __init__(self, hidden_size=32, out_size=128, pretrained=False, n_layers=5, batch_size=1):
        super().__init__()

        if pretrained:
            warnings.warn("Pretrained model is not available")

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        self.Conv1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=2)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.residul = ResidualBlock()

        self.gru_1 = torch.nn.GRU(64, hidden_size, 5, batch_first=True, bidirectional=True)


        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)

        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=64, out_features=128),
        )
        self.att_net.add_module("att_act", nn.ReLU())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=128, out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))


    def forward(self, inputs):
        # (b, 16, 256)
        conv_out = self.Conv1(inputs)       # [128, 32, 128]
        conv_out = self.act(conv_out)
        conv_out = self.avgpool(conv_out)   # [128, 64, 64]
        conv_out = self.residul(conv_out)   # [128, 64, 32]
        conv_out = conv_out.permute(0, 2, 1)    # [b, 32, 64]
        rnn_out, _ = self.gru_1(conv_out)         # (b, 32, *16)
        # outputs, _ = self.gru_2(outputs)

        # rnn_out = outputs.permute(0, 2, 1)      # (b, 16, 64)
        # print(rnn_out.shape)
        attention_score = self.att_net(rnn_out)
        out_att = torch.mul(rnn_out, attention_score)  # 使用注意力分数加权 torch.Size([128, 64, 32])
        conv_out_cat = torch.sum(out_att, dim=1)

        outputs = torch.cat((rnn_out[:, -1, :], conv_out_cat), dim=1)

        # outputs = self.fc1(out_att)           # (b,
        # outputs = self.fc2(outputs)           # (b,
        # outputs = self.fc3(outputs)           # (b,

        return outputs

    def init_hidden(self):
        # 这个函数写在这里，有一定迷惑性，这个不是模型的一部分，是每次第一个向量没有上下文，在这里捞一个上下文，仅此而已。
        hidden = torch.autograd.Variable(
            torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_size, device='cuda'))
        return hidden




class Res_AltsmFeatures(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model_net = Res_Altsm(pretrained=pretrained)
        self.__in_features = 128

    def forward(self, x):
        x = self.model_net(x)
        return x

    def output_num(self):
        return self.__in_features


# batch_size = 128
# input_dim = 256
# sequence_length = 16
# input_tensor = torch.randn(batch_size, sequence_length, input_dim)
#
# # 创建模型实例
# model = Res_Altsm()
#
# # 进行前向传播
# output = model(input_tensor)
#
# # 打印输出张量的形状
# print("Output shape:", output.shape)

