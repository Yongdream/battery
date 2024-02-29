import torch
import torch.nn as nn
import warnings


class ALSTMAdFeaturesNoSign(nn.Module):
    def __init__(self, d_feat=20, hidden_size=16, num_layers=16, dropout=0.0, rnn_type="GRU", pretrained=False):
        super().__init__()

        self.__in_features = 128

        if pretrained:
            warnings.warn("Pretrained model is not available")

        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type  # RNN类型，可以是"GRU"或"LSTM"
        self.rnn_layer = num_layers  # RNN的层数
        self._build_model()  # 构建模型

    def _build_model(self):
        try:  # try-except语句异常的代码检测
            klass = getattr(nn, self.rnn_type.upper())  # 根据指定的RNN类型获取相应的类
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e

        self.conv_net = nn.Sequential()
        self.conv_net.add_module('conv1', nn.Conv1d(12, 64, kernel_size=3, stride=2, padding=1))
        self.conv_net.add_module('conv1_act1', nn.ReLU())

        self.batch_norm = torch.nn.BatchNorm1d(num_features=64)

        self.avg_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.gru_1 = torch.nn.GRU(64, self.hid_size * 4, 5, batch_first=True, bidirectional=False)
        self.gru_2 = torch.nn.GRU(64, self.hid_size, 5, batch_first=True, bidirectional=True)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size * 2, out_features=int(self.hid_size)),
        )
        self.att_net.add_module("att_act", nn.ReLU())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

        self.fc1 = torch.nn.Linear(4800, 2048)
        self.fc2 = torch.nn.Linear(2048, 128)

        # self.single_out = nn.Sequential()
        # self.single_out.add_module(
        #
        # )

    def forward(self, inputs):
        # print(inputs.shape) torch.Size([128, 12, 300])
        out_conv = self.conv_net(inputs)

        out_conv = out_conv.permute(0, 2, 1)     # torch.Size([128, 150, 64])
        rnn_out, h2 = self.gru_1(out_conv)       # [128, 150, 64] [batch, seq_len, num_directions * hidden_size]
        rnn_out, h1 = self.gru_2(rnn_out)        # torch.Size([128, 300, 32])

        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1] 计算注意力分数

        out_att = torch.mul(rnn_out, attention_score)   # 使用注意力分数加权 torch.Size([128, 64, 64])

        out_att = out_att.flatten(1)  # ([128, 2000])
        out = self.fc1(out_att)
        out = self.fc2(out)
        return out

    def output_num(self):
        return self.__in_features


# batch_size = 128
# input_dim = 300
# sequence_length = 12
# input_tensor = torch.randn(batch_size, sequence_length, input_dim)
# print('input_tensor:')
# print(input_tensor.shape)
#
# # 创建模型实例
# model = ALSTMAdFeaturesNoSign()
# predicted_output = model(input_tensor)
# # 打印输出张量的形状
# print("Output shape:", predicted_output.shape)
