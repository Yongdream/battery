import torch
import torch.nn as nn
import warnings


class LSTMFeatures(nn.Module):
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

        # self.net = nn.Sequential()
        # self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        # self.net.add_module("act", nn.Tanh())

        self.conv_net = nn.Sequential()
        self.conv_net.add_module('conv1', nn.Conv1d(12, 32, kernel_size=7, stride=1, padding=3))
        self.conv_net.add_module('conv1_act1', nn.ReLU())
        self.conv_net.add_module('conv2', nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1))
        self.conv_net.add_module('conv2_act2', nn.ReLU())

        self.batch_norm = torch.nn.BatchNorm1d(num_features=64)

        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.gru_1 = torch.nn.GRU(64, self.hid_size * 4, 5, batch_first=True)
        self.gru_2 = torch.nn.GRU(64, self.hid_size, 5, batch_first=True)

        self.rnn_1 = torch.nn.LSTM(64, self.hid_size * 4, 5, batch_first=True)
        self.rnn_2 = torch.nn.LSTM(64, self.hid_size, 5, batch_first=True)

        self.fc1 = torch.nn.Linear(2400, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)


    def forward(self, inputs):
        # inputs: [batch_size, input_size*input_day]
        # inputs = inputs.view(len(inputs), self.input_size, -1)
        out_conv = self.conv_net(inputs)
        out_conv = self.avg_pool(out_conv)  # torch.Size([128, 64, 150])

        out_conv = out_conv.permute(0, 2, 1)    # torch.Size([128, 150, 64])
        rnn_out, h1 = self.gru_1(out_conv)
        rnn_out, h2 = self.gru_2(rnn_out)   # torch.Size([128, 150, 16])

        out_att = rnn_out.flatten(1)
        # print(out_att.shape)
        out = self.fc1(out_att)
        out = self.fc2(out)
        return out

    def output_num(self):
        return self.__in_features


batch_size = 128
input_dim = 300
sequence_length = 12
input_tensor = torch.randn(batch_size, sequence_length, input_dim)

# 创建模型实例
model = LSTMFeatures()
predicted_output = model(input_tensor)
# 打印输出张量的形状
print("Output shape:", predicted_output.shape)
