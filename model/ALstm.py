import torch
import torch.nn as nn
import warnings


class ALSTMAdFeatures(nn.Module):
    def __init__(self, d_feat=20, hidden_size=64, num_layers=5, dropout=0.0, rnn_type="GRU", pretrained=False):
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
        self.conv_net.add_module('conv1', nn.Conv1d(16, 32, kernel_size=6, stride=2, padding=2))
        self.conv_net.add_module('conv1_act1', nn.ReLU())
        # self.conv_net.add_module('avg', nn.AvgPool1d(kernel_size=3, stride=2, padding=1))
        self.conv_net.add_module('conv2', nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1))
        self.conv_net.add_module('conv2_act2', nn.ReLU())

        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=True
        )

        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size * 2, out_features=int(self.hid_size * 4)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.ReLU())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size * 4), out_features=128, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

        self.fc_out = nn.Linear(in_features=256, out_features=128)

    def forward(self, inputs):
        # inputs: [batch_size, input_size*input_day]
        # inputs = inputs.view(len(inputs), self.input_size, -1)
        out_conv = self.conv_net(inputs)
        out_conv = out_conv.permute(0, 2, 1)  # [batch, input_size, seq_len] -> [batch, seq_len, input_size]
        rnn_out, _ = self.rnn(out_conv)  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1] 计算注意力分数

        out_att = torch.mul(rnn_out, attention_score)  # 使用注意力分数加权
        out_att = torch.sum(out_att, dim=1)  # 沿着序列长度的维度求和

        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        # 将RNN层的最后一个时间步和注意力加权结果进行拼接，然后通过全连接层输出
        return out

    def output_num(self):
        return self.__in_features


batch_size = 128
input_dim = 256
sequence_length = 16
input_tensor = torch.randn(batch_size, sequence_length, input_dim)

# 创建模型实例
model = ALSTMAdFeatures()
predicted_output = model(input_tensor)
# 打印输出张量的形状
print("Output shape:", predicted_output.shape)
