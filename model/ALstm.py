import torch
import torch.nn as nn


class ALSTMModel(nn.Module):
    def __init__(self, d_feat=20, hidden_size=16, num_layers=5, dropout=0.0, rnn_type="GRU"):
        super().__init__()
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

        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )

        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)

        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        # inputs: [batch_size, input_size*input_day]
        # inputs = inputs.view(len(inputs), self.input_size, -1)
        inputs = inputs.permute(0, 2, 1)  # [batch, input_size, seq_len] -> [batch, seq_len, input_size]
        rnn_out, _ = self.rnn(inputs)  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1] 计算注意力分数

        out_att = torch.mul(rnn_out, attention_score)  # 使用注意力分数加权
        out_att = torch.sum(out_att, dim=1)  # 沿着序列长度的维度求和

        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        # 将RNN层的最后一个时间步和注意力加权结果进行拼接，然后通过全连接层输出
        return out[..., 0]  # 返回输出的第一列值


batch_size = 128
input_dim = 256
sequence_length = 16
input_tensor = torch.randn(batch_size, sequence_length, input_dim)

# 创建模型实例
model = ALSTMModel()
predicted_output = model(input_tensor)
# 打印输出张量的形状
print("Output shape:", predicted_output.shape)
