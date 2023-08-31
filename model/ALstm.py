import torch
import torch.nn as nn
import warnings


def z_score_normalize(tensor, dim=2):
    mean_val = torch.mean(tensor, dim=dim, keepdim=True)
    std_val = torch.std(tensor, dim=dim, keepdim=True)
    standardized_tensor = (tensor - mean_val) / std_val
    return standardized_tensor


def extract_features(data):
    """
    从时间序列数据中提取均值、方差、最大变化率。
    :param data: 时间序列数据，形状为(b, 16, 300)。
    :return: 提取的特征矩阵。
    """
    # 沿着时间轴计算均值和方差
    mean = torch.mean(data, dim=1, keepdim=True)    # (b, 1, 300)
    var = torch.var(data, dim=1, keepdim=True)      # (b, 1, 300)

    # 计算最大差分变化率
    diff = torch.diff(data, dim=2)                  # (b, 16, 299)
    cum_change_rate = torch.cumsum(diff, dim=2)     # (b, 16, 299)
    abs_cum_change_rate = torch.abs(cum_change_rate)
    max_abs_cum_change_rate, _ = torch.max(abs_cum_change_rate, dim=1, keepdim=True)  # 结果形状为(b, 1, 299)

    # 将最大累积变化率填充到原始形状（b, 1, 300）
    padding = torch.zeros((data.size(0), 1, 1), device=data.device)
    max_abs_cum_change_rate = torch.cat((padding, max_abs_cum_change_rate), dim=2)

    diff_mean = (data - mean) / mean
    # 计算 xi^2
    xi_squared = torch.mean(diff_mean ** 2, axis=1)
    xi_squared = xi_squared.unsqueeze(1)
    # 将所有特征合并
    features = torch.cat((mean, var, max_abs_cum_change_rate, xi_squared), dim=1)

    return features  # 结果形状为(b, 16, 302)


class ALSTMAdFeatures(nn.Module):
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
        # self.conv_net.add_module('avg', nn.AvgPool1d(kernel_size=3, stride=2, padding=1))
        self.conv_net.add_module('conv2', nn.Conv1d(32, 60, kernel_size=3, stride=1, padding=1))
        self.conv_net.add_module('conv2_act2', nn.ReLU())

        self.batch_norm = torch.nn.BatchNorm1d(num_features=64)

        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        # self.rnn = klass(
        #     input_size=64,
        #     hidden_size=self.hid_size,
        #     num_layers=self.rnn_layer,
        #     batch_first=True,
        #     dropout=self.dropout,
        #     bidirectional=True
        # )
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

        self.fc1 = torch.nn.Linear(9600, 2048)
        self.fc2 = torch.nn.Linear(2048, 128)

        # self.single_out = nn.Sequential()
        # self.single_out.add_module(
        #
        # )

    def forward(self, inputs):
        # inputs: [batch_size, input_size*input_day]
        # inputs = inputs.view(len(inputs), self.input_size, -1)
        out_conv = self.conv_net(inputs)
        s_fe = extract_features(inputs)     # torch.Size([128, 4, 600])
        s_fe = z_score_normalize(s_fe)

        out_conv = torch.cat((out_conv, s_fe), dim=1)
        # out_conv = self.batch_norm(out_conv)
        out_conv = self.avg_pool(out_conv)

        out_conv = out_conv.permute(0, 2, 1)    # torch.Size([128, 150, 64])
        rnn_out, _ = self.gru_1(out_conv)       # [batch, seq_len, num_directions * hidden_size]
        rnn_out, _ = self.gru_2(rnn_out)    # torch.Size([128, 300, 32])

        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1] 计算注意力分数

        out_att = torch.mul(rnn_out, attention_score)   # 使用注意力分数加权 torch.Size([128, 64, 64])
        # out_att = torch.sum(out_att, dim=1)             # 沿着序列长度的维度求和

        # 将RNN层的最后一个时间步和注意力加权结果进行拼接，然后通过全连接层输出
        # torch.Size([128, 300, 64])
        out_att = out_att.flatten(1)  # ([128, 2000])
        out = self.fc1(out_att)
        out = self.fc2(out)
        return out

    def output_num(self):
        return self.__in_features


batch_size = 128
input_dim = 600
sequence_length = 12
input_tensor = torch.randn(batch_size, sequence_length, input_dim)

# 创建模型实例
model = ALSTMAdFeatures()
predicted_output = model(input_tensor)
# 打印输出张量的形状
print("Output shape:", predicted_output.shape)
