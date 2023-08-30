import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
import numpy as np

# Hybrid Attention Mechanism with MMD


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

    # 将所有特征合并
    features = torch.cat((mean, var, max_abs_cum_change_rate), dim=1)

    return features  # 结果形状为(b, 16, 302)


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = F.softmax(Q @ K.transpose(-2, -1) / (self.embed_size ** 0.5), dim=-1)
        out = attention @ V
        return out


class InceptionA(nn.Module):

    def __init__(self, hidden_size=16, out_size=128, n_layers=5):

        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        self.branch1_gru = torch.nn.GRU(19, hidden_size, 3, batch_first=True, bidirectional=True)
        self.branch2_conv = nn.Conv1d(in_channels=19, out_channels=32, kernel_size=1, stride=1, padding=0)

    def forward(self, source, target):
        s_branch1_gru = source.permute(0, 2, 1)
        s_branch1_gru, _ = self.branch1_gru(s_branch1_gru)
        s_branch1_gru = s_branch1_gru.permute(0, 2, 1)      # torch.Size([b, 32, 300])

        s_branch2_conv = self.branch2_conv(source)          # torch.Size([b, 32, 300])

        t_branch1_gru = target.permute(0, 2, 1)
        t_branch1_gru, _ = self.branch1_gru(t_branch1_gru)
        t_branch1_gru = t_branch1_gru.permute(0, 2, 1)  # torch.Size([b, 32, 300])

        t_branch2_conv = self.branch2_conv(target)  # torch.Size([b, 32, 300])

        s_branch1_gru = torch.sum(s_branch1_gru, dim=1)
        s_branch2_conv = torch.sum(s_branch2_conv, dim=1)
        t_branch1_gru = torch.sum(t_branch1_gru, dim=1)
        t_branch2_conv = torch.sum(t_branch2_conv, dim=1)

        return s_branch1_gru, s_branch2_conv, t_branch1_gru, t_branch2_conv


class ATTFE(nn.Module):
    def __init__(self,  pretrained=False):
        super(ATTFE, self).__init__()

        self.__in_features = 600

        if pretrained:
            warnings.warn("Pretrained model is not available")

        self.attention = SelfAttention(embed_size=300)
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1)

        self.Inception = InceptionA()

    def forward(self, source, target, s_label):
        # Input shape: (batch_size, 16, 300)
        # s_att = self.attention(source)
        s_con = self.conv1(source)       # (batch_size, 3, 300)
        s_fe = extract_features(source)    # (batch_size, 3, 300)
        s_comb = torch.cat((source, s_fe), dim=1)  # torch.Size([b, 6, 300])

        # t_att = self.attention(target)
        t_con = self.conv1(target)       # (batch_size, 3, 300)
        t_fe = extract_features(target)    # (batch_size, 3, 300)
        t_comb = torch.cat((target, t_fe), dim=1)  # torch.Size([b, 6, 300])

        b1_source, b2_source, b1_target, b2_target = self.Inception(s_comb, t_comb)

        return b1_source, b2_source, b1_target, b2_target

    def output_num(self):
        return self.__in_features


# batch_size = 5  # 这只是一个随机选择的批处理大小，你可以根据需要更改它
# source_data = torch.randn(batch_size, 16, 300)
# target_data = torch.randn(batch_size, 16, 300)
# s_label = torch.randn(batch_size, 16, 300)  # 假设标签也是形状为(5, 16, 300)的张量
#
# # 创建模型实例
# model = ATTFE()
#
# # 将数据通过模型
# b1_source, b2_source, b1_target, b2_target = model(source_data, target_data, s_label)
#
# # 打印输出的形状以确保一切按计划进行
# print(b1_source.shape)
# print(b2_source.shape)
# print(b1_target.shape)
# print(b2_target.shape)
