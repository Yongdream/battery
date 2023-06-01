import torch
import torch.nn as nn
import warnings
from torch.nn import init


class PSA(nn.Module):
    def __init__(self, channel=64, reduction=4, S=4):
        super().__init__()
        """
        初始化PSA模块。

        参数：
            channel（int）：输入通道数
            reduction（int）：SE块的缩减因子
            S（int）：尺度的数量
        """
        self.S = S

        # 为每个尺度初始化卷积层
        self.convs = nn.ModuleList()
        for i in range(S):
            self.convs.append(nn.Conv1d(64 // S, 64 // S, kernel_size=2 * (i + 1) + 1, padding=i + 1))

        # 为每个尺度初始化SE块
        self.se_blocks = nn.ModuleList()
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(64 // S, 64 // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(64 // (S * reduction), 64 // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs):
        # PSA
        b, c, h = outputs.size()

        # 步骤1：SPC模块
        SPC_out = outputs.view(b, self.S, c // self.S, h).clone()  # 将输入张量重新形状为(b, S, c//S, h)
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :] = conv(SPC_out[:, idx, :, :].clone())     # 对每个尺度的输入应用卷积层

        # 步骤2：SE权重
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :]))    # 对每个尺度的输入应用SE块
        SE_out = torch.stack(se_out, dim=1)     # 将各个尺度的SE块输出堆叠起来
        SE_out = SE_out.expand_as(SPC_out)      # 将SE块输出的维度扩展为SPC模块输出的维度

        # Step3:Softmax
        softmax_out = self.softmax(SE_out)      # 对SE块输出应用Softmax函数，得到尺度权重

        # Step4:SPA
        PSA_out = torch.mul(SPC_out, softmax_out)   # 将SPC模块的输出与尺度权重相乘
        PSA_out = PSA_out.view(b, -1, h)            # 将结果重新形状为(b, c', h)

        return PSA_out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class BiGruAdPSA(nn.Module):
    def __init__(self, hidden_size=32, out_size=128, pretrained=False, n_layers=5, batch_size=1):
        super(BiGruAdPSA, self).__init__()

        if pretrained:
            warnings.warn("Pretrained model is not available")

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        self.conv1 = nn.Conv1d(16, 64, kernel_size=10, stride=2, padding=4)
        self.act1 = nn.ReLU()
        self.AvgPooling_1 = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)

        self.psa = PSA(channel=64, reduction=4, S=4)

        self.gru_1 = torch.nn.GRU(64, hidden_size, 5, batch_first=True, bidirectional=True)
        self.gru_2 = torch.nn.GRU(64, 16, 2, batch_first=True, bidirectional=True)

        self.fc1 = torch.nn.Linear(128, 512)
        self.fc2 = torch.nn.Linear(512, 64)
        self.fc3 = torch.nn.Linear(64, 128)

    def forward(self, inputs):
        # (b, 16, 256)

        outputs = self.conv1(inputs)     # (b, 64, 256)
        outputs = self.act1(outputs)
        outputs = self.AvgPooling_1(outputs)    # (b, 64, 256)
        outputs = self.psa(outputs)
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


class BiGruAdPSAFeatures(nn.Module):
    def __init__(self, pretrained=False):
        super(BiGruAdPSAFeatures, self).__init__()
        self.model_net = BiGruAdPSA(pretrained=pretrained)
        self.__in_features = 128

    def forward(self, outputs):
        outputs = self.model_net(outputs)
        return outputs

    def output_num(self):
        return self.__in_features


batch_size = 128
input_dim = 256
sequence_length = 16
input_tensor = torch.randn(batch_size, sequence_length, input_dim)

# 创建模型实例
model = BiGruAdPSAFeatures()

# 进行前向传播
output = model(input_tensor)

# 打印输出张量的形状
print("Output shape:", output.shape)

