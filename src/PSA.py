import numpy as np
import torch
from torch import nn
from torch.nn import init


class PSA(nn.Module):
    """
        金字塔空间注意力（PSA）模块
    """

    def __init__(self, channel=16, reduction=4, S=4):
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
        self.convs = []
        for i in range(S):
            self.convs.append(nn.Conv1d(channel//S, channel//S, kernel_size=2*(i+1)+1, padding=i+1))

        # 为每个尺度初始化SE块
        self.se_blocks = []
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channel//S, channel // (S*reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel // (S*reduction), channel//S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ))
        
        self.softmax = nn.Softmax(dim=1)

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

    def forward(self, x):
        b, c, h = x.size()

        # 步骤1：SPC模块
        SPC_out = x.view(b, self.S, c//self.S, h)  # bs, s, ci, h
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :] = conv(SPC_out[:, idx, :, :])

        # 步骤2：SE权重
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)

        # Step3:Softmax
        softmax_out = self.softmax(SE_out)

        # Step4:SPA
        PSA_out = SPC_out*softmax_out
        PSA_out = PSA_out.view(b, -1, h)

        return PSA_out
