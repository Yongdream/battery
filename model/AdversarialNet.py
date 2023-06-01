from torch import nn
import numpy as np


# # 计算系数的函数，用于计算对抗网络中的权重系数
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


# GRL（域逆向层）的hook函数，用于反向传播时对梯度进行处理
def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


# 对抗网络类
class AdversarialNet(nn.Module):
    def __init__(self, in_feature, hidden_size, max_iter=10000.0, trade_off_adversarial='Step', lam_adversarial=1.0):
        super(AdversarialNet, self).__init__()

        # in_feature: 输入特征的维度
        # hidden_size: 隐藏层的大小
        # max_iter: 最大迭代次数，默认为10000.0
        # trade_off_adversarial: 对抗损失权重的计算方式，默认为'Step'
        # lam_adversarial: 对抗损失的权重，默认为1.0

        self.ad_layer1 = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # parameters
        self.iter_num = 0
        self.alpha = 10         # 梯度反转层系数
        self.low = 0.0          # 梯度反转层系数下限
        self.high = 1.0
        self.max_iter = max_iter
        self.trade_off_adversarial = trade_off_adversarial
        self.lam_adversarial = lam_adversarial
        self.__in_features = 1

    def forward(self, x):
        if self.training:
            self.iter_num += 1

        # 根据设定的权重计算方式选择不同的权重系数
        if self.trade_off_adversarial == 'Cons':
            coeff = self.lam_adversarial
        elif self.trade_off_adversarial == 'Step':
            coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        else:
            raise Exception("loss not implement")
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return self.__in_features
