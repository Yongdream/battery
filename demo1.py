import torch

# 假设 phase 是一个表示当前阶段的字符串变量
phase = 'test'  # 假设当前阶段不是 'source_train'

# 在 source_train 阶段开启梯度计算，其他阶段关闭梯度计算
with torch.set_grad_enabled(phase == 'source_train'):
    # 这里的代码只会在 phase == 'source_train' 时执行
    print('1')

    if phase == 'source_train':
        print('112')

# 在其他阶段，梯度计算被关闭，这里可以执行其他代码
print("Do something else in other phases")
