import torch
import torch.nn as nn
import torch.optim as optim


class ALSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
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

        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())

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
        inputs = inputs.view(len(inputs), self.input_size, -1)
        inputs = inputs.permute(0, 2, 1)  # [batch, input_size, seq_len] -> [batch, seq_len, input_size]
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1] 计算注意力分数

        out_att = torch.mul(rnn_out, attention_score)  # 使用注意力分数加权
        out_att = torch.sum(out_att, dim=1)  # 沿着序列长度的维度求和

        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        # 将RNN层的最后一个时间步和注意力加权结果进行拼接，然后通过全连接层输出
        return out[..., 0]  # 返回输出的第一列值


model = ALSTMModel()
new_inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])  # 示例输入数据
print(new_inputs.shape)
predicted_output = model(new_inputs)
print("Predicted Output:", predicted_output.item())


# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
#
#
# def calculate_correlation_matrix(file_path, time_window=10, time_step=1):
#     data1 = pd.read_csv(file_path).values
#     data = data1.T
#     # 获取数据的通道数和序列长度
#     num_channels, sequence_length = data.shape
#     # 初始化相关系数矩阵
#     correlation_matrix = np.zeros((num_channels, sequence_length - time_window))
#
#     for i in range(num_channels):
#         j = i + 1
#         if j == num_channels:
#             idx = 0
#             for t in range(0, sequence_length - time_window, time_step):
#                 subsequence_i = data[0, t:t + time_window]
#                 subsequence_j = data[15, t:t + time_window]
#                 correlation = np.corrcoef(subsequence_i, subsequence_j)[0, 1]
#                 correlation_matrix[i, idx] = correlation
#                 idx += 1
#         elif j > i:
#             idx = 0
#             for t in range(0, sequence_length - time_window, time_step):
#                 subsequence_i = data[i, t:t + time_window]  # 取出当前通道的子序列
#                 subsequence_j = data[j, t:t + time_window]  # 取出另一个通道的子序列
#                 correlation = np.corrcoef(subsequence_i, subsequence_j)[0, 1]  # 计算相关系数
#                 correlation_matrix[i, idx] = correlation
#                 idx += 1
#
#     scaler = MinMaxScaler()
#     correlation_matrix_normalized = scaler.fit_transform(correlation_matrix.T)
#     return correlation_matrix_normalized, data1
#
#
# csv_file = 'data/fuds/FUDS_Isc_1ohm_0718.csv'
# correlation_matrix, org = calculate_correlation_matrix(csv_file, time_window=256, time_step=1)
# print(correlation_matrix.shape)




# import numpy as np
# import torch
#
#
# def summarize_confusion_matrix(all_labels, all_predicted_labels, num_classes, class_names, title):
#     labels = np.concatenate([t.cpu().numpy() for t in all_labels])
#     preds = np.concatenate([t.cpu().numpy() for t in all_predicted_labels])
#     # Rest of your code here
#
#     matrix = np.zeros((num_classes, num_classes))
#
#     for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
#         matrix[p, t] += 1
#     print(matrix)
#
#
# # Generate sample data
# num_classes = 5
# class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
# title = 'Confusion Matrix'
#
# all_labels = [torch.randint(0, num_classes, (128,)) for _ in range(100)]
# all_predicted_labels = [torch.randint(0, num_classes, (128,)) for _ in range(100)]
#
# # Call the function with the sample data
# summarize_confusion_matrix(all_labels, all_predicted_labels, num_classes, class_names, title)



# def calculate_label_recall(confMatrix, labelidx):
#     '''
#     计算某一个类标的召回率：
#     '''
#     label_total_sum = confMatrix.sum(axis=1)[labelidx]
#     label_correct_sum = confMatrix[labelidx][labelidx]
#     recall = 0
#     if label_total_sum != 0:
#         recall = round(100*float(label_correct_sum)/float(label_total_sum),2)
#     return recall
#
#
# def generate_confusion_matrix(actual_labels, predicted_labels):
#     unique_labels = sorted(list(set(actual_labels) | set(predicted_labels)))
#     num_labels = len(unique_labels)
#
#     label_to_index = {label: index for index, label in enumerate(unique_labels)}
#
#     confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)
#
#     for actual, predicted in zip(actual_labels, predicted_labels):
#         actual_index = label_to_index[actual]
#         predicted_index = label_to_index[predicted]
#         confusion_matrix[actual_index][predicted_index] += 1
#
#     return confusion_matrix


# actual_labels = [1, 0, 2, 1, 0, 2]
# predicted_labels = [1, 0, 1, 2, 0, 1]
#
# confMatrix = generate_confusion_matrix(actual_labels, predicted_labels)
# print(confMatrix)
# # 测试计算标签召回率
# labelidx = 1
# recall = calculate_label_recall(confMatrix, labelidx)
# print(recall)

# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
#
# writer = SummaryWriter(comment='test_tensorboard')
#
# for x in range(100):
#     writer.add_scalar('y=2x', x * 2, x)
#     writer.add_scalar('y=pow(2, x)', 2 ** x, x)
#
#     writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
#                                              "xcosx": x * np.cos(x),
#                                              "arctanx": np.arctan(x)}, x)
# writer.close()


# import numpy as np
# import pandas as pd
# import scipy.io
#
# # 加载MAT文件
# data = scipy.io.loadmat('dataorigin/UDDS_2600_data_41diya_1.mat')
#
# # 提取数据（假设数据按列存储）
# original_data = data['data']  # 根据MAT文件中的变量名提取数据
#
# # 将数据转为DataFrame
# df = pd.DataFrame(original_data)
#
# # 复制后8列，并添加到原始数据的末尾
# new_columns = df.iloc[:, -8:].copy()
# df = pd.concat([df, new_columns], axis=1)
#
# # 扩展为20列（如果原始数据不足12列，可以根据需要添加适当数量的空列）
# while df.shape[1] < 20:
#     df[f'New Column {df.shape[1] + 1}'] = ''
#
# # 保存为CSV文件
# output_path = 'data/fuds/FUDS_Cor_temp.csv'  # 输出文件的路径和名称
# df.to_csv(output_path, index=False)


# import torch
# import torch.nn as nn
# from utils import LabelSmoothing
#
# # 创建LabelSmoothing实例
# smoothing_factor = 0.1
# label_smoothing = LabelSmoothing(smoothing=smoothing_factor)
#
# # 生成随机输入和目标标签
# batch_size = 32
# num_classes = 10
# input_size = 100
# x = torch.randn(batch_size, input_size)
# target = torch.randint(0, num_classes, (batch_size,))
#
# # 计算损失
# loss = label_smoothing(x, target)
#
# print("Loss:", loss.item())

# import numpy as np
# import matplotlib.pyplot as plt
#
# file_path = "processed/udds/Isc/Isc_1230.npy"
# data_o = np.load(file_path)
#
#
# # 创建包含20个子图的图像
# fig, axes = plt.subplots(5, 4, figsize=(12, 10))
#
# # 遍历前20列，并在每个子图中输出图像
# for i, ax in enumerate(axes.flat):
#     data = data_o[:, i]
#
#     # fft_data = np.fft.fft(data)
#     fft_data = data
#
#     # 将频域数据 reshape 成二维图像
#     m = 15  # 图像的行数
#     n = len(data) // m  # 图像的列数，假设每行有 len(data) // m 列
#     fft_data_2d = np.reshape(fft_data, (m, n))
#
#     # 对二维图像进行整形操作
#     processed_data = np.log(fft_data_2d+0.001)
#     # processed_data = fft_data_2d
#
#     # 在当前子图中绘制预处理后的二维图像
#     ax.imshow(processed_data, cmap='hot', aspect='auto')
#     ax.set_title(f'Column {i+1}')
#
# # 调整子图间距和整体布局
# plt.tight_layout()
#
# # 显示图像
# plt.show()




# folder_path = r'E:/Sail/421'
# sub_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
#
# all_dataframes = []
# for sub_folder in sub_folders:
#     sub_folder_path = os.path.join(folder_path, sub_folder)
#     csv_files = [f for f in os.listdir(sub_folder_path) if f.endswith('.csv')]
#
#     dfs = []
#     for csv_file in csv_files:
#         csv_file_path = os.path.join(sub_folder_path, csv_file)
#         df = pd.read_csv(csv_file_path)



