import os
import json
import random
import torch

from matplotlib import pyplot as plt
from prettytable import PrettyTable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utlis.plotCFM import ConfusionMatrix
from sklearn.metrics import confusion_matrix


# Label Smoothing 标签平滑
class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, label_smooth=None, class_num=137):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)	# softmax + log
            target = F.one_hot(target, self.class_num)	# 转换成one-hot
            # label smoothing
            # 实现 1
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num
            # 实现 2
            # implement 2
            target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
            loss = -1*torch.sum(target*logprobs, 1)
        else:
            # standard cross entropy loss
            loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))
        return loss.mean()


def read_split_data(root: str, val_rate: float = 0.2, plot_image = False):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    state_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    state_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(state_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('../class_indices.json', 'w') as json_file:
        json_file.write(json_str)
        
    json_path = os.path.abspath('../class_indices.json')
    print('class_indices.json的绝对路径：', json_path)

    train_wintime_path = []
    train_wintime_label = []
    val_wintime_path = []
    val_wintime_label = []
    every_class_num = []
    supported = [".npy"]

    # 遍历每个文件夹下的文件
    for cla in state_class:
        cla_path = os.path.join(root, cla)
        states = [os.path.join(root, cla, i) 
                 for i in os.listdir(cla_path)
                 if os.path.splitext(i)[-1] in supported]
        
        time_class = class_indices[cla]
        every_class_num.append(len(states))
        val_path = random.sample(states, k=int(len(states) * val_rate))
        
        for state_path in states:
            if state_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_wintime_path.append(state_path)
                val_wintime_label.append(time_class)
            else:  # 否则存入训练集
                train_wintime_path.append(state_path)
                train_wintime_label.append(time_class)
    
    print("{} states were found in the dataset.".format(sum(every_class_num)))
    print("{} states for training.".format(len(train_wintime_path)))
    print("{} states for validation.".format(len(val_wintime_path)))

    if plot_image:
        #绘制每种类别个数柱状图
        plt.bar(range(len(state_class)), every_class_num, align='center')  # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(state_class)), state_class)  #在柱状图上添加数值标签for i, v in enumerate(every_class_num):plt.text(x=i,y=v + 5, s=str(v), ha='center')#设置x坐标
        plt.xlabel('state class')
        #设置y坐标
        plt.ylabel('number of state')
        #设置柱状图的标题
        plt.title('Class Distribution')
        plt.show()
    
    return train_wintime_path, train_wintime_label, val_wintime_path, val_wintime_label


def delete_files(folder_path, subfolders, deletion_rates):
    before_deletion = []
    after_deletion = []
    for i, subfolder in enumerate(subfolders):
        # Get the path of the current subfolder
        subfolder_path = os.path.join(folder_path, subfolder)
        # Get the list of files in the subfolder
        file_list = os.listdir(subfolder_path)

        # Calculate the number of files before deletion
        before_count = len(file_list)

        # Calculate the number of files to delete
        num_to_delete = int(deletion_rates[i] * before_count)

        # Randomly select files to delete
        files_to_delete = random.sample(file_list, num_to_delete)

        # Delete the selected files
        for file_name in files_to_delete:
            file_path = os.path.join(subfolder_path, file_name)
            os.remove(file_path)

        # Get the list of files after deletion
        file_list = os.listdir(subfolder_path)

        # Calculate the number of files after deletion
        after_count = len(file_list)

        # Append the file counts to the lists
        before_deletion.append(before_count)
        after_deletion.append(after_count)

    # Plot the comparison chart
    plt.bar(subfolders, before_deletion, label="Before Deletion")
    plt.bar(subfolders, after_deletion, label="After Deletion")
    plt.legend()
    plt.show()


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def summarize_confusion_matrix(all_labels, all_predicted_labels, num_classes, class_names, title):
    labels_n = np.concatenate([t.cpu().numpy() for t in all_labels])
    predicted_labels_n = np.concatenate([t.cpu().numpy() for t in all_predicted_labels])

    # labels_n = np.concatenate([t.cpu().numpy().reshape(1,) for t in all_labels])
    # predicted_labels_n = np.concatenate([t.cpu().numpy().reshape(1,) for t in all_predicted_labels])

    # 计算混淆矩阵
    confusion_matrix_obj = ConfusionMatrix(num_classes, class_names, title)
    confusion_matrix_obj.update(predicted_labels_n, labels_n)
    sk_confusion_matrix = confusion_matrix(labels_n, predicted_labels_n)

    # 打印sklearn生成的混淆矩阵
    print('sk_confusion_matrix:\n', sk_confusion_matrix)

    # 绘制混淆矩阵
    summary, _, matrix_plt = confusion_matrix_obj.plot()

    return summary, matrix_plt


def calculate_summary(matrix, labels):
    n = np.sum(matrix)
    sum_tp = np.sum(np.diag(matrix))
    acc = sum_tp / n

    sum_po = np.sum(np.diag(matrix))
    sum_pe = np.sum(np.sum(matrix, axis=0) * np.sum(matrix, axis=1))
    po = sum_po / n
    pe = sum_pe / (n * n)
    kappa = round((po - pe) / (1 - pe), 3)

    table = PrettyTable()
    table.field_names = ["", "Precision", "Recall", "Specificity", "F1 Score"]
    num_classes = len(labels)
    for i in range(num_classes):
        tp = matrix[i, i]
        fp = np.sum(matrix[i, :]) - tp
        fn = np.sum(matrix[:, i]) - tp
        tn = np.sum(matrix) - tp - fp - fn

        Precision = round(tp / (tp + fp), 3) if tp + fp != 0 else 0.
        Recall = round(tp / (tp + fn), 3) if tp + fn != 0 else 0.
        Specificity = round(tn / (tn + fp), 3) if tn + fp != 0 else 0.
        F1 = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.

        table.add_row([labels[i], Precision, Recall, Specificity, F1])

    summary_str = f"The model accuracy is {acc}\n"
    summary_str += f"The model kappa is {kappa}\n"
    summary_str += str(table) + '\n'

    return str(acc), summary_str, Recall


class ResultLogger:
    def __init__(self, op_mode):
        self.op_mode = op_mode
        self.log_dir = os.path.join('../logs', self.op_mode)

        self.log_file = os.path.join(self.log_dir, 'best_result.log')

        if not os.path.exists(self.log_file):
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            with open(self.log_file, 'w') as f:
                f.write('')  # Creating an empty file

    def log_best_result(self, acc, model_params, summary_val):
        with open(self.log_file, 'r') as f:
            lines = f.readlines()

        best_acc = 0.0
        for line in lines:
            if line.startswith('Best Accuracy:'):
                best_acc = float(line.split(':')[1].strip())
                break

        if acc > best_acc:
            with open(self.log_file, 'w') as f:
                f.write(f'Operating conditions: {self.op_mode}\n')
                f.write(f'Best Accuracy: {acc}\n')
                f.write('Model Parameters:\n')
                for param_name, param_value in model_params.items():
                    f.write(f'  {param_name}: {param_value}\n')
                f.write(f'summary: {summary_val}\n')

        print(self.op_mode + " success!")


def calculate_label_recall(confMatrix, labelidx):
    '''
    计算某一个类标的召回率：
    '''
    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100*float(label_correct_sum)/float(label_total_sum),2)
    return recall


def generate_confusion_matrix(actual_labels, predicted_labels):
    unique_labels = sorted(list(set(actual_labels) | set(predicted_labels)))
    num_labels = len(unique_labels)

    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)

    for actual, predicted in zip(actual_labels, predicted_labels):
        actual_index = label_to_index[actual]
        predicted_index = label_to_index[predicted]
        confusion_matrix[actual_index][predicted_index] += 1

    return confusion_matrix


# # 测试示例
# labels = ['A', 'B', 'C', 'D']
# matrix = np.array([[10, 2, 3, 0],
#                    [1, 12, 0, 1],
#                    [2, 1, 8, 0],
#                    [0, 3, 1, 7]])
#
# acc, summary, recall = calculate_summary(matrix, labels)
#
# print(f"Accuracy: {acc}")
# print(f"Summary:\n{summary}")
# print(f"Recall for class C: {recall[2]}")




