import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.optim.lr_scheduler import ExponentialLR
from matplotlib import pyplot as plt
from my_dataset import MyDataSet
import utils
from utils import read_split_data, get_parameter_number, delete_files, ResultLogger
from utils import LabelSmoothing, CELoss
from torch.utils.data import DataLoader
from model.models_yz import transformer, yang, Network
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter

from plot import ConfusionMatrix
from sklearn.metrics import confusion_matrix


def validate(model, val_loader, criterion, num_epochs):
    val_losses = []
    val_accs = []
    for epoch in range(num_epochs):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for step, data in enumerate(val_loader, start=0):
                states, labels = data
                logits = model(states.to(device))
                loss = criterion(logits, labels.to(device))
                predicted_labels = torch.argmax(logits, dim=1)
                correct += (predicted_labels == labels.to(device)).sum().item()
                total += len(labels)
                running_loss += loss.item()
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print("Validation Loss {:.4f}, Validation Accuracy {:.4f}".format(val_loss, val_acc))

    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.show()
    plt.plot(val_accs, label='validation accuracy')
    plt.legend()
    plt.show()
    return val_loss, val_acc


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_loss = float('inf')  # 初始化最小的损失值
    best_confusion_matrix = None  # 初始化最小损失时对应的混淆矩阵
    best_loss_val = float('inf')
    best_confusion_matrix_val = None

    # 定义当前patience计数
    patience = 12
    current_patience = 0

    writer = SummaryWriter('./logs/to')

    for epoch in range(num_epochs):

        # train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_logits = []

        all_predicted_labels = []
        all_labels = []

        all_predicted_labels_v = []
        all_labels_v = []

        for step, data in enumerate(train_loader, start=0):
            states, labels = data
            optimizer.zero_grad()
            logits = model(states.to(device))   # (B, 5)

            all_logits.append(logits)
            predicted_labels = torch.argmax(logits, dim=1)  # (B)
            # loss = criterion(logits, labels.to(device))     # (B,5) (B)
            loss = criterion(logits, labels.to(device))

            all_labels.append(labels)
            all_predicted_labels.append(predicted_labels)
            correct += (predicted_labels == labels.to(device)).sum().item()
            total += len(labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # t-sne图片显示
            # if epoch > (num_epochs * 0.95):
            #     tsne_visualization(logits, predicted_labels)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total 
        train_accs.append(epoch_acc)
        train_losses.append(epoch_loss)

        writer.add_scalar('train_acc', epoch_acc, epoch)
        writer.add_scalar('train_loss', epoch_loss, epoch)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_confusion_matrix = (all_labels, all_predicted_labels)

        # if epoch == num_epochs - 1:
        #     utils.summarize_confusion_matrix(best_confusion_matrix[0], best_confusion_matrix[1], 5,
        #                                      ['Cor', 'Isc', 'Noi', 'Nor', 'Sti'], title='Train')

        # validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for step, data in enumerate(val_loader, start=0):
                states, labels = data
                logits = model(states.to(device))
                # loss = criterion(logits, labels.to(device))
                loss = criterion(logits, labels.to(device))

                all_labels_v.append(labels)
                predicted_labels = torch.argmax(logits, dim=1)
                all_predicted_labels_v.append(predicted_labels)

                correct += (predicted_labels == labels.to(device)).sum().item()
                total += len(labels)
                running_loss += loss.item()

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        writer.add_scalar('Val_acc', val_acc, epoch)
        writer.add_scalar('Val_loss', val_loss, epoch)

        if val_loss < best_loss_val:
            best_loss_val = val_loss
            best_confusion_matrix_val = (all_labels_v, all_predicted_labels_v)
            current_patience = 0  # 重置耐心计数器
        else:
            current_patience += 1  # 验证损失没有改善，耐心计数器加1

        # if epoch == num_epochs - 1:
        #     utils.summarize_confusion_matrix(best_confusion_matrix_val[0], best_confusion_matrix_val[1], 5,
        #                                      ['Cor', 'Isc', 'Noi', 'Nor', 'Sti'], title='Train')

        print("Epoch {}: Training Loss {:.4f}, Training Accuracy {:.4f}, "
              "Validation Loss {:.4f}, Validation Accuracy {:.4f}"
              .format(epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc))

        # 判断是否达到耐心值，如果达到则停止训练
        if current_patience >= patience:
            print("Early stopping: Lack of improvement in loss.")
            break

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # 绘制训练损失和验证损失
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    min_val_loss = min(val_losses)
    min_val_loss_index = val_losses.index(min_val_loss)
    ax1.scatter(min_val_loss_index, min_val_loss, c='red', label='Min Validation Loss')
    ax1.plot([min_val_loss_index, min_val_loss_index], [min_val_loss+0.01, min_val_loss], 'r--', linewidth=0.7, alpha=0.7)
    ax1.plot([0, min_val_loss_index], [min_val_loss, min_val_loss], 'r--', linewidth=0.7, alpha=0.7)

    ax1.annotate(f'({min_val_loss_index}, {min_val_loss:.5f})', xy=(min_val_loss_index, min_val_loss), xytext=(-40, 20),
                 textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    # 绘制训练准确率和验证准确率
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    max_val_acc = max(val_accs)
    max_val_acc_index = val_accs.index(max_val_acc)
    ax2.scatter(max_val_acc_index, max_val_acc, c='red', label='Max Validation Accuracy')
    ax2.plot([max_val_acc_index, max_val_acc_index], [0.3, max_val_acc], 'r--', linewidth=0.7, alpha=0.7)
    ax2.plot([0, max_val_acc_index], [max_val_acc, max_val_acc], 'r--', linewidth=0.7, alpha=0.7)

    ax2.annotate(f'({max_val_acc_index}, {max_val_acc:.3%})', xy=(max_val_acc_index, max_val_acc), xytext=(-40, -30),
                 textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()

    # 最佳效果的混淆矩阵
    summary_tra = utils.summarize_confusion_matrix(best_confusion_matrix[0], best_confusion_matrix[1], 5,
                                     ['Cor', 'Isc', 'Noi', 'Nor', 'Sti'], title='Train')
    summary_val = utils.summarize_confusion_matrix(best_confusion_matrix_val[0], best_confusion_matrix_val[1], 5,
                                     ['Cor', 'Isc', 'Noi', 'Nor', 'Sti'], title='Valid')

    return max_val_acc, summary_val


op_mode = 'us06'
root = os.path.join('processed', op_mode)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print("using {} device.".format(device))

# # Set the deletion rates for each subfolder
# subfolders = [f.name for f in os.scandir(root) if f.is_dir()]
# deletion_rates = [0.75, 0.75, 0.75, 0, 0.75]
# delete_files(root, subfolders, deletion_rates)

train_wintime_path, train_wintime_label, val_wintime_path, val_wintime_label = read_split_data(root)
train_data_set = MyDataSet(timeWin_path=train_wintime_path,
                           timeWin_class=train_wintime_label)

val_data_set = MyDataSet(timeWin_path=val_wintime_path,
                         timeWin_class=val_wintime_label)
val_num = len(val_data_set)


nw = 0
batch_size = 128
num_epochs = 100
lr = 1e-3
model = Network().to(device)
# criterion = nn.CrossEntropyLoss()
criterion = CELoss(label_smooth=0.05, class_num=5)
optimizer = optim.Adam(model.parameters(), lr)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)   # 定义学习率调度程序
print('Model build successfully')
summary(model, input_size=(batch_size, 225, 20))

model_parameters = {'num_epochs': num_epochs, 'learning_rate': lr, 'batch_size': batch_size, 'model': model,
                    'criterion': criterion, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


train_loader = DataLoader(train_data_set,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=nw,
                          collate_fn=train_data_set.collate_fn,
                          drop_last=True)

val_loader = DataLoader(val_data_set,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=nw,
                        collate_fn=val_data_set.collate_fn,
                        drop_last=True)

logger = ResultLogger(op_mode)
best_accuracy, summary_val = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)
logger.log_best_result(best_accuracy, model_parameters, summary_val)
