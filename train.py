import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.optim.lr_scheduler import ExponentialLR
from matplotlib import pyplot as plt
from my_dataset import MyDataSet
import utils
from utils import read_split_data, get_parameter_number, delete_files
from utils import LabelSmoothing, CELoss
from torch.utils.data import DataLoader
from model.models_yz import transformer, yang, Network
from torchinfo import summary

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


def train(model, train_loader, val_loader ,criterion, optimizer, num_epochs):
    train_losses = []
    train_accs = []

    val_losses = []
    val_accs = []
    smoothing_factor = 0.1
    # label_smoothing = LabelSmoothing(smoothing=smoothing_factor)
    loss2 = CELoss(label_smooth=0.05, class_num=5)
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
            loss = criterion(logits, labels.to(device))     # (B,5) (B)
            # loss = loss2(logits, labels.to(device))

            all_labels.append(labels)
            all_predicted_labels.append(predicted_labels)
            correct += (predicted_labels == labels.to(device)).sum().item()
            total += len(labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # # t-sne图片显示
            # if epoch > (num_epochs * 0.95):
            #     tsne_visualization(logits, predicted_labels)
        if epoch == num_epochs - 1:
            utils.plot_and_summarize_confusion_matrix(all_labels, all_predicted_labels,
                                                      5, ['Cor', 'Isc', 'Noi', 'Nor', 'Sti'], title='Train')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total 
        train_accs.append(epoch_acc)
        train_losses.append(epoch_loss)

        # validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for step, data in enumerate(val_loader, start=0):
                states, labels = data
                logits = model(states.to(device))
                loss = criterion(logits, labels.to(device))
                # loss = loss2(logits, labels.to(device))

                all_labels_v.append(labels)
                predicted_labels = torch.argmax(logits, dim=1)
                all_predicted_labels_v.append(predicted_labels)

                correct += (predicted_labels == labels.to(device)).sum().item()
                total += len(labels)
                running_loss += loss.item()

            if epoch == num_epochs - 1:
                utils.plot_and_summarize_confusion_matrix(all_labels_v, all_predicted_labels_v, 5,
                                                          ['Cor', 'Isc', 'Noi', 'Nor', 'Sti'], title='Valid')

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print("Epoch {}: Training Loss {:.4f}, Training Accuracy {:.4f}, "
              "Validation Loss {:.4f}, Validation Accuracy {:.4f}"
              .format(epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc))

    # 创建包含两个子图的图形
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

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()
    return model, train_losses, train_accs, val_losses, val_accs


root = r'processed\us06'
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
num_epochs = 5
lr = 1e-3
model = Network().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)   # 定义学习率调度程序
print('Model build successfully')
summary(model, input_size=(batch_size, 225, 20))

best_acc = 0.0
save_path = 'logs/demo.pth'


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

train(model, train_loader, val_loader, criterion, optimizer, num_epochs)
