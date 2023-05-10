import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.optim.lr_scheduler import ExponentialLR
from matplotlib import pyplot as plt
from my_dataset import MyDataSet
from utils import read_split_data, get_parameter_number, delete_files
from torch.utils.data import DataLoader
from model.models_yz import transformer, yang
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

    for epoch in range(num_epochs):
        # train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_logits = []
        all_predicted_labels = []
        all_labels = []

        for step, data in enumerate(train_loader, start=0):
            states, labels = data
            optimizer.zero_grad()
            logits = model(states.to(device))
            # (128, 5)
            all_logits.append(logits)

            loss = criterion(logits, labels.to(device))
            predicted_labels = torch.argmax(logits, dim=1)

            all_labels.append(labels)
            all_predicted_labels.append(predicted_labels)
            correct += (predicted_labels == labels.to(device)).sum().item()
            total += len(labels)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            running_loss += loss.item()

            # # t-sne图片显示
            # if epoch > (num_epochs * 0.95):
            #     tsne_visualization(logits, predicted_labels)
        if epoch == num_epochs-1:
            # 混淆矩阵test
            confusion_matrix_obj = ConfusionMatrix(5, ['Cor', 'Isc', 'Noi', 'Nor', 'Sti'])
            labels_n = np.concatenate([t.cpu().numpy() for t in all_labels])
            predicted_labels_n = np.concatenate([t.cpu().numpy() for t in all_predicted_labels])
            confusion_matrix_obj.update(predicted_labels_n, labels_n)
            sk_confusion_matrix = confusion_matrix(labels_n, predicted_labels_n)
            print('sk_confusion_matrix:\n', sk_confusion_matrix)
            confusion_matrix_obj.plot()
            confusion_matrix_obj.summary()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total 
        train_accs.append(epoch_acc)
        train_losses.append(epoch_loss)

        # if epoch > (num_epochs*0.9):
        #     tsne_visualization(all_logits, all_predicted_labels)
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
                predicted_labels = torch.argmax(logits, dim=1)
                correct += (predicted_labels == labels.to(device)).sum().item()
                total += len(labels)
                running_loss += loss.item()
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print("Epoch {}: Training Loss {:.4f}, Training Accuracy {:.4f}, "
              "Validation Loss {:.4f}, Validation Accuracy {:.4f}"
              .format(epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc))

    plt.plot(train_losses, label='Training Loss')
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.show()
    return model, train_losses, train_accs, val_losses, val_accs


root = r'processed\udds'
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
batch_size = 256
num_epochs = 32
lr = 1e-4
model = yang().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)
lr_scheduler = ExponentialLR(optimizer, gamma=0.99)   # 定义学习率调度程序
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
