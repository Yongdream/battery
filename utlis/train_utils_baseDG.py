import logging
import os
import time
import warnings
import math
import torch
from torch import nn
from torch import optim
import numpy as np
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import datasets
import model
from torchinfo import summary

from utlis.utils import CELoss
from utlis.utils import summarize_confusion_matrix

from loss.DAN import DAN
from loss.JAN import JAN
from loss.CORAL import CORAL
from utlis.entropy_CDA import Entropy
from utlis.entropy_CDA import calc_coeff
from utlis.entropy_CDA import grl_hook
from utlis.plot_sne import plot_2D
from utlis.plot_3dsne import plot_3D


classes = ['Cor', 'Isc', 'Noi', 'Nor', 'Sti']


class TrainUtilsDG(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        if isinstance(args.transfer_task[0], str):
           #print(args.transfer_task)
           args.transfer_task = eval("".join(args.transfer_task))
        self.datasets['source_train'], self.datasets['source_val'], _, self.datasets['target_val'] = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=True)

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.num_workers,
                                                           drop_last=True,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['source_train', 'source_val', 'target_val']}
        self.model = getattr(model, args.model_name)(args.pretrained)
        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)

        if args.domain_adversarial:
            self.max_iter = len(self.dataloaders['source_train']) * (args.max_epoch - args.middle_epoch)
            if args.bottleneck:
                    self.AdversarialNet = getattr(model, 'AdversarialNet_multi')(
                                                                            in_feature=args.bottleneck_num,
                                                                            output_size=len(args.transfer_task[0]),
                                                                            hidden_size=args.hidden_size, max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )
            else:
                    self.AdversarialNet = getattr(model, 'AdversarialNet_multi')(
                                                                            in_feature=self.model.output_num(),
                                                                            output_size=len(args.transfer_task[0]),
                                                                            hidden_size=args.hidden_size,
                                                                            max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )


        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.bottleneck:
                self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)
            if args.domain_adversarial:
                self.AdversarialNet = torch.nn.DataParallel(self.AdversarialNet)
            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)

        # Define the learning parameters
        if args.domain_adversarial:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
        else:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'cos':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 20, 0)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        self.start_epoch = 0

        # Invert the model and define the loss
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.domain_adversarial:
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)


        if args.criterion == 'Entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif args.criterion == 'CeLoss':
            self.criterion = CELoss(label_smooth=0.05, class_num=5)
        else:
            raise Exception("Criterion not implement")

        logging.info(summary(self.model_all, input_size=(args.batch_size, 16, 256)))
        print('Model build successfully!')

    def train(self, cond):
        """
        Training process
        :return:
        """
        args = self.args

        best_confusion_matrix_val = None

        # 定义当前patience计数
        patience = args.patience
        current_patience = 0

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        iter_num = 0
        sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')

        writer = SummaryWriter(f'./logs/{args.method}/{cond}/{cond}{sub_dir}')

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                all_pred_labels_list = []
                all_true_labels_list = []

                feature_prep = torch.empty(0, 256, device=self.device)

                # Set model to train mode or test mode
                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    if args.domain_adversarial:
                        self.AdversarialNet.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    if args.domain_adversarial:
                        self.AdversarialNet.eval()
                    self.classifier_layer.eval()

                for batch_idx, (inputs, labels_temp) in enumerate(self.dataloaders[phase]):

                    inputs = inputs.to(self.device)
                    labels_temp = labels_temp.to(self.device)
                    labels = labels_temp.long() % 100
                    ds_labels = labels_temp.long()//100

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # forward
                        features = self.model(inputs)   # features (b,256)
                        if args.bottleneck:
                            features = self.bottleneck_layer(features)
                        outputs = self.classifier_layer(features)
                        if phase != 'source_train' or epoch < args.middle_epoch:
                            logits = outputs
                            classifier_loss = self.criterion(logits, labels)
                            loss = classifier_loss
                        else:
                            logits = outputs.narrow(0, 0, labels.size(0))
                            classifier_loss = self.criterion(logits, labels)

                            # Calculate the domain adversarial
                            if args.domain_adversarial:
                                adversarial_label = ds_labels.to(self.device)
                                adversarial_out = self.AdversarialNet(features)     # adversarial_out (b, 2)
                                adversarial_loss = self.criterion(adversarial_out, adversarial_label)
                            else:
                                adversarial_loss = 0
                            # 总损失
                            loss = classifier_loss + adversarial_loss

                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = classifier_loss.item() * labels.size(0)     # 只是一个分类的
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        all_true_labels_list.append(labels)
                        all_pred_labels_list.append(pred)

                        result_true = torch.cat(all_true_labels_list).view(-1)
                        result_prep = torch.cat(all_pred_labels_list).view(-1)
                        feature_prep = torch.cat((feature_prep, features), dim=0)

                        if phase == 'source_train':
                            # source_data = feature_prep
                            # source_label = result_true
                            source_data = features
                            source_label = labels
                        elif phase == 'target_val':
                            # target_data = feature_prep
                            # target_label = result_prep
                            target_data = features
                            target_label = labels

                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += labels.size(0)

                            # Print the training information
                            if step % args.print_step == 0:
                                # 记录每个batch的训练时间和总训练时间
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count

                                temp_time = time.time()  # 获取当前时间
                                train_time = temp_time - step_start
                                step_start = temp_time  # 开始时间重新计时

                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time

                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                            '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(labels), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # Print the train and val information via each epoch

                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

                writer.add_scalar(phase + '-acc', epoch_acc, epoch)
                writer.add_scalar(phase + '-loss', epoch_loss, epoch)

                # save the model
                if phase == 'target_val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model_all.state_dict()
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc and epoch > args.middle_epoch/2:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

                        current_patience = 0  # 重置耐心计数器
                        best_confusion_matrix_val = (all_true_labels_list, all_pred_labels_list)

                        source_data_best = source_data
                        source_label_best = source_label
                        target_data_best = target_data
                        target_label_best = target_label
                    else:
                        current_patience += 1  # 验证损失没有改善，耐心计数器加1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # 判断是否达到耐心值，如果达到则停止训练
            if current_patience >= patience:
                print("Early stopping: Lack of improvement in loss.")
                writer.close()
                break

        summary_confusion = summarize_confusion_matrix(best_confusion_matrix_val[0], best_confusion_matrix_val[1], 5,
                                                       ['Cor', 'Isc', 'Noi', 'Nor', 'Sti'],
                                                       title='Target_Valid')
        sne = plot_2D(source_data_best, source_label_best, target_data_best, target_label_best, classes)
        logging.info(summary_confusion)
        writer.add_figure('Source and Target Domains', sne)
        # writer.add_figure('Confusion_matrix', conf_plt)
        writer.close()



