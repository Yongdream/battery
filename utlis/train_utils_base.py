import logging
import os
import time
import warnings
import math

import numpy as np
import torch
from torch import nn
from torch import optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import datasets
import model
from torchinfo import summary

from utlis.utils import CELoss, generate_confusion_matrix, calculate_label_recall
from utlis.utils import summarize_confusion_matrix
from utlis.utils import calculate_summary

from loss.DAN import DAN
from loss.JAN import JAN
from loss.CORAL import CORAL
from loss.CMMD import CMMD
from loss.DSAN import DSAN
from utlis.entropy_CDA import Entropy
from utlis.entropy_CDA import calc_coeff
from utlis.entropy_CDA import grl_hook
from utlis.plot_sne import plot_label_2D, plot_domain_2D
from utlis.plot_3dsne import plot_3D


# lab_classes = ['Isc', 'Noi', 'Nor', 'Sti']
lab_classes = ['Inc', 'ISC', 'Noi', 'Nor', 'Sti']
# lab_classes = ['Cor', 'Isc', 'Nor']
dom_classes = ['Source', 'Target']


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()


class TrainUtilsDA(object):
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
        # getattr()函数是Python内置的一个函数，它用于获取对象的属性值
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        if isinstance(args.transfer_task[0], str):
           args.transfer_task = eval("".join(args.transfer_task))
        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'] = Dataset(args.data_dir, args.transfer_task).data_split(transfer_learning=True)

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.num_workers,
                                                           drop_last=True,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}

        # Define the model
        self.model = getattr(model, args.model_name)(args.pretrained)
        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)
            self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)
            self.model_all = nn.Sequential(self.model, self.classifier_layer)

        if args.domain_adversarial=='True':
            self.max_iter = len(self.dataloaders['source_train']) * (args.max_epoch - args.middle_epoch)
            if args.adversarial_loss == "CDA" or args.adversarial_loss == "CDA+E":
                if args.bottleneck:
                    self.AdversarialNet = getattr(model, 'AdversarialNet')(in_feature=args.bottleneck_num * Dataset.num_classes,
                                                                            hidden_size=args.hidden_size, max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial)
                else:
                    self.AdversarialNet = getattr(model, 'AdversarialNet')(
                                                                            in_feature=self.model.output_num() * Dataset.num_classes,
                                                                            hidden_size=args.hidden_size,
                                                                            max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )
            else:
                if args.bottleneck_num:
                    self.AdversarialNet = getattr(model, 'AdversarialNet')(in_feature=args.bottleneck_num,
                                                                            hidden_size=args.hidden_size,
                                                                            max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )
                else:
                    self.AdversarialNet = getattr(model, 'AdversarialNet')(in_feature=self.model.output_num(),
                                                                            hidden_size=args.hidden_size,
                                                                            max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )

        # if args.adabn:
        #     self.model_eval = getattr(model, args.model_name)(args.pretrained)
        #     self.model_eval.fc = torch.nn.Linear(self.model_eval.fc.in_features, Dataset.num_classes)

        # torch.nn.DataParallel 用于在多个 GPU 上同时运行模型的模块
        if self.device_count > 3:
            self.model = torch.nn.DataParallel(self.model)
            # if args.adabn:
            #     self.model_eval = torch.nn.DataParallel(self.model_eval)
            if args.bottleneck:
                self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)
            if args.domain_adversarial=='True':
                self.AdversarialNet = torch.nn.DataParallel(self.AdversarialNet)
            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)

        # Define the learning parameters
        if args.domain_adversarial=='True':
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
        # if args.adabn:
        #     self.model_eval.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.domain_adversarial=='True':
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)


        # Define the distance loss
        if args.distance_metric == "True":
            if args.distance_loss == 'MK-MMD':
                self.distance_loss = DAN
            elif args.distance_loss == "JMMD":
                ## add additional network for some methods
                self.softmax_layer = nn.Softmax(dim=1)
                self.softmax_layer = self.softmax_layer.to(self.device)
                self.distance_loss = JAN
            elif args.distance_loss == "CORAL":
                self.distance_loss = CORAL
            elif args.distance_loss == "DSAN":
                self.softmax_layer = nn.Softmax(dim=1)
                self.softmax_layer = self.softmax_layer.to(self.device)
                self.distance_loss = DSAN
            else:
                raise Exception("loss not implement")
        else:
            self.distance_loss = None

        # Define the adversarial loss
        if args.domain_adversarial=="True":
            if args.adversarial_loss == 'DA':
                self.adversarial_loss = nn.BCELoss()
            elif args.adversarial_loss == "CDA" or args.adversarial_loss == "CDA+E":
                ## add additional network for some methods
                self.softmax_layer_ad = nn.Softmax(dim=1)
                self.softmax_layer_ad = self.softmax_layer_ad.to(self.device)
                self.adversarial_loss = nn.BCELoss()
            else:
                raise Exception("loss not implement")
        else:
            self.adversarial_loss = None

        if args.criterion == 'Entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif args.criterion == 'CeLoss':
            # self.criterion = CELoss(label_smooth=0.05, class_num=5)
            self.criterion = CELoss(label_smooth=0.05, class_num=5)
            # self.criterion = CELoss(label_smooth=0.05, class_num=5)
        else:
            raise Exception("Criterion not implement")

        logging.info(summary(self.model_all, input_size=(args.batch_size, 12, 300)))
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
        best_recall = 0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        iter_num = 0
        sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.cond = cond
        writer = SummaryWriter(f'./logs/{args.method}/{cond}/{cond}{sub_dir}')

        dir_str = f"{args.model_name}_{sub_dir}"

        if args.wandb:
            import wandb
            os.environ['WANDB_SILENT'] = "true"
            wandb.login()
            save_name = f"{args.method}_{cond}"
            wandb.init(project=save_name, entity='yang7hi', name=dir_str)

            wandb.watch(self.model_all)
            for k, v in args.__dict__.items():
                wandb.config[k] = v

        arr_middle_epoch = False

        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            iter_target = iter(self.dataloaders['target_train'])
            len_target_loader = len(self.dataloaders['target_train'])

            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                all_pred_labels_list = []
                all_true_labels_list = []
                total_recall = 0
                best_recall_epoch = 0

                feature_all = torch.empty(0, args.bottleneck_num, device=self.device)

                # Set model to train mode or test mode
                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    if args.domain_adversarial=='True':
                        self.AdversarialNet.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    if args.domain_adversarial=='True':
                        self.AdversarialNet.eval()
                    self.classifier_layer.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if phase != 'source_train' or epoch < args.middle_epoch:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        source_label = labels
                        target_inputs, target_labels = iter_target.__next__()

                        # target_domain的域标签需要+源域标签数目 (未使用)
                        dt_labels = ((target_labels.long()) // 100).to(self.device) + len(args.transfer_task[0])

                        if args.model_name == 'ATTFE':
                            source_inputs = source_inputs.to(self.device)
                            target_inputs = target_inputs.to(self.device)
                        else:
                            inputs = torch.cat((source_inputs, target_inputs), dim=0)
                            inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                    if (step + 1) % len_target_loader == 0:
                        iter_target = iter(self.dataloaders['target_train'])

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # forward
                        if args.model_name == 'ATTFE':
                            b1_source, \
                            b2_source, \
                            b1_target, \
                            b2_target \
                                = self.model(source_inputs, target_inputs, source_label)
                        else:
                            features = self.model(inputs)

                        if args.bottleneck:
                            features = self.bottleneck_layer(features)

                        outputs = self.classifier_layer(features)
                        if phase != 'source_train' or epoch < args.middle_epoch:
                        # if phase != 'source_train':
                            logits = outputs
                            loss = self.criterion(logits, labels)
                        else:
                            # 如果阶段是'source_train'并且当前轮次大于等于args.middle_epoch
                            logits = outputs.narrow(0, 0, labels.size(0))   # 虽然引入了目标域 但是并不预测其的label，只是以此来对源域算loss
                            classifier_loss = self.criterion(logits, labels)

                            # Calculate the distance metric
                            if self.distance_loss is not None:
                                if args.distance_loss == 'MK-MMD':
                                    distance_loss = self.distance_loss(features.narrow(0, 0, labels.size(0)),
                                                                       features.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)))
                                elif args.distance_loss == 'JMMD':
                                    softmax_out = self.softmax_layer(outputs)
                                    distance_loss = self.distance_loss([features.narrow(0, 0, labels.size(0)), softmax_out.narrow(0, 0, labels.size(0))],
                                                                        [features.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)),
                                                                        softmax_out.narrow(0, labels.size(0), inputs.size(0) - labels.size(0))],)
                                elif args.distance_loss == 'CORAL':
                                    distance_loss = self.distance_loss(outputs.narrow(0, 0, labels.size(0)),
                                                                        outputs.narrow(0, labels.size(0),
                                                                        inputs.size(0) - labels.size(0)))
                                elif args.distance_loss == 'DSAN':
                                    softmax_out = self.softmax_layer(outputs)
                                    distance_loss = self.distance_loss([features.narrow(0, 0, labels.size(0)),
                                                                        outputs.narrow(0, 0, labels.size(0))],
                                                                       [features.narrow(0, labels.size(0),
                                                                                        inputs.size(0) - labels.size(
                                                                                            0)),
                                                                        outputs.narrow(0, labels.size(0),
                                                                                           inputs.size(0) - labels.size(
                                                                                               0))], )
                                elif args.distance_loss == 'CMMD':
                                    if args.model_name == 'ATTFE':
                                        distance_loss = self.distance_loss(s_features1, t_features1, source_label, t_label)
                                        distance_loss += self.distance_loss((s_features2, t_features2, source_label, t_label))
                                    else:
                                        print('Other models are not suitable for the CMMD method, please use the AAFE!')
                                else:
                                    raise Exception("loss not implement")

                            else:
                                distance_loss = 0

                            # Calculate the domain adversarial
                            if self.adversarial_loss is not None:
                                if args.adversarial_loss == 'DA':
                                    domain_label_source = torch.ones(labels.size(0)).float()
                                    domain_label_target = torch.zeros(inputs.size(0) - labels.size(0)).float()
                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(self.device)
                                    adversarial_out = self.AdversarialNet(features)
                                    adversarial_loss = self.adversarial_loss(adversarial_out.squeeze(), adversarial_label)
                                elif args.adversarial_loss == 'CDA':
                                    softmax_out = self.softmax_layer_ad(outputs).detach()

                                    # 将softmax_out和features进行扩展和批量矩阵乘法，用于输入到对抗网络中
                                    op_out = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))
                                    # 将op_out进行形状变换后输入到对抗网络中得到对抗网络的输出
                                    adversarial_out = self.AdversarialNet(op_out.view(-1, softmax_out.size(1) * features.size(1)))

                                    # 创建源域和目标域的标签，并将其拼接为对抗损失的标签
                                    domain_label_source = torch.ones(labels.size(0)).float()
                                    domain_label_target = torch.zeros(inputs.size(0) - labels.size(0)).float()
                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(self.device)
                                    adversarial_loss = self.adversarial_loss(adversarial_out.squeeze(), adversarial_label)
                                elif args.adversarial_loss == "CDA+E":
                                    softmax_out = self.softmax_layer_ad(outputs)
                                    coeff = calc_coeff(iter_num, self.max_iter)
                                    entropy = Entropy(softmax_out)
                                    entropy.register_hook(grl_hook(coeff))
                                    entropy = 1.0 + torch.exp(-entropy)
                                    entropy_source = entropy.narrow(0, 0, labels.size(0))
                                    entropy_target = entropy.narrow(0, labels.size(0), inputs.size(0) - labels.size(0))

                                    softmax_out = softmax_out.detach()
                                    op_out = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))
                                    adversarial_out = self.AdversarialNet(
                                        op_out.view(-1, softmax_out.size(1) * features.size(1)))

                                    # 源域标签
                                    domain_label_source = torch.ones(labels.size(0)).float().to(
                                        self.device)    # (b,)
                                    # 目标域标签
                                    domain_label_target = torch.zeros(inputs.size(0) - labels.size(0)).float().to(
                                        self.device)    # (b,)
                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(
                                        self.device)    # (b*2,)
                                    weight = torch.cat((entropy_source / torch.sum(entropy_source).detach().item(),
                                                        entropy_target / torch.sum(entropy_target).detach().item()), dim=0)
                                    # (b*2,)

                                    # adversarial_out 和 adversarial_label 中获取输入数据和标签
                                    adversarial_loss = torch.sum(weight.view(-1, 1) * self.adversarial_loss(adversarial_out.squeeze(), adversarial_label)) / torch.sum(weight).detach().item()
                                    iter_num += 1

                                else:
                                    raise Exception("loss not implement")
                            else:
                                adversarial_loss = 0

                            # Calculate the trade off parameter lam
                            if args.trade_off_distance == 'Cons':
                                lam_distance = args.lam_distance
                            elif args.trade_off_distance == 'Step':
                                lam_distance = 2 / (1 + math.exp(-10 * ((epoch - args.middle_epoch) /
                                                                        (args.max_epoch - args.middle_epoch)))) - 1
                            else:
                                raise Exception("trade_off_distance not implement")

                            # 总损失
                            loss = classifier_loss + lam_distance * distance_loss + adversarial_loss

                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        all_true_labels_list.append(labels)
                        all_pred_labels_list.append(pred)
                        confusion_matrix_val = (all_true_labels_list, all_pred_labels_list)

                        matrix = generate_confusion_matrix(all_true_labels_list[-1].cpu().numpy(), all_pred_labels_list[-1].cpu().numpy())
                        recall = calculate_label_recall(matrix, 1)
                        total_recall += recall
                        average_recall = total_recall/(batch_idx+1)

                        if average_recall > best_recall_epoch:
                            best_recall_epoch = average_recall
                            # best_matrix = (all_true_labels_list[-1], all_pred_labels_list[-1])

                        # confMatrix = generate_confusion_matrix(all_true_labels_list, all_pred_labels_list)

                        result_true = torch.cat(all_true_labels_list).view(-1)
                        result_prep = torch.cat(all_pred_labels_list).view(-1)

                        if phase == 'source_train':
                            if epoch >= args.middle_epoch:
                                source_feature_num = len(features) // 2
                                feature_all = torch.cat((feature_all, features[:source_feature_num, :]), dim=0)
                                source_data = feature_all
                                source_label = result_prep
                            else:
                                feature_all = torch.cat((feature_all, features), dim=0)
                                source_data = feature_all
                                source_label = result_prep
                        elif phase == 'target_val':
                            feature_all = torch.cat((feature_all, features), dim=0)
                            target_data = feature_all
                            target_label = result_prep

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
                                              epoch, batch_idx * len(inputs), len(self.dataloaders[phase].dataset),
                                              batch_loss, batch_acc, sample_per_sec, batch_time
                                              ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # Print the train and val information via each epoch

                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length

                cost_t = time.time() - epoch_start
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, cost_t
                ))
                if args.wandb:
                    wandb.log({
                        'Epoch': epoch,
                        '{}-Loss'.format(phase): epoch_loss,
                        '{}-Acc'.format(phase): epoch_acc,
                        '{}-Cost'.format(phase): cost_t,
                        "best_acc": best_acc
                    })

                writer.add_scalar(phase + '-acc', epoch_acc, epoch)
                writer.add_scalar(phase + '-loss', epoch_loss, epoch)

                # save the model
                if phase == 'target_val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model_all.state_dict()

                    if epoch == 1:
                        source_data_origin = source_data
                        source_label_origin = source_label
                        target_data_origin = target_data
                        target_label_origin = target_label

                        source_d_label_origin = torch.from_numpy(np.zeros(source_data_origin.shape[0])).to(self.device)
                        target_d_label_origin = torch.from_numpy(np.ones(target_data_origin.shape[0])).to(self.device)
                    elif epoch == args.middle_epoch:
                        source_data_mid = source_data
                        source_label_mid = source_label
                        target_data_mid = target_data
                        target_label_mid = target_label

                        source_d_label_mid = torch.from_numpy(np.zeros(source_data_mid.shape[0])).to(self.device)
                        target_d_label_mid = torch.from_numpy(np.ones(source_data_mid.shape[0])).to(self.device)

                    # save the best model according to the val accuracy
                    # print(f"Isc recall: {best_recall_epoch}")
                    if epoch_acc > best_acc and epoch > args.middle_epoch/2:
                        if best_recall_epoch > 0:
                            best_acc = epoch_acc
                            logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                            print("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                            torch.save(model_state_dic,
                                       os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

                            current_patience = 0  # 重置耐心计数器
                            best_confusion_matrix_val = (all_true_labels_list, all_pred_labels_list)

                            source_data_best = source_data
                            source_label_best = source_label
                            target_data_best = target_data
                            target_label_best = target_label

                            source_domain_label = torch.from_numpy(np.zeros(source_data_best.shape[0])).to(self.device)
                            target_domain_label = torch.from_numpy(np.ones(target_data_best.shape[0])).to(self.device)
                    else:
                        current_patience += 1  # 验证损失没有改善，耐心计数器加1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # 判断是否达到耐心值，如果达到则停止训练
            if current_patience >= patience:
                print("Early stopping: Lack of improvement in loss.")
                writer.close()
                break

        summary_confusion, matrix_plt = summarize_confusion_matrix(best_confusion_matrix_val[0],
                                                                   best_confusion_matrix_val[1], 5,
                                                                   ['Inc', 'ISC', 'Noi', 'Nor', 'Sti'],
                                                                   title='Target_Valid')

        sne_ori = plot_label_2D(source_data_origin, source_label_origin, target_data_origin, target_label_origin, lab_classes)
        sne_domain_ori = plot_domain_2D(source_data_origin, source_d_label_origin, target_data_origin, target_d_label_origin,
                                    dom_classes)

        sne_mid = plot_label_2D(source_data_mid, source_label_mid, target_data_mid, target_label_mid, lab_classes)
        sne_domain_mid = plot_domain_2D(source_data_mid, source_d_label_mid, target_data_mid, target_d_label_mid,
                                    dom_classes)

        sne = plot_label_2D(source_data_best, source_label_best, target_data_best, target_label_best, lab_classes)
        sne_domain = plot_domain_2D(source_data_best, source_domain_label, target_data_best, target_domain_label,
                                   dom_classes)

        logging.info(summary_confusion)
        if args.wandb:

            if args.vis_train:
                sne_ori_wandb = wandb.Image(
                    sne_ori, caption="Source and Target Labels")
                sne_domain_ori_wandb = wandb.Image(
                    sne_domain_ori, caption="Source and Target Labels")

                sne_mid_wandb = wandb.Image(
                    sne_mid, caption="Source and Target Labels")
                sne_domain_mid_wandb = wandb.Image(
                    sne_domain_mid, caption="Source and Target Labels")

                sne_wandb = wandb.Image(
                    sne, caption="Source and Target Labels")
                sne_domain_wandb = wandb.Image(
                    sne_domain, caption="Source and Target Labels")

                wandb.log({
                    'Source and Target Labels origin ': sne_ori_wandb,
                    'Source and Target Domains origin': sne_domain_ori_wandb,
                })

                wandb.log({
                    'Source and Target Labels mid ': sne_mid_wandb,
                    'Source and Target Domains mid': sne_domain_mid_wandb,
                })

                wandb.log({
                    'Source and Target Labels ': sne_wandb,
                    'Source and Target Domains ': sne_domain_wandb,
                })

            wandb.log({'summary_confusion':summary_confusion})
            matrix_plt_wandb = wandb.Image(matrix_plt, caption="Source and Target Labels")
            wandb.log({'Confusion Matrix': matrix_plt_wandb})

        writer.add_figure('Source and Target Labels ', sne)
        writer.add_figure('Source and Target Domains ', sne_domain)
        writer.add_figure('Confusion Matrix', matrix_plt)

        writer.close()



