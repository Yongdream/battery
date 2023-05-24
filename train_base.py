import logging
from utlis.train_utils_base import TrainUtils
import os
import torch
from datetime import datetime
from matplotlib import pyplot as plt
from utlis import utils
from utlis.logger import setlogger
from torch.utils.tensorboard import SummaryWriter
import argparse

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # model and data parameters
    parser.add_argument('--model_name', type=str, default='netFeatures', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='Battery', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='E:/Galaxy/yang7hi_battery/processed', help='the directory of the data')

    parser.add_argument('--transfer_task', type=list, default=[[1], [2]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')

    # adabn parameters
    parser.add_argument('--adabn', type=bool, default=True, help='whether using adabn')            # Adabn 批量归一化
    parser.add_argument('--eval_all', type=bool, default=False, help='whether using all samples to update the results')
    parser.add_argument('--adabn_epochs', type=int, default=3, help='the number of training process')

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=128, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    parser.add_argument('--patience', type=int, default=50, help='Early Stopping')
    parser.add_argument('--bottleneck', type=bool, default=True, help='whether using the bottleneck layer')
    parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')
    parser.add_argument('--last_batch', type=bool, default=False, help='whether using the last batch')

    #
    parser.add_argument('--distance_metric', type=bool, default=True, help='whether use distance metric')
    parser.add_argument('--distance_loss', type=str, choices=['MK-MMD', 'JMMD', 'CORAL'], default='MK-MMD', help='which distance loss you use')
    parser.add_argument('--trade_off_distance', type=str, default='Step', help='')
    parser.add_argument('--lam_distance', type=float, default=1, help='this is used for Cons')
    #
    parser.add_argument('--domain_adversarial', type=bool, default=True, help='whether use domain_adversarial')
    parser.add_argument('--adversarial_loss', type=str, choices=['DA', 'CDA', 'CDA+E'], default='CDA+E', help='which adversarial loss you use')
    parser.add_argument('--hidden_size', type=int, default=1024, help='whether using the last batch')
    parser.add_argument('--trade_off_adversarial', type=str, default='Step', help='')
    parser.add_argument('--lam_adversarial', type=float, default=1, help='this is used for Cons')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-2, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='exp', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.98, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='150, 250', help='the learning rate decay for step and stepLR')

    parser.add_argument('--criterion', type=str, choices=['Entropy', 'CeLoss'], default='CeLoss', help='')

    # save, load and display information
    parser.add_argument('--middle_epoch', type=int, default=20, help='max number of epoch')
    parser.add_argument('--max_epoch', type=int, default=128, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=600, help='the interval of log training information')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = TrainUtils(args, save_dir)
    trainer.setup()
    trainer.train()

    # op_mode = args.data_name
    # root = os.path.join('processed', op_mode)
    #
    # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    # print("using {} device.".format(device))
    #
    # # # Set the deletion rates for each subfolder
    # # subfolders = [f.name for f in os.scandir(root) if f.is_dir()]
    # # deletion_rates = [0.75, 0.75, 0.75, 0, 0.75]
    # # delete_files(root, subfolders, deletion_rates)
    #
    # train_wintime_path, train_wintime_label, val_wintime_path, val_wintime_label = read_split_data(root)
    # train_data_set = MyDataSet(timeWin_path=train_wintime_path,
    #                            timeWin_class=train_wintime_label)
    #
    # val_data_set = MyDataSet(timeWin_path=val_wintime_path,
    #                          timeWin_class=val_wintime_label)
    # val_num = len(val_data_set)
    #
    # nw = args.num_workers
    # batch_size = args.batch_size
    # num_epochs = args.max_epoch
    # lr = args.lr
    # model = Network().to(device)
    # # criterion = nn.CrossEntropyLoss()
    # criterion = CELoss(label_smooth=0.05, class_num=5)
    # optimizer = optim.Adam(model.parameters(), lr)
    # lr_scheduler = ExponentialLR(optimizer, gamma=0.95)   # 定义学习率调度程序
    # print('Model build successfully')
    # summary(model, input_size=(batch_size, 225, 20))
    #
    # model_parameters = {'num_epochs': num_epochs, 'learning_rate': lr, 'batch_size': batch_size, 'model': model,
    #                     'criterion': criterion, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    #
    # train_loader = DataLoader(train_data_set,
    #                           batch_size=batch_size,
    #                           shuffle=True,
    #                           pin_memory=True,
    #                           num_workers=nw,
    #                           collate_fn=train_data_set.collate_fn,
    #                           drop_last=True)
    #
    # val_loader = DataLoader(val_data_set,
    #                         batch_size=batch_size,
    #                         shuffle=True,
    #                         pin_memory=True,
    #                         num_workers=nw,
    #                         collate_fn=val_data_set.collate_fn,
    #                         drop_last=True)
    #
    # logger = ResultLogger(op_mode)
    # best_accuracy, summary_val = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    # logger.log_best_result(best_accuracy, model_parameters, summary_val)
