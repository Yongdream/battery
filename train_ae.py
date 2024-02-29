import argparse
import os
from datetime import datetime
from utlis.logger import setlogger
import logging
from utlis.train_utils_ae import TrainUtilsDA
from utlis.train_utils_baseDG import TrainUtilsDG
import torch
import warnings
import wandb

os.environ['WANDB_SILENT'] = "true"
wandb.login()
wandb.init(project="Battery", entity='yang7hiv1')  # 初始化

warnings.filterwarnings('ignore')

torch.autograd.set_detect_anomaly(True)

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # model and data parameters
    parser.add_argument('--method', type=str, default='DA', choices=['DG', 'DA'], help='the name of the method')
    parser.add_argument('--model_name', type=str, default='VareFea', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='Battery', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='../processed', help='the directory of the data')

    parser.add_argument('--transfer_task', type=list, default=[[1], [2]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')

    # adabn parameters
    parser.add_argument('--adabn', type=bool, default=False, help='whether using adabn')            # Adabn 批量归一化
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
    parser.add_argument('--bottleneck_num', type=int, default=128, help='whether using the bottleneck layer')
    parser.add_argument('--last_batch', type=bool, default=False, help='whether using the last batch')

    # Variational Autoencoder parameters
    parser.add_argument('--number_of_features', type=int, default=16, help='')
    parser.add_argument('--sequence_length', type=int, default=256, help='')
    parser.add_argument('--hidden_layer_depth', type=int, default=5, help='')
    parser.add_argument('--latent_length', type=int, default=20, help='')
    # RNN parameters
    parser.add_argument('--block', type=str, default='GRU', help='')
    parser.add_argument('--rnn_hidden_size', type=int, default=32, help='')

    # Distance_metric parameters
    parser.add_argument('--distance_metric', type=bool, default=True, help='whether use distance metric')
    parser.add_argument('--distance_loss', type=str, choices=['MK-MMD', 'JMMD', 'CORAL'], default='JMMD', help='which distance loss you use')
    parser.add_argument('--trade_off_distance', type=str, default='Step', help='')
    parser.add_argument('--lam_distance', type=float, default=1.2, help='this is used for Cons')

    # Domain_adversarial parameters
    parser.add_argument('--domain_adversarial', type=bool, default=False, help='whether use domain_adversarial')
    parser.add_argument('--adversarial_loss', type=str, choices=['DA', 'CDA', 'CDA+E'], default='CDA', help='which adversarial loss you use')
    parser.add_argument('--hidden_size', type=int, default=1024, help='whether using the last batch')
    parser.add_argument('--trade_off_adversarial', type=str, default='Step', help='')
    parser.add_argument('--lam_adversarial', type=float, default=2, help='this is used for Cons')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.8, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='80, 95, 105', help='the learning rate decay for step and stepLR')

    parser.add_argument('--criterion', type=str, choices=['Entropy', 'CeLoss'], default='CeLoss', help='')

    # save, load and display information
    parser.add_argument('--middle_epoch', type=int, default=70, help='middle epoch')
    parser.add_argument('--middle_vae_epoch', type=int, default=256, help='middle epoch')
    parser.add_argument('--max_epoch', type=int, default=128, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=600, help='the interval of log training information')

    args_s = parser.parse_args()
    return args_s


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')

    if isinstance(args.transfer_task[0], str):
        str_list = eval("".join(args.transfer_task))
    else:
        str_list = args.transfer_task

    condition = ''
    for i in range(len(str_list[0])):
        if i == 0:
            condition = 's'+'_'+str(str_list[0][i])
        else:
            condition = condition+'_'+str(str_list[0][i])

    condition = condition + '_t' + str(str_list[1][0])
    save_dir = os.path.join(args.checkpoint_dir, args.method, condition, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    if args.method == 'DG':
        trainer = TrainUtilsDG(args, save_dir)
    elif args.method == 'DA':
        trainer = TrainUtilsDA(args, save_dir)
    else:
        raise 'The method does not exist.'

    trainer.setup()
    trainer.train(cond=condition)
