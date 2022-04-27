import os
import sys
import copy
import logging
import random
import argparse
from tqdm import tqdm
import numpy as np
import math

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from FedAvg import FedAvg
from dataset.baseset import *
from val import compute_bacc
from networks.networks import Network
from local_training import LocalUpdate


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/home/wnn/dataset/isic2018classification/', help='root path')
    parser.add_argument('--exp', type=str,
                        default='FedRare', help='experiment name')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=5e-4,
                        help='base learning rate')
    parser.add_argument('--num_users', type=int,  default=10,
                        help='number of users')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
    parser.add_argument('--local_ep', type=int,
                        default=1, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=70, help='rounds')
    parser.add_argument('--tta', type=int,  default=16, help='tta times')
    parser.add_argument('--T', type=float, default=1, help='temperature')
    parser.add_argument('--dim', type=int, default=256, help='dimension')
    args = parser.parse_args()
    return args


def split(dataset, num_users):
    # assert num_users >= 4
    class_dict = dataset._get_class_dict()
    class_num = dataset.get_num_classes()
    dict_users = {}
    index_set = []
    for i in range(class_num):
        random.shuffle(class_dict[i])
        index_set.append(np.array_split(class_dict[i], num_users))
    for id in range(num_users):
        dict_users[id] = list(index_set[0][id])
        if id < 4:
            for i in range(1, class_num):
                dict_users[id] += list(index_set[i][id])
        else:
            for i in range(1, 5):
                dict_users[id] += list(index_set[i][id])
    return dict_users


if __name__ == '__main__':
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # ------------------------------ output files ------------------------------
    outputs_dir = 'outputs'
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    exp_dir = os.path.join(outputs_dir, args.exp)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(tensorboard_dir)

    # ------------------------------ dataset and dataloader ------------------------------
    train_dataset = BaseSet(root=args.root, mode='train')
    logging.info(np.array(train_dataset.get_num_class_list()))
    val_dataset = BaseSet(root=args.root, mode='valid', tta=args.tta)
    logging.info(np.array(val_dataset.get_num_class_list())/args.tta)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # ------------------------------ global and local settings ------------------------------
    net_glob = Network(
        mode='train', network='Efficient_b0', num_classes=7, project=True, args=args).cuda()
    net_glob.train()
    w_glob = net_glob.state_dict()
    w_locals = []
    trainer_locals = []
    net_locals = []
    optim_locals = []
    local_data_num = []
    user_id = list(range(args.num_users))
    dict_users = split(train_dataset, args.num_users)
    logging.info(dict_users)

    for id in user_id:
        trainer_locals.append(LocalUpdate(
            args, id, train_dataset, dict_users[id]))
        w_locals.append(copy.deepcopy(w_glob))
        net_locals.append(copy.deepcopy(net_glob).cuda())
        optimizer = torch.optim.Adam(net_locals[id].parameters(), lr=args.base_lr,
                                     betas=(0.9, 0.999), weight_decay=1e-4)
        optim_locals.append(copy.deepcopy(optimizer.state_dict()))
        local_data_num.append(len(dict_users[id]))

    # ------------------------------ begin training ------------------------------
    best_performance = 0.
    features_bank = torch.rand((args.num_users, 7, args.dim))
    confidence_bank = torch.rand((args.num_users, 7))
    lr = args.base_lr
    for com_round in range(args.rounds):
        logging.info(f'\n---------------round: {com_round}---------------')
        loss_locals = []
        # lr schedule
        if com_round >= 30 and com_round % 10 == 0:
            lr = lr * 0.1
        writer.add_scalar('train/lr', lr, com_round)

        # features_avg
        feature_avg = features_bank[:4]
        confidence = confidence_bank[:4]
        weight = torch.softmax(confidence/args.T, dim=0).unsqueeze(-1)
        logging.info(weight.float())
        feature_avg = feature_avg * weight
        feature_avg = feature_avg.sum(0).detach().cuda()
        feature_avg = F.normalize(feature_avg, dim=1)

        # local train and val
        for idx in user_id:
            trainer_locals[idx].lr = lr
            local = trainer_locals[idx]
            optimizer = optim_locals[idx]
            net_local, w, loss, op = local.train_FedRare(
                args, net_locals[idx], optimizer, writer, copy.deepcopy(feature_avg))
            w_locals[idx] = copy.deepcopy(w)
            optim_locals[idx] = copy.deepcopy(op)
            loss_locals.append(copy.deepcopy(loss))

        # upload and download
        with torch.no_grad():
            w_glob = FedAvg(w_locals, local_data_num)
        net_glob.load_state_dict(w_glob)
        for i in user_id:
            net_locals[i].load_state_dict(w_glob)
            features_bank[i] = trainer_locals[i].feature_pool
            confidence_bank[i] = trainer_locals[i].confidence_pool

        # global val
        net_glob = net_glob.cuda()
        bacc_g, conf_matrix = compute_bacc(
            net_glob, val_loader, get_confusion_matrix=True, args=args)
        writer.add_scalar(
            f'glob/bacc_val', bacc_g, com_round)
        logging.info('global conf_matrix')
        logging.info(conf_matrix)
        net_glob = net_glob.cpu()

        # save model
        if bacc_g > best_performance:
            best_performance = bacc_g
            torch.save(net_glob.state_dict(),  models_dir +
                       f'/best_model_{com_round}_{best_performance}.pth')
            torch.save(net_glob.state_dict(),  models_dir+'/best_model.pth')
        logging.info(f'best bacc: {best_performance}, now bacc: {bacc_g}')

    writer.close()
