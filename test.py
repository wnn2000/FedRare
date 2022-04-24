import os
import sys
import copy
import logging
import random
import time
import argparse
from tqdm import tqdm
import numpy as np
import csv

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
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
                        default='inter', help='experiment name')
    parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
    parser.add_argument('--tta', type=int,  default=16, help='tta times')
    parser.add_argument('--mode', type=str,
                        default='test', help='test or valid')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic trainin g')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--dim', type=int, default=256, help='dimension')
    args = parser.parse_args()
    return args


def get_test_result(pred, num_classes, args):
    val_sample_repeat_num = args.tta
    pred = pred.reshape(-1, val_sample_repeat_num, num_classes)
    pred = np.transpose(pred, (0, 2, 1))
    predictions = np.mean(pred, 2)

    return predictions


def test_model(data_loader, model, args):
    model.eval()
    assert data_loader.dataset.tta_num == args.tta
    num_classes = data_loader.dataset.get_num_classes()
    func = torch.nn.Softmax(dim=1)

    all_preds = np.zeros([data_loader.dataset.get_num_images(), num_classes])
    ii = 0
    print("\n-------  Start testing  -------")
    pbar = tqdm(total=len(data_loader))
    with torch.no_grad():
        for i, (image, label) in enumerate(data_loader):
            image = image.cuda()
            if model.project == False:
                output = model(image)
            else:
                _, output = model(image)
            now_pred = func(output)
            new_pred = get_test_result(
                now_pred.cpu().numpy(), num_classes, args)
            all_preds[ii:ii + new_pred.shape[0], :] = new_pred
            ii += new_pred.shape[0]
            pbar.update(1)

    pbar.close()
    image_id_list = data_loader.dataset.get_image_id_list()

    # save the predictions to csv file
    csv_file = 'pridiction.csv'
    csv_file = os.path.join('outputs', args.exp, csv_file)
    csv_fp = open(csv_file, 'w')
    csv_writer = csv.writer(csv_fp)
    table_head = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    csv_writer.writerow(table_head)
    for i in range(len(image_id_list)):
        # change predictions to binary value([0, 1])
        pred = all_preds[i]
        pred = pred.tolist()
        # for every row, change the max prediction to 0.51 if the max prediction is lower than 0.5
        row_new = pred
        if max(pred) <= 0.5:
            row_new[pred.index(max(pred))] = 0.51
        row_new = [image_id_list[i]] + row_new
        csv_writer.writerow(row_new)

    # close the csv file
    csv_fp.close()


if __name__ == "__main__":
    args = args_parser()
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # ------------------------------ load model ------------------------------
    model = Network(mode='train', network='Efficient_b0',
                    num_classes=7, project=False, args=args).cuda()
    model_dir = os.path.join('outputs', args.exp,
                             'models', 'best_model.pth')
    model.load_state_dict(torch.load(model_dir))
    print("have loaded the best model from {}".format(model_dir))

    # ------------------------------ dataset and dataloader ------------------------------
    test_dataset = BaseSet(root=args.root, mode=args.mode, tta=args.tta)
    test_batch_size = 128
    val_test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=32,
    )

    # ------------------------------ test or valid ------------------------------
    if args.mode == 'test':
        test_model(val_test_loader, model, args)
    elif args.mode == 'valid':
        start = time.time()
        acc, conf_matrix = compute_bacc(
            model, val_test_loader, get_confusion_matrix=True, args=args)
        end = time.time()
        print(end-start)
        recall = []
        for i in range(7):
            recall.append(conf_matrix[i, i]/np.sum(conf_matrix[i]))
        print(conf_matrix)
        print(f'bacc: {acc}')
