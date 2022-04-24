import logging
from cv2 import mean
import numpy as np
import random
import math
import copy
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.loss import CrossEntropyLoss
from feature_pool import get_feature_pool_with_metric
import copy
from tqdm import tqdm
from losses import SupConLoss, SupConLoss_inter


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    def get_num_class_list(self):
        class_sum = np.array([0] * 7)
        for idx in self.idxs:
            _, label = self.dataset[idx]
            class_sum[label] += 1
        return class_sum.tolist()


class LocalUpdate(object):
    def __init__(self, args, id, dataset, idxs):
        self.id = id
        self.local_dataset = DatasetSplit(dataset, idxs)
        self.class_num_list = self.local_dataset.get_num_class_list()
        logging.info(
            f'client{id} each class num: {self.class_num_list}')
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        self.epoch = 0
        self.iter_num = 0
        self.lr = args.base_lr

        if self.class_num_list[-1] != 0:
            loss_weight = 1.0 / \
                np.array(self.class_num_list).astype(np.float64)
            loss_weight = [math.pow(num, 1.2) for num in loss_weight]
            loss_weight = loss_weight / np.sum(loss_weight) * len(loss_weight)
        else:
            loss_weight = 1.0 / \
                np.array(self.class_num_list[:5]).astype(np.float64)
            loss_weight = [math.pow(num, 1.2) for num in loss_weight]
            loss_weight = loss_weight / np.sum(loss_weight) * len(loss_weight)
            loss_weight = np.append(loss_weight, np.array([1000., 1000.]))
        self.weight = torch.FloatTensor(loss_weight).cuda()
        logging.info(f'loss weight: {self.weight}')

        self.feature_pool = torch.zeros((7, 256)).cuda()
        self.queue_5 = []
        self.queue_6 = []

    def enqueue(self, features, labels):
        features_5 = features[labels == 5]
        for i in range(features_5.shape[0]):
            self.queue_5.append(features_5[i].unsqueeze(0).detach())

        if len(self.queue_5) > 10:
            self.queue_5 = self.queue_5[-10:]

        features_6 = features[labels == 6]
        for i in range(features_6.shape[0]):
            self.queue_6.append(features_6[i].unsqueeze(0).detach())

        if len(self.queue_6) > 10:
            self.queue_6 = self.queue_6[-10:]

    def train_FedRare(self, args, net, op_dict, writer, feature_avg):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        self.optimizer.load_state_dict(op_dict)
        self.optimizer.param_groups[0]['lr'] = self.lr
        print(f'epoch: {self.epoch}, lr: {self.lr}')
        supconloss = SupConLoss()
        supconloss_inter = SupConLoss_inter()

        # create lists to store
        feature_list = []
        label_list = []
        pred_list = []

        # train and update
        epoch_loss = []
        iter_max = len(self.ldr_train)
        for epoch in range(args.local_ep):
            batch_loss = []
            for (images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                if (labels == 5).float().sum() == 1 and len(self.queue_5) > 0:
                    images = torch.cat(
                        [images, self.queue_5[0].cuda()], dim=0)
                    labels = torch.cat(
                        [labels, torch.tensor([5]).cuda()], dim=0)
                    self.queue_5 = self.queue_5[1:]

                if (labels == 6).float().sum() == 1 and len(self.queue_6) > 0:
                    images = torch.cat(
                        [images, self.queue_6[0].cuda()], dim=0)
                    labels = torch.cat(
                        [labels, torch.tensor([6]).cuda()], dim=0)
                    self.queue_6 = self.queue_6[1:]

                features, outputs = net(images)
                features = F.normalize(features, dim=1)

                loss_ce = F.cross_entropy(outputs, labels, weight=self.weight)
                loss_intra = supconloss(
                    features.unsqueeze(1), labels) if labels.shape[0] > 1 else 0.

                features_ = torch.cat([features, feature_avg], dim=0)
                labels_ = torch.cat(
                    [labels, torch.tensor(list(range(7))).cuda()], dim=0)
                loss_inter = supconloss_inter(features_.unsqueeze(
                    1), labels_, mean=False, local_size=features.shape[0])[:, :features.shape[0]].mean()

                w = 2. if self.epoch > 10 else 0.
                loss = loss_ce + loss_intra + w*loss_inter

                self.enqueue(images, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/loss_ce', loss_ce, self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/loss_intra', loss_intra, self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/loss_inter', loss_inter, self.iter_num)

                with torch.no_grad():
                    feature_list.append(features.detach())
                    label_list.append(labels.detach())
                    pred_list.append(torch.softmax(outputs, dim=1).detach())

                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        feature_local = torch.cat(feature_list)
        label_local = torch.cat(label_list)
        pred_local = torch.cat(pred_list)

        assert feature_local.shape[0] == label_local.shape[0] and feature_local.shape[0] == pred_local.shape[0]

        self.feature_pool, self.confidence_pool = get_feature_pool_with_metric(
            feature_local, label_local, pred_local, args)
        self.feature_pool = F.normalize(self.feature_pool, dim=1)

        return net, net.state_dict(), np.array(epoch_loss).mean(), copy.deepcopy(self.optimizer.state_dict())
