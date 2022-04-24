from torch import nn
import torch
import copy
import numpy as np


def FedAvg(w, data_num):
    weight = np.array(data_num) / sum(data_num)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k]*weight[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]*weight[i]
    return w_avg
