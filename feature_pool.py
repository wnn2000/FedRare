import logging
import torch
import numpy as np
import torch.nn.functional as F
NUM_CLASSES = 7


def get_feature_pool_with_metric(features, labels, pred, args):
    feature_pool = []
    confidence_pool = []
    metric_value = (torch.max(pred, dim=1)[0].unsqueeze(1)-pred).sum(1)/6
    for i in range(NUM_CLASSES):
        mask = ((labels == i).float()
                * (torch.argmax(pred, dim=1) == labels).float())
        feature_avg = torch.sum(features * mask.unsqueeze(-1), dim=0) / \
            (torch.sum(mask.unsqueeze(-1))+1e-9)
        confidence_avg = torch.sum(metric_value * mask, dim=0) / \
            (torch.sum(mask)+1e-9)
        feature_pool.append(feature_avg)
        confidence_pool.append(confidence_avg)

    return torch.stack(feature_pool), torch.stack(confidence_pool)