import torch
import torch.nn as nn
import networks.all_models as all_models


class Network(nn.Module):
    def __init__(self, mode='train', network='Efficient_b0', num_classes=7, project=False, args=None):
        super(Network, self).__init__()
        pretrain = True
        self.network = network
        self.num_classes = num_classes
        self.project = project
        model = all_models.get_model(self.network, pretrain)
        self.model, self.last_layer = all_models.modify_last_layer(
            self.network, model, self.num_classes)
        self.projector = nn.Sequential(
            nn.Linear(self.model._fc.in_features, 1024),
            nn.Linear(1024, args.dim)
        )

    def forward(self, x, tsne=False, **kwargs):
        # Convolution layers
        x = self.model.extract_features(x)
        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        y = self.model._dropout(x)
        y = self.model._fc(y)
        if self.project:
            feature = self.projector(x)
            return feature, y
        else:
            return y
