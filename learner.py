from __future__ import division, print_function, absolute_import

import pdb
import copy

import torch
import torch.nn as nn
import numpy as np

class Learner(nn.Module):

    def __init__(self, image_size, bn_eps, bn_momentum, n_classes):
        super(Learner, self).__init__()
        self.model = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32, bn_eps, bn_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32, bn_eps, bn_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32, bn_eps, bn_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32, bn_eps, bn_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )])

        clr_in = image_size // 2**4
        self.model.append(nn.Linear(32 * clr_in * clr_in, n_classes))
        self.criterion = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        x = self.model[0](x)
        x = torch.reshape(x, [x.size(0), -1])
        outputs = self.model[1](x)
        return outputs

    @staticmethod
    def get_params(model):
        return torch.cat([p.data.view(-1) for p in model.parameters()], 0)

    @staticmethod
    def set_params(model, params):
        idx = 0
        for p in model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(params[idx: idx+plen].view_as(p))
            idx += plen

