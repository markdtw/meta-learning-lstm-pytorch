from __future__ import division, print_function, absolute_import

import pdb
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

class Learner(nn.Module):

    def __init__(self, image_size, bn_eps, bn_momentum, n_classes):
        super(Learner, self).__init__()
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, padding=1)),
            ('norm1', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu1', nn.ReLU(inplace=False)),
            ('pool1', nn.MaxPool2d(2)),

            ('conv2', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm2', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu2', nn.ReLU(inplace=False)),
            ('pool2', nn.MaxPool2d(2)),

            ('conv3', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm3', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu3', nn.ReLU(inplace=False)),
            ('pool3', nn.MaxPool2d(2)),

            ('conv4', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm4', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu4', nn.ReLU(inplace=False)),
            ('pool4', nn.MaxPool2d(2))]))
        })

        clr_in = image_size // 2**4
        self.model.update({'cls': nn.Linear(32 * clr_in * clr_in, n_classes)})
        self.criterion = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        x = self.model.features(x)
        x = torch.reshape(x, [x.size(0), -1])
        outputs = self.model.cls(x)
        return outputs

    def get_flat_params(self):
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def set_params(self, mode, flat_params=None):
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                blen = m._parameters['bias'].view(-1).size(0)
                if mode == 0:   # set parameters as tensors
                    m._parameters['weight'] = m._parameters['weight'].data
                    m._parameters['bias'] = m._parameters['bias'].data
                elif mode == 1: # simply copy the data from tensor to parameters
                    m._parameters['weight'].data.copy_(flat_params[idx: idx+wlen].view_as(m._parameters['weight']))
                    m._parameters['bias'].data.copy_(flat_params[idx+wlen: idx+wlen+blen].view_as(m._parameters['bias']))
                elif mode == 2: # assign the tensors (retain grad_fn and the backward properties)
                    m._parameters['weight'] = flat_params[idx: idx+wlen].view_as(m._parameters['weight'])
                    m._parameters['bias'] = flat_params[idx+wlen: idx+wlen+blen].view_as(m._parameters['bias'])
                else:
                    raise ValueError("mode = {} not supported".format(mode))
                idx += wlen + blen

