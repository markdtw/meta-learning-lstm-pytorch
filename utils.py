from __future__ import division, print_function, absolute_import

import os
import pdb
import time
import logging

import torch
import numpy as np
import pandas as pd
import PIL.Image as PILI
import matplotlib.pyplot as plt


class GOATLogger:

    def __init__(self, mode, save_root, log_freq=100):
        self.mode = mode
        self.save_root = save_root
        self.log_freq = log_freq

        if self.mode == 'train':
            if not os.path.exists(self.save_root):
                os.mkdir(self.save_root)
            filename = os.path.join(self.save_root, 'console.log')
            logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s.%(msecs)03d - %(message)s',
                datefmt='%b-%d %H:%M:%S',
                filename=filename,
                filemode='w')
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter('%(message)s'))
            logging.getLogger('').addHandler(console)

            logging.info("Logger created at {}".format(filename))
        else:
            logging.basicConfig(level=logging.INFO,
                format='%(asctime)s.%(msecs)03d - %(message)s',
                datefmt='%b-%d %H:%M:%S')

    def batch_info(self, **kwargs):
        if kwargs['phase'] == 'train':
            if kwargs['eps'] % self.log_freq == 0 and kwargs['eps'] != 0:
                strout = '[{:5d}/{:5d}] loss: {:6.4f}, acc: {:6.3f}%'.format(\
                    kwargs['eps'], kwargs['eps_total'], kwargs['loss'], kwargs['acc'])
                self.loginfo(strout)
        else:
            strout = '[{:5d}] Eval ({:3d} episode) - loss: {:6.4f}, acc: {:6.3f}%'.format(\
                kwargs['eps'], kwargs['eps_total'], kwargs['loss'], kwargs['acc'])
            self.loginfo(strout)

    def save_stats(self):
        return
    def logdebug(self, strout):
        logging.debug(strout)
    def loginfo(self, strout):
        logging.info(strout)


def torch_tensor_to_pil(torch_tensor):
    """Simply convert a torch tensor t oa pillow savable image object
    Args:
        torch_tensor (torch.FloatTensor): of torch.Size([n, c, h, w])

    Return:
        array_pil (Pillow Image)
    """
    array_np = torch_tensor.cpu().numpy()
    array_np = array_np.transpose([0, 2, 3, 1]) # (n, h, w, c)
    array_np = (array_np * 255).astype(np.uint8)

    if array_np.shape[-1] == 3:
        array_pil = PILI.fromarray(array_np[0], mode='RGB')
    elif array_np.shape[-1] == 1:
        array_pil = PILI.fraomarray(array_np[0, :, :, 0], mode='P')

    return array_pil


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0].item() if len(topk) == 1 else [r.item() for r in res]


def save_ckpt(episode, metalearner, optim, save):
    if not os.path.exists(os.path.join(save, 'ckpts')):
        os.mkdir(os.path.join(save, 'ckpts'))

    torch.save({
        'episode': episode + 1,
        'metalearner': metalearner.state_dict(),
        'optim': optim.state_dict()
    }, os.path.join(save, 'ckpts', 'meta-learner-{}.pth.tar'.format(episode)))


def resume_ckpt(metalearner, optim, resume, device):
    ckpt = torch.load(resume, map_location=device)
    metalearner.load_state_dict(ckpt['metalearner'])
    optim.load_state_dict(ckpt['optim'])
    return metalearner, optim


def preprocess_grad_loss(x):
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)

    # preproc1
    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
    # preproc2
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x

    return torch.stack((x_proc1, x_proc2), 1)

