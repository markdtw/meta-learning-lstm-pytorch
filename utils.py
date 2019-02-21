from __future__ import division, print_function, absolute_import

import os
import pdb
import time
import logging

import torch
import numpy as np
import PIL.Image as PILI
import matplotlib.pyplot as plt


class GOATLogger:

    def __init__(self, mode, save_root, log_freq=100):
        self.mode = mode
        self.save_root = save_root
        self.log_freq = log_freq

        if self.mode == 'train':
            self.stats = {'train': {'loss': [], 'acc': []},
                          'eval': {'loss': [], 'acc': []}}
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
            self.stats = {'eval': {'loss': [], 'acc': []}}
            logging.basicConfig(level=logging.INFO,
                format='%(asctime)s.%(msecs)03d - %(message)s',
                datefmt='%b-%d %H:%M:%S')

    def batch_info(self, **kwargs):
        if kwargs['phase'] == 'train':
            self.stats['train']['loss'].append(kwargs['loss'])
            self.stats['train']['acc'].append(kwargs['acc'])

            if kwargs['eps'] % self.log_freq == 0 and kwargs['eps'] != 0:
                self.draw_stats()
                self.loginfo("[{:5d}/{:5d}] loss: {:6.4f}, acc: {:6.3f}%".format(\
                    kwargs['eps'], kwargs['totaleps'], kwargs['loss'], kwargs['acc']))

        elif kwargs['phase'] == 'eval':
            self.stats['eval']['loss'].append(kwargs['loss'])
            self.stats['eval']['acc'].append(kwargs['acc'])

        elif kwargs['phase'] == 'evaldone':
            loss_mean = np.mean(self.stats['eval']['loss'])
            loss_std = np.std(self.stats['eval']['loss'])
            acc_mean = np.mean(self.stats['eval']['acc'])
            acc_std = np.std(self.stats['eval']['acc'])
            self.loginfo("[{:5d}] Eval ({:3d} episode) - loss: {:6.4f} +- {:6.4f}, acc: {:6.3f}% +- {:5.3f}%".format(\
                kwargs['eps'], kwargs['totaleps'], loss_mean, loss_std, acc_mean, acc_std))

            self.stats['eval']['loss'] = []
            self.stats['eval']['acc'] = []

        else:
            raise ValueError("phase {} not supported".format(kwargs['phase']))

    def draw_stats(self):
        plt.style.use('seaborn-darkgrid')
        for item in ['loss', 'acc']:
            plt.plot(np.arange(len(self.stats['train'][item])), self.stats['train'][item])
            plt.xlabel('training episodes')
            plt.ylabel(item)
            plt.title('Training {}'.format(item))
            plt.savefig(os.path.join(self.save_root, '{}.png'.format(item)))
            plt.clf()

    def logdebug(self, strout):
        logging.debug(strout)
    def loginfo(self, strout):
        logging.info(strout)


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
        'episode': episode,
        'metalearner': metalearner.state_dict(),
        'optim': optim.state_dict()
    }, os.path.join(save, 'ckpts', 'meta-learner-{}.pth.tar'.format(episode)))


def resume_ckpt(metalearner, optim, resume, device):
    ckpt = torch.load(resume, map_location=device)
    last_episode = ckpt['episode']
    metalearner.load_state_dict(ckpt['metalearner'])
    optim.load_state_dict(ckpt['optim'])
    return last_episode, metalearner, optim


def preprocess_grad_loss(x):
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)

    # preproc1
    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
    # preproc2
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    return torch.stack((x_proc1, x_proc2), 1)

