from __future__ import division, print_function, absolute_import

import os
import pdb
import copy
import random
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from learner import Learner
from metalearner import MetaLearner
from dataloader import prepare_data
from utils import *


FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode', choices=['train', 'test'])
# Hyper-parameters
FLAGS.add_argument('--n-shot', type=int,
                   help="How many examples per class for training (k, n_support)")
FLAGS.add_argument('--n-eval', type=int,
                   help="How many examples per class for evaluation (n_query)")
FLAGS.add_argument('--n-class', type=int,
                   help="How many classes (N, n_way)")
FLAGS.add_argument('--input-size', type=int,
                   help="Input size for the first LSTM")
FLAGS.add_argument('--hidden-size', type=int,
                   help="Hidden size for the first LSTM")
FLAGS.add_argument('--lr', type=float,
                   help="Learning rate")
FLAGS.add_argument('--episode', type=int,
                   help="Episodes to train")
FLAGS.add_argument('--episode-val', type=int,
                   help="Episodes to eval")
FLAGS.add_argument('--epoch', type=int,
                   help="Epoch to train for an episode")
FLAGS.add_argument('--batch-size', type=int,
                   help="Batch size when training an episode")
FLAGS.add_argument('--image-size', type=int,
                   help="Resize image to this size")
FLAGS.add_argument('--grad-clip', type=float,
                   help="Clip gradients larger than this number")
FLAGS.add_argument('--bn-momentum', type=float,
                   help="Momentum parameter in BatchNorm2d")
FLAGS.add_argument('--bn-eps', type=float,
                   help="Eps parameter in BatchNorm2d")

# Paths
FLAGS.add_argument('--data', choices=['miniimagenet'],
                   help="Name of dataset")
FLAGS.add_argument('--data-root', type=str,
                   help="Location of data")
FLAGS.add_argument('--resume', type=str,
                   help="Location to pth.tar")
FLAGS.add_argument('--save', type=str, default='logs',
                   help="Location to logs and ckpts")
# Others
FLAGS.add_argument('--cpu', action='store_true',
                   help="Set this to use CPU, default use CUDA")
FLAGS.add_argument('--n-workers', type=int, default=4,
                   help="How many processes for preprocessing")
FLAGS.add_argument('--pin-mem', type=bool, default=False,
                   help="DataLoader pin_memory")
FLAGS.add_argument('--log-freq', type=int, default=100,
                   help="Logging frequency")
FLAGS.add_argument('--val-freq', type=int, default=1000,
                   help="Validation frequency")
FLAGS.add_argument('--seed', type=int, default=420,
                   help="Random seed")


def train_learner(flat_learner, dummy_learner, metalearner, train_input, train_target, args):
    hs = [[None, [None, None, flat_learner.unsqueeze(1)]]]
    for _ in range(args.epoch):
        for i in range(0, len(train_input), args.batch_size):
            x = train_input[i:i+args.batch_size]
            y = train_target[i:i+args.batch_size]

            # get the loss/grad from dummy learner
            dummy_learner.set_params(1, flat_learner)
            output = dummy_learner(x)
            loss = dummy_learner.criterion(output, y)
            dummy_learner.zero_grad()
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) / args.batch_size for p in dummy_learner.parameters()], 0)

            # preprocess grad & loss and metalearner forward
            grad_prep = preprocess_grad_loss(grad)  # [n_learner_params, 2]
            loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2]
            metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
            flat_learner, h = metalearner(metalearner_input, hs[-1])
            hs.append(h)

            #print("loss: {}".format(loss))

    return flat_learner


def meta_test(eps, val_loader, learner, dummy_learner, metalearner, args, logger):
    for subeps, (d_episode_x, d_episode_y) in enumerate(tqdm(val_loader, ascii=True)):
        train_input = d_episode_x[:, :args.n_shot].reshape(-1, *d_episode_x.shape[-3:]).to(args.dev) # [n_class * n_shot, :]
        train_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev) # [n_class * n_shot]
        test_input = d_episode_x[:, args.n_shot:].reshape(-1, *d_episode_x.shape[-3:]).to(args.dev) # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.dev) # [n_class * n_eval]

        # Train learner with metalearner
        metalearner.eval()
        learner.set_params(mode=0)
        flat_learner = train_learner(learner.get_flat_params(), dummy_learner, metalearner, train_input, train_target, args)

        # Train meta-learner with validation loss
        metalearner.train()
        learner.set_params(2, flat_learner)
        output = learner(test_input)
        loss = learner.criterion(output, test_target)
        acc = accuracy(output, test_target)
 
        logger.batch_info(loss=loss.item(), acc=acc, phase='eval')

    logger.batch_info(eps=eps, totaleps=args.episode_val, phase='evaldone')


def main():

    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    logger = GOATLogger(args.mode, args.save, args.log_freq)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cpu:
        args.dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args.dev = torch.device('cuda')

    # Get data
    train_loader, val_loader, test_loader = prepare_data(args)

    # Set up learner, meta-learner
    learner = Learner(args.image_size, args.bn_eps, args.bn_momentum, args.n_class).to(args.dev)
    dummy_learner = copy.deepcopy(learner)
    metalearner = MetaLearner(args.input_size, args.hidden_size).to(args.dev)

    # Set up loss, optimizer, learning rate scheduler
    optim = torch.optim.Adam(metalearner.parameters(), args.lr)

    if args.resume:
        logger.loginfo("Initialized from: {}".format(args.resume))
        last_eps, metalearner, optim = resume_ckpt(metalearner, optim, args.resume, args.dev)

    if args.mode == 'test':
        meta_test(last_eps, val_loader, learner, dummy_learner, metalearner, args, logger)
        return

    logger.loginfo("Start training")
    # Meta-training
    for eps, (d_episode_x, d_episode_y) in enumerate(train_loader):
        # d_episode_x.shape = [n_class, n_shot + n_eval, c, h, w]
        # d_episode_y.shape = [n_class, n_shot + n_eval] --> NEVER USED
        train_input = d_episode_x[:, :args.n_shot].reshape(-1, *d_episode_x.shape[-3:]).to(args.dev) # [n_class * n_shot, :]
        train_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev) # [n_class * n_shot]
        test_input = d_episode_x[:, args.n_shot:].reshape(-1, *d_episode_x.shape[-3:]).to(args.dev) # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.dev) # [n_class * n_eval]

        # Train learner with metalearner
        metalearner.eval()
        learner.set_params(mode=0)
        flat_learner = train_learner(learner.get_flat_params(), dummy_learner, metalearner, train_input, train_target, args)

        # Train meta-learner with validation loss
        metalearner.train()
        learner.set_params(2, flat_learner)
        output = learner(test_input)
        loss = learner.criterion(output, test_target)
        acc = accuracy(output, test_target)
        
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(metalearner.parameters(), args.grad_clip)
        optim.step()

        logger.batch_info(eps=eps, totaleps=args.episode, loss=loss.item(), acc=acc, phase='train')

        # Meta-validation
        if eps % args.val_freq == 0 and eps != 0:
            save_ckpt(eps, metalearner, optim, args.save)
            meta_test(eps, val_loader, learner, dummy_learner, metalearner, args, logger)

    logger.loginfo("Done")


if __name__ == '__main__':
    main()
