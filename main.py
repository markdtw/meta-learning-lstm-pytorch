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
FLAGS.add_argument('--seed', type=int,
                   help="Random seed")


def meta_test(eps, eval_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger):
    for subeps, (episode_x, episode_y) in enumerate(tqdm(eval_loader, ascii=True)):
        train_input = episode_x[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_shot, :]
        train_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev) # [n_class * n_shot]
        test_input = episode_x[:, args.n_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.dev) # [n_class * n_eval]

        # Train learner with metalearner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        learner_w_grad.train()
        learner_wo_grad.eval()
        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, args)

        learner_wo_grad.transfer_params(learner_w_grad, cI)
        output = learner_wo_grad(test_input)
        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)
 
        logger.batch_info(loss=loss.item(), acc=acc, phase='eval')

    return logger.batch_info(eps=eps, totaleps=args.episode_val, phase='evaldone')


def train_learner(learner_w_grad, metalearner, train_input, train_target, args):
    cI = metalearner.metalstm.cI.data
    hs = [None]
    for _ in range(args.epoch):
        for i in range(0, len(train_input), args.batch_size):
            x = train_input[i:i+args.batch_size]
            y = train_target[i:i+args.batch_size]

            # get the loss/grad
            learner_w_grad.copy_flat_params(cI)
            output = learner_w_grad(x)
            loss = learner_w_grad.criterion(output, y)
            acc = accuracy(output, y)
            learner_w_grad.zero_grad()
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) / args.batch_size for p in learner_w_grad.parameters()], 0)

            # preprocess grad & loss and metalearner forward
            grad_prep = preprocess_grad_loss(grad)  # [n_learner_params, 2]
            loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2]
            metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
            cI, h = metalearner(metalearner_input, hs[-1])
            hs.append(h)

            #print("training loss: {:8.6f} acc: {:6.3f}, mean grad: {:8.6f}".format(loss, acc, torch.mean(grad)))

    return cI


def main():

    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    if args.seed is None:
        args.seed = random.randint(0, 1e3)
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

    logger = GOATLogger(args)

    # Get data
    train_loader, val_loader, test_loader = prepare_data(args)
    
    # Set up learner, meta-learner
    learner_w_grad = Learner(args.image_size, args.bn_eps, args.bn_momentum, args.n_class).to(args.dev)
    learner_wo_grad = copy.deepcopy(learner_w_grad)
    metalearner = MetaLearner(args.input_size, args.hidden_size, learner_w_grad.get_flat_params().size(0)).to(args.dev)
    metalearner.metalstm.init_cI(learner_w_grad.get_flat_params())

    # Set up loss, optimizer, learning rate scheduler
    optim = torch.optim.Adam(metalearner.parameters(), args.lr)

    if args.resume:
        logger.loginfo("Initialized from: {}".format(args.resume))
        last_eps, metalearner, optim = resume_ckpt(metalearner, optim, args.resume, args.dev)

    if args.mode == 'test':
        _ = meta_test(last_eps, test_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger)
        return

    best_acc = 0.0
    logger.loginfo("Start training")
    # Meta-training
    for eps, (episode_x, episode_y) in enumerate(train_loader):
        # episode_x.shape = [n_class, n_shot + n_eval, c, h, w]
        # episode_y.shape = [n_class, n_shot + n_eval] --> NEVER USED
        train_input = episode_x[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_shot, :]
        train_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev) # [n_class * n_shot]
        test_input = episode_x[:, args.n_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.dev) # [n_class * n_eval]

        # Train learner with metalearner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        learner_w_grad.train()
        learner_wo_grad.train()
        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, args)

        # Train meta-learner with validation loss
        learner_wo_grad.transfer_params(learner_w_grad, cI)
        output = learner_wo_grad(test_input)
        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)
        
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(metalearner.parameters(), args.grad_clip)
        optim.step()

        logger.batch_info(eps=eps, totaleps=args.episode, loss=loss.item(), acc=acc, phase='train')

        # Meta-validation
        if eps % args.val_freq == 0 and eps != 0:
            save_ckpt(eps, metalearner, optim, args.save)
            acc = meta_test(eps, val_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger)
            if acc > best_acc:
                best_acc = acc
                logger.loginfo("* Best accuracy so far *\n")

    logger.loginfo("Done")


if __name__ == '__main__':
    main()
