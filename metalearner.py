from __future__ import division, print_function, absolute_import

import pdb
import torch
import torch.nn as nn
import numpy as np

from learner import Learner


class MetaLSTM(nn.Module):
    """C_t = f_t * C_{t-1} + i_t * \tilde{C_t}"""
    def __init__(self, n_inputs, n_params, batch_first=False):
        super(MetaLSTM, self).__init__()
        """Args:
            n_inputs (int): cell input size, default = 20
            n_params (int): number of learner's parameters
        """
        self.n_inputs = n_inputs
        self.n_params = n_params
        self.batch_first = batch_first
        self.WF = nn.Parameter(torch.Tensor(n_inputs + 2, 1))
        self.WI = nn.Parameter(torch.Tensor(n_inputs + 2, 1))
        self.cI = nn.Parameter(torch.Tensor(n_params, 1))
        self.bI = nn.Parameter(torch.Tensor(1, 1))
        self.bF = nn.Parameter(torch.Tensor(1, 1))

        self.reset_parameters()

    def reset_parameters(self, flat_learner=None):
        for weight in self.parameters():
            nn.init.constant_(weight, 0.0)

        # want initial forget value to be high and input value to be low so that 
        #  model starts with gradient descent
        nn.init.uniform_(self.bF, 8, 10)
        nn.init.uniform_(self.bI, -5, -4)
        # set initial cell state = learner's initial parameters
        if flat_learner is not None:
            self.cI.data.copy_(flat_learner.unsqueeze(1))

    def forward(self, inputs, hx=None):
        """Args:
            inputs = [x_all, grad]:
                x_all (torch.Tensor of size [n_params, 1, n_inputs]): outputs from previous LSTM, batch first
                grad (torch.Tensor of size [n_params]): gradients from learner
            hx (list of f, i, c):
                f (torch.Tensor of size [batch, 1]): forget gate
                i (torch.Tensor of size [batch, 1]): input gate
                c (torch.Tensor of size [n_params, 1]): cell state
        """
        x_all = inputs[0]
        grad = inputs[1]  # \tilde{C_t} = grad_t
        
        if self.batch_first:
            x_all.transpose_(0, 1)

        steps, batch, _ = x_all.size()

        if hx is None:
            f0 = torch.zeros((batch, 1)).to(self.cI.device)
            i0 = torch.zeros((batch, 1)).to(self.cI.device)
            c0 = self.cI    # C_{t-1} = theta_{t-1}
            hx = [f0, i0, c0]

        f_s, i_s, c_s = [hx[0]], [hx[1]], [hx[2]]
        
        for i in range(steps):
            f_prev, i_prev, c_prev = f_s[i], i_s[i], c_s[i]
            
            # next forget, input gate
            # f_t = sigmoid(W_f * [grad_t, loss_t, theta_{t-1}, f_{t-1}] + b_f)
            f_next = torch.mm(torch.cat((x_all[i], c_prev, f_prev), 1), self.WF) + self.bF.expand(batch, 1)
            # i_t = sigmoid(W_i * [grad_t, loss_t, theta_{t-1}, i_{t-1}] + b_i)
            i_next = torch.mm(torch.cat((x_all[i], c_prev, i_prev), 1), self.WI) + self.bI.expand(batch, 1)

            # next cell/params
            c_next = torch.sigmoid(f_next).mul(c_prev) - torch.sigmoid(i_next).mul(grad[i])

            f_s.append(f_next)
            i_s.append(i_next)
            c_s.append(c_next)

        cx = c_next
        hx = [f_next, i_next, cx]
        return cx, hx

    def extra_repr(self):
        s = 'n_inputs={n_inputs}, n_params={n_params}'
        if self.batch_first:
            s += ', batch_first={batch_first}'
        return s.format(**self.__dict__)


class MetaLearner(nn.Module):

    def __init__(self, input_size, hidden_size, flat_learner):
        super(MetaLearner, self).__init__()
        """Args:
            input_size (int): for the first LSTM layer, default = 4
            hidden_size (int): for the first LSTM layer, default = 20
            flat_learner (torch.Tensor): flattened learner
        """
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.metalstm = MetaLSTM(n_inputs=hidden_size, n_params=flat_learner.size(0), batch_first=True)

        self.reset_parameters(flat_learner)

    def reset_parameters(self, flat_learner=None):
        self.lstm.reset_parameters()
        self.metalstm.reset_parameters(flat_learner)

    def forward(self, inputs, hxs=None):
        """Args:
            inputs = [loss, grad_prep, grad]
                loss (torch.Tensor of size [1, 2])
                grad_prep (torch.Tensor of size [n_learner_params, 2])
                grad (torch.Tensor of size [n_learner_params])
            hxs = [(lstm_hn, lstm_cn), [mlstm_fn, mlstm_in, mlstm_cn]]
        """
        loss, grad_prep, grad = inputs[0], inputs[1], inputs[2]
        grad_prep.unsqueeze_(1) # [n_learner_params, 1, 2]
        loss = loss.expand_as(grad_prep)
        lstm_inputs = torch.cat((loss, grad_prep), 2)   # [n_learner_params, 1, 4] (batch, seqlen, input_size)
        
        if hxs is None:
            hxs = [None, None]

        lstm_outputs, lstm_hn_cn = self.lstm(lstm_inputs, hxs[0])
        cI, mlstm_fn_in_cn = self.metalstm([lstm_outputs, grad], hxs[1])

        return cI, [lstm_hn_cn, mlstm_fn_in_cn]

