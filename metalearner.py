from __future__ import division, print_function, absolute_import

import pdb
import torch
import torch.nn as nn


class MetaLSTMCell(nn.Module):
    """C_t = f_t * C_{t-1} + i_t * \tilde{C_t}"""
    def __init__(self, input_size, hidden_size=1):
        super(MetaLSTMCell, self).__init__()
        """Args:
            input_size (int): cell input size, default = 20
            hidden_size (int): should be 1
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.WF = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.WI = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.bI = nn.Parameter(torch.Tensor(1, hidden_size))
        self.bF = nn.Parameter(torch.Tensor(1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.constant_(weight, 0.0)

        # want initial forget value to be high and input value to be low so that 
        #  model starts with gradient descent
        nn.init.uniform_(self.bF, 4, 5)
        nn.init.uniform_(self.bI, -5, -4)

    def forward(self, inputs, hx):
        """Args:
            inputs = [x_all, grad]:
                x_all (torch.Tensor of size [n_learner_params, input_size]): outputs from previous LSTM
                grad (torch.Tensor of size [n_learner_params]): gradients from learner
            hx = [f_prev, i_prev, c_prev]:
                f (torch.Tensor of size [n_learner_params, 1]): forget gate
                i (torch.Tensor of size [n_learner_params, 1]): input gate
                c (torch.Tensor of size [n_learner_params, 1]): flat learner parameters (from real learner, not dummy)
        """
        x_all, grad = inputs
        batch, _ = x_all.size()

        f_prev, i_prev, c_prev = hx
        if f_prev is None:
            f_prev = torch.zeros((batch, self.hidden_size)).to(self.WF.device)
        if i_prev is None:
            i_prev = torch.zeros((batch, self.hidden_size)).to(self.WI.device)
        
        # f_t = sigmoid(W_f * [grad_t, loss_t, theta_{t-1}, f_{t-1}] + b_f)
        f_next = torch.mm(torch.cat((x_all, c_prev, f_prev), 1), self.WF) + self.bF.expand_as(f_prev)
        # i_t = sigmoid(W_i * [grad_t, loss_t, theta_{t-1}, i_{t-1}] + b_i)
        i_next = torch.mm(torch.cat((x_all, c_prev, i_prev), 1), self.WI) + self.bI.expand_as(i_prev)
        # next cell/params
        c_next = torch.sigmoid(f_next).mul(c_prev) - torch.sigmoid(i_next).mul(grad)

        return c_next, [f_next, i_next, c_next]

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        return s.format(**self.__dict__)


class MetaLearner(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MetaLearner, self).__init__()
        """Args:
            input_size (int): for the first LSTM layer, default = 4
            hidden_size (int): for the first LSTM layer, default = 20
        """
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.metalstm = MetaLSTMCell(input_size=hidden_size, hidden_size=1)

    def forward(self, inputs, hs=None):
        """Args:
            inputs = [loss, grad_prep, grad]
                loss (torch.Tensor of size [1, 2])
                grad_prep (torch.Tensor of size [n_learner_params, 2])
                grad (torch.Tensor of size [n_learner_params])

            hs = [(lstm_hn, lstm_cn), [metalstm_fn, metalstm_in, flat_learner]]
                = [None, [None, None, flat_learner]] (initially)
        """
        loss, grad_prep, grad = inputs
        loss = loss.expand_as(grad_prep)
        inputs = torch.cat((loss, grad_prep), 1)   # [n_learner_params, 4]

        lstmhx, lstmcx = self.lstm(inputs, hs[0])
        flat_learner_unsqzd, metalstm_hs = self.metalstm([lstmhx, grad], hs[1])

        return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]

