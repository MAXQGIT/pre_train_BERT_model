import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath('..'))
from config import Config

config = Config()


class Mask():
    def __init__(self):
        pass

    def padding_mask(self, seq_k, seq_q):
        seq_len = seq_q.size(1)
        pad_mask = seq_k.eq(config.pad_idx)
        return pad_mask.unsqueeze(1).expand(-1, seq_len, -1)

    def no_padding_mask(self, seq):
        return seq.ne(config.pad_idx).type(torch.float).unsqueeze(-1)


import math


# gelu()激活函数，区别于relu()激活函数
def gelu(x):
    """
    区别于relu()激活函数的gelu()激活函数
    :param x: 要激活的神经元
    :return:
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SpecialOptimizer():
    def __init__(self, optimizer, warmup_steps, d_model, step_num=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = step_num

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step_and_update_learning_rate(self):
        self.step_num += 1
        lr = np.power(self.d_model, -0.5) * np.min(
            [np.power(self.step_num, -0.5), np.power(self.warmup_steps, -1.5) * self.step_num])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()
        return lr
