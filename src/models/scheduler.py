#
# WildCLIP by Gabeff et al.
# © ECEO and A. Mathis Lab
# https://github.com/amathislab/wildclip
#
# Licensed under GNU Lesser General Public License v3.0
#

"""Pytorch Scheduler

from https://github.com/mlfoundations/open_clip/blob/37b729bc69068daa7e860fb7dbcf1ef1d03a4185/src/training/scheduler.py#L9
"""

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


class CosineLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_length, steps, eta_min=0) -> None:
        self.current_step = 0
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_length = warmup_length
        self.steps = steps
        self.eta_min = eta_min

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_length:
            lr = _warmup_lr(self.base_lr, self.warmup_length, self.current_step)
        else:
            e = self.current_step - self.warmup_length
            es = self.steps - self.warmup_length
            lr = max(0.5 * (1 + np.cos(np.pi * e / es)) * self.base_lr, self.eta_min)
        assign_learning_rate(self.optimizer, lr)
        return lr
