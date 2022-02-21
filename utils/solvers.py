'''
files relevant to ooptimizers, lr_schedulers
'''

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(optim_config, params):
    return getattr(torch.optim, optim_config['type'])(
        params, **optim_config['options'])

def build_lr_scheduler(lr_config, optimizer):
    if lr_config['type'] == 'PolyLR':
        return PolyLR(optimizer, **lr_config['options'])
    else:
        # try getting optimizer form torch.optim
        return getattr(torch.optim.lr_scheduler, lr_config['type'])(
          optimizer, **lr_config['options'])


class PolyLR(LambdaLR):
  """DeepLab learning rate policy"""
  def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
      super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)




