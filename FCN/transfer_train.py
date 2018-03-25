
import datetime
import math
import os
import os.path as osp
import shutil

import fcn
import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

import torchfcn

def domain_classifer_loss(source, target):

  assert source.size() == target.size()
  source_p_theta = F.softmax(source, dim=1)
  target_p_theta = F.softmax(target, dim=1)

  log_source = -F.nll_loss(source_p_theta)
  log_target = -F.nll_loss(1 - target_p_theta.data)

  log_source = Variable(torch.from_tensor(log_source))
  log_target = Variable(torch.from_tensor(log_target))

  sum_source = log_source.sum()
  sum_target = log_target.sum()

  return -sum_source - sum_target

def domain_adversarial_loss(source, target):
  
  assert source.size() == target.size()
  
  source_p_theta = F.softmax(source, dim=1)
  target_p_theta = F.softmax(target, dim=1)

  log_source = -F.nll_loss(1 - source_p_theta.data)
  log_target = -F.nll_loss(target_p_theta.data)

  log_source = Variable(torch.from_tensor(log_source))
  log_target = Variable(torch.from_tensor(log_target))

  sum_source = log_source.sum()
  sum_target = log_target.sum()

  return -sum_source - sum_target


class Transfer_train(object):

  def __init__(self, cuda, model, optimizer,
         train_loader, val_loader, out, max_iter,
         size_average=False, interval_validate=None):
    
    self.cuda = cuda 

    self.source_model = model
    self.target_model = model

    self.train_loader = train_loader
    self.val_loader = val_loader

    self.timestamp_start = \
      datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    self.size_average = size_average

    if interval_validate is None:
      self.interval_validate = len(self.train_loader)
    else:
      self.interval_validate = interval_validate

    self.out = out

    if not osp.exists(self.out):
      os.makedirs(out)

