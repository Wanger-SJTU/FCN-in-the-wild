
import os
import math
import tqdm
import torch
import visdom
import shutil
import os.path as osp
import numpy as np

from torch.autograd import Variable
import torch.nn.functional as F

from data.GTA5 import GTA5
from data.data_utils import get_label_classes
from data.data_utils import index2rgb
from utils.criterion import CrossEntropyLoss2d
from utils.util import label_accuracy_score

#visdom for visualization
vis = visdom.Visdom()

win0 = vis.image(torch.zeros(3, 100, 100))
win1 = vis.image(torch.zeros(3, 100, 100))
win2 = vis.image(torch.zeros(3, 100, 100))
win3 = vis.image(torch.zeros(3, 100, 100))

#subtract mean_bgr
mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

class Trainer(object):
  def __init__(self, cuda, model, optimizer,
         train_loader, val_loader, out, max_iter,
         size_average=False, interval_validate=None):
    
    self.cuda = cuda
    self.model = model
    self.optim = optimizer

    self.train_loader = train_loader
    self.val_loader = val_loader
    self.size_average = size_average

    if interval_validate is None:
      self.interval_validate = len(self.train_loader)
    else:
      self.interval_validate = interval_validate

    self.out = out
    if not osp.exists(self.out):
      os.makedirs(self.out)

    self.checkpoint_dir = 'FCN\checkpoints'
    if not osp.exists(self.checkpoint_dir):
      os.makedirs(self.checkpoint_dir)

    self.epoch = 0
    self.iteration = 0
    self.max_iter = max_iter
    self.best_mean_iu = 0
    self.val_loss_list = []
    self.train_loss_list =[]

  def validate(self):
    training = self.model.training
    self.model.eval()

    n_class = max(get_label_classes())

    val_loss = 0
    visualizations = []
    label_trues, label_preds = [], []
    
    for batch_idx, (data, target) in tqdm.tqdm(
        enumerate(self.val_loader), total=len(self.val_loader),
        desc='Valid iteration=%d' % self.iteration, ncols=80,
        leave=False):
      
      if self.cuda:
        data, target = data.cuda(), target.cuda()
      data, target = Variable(data, volatile=True), Variable(target)
      score = self.model(data)

      loss = CrossEntropyLoss2d(score, target)
      if np.isnan(float(loss.data[0])):
        raise ValueError('loss is nan while validating')
      
      val_loss += float(loss.data[0]) / len(data)

      imgs = data.data.cpu()
      lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
      lbl_true = target.data.cpu()
      
      for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
        img, lt = self.val_loader.dataset.untransform(img, lt)
        label_trues.append(lt)
        label_preds.append(lp)
            
    metrics = label_accuracy_score(label_trues, label_preds, n_class)

    val_loss /= len(self.val_loader)

    mean_iu = metrics[2]
    is_best = mean_iu > self.best_mean_iu
    if is_best:
      self.best_mean_iu = mean_iu
    
    filename = ('%s/epoch-%d.pth' \
                    % (self.checkpoint_dir, self.epoch))
    torch.save({
      'epoch': self.epoch,
      'iteration': self.iteration,
      'arch': self.model.__class__.__name__,
      'optim_state_dict': self.optim.state_dict(),
      'model_state_dict': self.model.state_dict(),
      'best_mean_iu': self.best_mean_iu,
    }, filename)
    
    if is_best:
      shutil.copy(osp.join(self.out, filename),
            osp.join(self.out, 'model_best.pth.tar'))

    metrics ={
        'loss':loss.data[0],
        'acc' : metrics[0],
        'acc_cls':metrics[1],
        'mean_iu':metrics[2],
        'fwavacc':metrics[3]
    }
    self.val_loss_list.append(metrics)

    if training:
      self.model.train()

  def train_epoch(self):
    self.model.train()

    n_class = max(get_label_classes())
    train_loss = 0
    for batch_idx, (data, target) in tqdm.tqdm(
        enumerate(self.train_loader), total=len(self.train_loader),
        desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
      
      iteration = batch_idx + self.epoch * len(self.train_loader)
      
      if self.iteration != 0 and (iteration-1) != self.iteration:
        continue  # for resuming
      
      self.iteration = iteration

      if self.iteration % self.interval_validate == 0:
        self.validate()

      assert self.model.training

      if self.cuda:
        data, target = data.cuda(), target.cuda()
      
      data, target = Variable(data), Variable(target)
      self.optim.zero_grad()
      
      score = self.model(data)
      loss = CrossEntropyLoss2d(score, target)
      loss /= len(data)
      
      if np.isnan(float(loss.data[0])):
        raise ValueError('loss is nan while training')
      
      loss.backward()
      self.optim.step()

      train_loss += loss.data[0]
      
      metrics = []
      
      lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
      lbl_true = target.data.cpu().numpy()
      
      for lt, lp in zip(lbl_true, lbl_pred):
        acc, acc_cls, mean_iu, fwavacc = \
          label_accuracy_score([lt], [lp], n_class=n_class)
        metrics.append((acc, acc_cls, mean_iu, fwavacc))
      metrics = np.mean(metrics, axis=0)

      #visualize
      if self.iteration % 100 == 0:
        metrics ={
        'loss':train_loss / 100,
        'acc' : metrics[0],
        'acc_cls':metrics[1],
        'mean_iu':metrics[2],
        'fwavacc':metrics[3]}
        self.train_loss_list.append(metrics)

        # print('train loss: %.4f (epoch: %d, step: %d)' % \
        #   (metrics[-1]['loss'],self.epoch, self.iteration%len(self.train_loader))
        
        image = data[0].data.cpu()
        image[0] = image[0] + 122.67891434
        image[1] = image[1] + 116.66876762
        image[2] = image[2] + 104.00698793
        step = self.iteration % len(self.train_loader)
        title = 'input: (epoch: %d, step: %d)' % (self.epoch,step)
        vis.image(image, win=win1, env='fcn', opts=dict(title=title))
        title = 'output (epoch: %d, step: %d)' % (self.epoch,step)
        vis.image(index2rgb(lbl_pred[0]),
                  win=win2, env='fcn', opts=dict(title=title))
        title = 'target (epoch: %d, step: %d)' % (self.epoch,step)
        vis.image(index2rgb(lbl_true[0]),
                  win=win3, env='fcn', opts=dict(title=title))
        epoch_loss = train_loss / 100
        x = np.arange(1, len(epoch_loss) + 1, 1)
        title = 'loss (epoch: %d, step: %d)' % (self.epoch,step)
        vis.line(np.array(epoch_loss), x, env='fcn', win=win0,
                 opts=dict(title=title))
      
        train_loss = 0
      if self.iteration >= self.max_iter:
        break

  def train(self):
    max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
    for epoch in tqdm.trange(self.epoch, max_epoch,
                 desc='Train', ncols=80):
      self.epoch = epoch
      self.train_epoch()
      if self.iteration >= self.max_iter:
        break