import torch
import torch.nn as nn
import torch.nn.functional as F

# from data.data_utils import labelClasses
# labels = labelClasses()
# class CrossEntropyLoss2d(nn.Module):

#   def __init__(self, weight=None):
#     super(CrossEntropyLoss2d,self).__init__()
#     self.loss = nn.NLLLoss2d(weight)

#   def forward(self, outputs, targets):
#     # n,c,w,h = outputs.size()
#     # for i in range(c):
#     #   if i in labels:
#     #     continue
#     #   else:
#     #     outputs[:,i,:,:] = 0
#     return self.loss(F.log_softmax(outputs), targets)

def CrossEntropyLoss2d(outputs, targets):
  loss = nn.NLLLoss2d()
  return loss(F.log_softmax(outputs), targets)
  
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