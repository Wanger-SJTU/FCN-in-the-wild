

import torch
import torch.nn as nn
import numpy as np
# import torchvision.models as models

from torch.autograd import Variable
from data.data_utils import get_label_classes
# from torchvision import datasets, models, transforms


class FCN(nn.Module):

  def __init__(self, n_classes = max(get_label_classes())):
    super(FCN, self).__init__()
    
    self.front_end = nn.Sequential(
      # layer_1
      nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      #layer_2
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      #layer_3                        
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=True),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      #layer_4
      nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
      nn.ReLU(inplace=True),
      #layer_5            
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
      nn.ReLU(inplace=True),
      
      # fc6 layer
      nn.Conv2d(512, 4096, kernel_size=3, stride=1, padding=0, bias=True, dilation=4), 
      nn.ReLU(inplace=True),
      
      # nn.Dropout(),
      
      # fc7 layer
      nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=True), 
      nn.ReLU(inplace=True),
     
      # nn.Dropout(),
      
      # final layer
      nn.Conv2d(4096, n_classes, kernel_size=1, stride=1, padding=0, bias=True),
      nn.ReLU(inplace=True),
     )

    self.upsample = nn.Sequential(       
        # nn.Conv2d(19, 19, kernel_size=1, stride=1, padding=0, bias=True),
        nn.Upsample(size=(1000), mode='bilinear'),    
    )

  def forward(self, x, transfer=False):
    
    feature_map = self.front_end(x)
    if transfer:
      return feature_map
    
    predict = self.upsample(feature_map)

    return predict

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.zero_()
        if m.bias is not None:
          m.bias.data.zero_()

  def copy_para_from_vgg16(self, vgg16):
    for name, l1 in vgg16.named_children():
      try:
        l2 = getattr(self, name)
        l2.weight
      except Exception as e:
        continue
      assert l1.weight.size() == l2.weight.size()
      assert l1.bias.size() == l2.bias.size()
      l2.weight.data.copy_(l1.weight.data)
      l2.bias.data.copy_(l1.bias.data)