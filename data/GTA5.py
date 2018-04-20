#!/usr/bin/env python

import torch
import os
import sys
import collections
import os.path as osp
import numpy as np 
import scipy.io as sio

from PIL import Image as image
from torch.utils import data
from data.data_utils import get_label_classes
from data.data_utils import resize_input


mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

class GTA5(data.Dataset):
  """dataloader for GTA5"""
  
  def __init__(self, root, split='train', transform=False):
    super(GTA5, self).__init__()
    
    self.root = osp.join(root, 'GTA5')
    if split != 'train':
      self.split = 'val'
    else:
      self.split = 'train'
    self._transform = transform

    # GTA5 dataset dir
    # GTA5 dataset direcory structer
    
    dataset_dir = osp.join(self.root, self.split)
    self.files = []

    # for split in ['train', 'val']:
    imageset_files = osp.join(dataset_dir,
      '{file}.txt'.format(file=self.split))
    with open(imageset_files) as f:
      for file in f:
        img_file = osp.join(dataset_dir,
          'images/{file_name}'.format(file_name=file))
        lbl_file = osp.join(dataset_dir,
          'labels/{file_name}'.format(file_name=file))
        self.files.append({
          'img':img_file,
          'lbl':lbl_file,})
    
  def __len__(self):
    return len(self.files)


  def __getitem__(self, index):
    data_file = self.files[index]
    # load image
    img_file = data_file['img'].strip()
    img = image.open(img_file)
    img = resize_input(img)
    img = np.array(img, dtype=np.uint8)
    # load label
    lbl_file = data_file['lbl'].strip()
    lbl = image.open(lbl_file)
    lbl = resize_input(lbl)
    lbl = np.array(lbl, dtype=np.uint8)
    if self._transform:
      return self.transform(img, lbl)
    else:
      return img, lbl

  def transform(self, img, lbl):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= mean_bgr
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    lbl = torch.from_numpy(lbl).long()
    return img, lbl

  def untransform(self, img, lbl):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img += mean_bgr
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]
    lbl = lbl.numpy()
    return img, lbl
