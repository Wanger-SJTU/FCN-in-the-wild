#!/usr/bin/env python

import torch
import os
import collections
import os.path as osp
import numpy as np 
import scipy.io as sio

from PIL import Image as image
from torch.utils import data


mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

class GTA5(data.Dataset):
	"""dataloader for GTA5"""
	
	def __init__(self, root, split='train', transform=False):
		super(GTA5, self).__init__()
		
		self.root = root
		self.split = split
		self._transform = transform

		#GTA5 dataset dir
		# GTA5 dataset direcory structer
		
		dataset_dir = osp.join(self.root, 'GTA5')
		self.files = collections.defaultdict(list)

		for split in ['train', 'val']:
			imageset_files = osp.join(dataset_dir,
				'{dir}/{file}.txt'.format(dir=split, file=split))
			with open(imageset_files) as f:
				for file in f:
					img_file = osp.join(dataset_dir,
						'{split}/images/{file_name}'.format(split=split,file_name=file))
					lbl_file = osp.join(dataset_dir,
						'{split}/labels/{file_name}'.format(split=split,file_name=file))
					self.files[split].append({
						'img':img_file,
						'lbl':lbl_file,})
		img_dir  = osp.join(dataset_dir,'train', 'labels')
		img_path = os.listdir(img_dir)[0]
		img = Image.open(osp.join(dataset_dir,'GTA5','train','labels',img_path))
		self.pa
	def __len__(self):
		return len(self.files[self.split])


	def __getitem__(self, index):
		data_file = self.files[self.split][index]
    # load image
    img_file = data_file['img']
    img = PIL.Image.open(img_file)
    img = np.array(img, dtype=np.uint8)
    # load label
    lbl_file = data_file['lbl']
    lbl = PIL.Image.open(lbl_file)
    lbl = np.array(lbl, dtype=np.uint8)
    if self._transform:
      return self.transform(img, lbl)
    else:
      return img, lbl

  def transform(self, img, lbl):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= self.mean_bgr
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    lbl = torch.from_numpy(lbl).long()
    return img, lbl

  def untransform(self, img, lbl):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img += self.mean_bgr
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]
    lbl = lbl.numpy()
    return img, lbl