#!/usr/bin/env python

import argparse
import os
import os.path as osp

import torch
import datetime

from data.GTA5 import GTA5
from FCN.model import FCN
from FCN.trainer import Trainer
from data.data_utils  import get_label_classes
from FCN.vgg import VGG16
import pdb 

configurations = {
	1: dict(
		max_iteration=200000,
		lr=1.0e-12,
		momentum=0.99,
		weight_decay=0.0005,
		interval_validate=5000,
	)
}


here = osp.dirname(osp.abspath(__file__))

def get_parameters(model, bias=False):
	import torch.nn as nn
	modules_skipped = (
		FCN,
		nn.ReLU,
		nn.MaxPool2d,
		nn.Dropout2d,
		nn.Sequential,
		nn.Upsample
	)
	#pdb.set_trace()
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			if bias:
				yield m.bias
			else:
				yield m.weight
		elif isinstance(m, nn.ConvTranspose2d):
			# weight is frozen because it is just a bilinear upsampling
			if bias:
				assert m.bias is None
		elif isinstance(m, modules_skipped):
			continue
		else:
			raise ValueError('Unexpected module: %s' % str(m))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', type=int, default=1,
						choices=configurations.keys())
	parser.add_argument('--resume', help='Checkpoint path', default = 'FCN/checkpoints/')
	parser.add_argument('-transfer', type=bool, default=False)

	args = parser.parse_args()

	resume = args.resume
	cfg = configurations[args.config]
	out = os.getcwd()
	
	cuda = torch.cuda.is_available()
	
	if cuda:
		torch.cuda.manual_seed(1123)
	else:
		torch.manual_seed(1123)
	
	# load dataset
	root = osp.join(here, 'data')
	kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}


	train_loader = torch.utils.data.DataLoader(
		GTA5(root, split='train', transform=True),
		batch_size=1, shuffle=True, **kwargs)
	val_loader = torch.utils.data.DataLoader(
		GTA5(root, split='val', transform=True),
		batch_size=1, shuffle=False, **kwargs)

	# model
	model = FCN()
	start_epoch = 0
	start_iteration = 0

	check_point = None
	## resume 
	if resume:
		if os.listdir(resume):
			check_points = os.listdir(resume)
			check_point = osp.join(resume, check_points[-1])
			check_point = torch.load(check_point)
			model.load_state_dict(check_point['model_state_dict'])
			start_epoch = check_point['epoch']
			start_iteration = check_point['iteration']
	else:
		#initialize the model
		vgg = VGG16(pretrained=True)
		model.copy_para_from_vgg16(vgg)

	if cuda:
		model = model.cuda()

	# optimizer
	optim = torch.optim.SGD(
		[ 
			{'params': get_parameters(model, bias=False)},
			{'params': get_parameters(model, bias=True),
			 'lr': cfg['lr'] * 2, 'weight_decay': 0},
		],
		lr=cfg['lr'],
		momentum=cfg['momentum'],
		weight_decay=cfg['weight_decay']
		)
	# if resume:
	# 	optim.load_state_dict(check_point['optim_state_dict'])

	trainer = Trainer(
		cuda=cuda,
		model=model,
		optimizer=optim,
		train_loader=train_loader,
		val_loader=val_loader,
		out=out,
		max_iter=cfg['max_iteration'],
		interval_validate=cfg.get('interval_validate', len(train_loader)),
	)
	trainer.epoch = start_epoch
	trainer.iteration = start_iteration
	trainer.train()

if __name__ == '__main__':
	from __init_path__ import add_full_path
	add_full_path()
	from utils.util import label_accuracy_score
	main()
	