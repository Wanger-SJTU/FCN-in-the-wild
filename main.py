#!/usr/bin/env python

import argparse
import os
import os.path as osp

import torch

from data.GTA5 import GTA5
from FCN.model import FCN
from FCN.trainer import Trainer

# from train_fcn32s import get_log_dir
# from train_fcn32s import get_parameters


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        lr=1.0e-12,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
        # fcn32s_pretrained_model=torchfcn.models.FCN32s.download(),
    )
}


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    parser.add_argument('-transfer', type=bool, default=False)

    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    out = get_log_dir('fcn16s', args.config, cfg)
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1123)
    
    if cuda:
        torch.cuda.manual_seed(1123)

    # 1. dataset
    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
        GTA5(root, split='train', transform=False),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        GTA5(root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model
    model = FCN(n_class=34)
    start_epoch = 0
    start_iteration = 0
    
    ######### todo
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        fcn32s = torchfcn.models.FCN32s()
        fcn32s.load_state_dict(torch.load(cfg['fcn32s_pretrained_model']))
        model.copy_params_from_fcn32s(fcn32s)
    if cuda:
        model = model.cuda()

    # 3. optimizer
    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': cfg['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])
    
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

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
    main()
