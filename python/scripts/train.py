import multiprocessing

import torch
from scene3d import config
from scene3d import io_utils
from os import path
import itertools
import cv2
import time
from scene3d.dataset import v1
from torch.utils import data
import time
from torch.backends import cudnn
from scene3d.net import deeplab
from scene3d.net import unet
from scene3d import log
from torch import optim
from torch import nn
from torch import autograd
import argparse

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--num_data_workers', type=int, default=5)
parser.add_argument('--save_dir', type=str, default='/data2/out/scene3d/v1/default')
parser.add_argument('--experiment', type=str, default='multi-layer')
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--save_every', type=int, default=10000)
parser.add_argument('--first_n', type=int, default=0)
parser.add_argument('--model', type=str, default='deeplab')
args = parser.parse_args()

available_experiments = ['multi-layer', 'single-layer']
available_models = ['unet_v0']


def loss_calc(pred, target):
    assert pred.shape[1] > 1
    mask = ~torch.isnan(target)
    return (torch.log2(target[mask] + 0.5) - pred[mask]).abs().mean()


def loss_calc_single_depth(pred, target):
    assert pred.shape[1] == 1
    target_single_depth = target[:, 0, None] - target[:, 1, None]
    mask = ~torch.isnan(target_single_depth)
    return (torch.log2(target_single_depth[mask] + 0.5) - pred[mask]).abs().mean()


def main():
    log.info(args)
    assert args.save_dir.startswith('/'), args.save_dir
    io_utils.ensure_dir_exists(args.save_dir)

    assert args.experiment in available_experiments
    assert args.model in available_models

    cudnn.enabled = True
    cudnn.benchmark = True

    depth_dataset = v1.MultiLayerDepth(train=True, subtract_mean=True, image_hw=(240, 320), first_n=args.first_n, rgb_scale=1.0 / 255)

    loader = data.DataLoader(depth_dataset, batch_size=args.batch_size, num_workers=args.num_data_workers,
                             shuffle=True, drop_last=True, pin_memory=True)

    log.info('Number of examples: %d', len(depth_dataset))

    if args.model == 'deeplab':
        raise NotImplementedError()
    elif args.model == 'unet_v0':
        if args.experiment == 'multi-layer':
            model = unet.Unet0(out_channels=2)
        elif args.experiment == 'single-layer':
            model = unet.Unet0(out_channels=1)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    model.train()
    model.cuda()

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    log.info('Number of torch parameters %d', len(params))
    optimizer = optim.Adam(params, lr=0.0005)
    optimizer.zero_grad()

    # interp = nn.Upsample(size=depth_dataset.image_hw, mode='bilinear', align_corners=True)

    log.info('Initialized model.')

    global_step = 0
    for i_epoch in range(args.max_epochs):
        for i_iter, batch in enumerate(loader):
            example_name, in_rgb, target_depth = batch
            in_rgb = in_rgb.cuda()
            target_depth = target_depth.cuda()
            optimizer.zero_grad()

            pred = model(in_rgb)

            if args.experiment == 'multi-layer':
                loss = loss_calc(pred, target_depth)
            elif args.experiment == 'single-layer':
                loss = loss_calc_single_depth(pred, target_depth)

            loss.backward()
            optimizer.step()
            log.info('%08d, %03d, %07d, %.5f', global_step, i_epoch, i_iter, loss)

            if global_step % args.save_every == 0:
                save_filename = path.join(args.save_dir, 'v1_{:03d}_{:07d}_{:08d}.pth'.format(i_epoch, i_iter, global_step))
                log.info('Saving %s', save_filename)
                with open(save_filename, 'wb') as f:
                    torch.save(model, f)
            global_step += 1


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
