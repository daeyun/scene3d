import multiprocessing

import torch
from scene3d import config
from scene3d import io_utils
from os import path
import itertools
import cv2
import imageio
import time
from scene3d.dataset import v1
from torch.utils import data
import time
from torch.backends import cudnn
from scene3d.net import deeplab
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
parser.add_argument('--save_every', type=int, default=18000)
args = parser.parse_args()

available_experiments = ['multi-layer', 'single-layer']


def loss_calc(pred, target):
    assert pred.shape[1] > 1

    mask = ~torch.isnan(target)
    target_log = torch.log2(target + 0.00001)

    target_depth_bg = target_log[:, 1, None]
    target_depth_diff = target_depth_bg - target_log[:, 0, None]

    new_target = torch.cat((target_depth_bg, target_depth_diff), dim=1)

    target_variable = autograd.Variable(new_target).cuda()
    return (target_variable - pred).abs()[mask].mean()


def loss_calc_single_depth(pred, target):
    assert pred.shape[1] == 1
    target_single_depth = target[:, 0, None]
    mask = ~torch.isnan(target_single_depth)
    target_log = torch.log2(target_single_depth + 0.00001)
    target_torch = autograd.Variable(target_log).cuda()
    return (target_torch - pred).abs()[mask].mean()


def main():
    log.info(args)
    assert args.save_dir.startswith('/'), args.save_dir
    io_utils.ensure_dir_exists(args.save_dir)

    cudnn.enabled = True
    cudnn.benchmark = True

    depth_dataset = v1.MultiLayerDepth(train=True)
    loader = data.DataLoader(depth_dataset, batch_size=args.batch_size, num_workers=args.num_data_workers,
                             shuffle=True, drop_last=True, pin_memory=True)

    log.info('Number of examples: %d', len(depth_dataset))

    if args.experiment == 'multi-layer':
        model = deeplab.Res_Deeplab(num_classes=2)
    elif args.experiment == 'single-layer':
        model = deeplab.Res_Deeplab(num_classes=1)
    else:
        raise NotImplementedError()

    model.train()
    model.cuda()

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    log.info('Number of torch parameters %d', len(params))
    optimizer = optim.Adam(params, lr=0.0005)
    optimizer.zero_grad()

    interp = nn.Upsample(size=depth_dataset.input_image_size, mode='bilinear', align_corners=True)

    log.info('Initialized model.')

    for i_epoch in range(args.max_epochs):
        for i_iter, batch in enumerate(loader):
            example_name, in_rgb, target_depth = batch
            in_rgb = autograd.Variable(in_rgb).cuda()
            optimizer.zero_grad()

            pred = interp(model(in_rgb))

            if args.experiment == 'multi-layer':
                loss = loss_calc(pred, target_depth)
            elif args.experiment == 'single-layer':
                loss = loss_calc_single_depth(pred, target_depth)

            loss.backward()
            optimizer.step()
            log.info('%d, %.5f', i_iter, loss)

            if i_iter % args.save_every == 0:
                save_filename = path.join(args.save_dir, 'v1_{:03d}_{:07d}.pth'.format(i_epoch, i_iter))
                log.info('Saving %s', save_filename)
                with open(save_filename, 'wb') as f:
                    torch.save(model, f)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
