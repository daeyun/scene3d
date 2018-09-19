import multiprocessing

import torch
import numpy as np
from scene3d import config
import copy
from scene3d import io_utils
from os import path
import itertools
import cv2
import time
import typing
import re
from scene3d.dataset import v1
from scene3d.dataset import v2
from scene3d.dataset import v8
import torch.utils.data
import time
from torch.backends import cudnn
from scene3d.net import deeplab
from scene3d.net import unet
from scene3d.net import unet_overhead
from scene3d.net import unet_no_bn
from scene3d import log
from scene3d import loss_fn
from scene3d import torch_utils
from torch import optim
from torch import nn
from torch import autograd
import argparse

"""
When you add a new experiment, give it a unique experiment name in `available_experiments`.
Every experiment is uniquely identified by (experiment_name, model_name).

Then youo need to edit the following three functions:
`get_dataset`, `get_pytorch_model_and_optimizer`, `compute_loss`
"""

available_experiments = [
    'multi-layer',
    'single-layer',
    'nyu40-segmentation',
    'multi-layer-and-segmentation',
    'single-layer-and-segmentation',
    'multi-layer-3',
    'multi-layer-3',
    'overhead-features-01-log-l1-loss',
    'overhead-features-01-l1-loss',
]

available_models = [
    'unet_v0',
    'unet_v0_no_bn',
    'unet_v0_overhead',
]


def get_dataset(experiment_name, split_name) -> torch.utils.data.Dataset:
    """
    :param experiment_name:
    :param split_name: Either 'train', 'test', or 'all'.  Whether 'test' means validation or final test split is determined by the Dataset object implementation.
    :return: A pytorch Dataset object. Training split only.
    """
    assert split_name in ('train', 'test', 'all')

    # Used for debugging. e.g. When you want to make sure the model can overfit given a small dataset.
    first_n = None

    if experiment_name == 'multi-layer':
        assert split_name in ('train', 'test')
        dataset = v1.MultiLayerDepth(train=split_name == 'train', subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255)
    elif experiment_name == 'single-layer':
        assert split_name in ('train', 'test')
        dataset = v1.MultiLayerDepth(train=split_name == 'train', subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255)
    elif experiment_name == 'nyu40-segmentation':
        assert split_name in ('train', 'test')
        dataset = v1.NYU40Segmentation(train=split_name == 'train', subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255)
    elif experiment_name == 'multi-layer-and-segmentation':
        assert split_name in ('train', 'test')
        dataset = v1.MultiLayerDepthNYU40Segmentation(train=split_name == 'train', subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255)
    elif experiment_name == 'single-layer-and-segmentation':
        assert split_name in ('train', 'test')
        dataset = v1.MultiLayerDepthNYU40Segmentation(train=split_name == 'train', subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255)
    elif experiment_name == 'multi-layer-3':
        assert split_name in ('train', 'test')
        dataset = v2.MultiLayerDepth_0(train=split_name == 'train', subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255)
    elif experiment_name == 'multi-layer-d-3':
        assert split_name in ('train', 'test')
        dataset = v2.MultiLayerDepth_1(train=split_name == 'train', subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255)
    # End of legacy code.

    elif experiment_name.startswith('overhead-features-01'):
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('features_overhead', 'depth_overhead'))

    else:
        raise NotImplementedError()

    return dataset


def get_pytorch_model_and_optimizer(model_name: str, experiment_name: str) -> typing.Tuple[nn.Module, optim.Optimizer]:
    if model_name == 'deeplab':
        raise NotImplementedError()
    elif model_name == 'unet_v0':
        if experiment_name == 'multi-layer':
            model = unet.Unet0(out_channels=2)
        elif experiment_name == 'single-layer':
            model = unet.Unet0(out_channels=1)
        elif experiment_name == 'nyu40-segmentation':
            model = unet.Unet0(out_channels=40)
        elif experiment_name == 'multi-layer-and-segmentation':
            model = unet.Unet0(out_channels=42)
        elif experiment_name == 'single-layer-and-segmentation':
            model = unet.Unet0(out_channels=41)
        elif experiment_name == 'multi-layer-3':
            model = unet.Unet0(out_channels=3)
        elif experiment_name == 'multi-layer-d-3':
            model = unet.Unet0(out_channels=3)
        else:
            raise NotImplementedError()
    elif model_name == 'unet_v0_no_bn':
        if experiment_name == 'multi-layer':
            model = unet_no_bn.Unet0(out_channels=2)
        elif experiment_name == 'single-layer':
            model = unet_no_bn.Unet0(out_channels=1)
        elif experiment_name == 'nyu40-segmentation':
            model = unet_no_bn.Unet0(out_channels=40)
        elif experiment_name == 'multi-layer-and-segmentation':
            model = unet_no_bn.Unet0(out_channels=42)
        elif experiment_name == 'single-layer-and-segmentation':
            model = unet_no_bn.Unet0(out_channels=41)
        elif experiment_name == 'multi-layer-3':
            model = unet_no_bn.Unet0(out_channels=3)
        elif experiment_name == 'multi-layer-d-3':
            model = unet_no_bn.Unet0(out_channels=3)
        else:
            raise NotImplementedError()
    elif model_name == 'unet_v0_overhead':
        if experiment_name.startswith('overhead-features-01'):
            model = unet_overhead.Unet0(out_channels=1)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    log.info('Number of pytorch parameter tensors %d', len(params))
    optimizer = optim.Adam(params, lr=0.0005)
    optimizer.zero_grad()

    return model, optimizer


def compute_loss(pytorch_model: nn.Module, batch, experiment_name: str) -> torch.Tensor:
    if experiment_name == 'multi-layer':
        example_name, in_rgb, target = batch
        in_rgb = in_rgb.cuda()
        pred = pytorch_model(in_rgb)
        target = target.cuda()
        loss_all = loss_fn.loss_calc(pred, target)
    elif experiment_name == 'single-layer':
        example_name, in_rgb, target = batch
        in_rgb = in_rgb.cuda()
        pred = pytorch_model(in_rgb)
        target = target.cuda()
        loss_all = loss_fn.loss_calc_single_depth(pred, target)
    elif experiment_name == 'nyu40-segmentation':
        example_name, in_rgb, target = batch
        in_rgb = in_rgb.cuda()
        pred = pytorch_model(in_rgb)
        target = target.cuda()
        loss_all = loss_fn.loss_calc_classification(pred, target)
    elif experiment_name == 'multi-layer-and-segmentation':
        example_name, in_rgb, target = batch
        in_rgb = in_rgb.cuda()
        pred = pytorch_model(in_rgb)
        target_depth = target[0].cuda()
        target_category = target[1].cuda()
        loss_category = loss_fn.loss_calc_classification(pred[:, :40], target_category)
        loss_depth = loss_fn.loss_calc(pred[:, 40:], target_depth)
        loss_all = loss_category * 0.4 + loss_depth
    elif experiment_name == 'single-layer-and-segmentation':
        example_name, in_rgb, target = batch
        in_rgb = in_rgb.cuda()
        pred = pytorch_model(in_rgb)
        target_depth = target[0].cuda()
        target_category = target[1].cuda()
        loss_category = loss_fn.loss_calc_classification(pred[:, :40], target_category)
        loss_depth = loss_fn.loss_calc_single_depth(pred[:, 40:], target_depth)
        loss_all = loss_category * 0.4 + loss_depth
    elif experiment_name == 'multi-layer-3':
        example_name, in_rgb, target, _, _ = batch
        in_rgb = in_rgb.cuda()
        pred = pytorch_model(in_rgb)
        target = target.cuda()
        loss_all = loss_fn.loss_calc(pred, target)
    elif experiment_name == 'multi-layer-d-3':
        example_name, in_rgb, target, _, _ = batch
        in_rgb = in_rgb.cuda()
        pred = pytorch_model(in_rgb)
        target = target.cuda()
        loss_all = loss_fn.loss_calc(pred, target)
    # End of legacy code.

    elif experiment_name == 'overhead-features-01-l1-loss':
        example_name = batch['name']
        # Excluding RGB features. 64 channels
        input_features = batch['features_overhead'][:, 3:].cuda()
        target_depth = batch['depth_overhead'][:, :1].cuda()
        pred = pytorch_model(input_features)
        loss_all = loss_fn.loss_calc_overhead_single_raw(pred, target_depth)
    elif experiment_name == 'overhead-features-01-log-l1-loss':
        example_name = batch['name']
        # Excluding RGB features. 64 channels
        input_features = batch['features_overhead'][:, 3:].cuda()
        target_depth = batch['depth_overhead'][:, :1].cuda()
        pred = pytorch_model(input_features)
        loss_all = loss_fn.loss_calc_overhead_single_log(pred, target_depth)
    else:
        raise NotImplementedError()

    return loss_all


def load_checkpoint(filename, use_cpu=False) -> typing.Tuple[nn.Module, optim.Optimizer, dict]:
    """
    :return: A tuple of (pytorch_model, optimizer, metadata_dict)
    `metadata_dict` contains `global_step`, etc.
    See `save_checkpoint`.
    """
    assert path.isfile(filename)
    assert filename.endswith('.pth')  # Sanity check. Not a requirement.
    log.info('Loading from checkpoint file {}'.format(filename))
    loaded_dict = torch_utils.load_torch_model(filename, use_cpu=use_cpu)
    if not isinstance(loaded_dict, dict):
        log.error('Loaded object:\n{}'.format(loaded_dict))
        raise RuntimeError()

    metadata_dict = loaded_dict['metadata']
    pytorch_model, optimizer = get_pytorch_model_and_optimizer(
        model_name=metadata_dict['model_name'],
        experiment_name=metadata_dict['experiment_name'],
    )
    pytorch_model.load_state_dict(loaded_dict['model_state_dict'])
    optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

    return pytorch_model, optimizer, metadata_dict


def save_checkpoint(save_dir: str, pytorch_model: nn.Module, optimizer: torch.optim.Optimizer, metadata: dict):
    assert isinstance(pytorch_model, nn.Module)
    assert 'global_step' in metadata
    assert 'epoch' in metadata
    assert 'iter' in metadata
    assert 'experiment_name' in metadata
    assert 'model_name' in metadata
    assert 'timestamp' in metadata
    assert 'model' not in metadata  # Sanity check.

    io_utils.ensure_dir_exists(save_dir)

    save_filename = path.join(save_dir, '{:08d}_{:03d}_{:07d}.pth'.format(metadata['global_step'], metadata['epoch'], metadata['iter']))
    log.info('Saving %s', save_filename)

    saved_dict = {
        'model_state_dict': pytorch_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata,
    }

    with open(save_filename, 'wb') as f:
        torch.save(saved_dict, f)

    return save_filename


class BottleneckDetector(object):
    def __init__(self, name, logger, check_every=20, threshold_seconds=0.2):
        self.check_every = check_every
        self.last_seen = time.time()
        self.delay_count = 0
        self.count = 0
        self.threshold_seconds = threshold_seconds
        self.logger = logger
        self.name = name
        self.delay_times = []

    def tic(self):
        self.last_seen = time.time()

    def toc(self):
        delay = time.time() - self.last_seen
        if delay > self.threshold_seconds:
            self.delay_times.append(delay)
            self.delay_count += 1
        self.count += 1

        if self.count >= self.check_every:
            # If delay happens more than half the times, log warning.
            if self.delay_count > 0.5 * self.count:
                self.logger.warning('{} bottleneck detected: {} out of {}.  Mean delay: {:.3f}'.format(self.name, self.delay_count, self.count, np.mean(self.delay_times)))
            self.count = 0
            self.delay_count = 0
            self.delay_times.clear()


class Trainer(object):
    def __init__(self, args: argparse.Namespace):
        assert args.save_dir.startswith('/'), args.save_dir
        io_utils.ensure_dir_exists(args.save_dir)

        assert args.experiment in available_experiments
        assert args.model in available_models
        assert args.batch_size > 0
        assert args.save_every > 0

        self.experiment_name = args.experiment
        self.model_name = args.model
        self.save_every = args.save_every
        self.max_epochs = args.max_epochs
        self.load_checkpoint = args.load_checkpoint
        self.save_dir = args.save_dir
        self.num_data_workers = args.num_data_workers
        self.batch_size = args.batch_size
        self.use_cpu = args.use_cpu
        self.log_filename = path.join(self.save_dir, '{}_{}.log'.format(self.experiment_name, self.model_name))

        self.logger = log.make_logger('trainer', level=log.DEBUG)
        log.add_stream_handler(self.logger, level=log.INFO)
        log.add_file_handler(self.logger, filename=self.log_filename, level=log.DEBUG)

        self.logger.info('Initializing Trainer:\n{}'.format(args))

        if not self.use_cpu:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.logger.info('cudnn version: {}'.format(cudnn.version()))
            assert torch.cuda.is_available()

        self.dataset = get_dataset(experiment_name=self.experiment_name, split_name='train')
        self.logger.info('Number of examples: %d', len(self.dataset))

        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_data_workers, shuffle=True, drop_last=True, pin_memory=True)
        self.logger.info('Initialized data loader.')

        self.global_step = 0  # Total number of steps. Preserved across checkpoints.
        self.epoch = 0
        self.iter = 0  # Step within the current epoch.
        self.loaded_metadata = None

        if self.load_checkpoint:
            self.model, self.optimizer, self.loaded_metadata = load_checkpoint(self.load_checkpoint, use_cpu=self.use_cpu)
            self.logger.info('Loaded metadata from {}:\n{}'.format(self.load_checkpoint, self.loaded_metadata))

            assert self.experiment_name == self.loaded_metadata['experiment_name']
            assert self.model_name == self.loaded_metadata['model_name']
            if self.batch_size != self.loaded_metadata['batch_size']:
                self.logger.info('batch_size changed from {} to {}'.format(self.loaded_metadata['batch_size'], self.batch_size))
            self.global_step = self.loaded_metadata['global_step']
            # Other attributes are "overwritten" by the values given in `args`.
        else:
            self.model, self.optimizer = get_pytorch_model_and_optimizer(model_name=self.model_name, experiment_name=self.experiment_name)
            # Immediately save a checkpoint at global step 0.
            assert self.try_save_checkpoint()

        self.model.train()

        if not self.use_cpu:
            self.model.cuda()
            torch.set_default_tensor_type('torch.FloatTensor')  # Back to defaults.

        self.logger.info('Initialized model. Current global step: {}'.format(self.global_step))

    def metadata(self) -> dict:
        return {
            'experiment_name': self.experiment_name,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'iter': self.iter,
            'timestamp': io_utils.timestamp_now_isoformat(timezone='America/Los_Angeles')
        }

    def try_save_checkpoint(self):
        if self.global_step % self.save_every == 0:
            metadata_to_save = self.metadata()
            saved_filename = save_checkpoint(save_dir=self.save_dir, pytorch_model=self.model, optimizer=self.optimizer, metadata=metadata_to_save)
            self.logger.info('Saved {}'.format(saved_filename))
            return True
        return False

    def train(self):
        if self.max_epochs is None or self.max_epochs == 0:
            max_epochs = int(1e22)
        else:
            max_epochs = self.max_epochs

        for i_epoch in range(max_epochs):
            self.epoch = i_epoch

            bottleneck_detector = BottleneckDetector(name='IO', logger=self.logger, check_every=20, threshold_seconds=0.15)
            for i_iter, batch in enumerate(self.data_loader):
                bottleneck_detector.toc()

                self.optimizer.zero_grad()
                loss_all = compute_loss(pytorch_model=self.model, batch=batch, experiment_name=self.experiment_name)
                loss_all.backward()
                self.optimizer.step()

                self.global_step += 1
                self.iter = i_iter + 1
                self.logger.info('%08d, %03d, %07d, %.5f', self.global_step, self.epoch, self.iter, loss_all)

                self.try_save_checkpoint()
                bottleneck_detector.tic()
