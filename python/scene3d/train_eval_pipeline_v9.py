import argparse
import random
import io
import warnings
import gzip
import string
import scipy.misc
import portalocker
import pickle
import shutil
import glob
import time
import typing
from scene3d import depth_mesh_utils_cpp
import collections
import os
from os import path
import scipy
import scipy.io as sio

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch import optim
from torch.backends import cudnn

from scene3d import config
from scene3d import feat
from scene3d import voxel
from scene3d import io_utils
from scene3d import log
from scene3d import pbrs_utils
from scene3d import loss_fn
from scene3d import torch_utils
from scene3d import transforms
from scene3d.dataset import dataset_utils
from scene3d.dataset import v1
from scene3d.dataset import v2
from scene3d.dataset import v8
from scene3d.net import unet
from scene3d.net import unet_no_bn
from scene3d.net import unet_overhead
from multiprocessing.pool import ThreadPool
import itertools

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
    'overhead_features_02_all',
    'v8-multi_layer_depth',
    'v8-multi_layer_depth_aligned_background',
    'v8-multi_layer_depth_replicated_background',
    'v8-multi_layer_depth_aligned_background-unet_v1',
    'v8-multi_layer_depth_replicated_background-unet_v1',
    'v8-multi_layer_depth_aligned_background_multi_branch',
    'v8-multi_layer_depth_replicated_background_multi_branch',
    'v8-multi_layer_depth_aligned_background_multi_branch_32',
    'v8-multi_layer_depth_aligned_background_multi_branch_nolog',
    'v8-category_nyu40-1l',
    'v8-category_nyu40_merged_background-1l',
    'v8-category_nyu40_merged_background-2l',
    'v8-category_nyu40_merged_background-2l-solo',
    'v8-normals',
    'v8-normals-acos',
    'v8-normal_direction_volume',
    'v8-multi_layer_depth_multi_branch-from_rgbd',
    'v8-multi_layer_depth_multi_branch-from_d',
    'v8-multi_layer_depth_aligned_background_multi_branch-from_rgbd',
    'v8-multi_layer_depth_aligned_background_multi_branch-from_d',
    'v8-category_nyu40-1l-from_rgbd',
    'v8-category_nyu40-1l-from_d',
    'v8-category_nyu40_merged_background-2l-from_rgbd',
    'v8-category_nyu40_merged_background-2l-from_d',
    'overfit-v8-multi_layer_depth-unet_v1',
    'overfit-v8-multi_layer_depth-unet_v2',
    'v8-single_layer_depth',
    'v8-two_layer_depth',
    'v8-overhead_camera_pose',
    'v8-overhead_camera_pose_4params',
    'OVERHEAD_OTF_01',  # all features, predicted geometry
    'OVERHEAD_offline_01',  # all features, predicted geometry
    'OVERHEAD_offline_02',  # no semantics, predicted geometry
    'OVERHEAD_offline_03',  # no semantics, no depth.
]

available_models = [
    'unet_v0',
    'unet_v0_no_bn',
    'unet_v0_overhead',
    'unet_v1_overhead',
    'unet_v1',
    'unet_v2',
    'unet_v2_regression',
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
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('overhead_features', 'multi_layer_overhead_depth'))
    elif experiment_name == 'overhead_features_02_all':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('overhead_features_v2', 'multi_layer_overhead_depth'))
    elif experiment_name == 'OVERHEAD_OTF_01':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'overhead_camera_pose_4params', 'camera_filename', 'multi_layer_overhead_depth'))
    elif experiment_name == 'OVERHEAD_offline_01':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'overhead_features_v3', 'camera_filename', 'multi_layer_overhead_depth'))
    elif experiment_name == 'OVERHEAD_offline_02':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'overhead_features_v3', 'camera_filename', 'multi_layer_overhead_depth'))
    elif experiment_name == 'OVERHEAD_offline_03':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'overhead_features_v3', 'camera_filename', 'multi_layer_overhead_depth'))
    elif experiment_name.startswith('overhead-features-eval-01'):
        """This is not actually one of the available experiments, but this has RGB images for evaluation or visualization mode. This can be deleted later. This is a preliminary experiment anyway.
        """
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255,
                                     fields=('overhead_features', 'multi_layer_depth', 'multi_layer_overhead_depth', 'rgb'))
    elif experiment_name == 'v8-multi_layer_depth':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth'))
    elif experiment_name == 'v8-multi_layer_depth_aligned_background':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_aligned_background'))
    elif experiment_name == 'v8-multi_layer_depth_replicated_background':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_replicated_background'))
    elif experiment_name == 'v8-multi_layer_depth_aligned_background-unet_v1':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_aligned_background'))
    elif experiment_name == 'v8-multi_layer_depth_replicated_background-unet_v1':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_replicated_background'))
    elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_aligned_background'))
    elif experiment_name == 'v8-multi_layer_depth_replicated_background_multi_branch':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_replicated_background'))
    elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch_32':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_aligned_background'))
    elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch_nolog':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_aligned_background'))
    elif experiment_name == 'v8-category_nyu40-1l':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'category_nyu40'))
    elif experiment_name == 'v8-category_nyu40_merged_background-1l':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'category_nyu40_merged_background'))
    elif experiment_name == 'v8-category_nyu40_merged_background-2l':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'category_nyu40_merged_background'))
    elif experiment_name == 'v8-category_nyu40_merged_background-2l-solo':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'category_nyu40_merged_background_replicated'))
    elif experiment_name == 'v8-normals':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'normals'))
    elif experiment_name == 'v8-normals-acos':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'normals'))
    elif experiment_name == 'v8-normal_direction_volume':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'normal_direction_volume'))
    elif experiment_name == 'v8-multi_layer_depth_multi_branch-from_rgbd':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_and_input_depth'))
    elif experiment_name == 'v8-multi_layer_depth_multi_branch-from_d':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_and_input_depth'))
    elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch-from_rgbd':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_aligned_background_and_input_depth'))
    elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch-from_d':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_aligned_background_and_input_depth'))
    elif experiment_name == 'v8-category_nyu40-1l-from_rgbd':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'category_nyu40', 'input_depth'))
    elif experiment_name == 'v8-category_nyu40-1l-from_d':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'category_nyu40', 'input_depth'))
    elif experiment_name == 'v8-category_nyu40_merged_background-2l-from_rgbd':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'category_nyu40_merged_background', 'input_depth'))
    elif experiment_name == 'v8-category_nyu40_merged_background-2l-from_d':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'category_nyu40_merged_background', 'input_depth'))
    elif experiment_name == 'overfit-v8-multi_layer_depth-unet_v1':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth'))
    elif experiment_name == 'overfit-v8-multi_layer_depth-unet_v2':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth'))
    elif experiment_name == 'v8-single_layer_depth':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth'))
    elif experiment_name == 'v8-two_layer_depth':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_aligned_background'))
    elif experiment_name == 'v8-overhead_camera_pose':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'overhead_camera_pose_3params'))
    elif experiment_name == 'v8-overhead_camera_pose_4params':
        dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=first_n, rgb_scale=1.0 / 255, fields=('rgb', 'overhead_camera_pose_4params'))
    else:
        raise NotImplementedError()

    return dataset


def get_pytorch_model_and_optimizer(model_name: str, experiment_name: str) -> typing.Tuple[nn.Module, optim.Optimizer, nn.Module]:
    frozen_model = None

    # This is just a default learning rate. It will be overridden by --learning_rate.
    learning_rate = 0.0005

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
    elif model_name == 'unet_v1_overhead':
        if experiment_name == 'overhead_features_02_all':
            model = unet_overhead.Unet1(in_channels=117, out_channels=1)
        elif experiment_name == 'OVERHEAD_OTF_01':
            frozen_model = [
                feat.Transformer(
                    depth_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00700000_008_0001768.pth'),
                    segmentation_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-category_nyu40_merged_background-2l/0/00800000_022_0016362.pth'),
                    device_id=list(range(torch.cuda.device_count()))[-1],  # use last n gpus
                ),
                # feat.Transformer(
                #     depth_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00700000_008_0001768.pth'),
                #     segmentation_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-category_nyu40_merged_background-2l/0/00800000_022_0016362.pth'),
                #     device_id=list(range(torch.cuda.device_count()))[-2],  # use last n gpus
                # ),
            ]
            model = unet_overhead.Unet1(in_channels=117, out_channels=1)
            learning_rate = 0.002
        elif experiment_name == 'OVERHEAD_offline_01':
            model = unet_overhead.Unet1(in_channels=117, out_channels=1)
            learning_rate = 0.002
        elif experiment_name == 'OVERHEAD_offline_02':
            # exclude semantic segmentation features
            model = unet_overhead.Unet1(in_channels=117 - 64, out_channels=1)
            learning_rate = 0.002
        elif experiment_name == 'OVERHEAD_offline_03':
            # exclude semantic segmentation and depth features
            model = unet_overhead.Unet1(in_channels=117 - 64 - 48, out_channels=1)
            learning_rate = 0.002
        else:
            raise NotImplementedError()
    elif model_name == 'unet_v1':
        if experiment_name == 'v8-multi_layer_depth':
            model = unet.Unet0(out_channels=4)
        elif experiment_name == 'v8-multi_layer_depth_aligned_background':
            model = unet.Unet0(out_channels=4)
        elif experiment_name == 'v8-multi_layer_depth_replicated_background':
            model = unet.Unet0(out_channels=4)
        elif experiment_name == 'v8-multi_layer_depth_aligned_background-unet_v1':
            model = unet.Unet1(out_channels=4)
        elif experiment_name == 'v8-multi_layer_depth_replicated_background-unet_v1':
            model = unet.Unet1(out_channels=4)
        elif experiment_name == 'v8-category_nyu40-1l':
            model = unet.Unet1(out_channels=40)  # category 34 is wall=background.
        elif experiment_name == 'v8-category_nyu40_merged_background-1l':
            model = unet.Unet1(out_channels=40)  # category 34 is wall=background.
        elif experiment_name == 'v8-category_nyu40_merged_background-2l':
            model = unet.Unet1(out_channels=80)  # two layer segmentation
        elif experiment_name == 'v8-category_nyu40_merged_background-2l-solo':
            model = unet.Unet1(out_channels=40)  # layer 2 segmentation
        elif experiment_name == 'v8-normals':
            model = unet.Unet1(out_channels=3)
        elif experiment_name == 'v8-normals-acos':
            model = unet.Unet1(out_channels=3)
        elif experiment_name == 'v8-normal_direction_volume':
            model = unet.Unet1(out_channels=1)
        elif experiment_name == 'v8-category_nyu40-1l-from_rgbd':
            model = unet.Unet1(out_channels=40, in_channels=4)  # category 34 is wall=background.
        elif experiment_name == 'v8-category_nyu40-1l-from_d':
            model = unet.Unet1(out_channels=40, in_channels=1)  # category 34 is wall=background.
        elif experiment_name == 'v8-category_nyu40_merged_background-2l-from_rgbd':
            model = unet.Unet1(out_channels=80, in_channels=4)  # category 34 is wall=background.
        elif experiment_name == 'v8-category_nyu40_merged_background-2l-from_d':
            model = unet.Unet1(out_channels=80, in_channels=1)  # category 34 is wall=background.
        elif experiment_name == 'overfit-v8-multi_layer_depth-unet_v1':
            model = unet.Unet1(out_channels=4)
        else:
            raise NotImplementedError()
    elif model_name == 'unet_v2':
        if experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch':
            model = unet.Unet2(out_channels=4)
        elif experiment_name == 'v8-multi_layer_depth_replicated_background_multi_branch':
            model = unet.Unet2(out_channels=4)
        elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch_32':
            model = unet.Unet2(out_channels=4, ch=(32, 64, 64, 256, 512), ch_branch=24)
        elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch_nolog':
            model = unet.Unet2(out_channels=4)
        elif experiment_name == 'v8-multi_layer_depth_multi_branch-from_rgbd':
            model = unet.Unet2(out_channels=4, in_channels=4)
        elif experiment_name == 'v8-multi_layer_depth_multi_branch-from_d':
            model = unet.Unet2(out_channels=4, in_channels=1)
        elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch-from_rgbd':
            model = unet.Unet2(out_channels=4, in_channels=4)
        elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch-from_d':
            model = unet.Unet2(out_channels=4, in_channels=1)
        elif experiment_name == 'overfit-v8-multi_layer_depth-unet_v2':
            model = unet.Unet2(out_channels=4)
        elif experiment_name == 'v8-single_layer_depth':
            model = unet.Unet2(out_channels=1)
        elif experiment_name == 'v8-two_layer_depth':
            model = unet.Unet2(out_channels=2)
        else:
            raise NotImplementedError()
    elif model_name == 'unet_v2_regression':
        if experiment_name == 'v8-overhead_camera_pose':
            frozen_model_checkpoint = path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00416000_007_0000297.pth')
            # frozen_model_checkpoint = '/home/daeyuns/scene3d_out/out/scene3d/v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00416000_007_0000297.pth'
            frozen_model, frozen_model_metadata = load_checkpoint_as_frozen_model(frozen_model_checkpoint)
            log.info('Frozen model loaded: {}'.format(frozen_model_metadata))
            model = unet.Unet2Regression(out_features=3)
        elif experiment_name == 'v8-overhead_camera_pose_4params':
            frozen_model_checkpoint = path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00416000_007_0000297.pth')
            # frozen_model_checkpoint = '/home/daeyuns/scene3d_out/out/scene3d/v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00416000_007_0000297.pth'
            frozen_model, frozen_model_metadata = load_checkpoint_as_frozen_model(frozen_model_checkpoint)
            log.info('Frozen model loaded: {}'.format(frozen_model_metadata))
            model = unet.Unet2Regression(out_features=4)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    log.info('Number of pytorch parameter tensors %d', len(params))
    optimizer = optim.Adam(params, lr=learning_rate)
    optimizer.zero_grad()

    return model, optimizer, frozen_model


def compute_loss(pytorch_model: nn.Module, batch, experiment_name: str, frozen_model) -> torch.Tensor:
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
        input_features = batch['overhead_features'][:, 3:].cuda()
        target_depth = batch['multi_layer_overhead_depth'][:, :1].cuda()
        pred = pytorch_model(input_features)
        loss_all = loss_fn.loss_calc_overhead_single_raw(pred, target_depth)
    elif experiment_name == 'overhead-features-01-log-l1-loss':
        example_name = batch['name']
        # Excluding RGB features. 64 channels
        input_features = batch['overhead_features'][:, 3:].cuda()
        target_depth = batch['multi_layer_overhead_depth'][:, :1].cuda()
        pred = pytorch_model(input_features)
        loss_all = loss_fn.loss_calc_overhead_single_log(pred, target_depth)

    elif experiment_name == 'overhead_features_02_all':
        example_name = batch['name']
        # 117 channels
        input_features = batch['overhead_features_v2'].cuda()
        target_depth = batch['multi_layer_overhead_depth'][:, :1].cuda()
        pred = pytorch_model(input_features)
        loss_all = loss_fn.loss_calc_overhead_single_raw(pred, target_depth)

    elif experiment_name == 'OVERHEAD_OTF_01':
        assert 'from_prev_stage' in batch
        example_name = batch['name']

        input_features = batch['from_prev_stage']['overhead_features']
        target_depth = batch['multi_layer_overhead_depth'][:, :1].contiguous().cuda(async=True)  # check how long this takes.
        loss_per_gpu = pytorch_model([input_features, target_depth])
        loss_all = loss_per_gpu

    elif experiment_name == 'OVERHEAD_offline_01':
        example_name = batch['name']

        input_features = batch['overhead_features_v3'].cuda()
        target_depth = batch['multi_layer_overhead_depth'][:, :1].contiguous().cuda(async=True)  # check how long this takes.
        loss_all = pytorch_model([input_features, target_depth])

    elif experiment_name == 'OVERHEAD_offline_02':
        example_name = batch['name']

        input_features_all = batch['overhead_features_v3']
        assert input_features_all.shape[1] == 117
        assert input_features_all.dim() == 4

        # 0: best guess depth
        # 1: frustum visibility map
        # 2-5: rgb features
        # 5-53 depth features
        # 53-117 semantic segmentation features

        # exclude the last 64 channels
        input_features = input_features_all[:, :-64].cuda()
        assert input_features.shape[1] == 117 - 64, input_features.shape[1]
        target_depth = batch['multi_layer_overhead_depth'][:, :1].contiguous().cuda(async=True)  # check how long this takes.
        loss_all = pytorch_model([input_features, target_depth])

    elif experiment_name == 'OVERHEAD_offline_03':
        example_name = batch['name']

        input_features_all = batch['overhead_features_v3']
        assert input_features_all.shape[1] == 117
        assert input_features_all.dim() == 4

        # 0: best guess depth
        # 1: frustum visibility map
        # 2-5: rgb features
        # 5-53 depth features
        # 53-117 semantic segmentation features

        # exclude the last 64+48 channels
        input_features = input_features_all[:, :-(64 + 48)].cuda()
        assert input_features.shape[1] == 117 - 64 - 48, input_features.shape[1]
        target_depth = batch['multi_layer_overhead_depth'][:, :1].contiguous().cuda(async=True)  # check how long this takes.
        loss_all = pytorch_model([input_features, target_depth])

    # v8
    elif experiment_name == 'v8-multi_layer_depth':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth'].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-multi_layer_depth_aligned_background':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth_aligned_background'].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-multi_layer_depth_replicated_background':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth_replicated_background'].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-multi_layer_depth_aligned_background-unet_v1':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth_aligned_background'].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-multi_layer_depth_replicated_background-unet_v1':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth_replicated_background'].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth_aligned_background'].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-multi_layer_depth_replicated_background_multi_branch':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth_replicated_background'].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch_32':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth_aligned_background'].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch_nolog':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth_aligned_background'].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=False)
    elif experiment_name == 'v8-category_nyu40-1l':
        in_rgb = batch['rgb'].cuda()
        target = batch['category_nyu40'][:, 0].cuda()  # (B, 240, 320)
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        assert pred.shape[1] == 40
        loss_all = loss_fn.loss_calc_classification(pred, target, ignore_index=65535)  # ignore empty
    elif experiment_name == 'v8-category_nyu40_merged_background-1l':
        in_rgb = batch['rgb'].cuda()
        target = batch['category_nyu40_merged_background'][:, 0].cuda()  # (B, 240, 320)
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        assert pred.shape[1] == 40
        loss_all = loss_fn.loss_calc_classification(pred, target, ignore_index=65535)  # ignore empty. background is merged to the wall category (34), which is not ignored.
    elif experiment_name == 'v8-category_nyu40_merged_background-2l':
        in_rgb = batch['rgb'].cuda()
        target1 = batch['category_nyu40_merged_background'][:, 0].cuda()  # (B, 240, 320)
        target2 = batch['category_nyu40_merged_background'][:, 2].cuda()  # (B, 240, 320)
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        assert pred.shape[1] == 80
        loss1 = loss_fn.loss_calc_classification(pred[:, :40], target1, ignore_index=65535)  # ignore empty. background is merged to the wall category (34), which is not ignored.
        loss2 = loss_fn.loss_calc_classification(pred[:, 40:], target2, ignore_index=65535)  # ignore empty. background is ignored
        loss_all = (loss1 + loss2) / 2
    elif experiment_name == 'v8-category_nyu40_merged_background-2l-solo':
        in_rgb = batch['rgb'].cuda()
        target = batch['category_nyu40_merged_background_replicated'][:, 2].cuda()  # (B, 240, 320)
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        assert pred.shape[1] == 40
        loss_all = loss_fn.loss_calc_classification(pred, target, ignore_index=65535)  # ignore empty. background is merged to the wall category (34), which is not ignored.
    elif experiment_name == 'v8-normals':
        in_rgb = batch['rgb'].cuda()
        target = batch['normals'].cuda()  # (B, 3, 240, 320)
        pred = pytorch_model(in_rgb)  # (B, 3, 240, 320)
        loss_all = loss_fn.compute_masked_surface_normal_loss(pred, target=target, use_inverse_cosine=False)
    elif experiment_name == 'v8-normals-acos':
        in_rgb = batch['rgb'].cuda()
        target = batch['normals'].cuda()  # (B, 3, 240, 320)
        pred = pytorch_model(in_rgb)  # (B, 3, 240, 320)
        loss_all = loss_fn.compute_masked_surface_normal_loss(pred, target=target, use_inverse_cosine=True)
    elif experiment_name == 'v8-normal_direction_volume':
        in_rgb = batch['rgb'].cuda()
        target = batch['normal_direction_volume'].cuda()  # (B, 1, 240, 320)
        pred = pytorch_model(in_rgb)  # (B, 1, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=False)
    elif experiment_name == 'v8-multi_layer_depth_multi_branch-from_rgbd':
        rgb = batch['rgb']
        d_ldi = batch['multi_layer_depth_and_input_depth']
        assert d_ldi.shape[1] == 5
        depth = d_ldi[:, :1]
        in_rgbd = torch.cat([rgb, depth], dim=1).cuda()
        target = d_ldi[:, 1:].cuda()
        assert in_rgbd.shape[1] == 4
        assert target.shape[1] == 4
        pred = pytorch_model(in_rgbd)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-multi_layer_depth_multi_branch-from_d':
        d_ldi = batch['multi_layer_depth_and_input_depth'].cuda()
        assert d_ldi.shape[1] == 5
        in_depth = d_ldi[:, :1]
        target = d_ldi[:, 1:]
        assert target.shape[1] == 4
        pred = pytorch_model(in_depth)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch-from_rgbd':
        rgb = batch['rgb']
        d_ldi = batch['multi_layer_depth_aligned_background_and_input_depth']
        assert d_ldi.shape[1] == 5
        depth = d_ldi[:, :1]  # (B, 1, H, W)
        in_rgbd = torch.cat([rgb, depth], dim=1).cuda()
        target = d_ldi[:, 1:].cuda()  # (B, 4, H, W)
        assert target.shape[1] == 4
        assert in_rgbd.shape[1] == 4
        pred = pytorch_model(in_rgbd)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-multi_layer_depth_aligned_background_multi_branch-from_d':
        d_ldi = batch['multi_layer_depth_aligned_background_and_input_depth'].cuda()
        assert d_ldi.shape[1] == 5
        in_depth = d_ldi[:, :1]  # (B, 1, H, W)
        target = d_ldi[:, 1:]  # (B, 4, H, W)
        assert target.shape[1] == 4
        pred = pytorch_model(in_depth)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-category_nyu40-1l-from_rgbd':
        rgb = batch['rgb']
        depth = batch['input_depth']
        in_rgbd = torch.cat([rgb, depth], dim=1).cuda()
        assert in_rgbd.shape[1] == 4
        target = batch['category_nyu40'][:, 0].cuda()  # (B, 240, 320)
        pred = pytorch_model(in_rgbd)  # (B, C, 240, 320)
        assert pred.shape[1] == 40
        loss_all = loss_fn.loss_calc_classification(pred, target, ignore_index=65535)  # ignore empty
    elif experiment_name == 'v8-category_nyu40-1l-from_d':
        in_depth = batch['input_depth'].cuda()
        assert in_depth.shape[1] == 1
        target = batch['category_nyu40'][:, 0].cuda()  # (B, 240, 320)
        pred = pytorch_model(in_depth)  # (B, C, 240, 320)
        assert pred.shape[1] == 40
        loss_all = loss_fn.loss_calc_classification(pred, target, ignore_index=65535)  # ignore empty
    elif experiment_name == 'v8-category_nyu40_merged_background-2l-from_rgbd':
        rgb = batch['rgb']
        depth = batch['input_depth']
        in_rgbd = torch.cat([rgb, depth], dim=1).cuda()
        assert in_rgbd.shape[1] == 4
        target1 = batch['category_nyu40_merged_background'][:, 0].cuda()  # (B, 240, 320)
        target2 = batch['category_nyu40_merged_background'][:, 2].cuda()  # (B, 240, 320)
        pred = pytorch_model(in_rgbd)  # (B, C, 240, 320)
        assert pred.shape[1] == 80
        loss1 = loss_fn.loss_calc_classification(pred[:, :40], target1, ignore_index=65535)  # ignore empty. background is merged to the wall category (34), which is not ignored.
        loss2 = loss_fn.loss_calc_classification(pred[:, 40:], target2, ignore_index=65535)  # ignore empty. background is ignored
        loss_all = (loss1 + loss2) / 2
    elif experiment_name == 'v8-category_nyu40_merged_background-2l-from_d':
        in_depth = batch['input_depth'].cuda()
        assert in_depth.shape[1] == 1
        target1 = batch['category_nyu40_merged_background'][:, 0].cuda()  # (B, 240, 320)
        target2 = batch['category_nyu40_merged_background'][:, 2].cuda()  # (B, 240, 320)
        pred = pytorch_model(in_depth)  # (B, C, 240, 320)
        assert pred.shape[1] == 80
        loss1 = loss_fn.loss_calc_classification(pred[:, :40], target1, ignore_index=65535)  # ignore empty. background is merged to the wall category (34), which is not ignored.
        loss2 = loss_fn.loss_calc_classification(pred[:, 40:], target2, ignore_index=65535)  # ignore empty. background is ignored
        loss_all = (loss1 + loss2) / 2
    elif experiment_name == 'overfit-v8-multi_layer_depth-unet_v1':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth'].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'overfit-v8-multi_layer_depth-unet_v2':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth'].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-single_layer_depth':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth'][:, :1].cuda()  # (B, 1, 240, 320)
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-two_layer_depth':
        in_rgb = batch['rgb'].cuda()
        target = batch['multi_layer_depth_aligned_background'][:, (0, 3)].cuda()
        pred = pytorch_model(in_rgb)  # (B, C, 240, 320)
        loss_all = loss_fn.compute_masked_smooth_l1_loss(pred=pred, target=target, apply_log_to_target=True)
    elif experiment_name == 'v8-overhead_camera_pose':
        assert frozen_model is not None
        in_rgb = batch['rgb'].cuda()
        target = batch['overhead_camera_pose_3params'].cuda()

        # (B, 48, 240, 320),  (B, 768, 15, 20)
        features, encoding = unet.get_feature_map_output_v2(frozen_model, in_rgb)

        pred = pytorch_model((features, encoding))
        assert target.shape == pred.shape, (target, pred)

        loss_translation = ((target[:, 0] - pred[:, 0]) ** 2 + (target[:, 1] - pred[:, 1]) ** 2).sqrt().mean()  # mean of tensor of shape (B)
        loss_scale = (target[:, 2] - pred[:, 2]).abs().mean()
        loss_all = loss_translation + loss_scale
    elif experiment_name == 'v8-overhead_camera_pose_4params':
        assert frozen_model is not None
        in_rgb = batch['rgb'].cuda()
        target = batch['overhead_camera_pose_4params'].cuda()

        # (B, 48, 240, 320),  (B, 768, 15, 20)
        features, encoding = unet.get_feature_map_output_v2(frozen_model, in_rgb)

        pred = pytorch_model((features, encoding))
        assert target.shape == pred.shape, (target, pred)

        loss_translation = ((target[:, 0] - pred[:, 0]) ** 2 + (target[:, 1] - pred[:, 1]) ** 2).sqrt().mean()  # mean of tensor of shape (B)
        loss_scale = (target[:, 2] - pred[:, 2]).abs().mean()
        loss_theta = (target[:, 3] - pred[:, 3]).abs().mean()
        loss_all = loss_translation + loss_scale + loss_theta
    else:
        raise NotImplementedError()

    return loss_all


def get_output_and_target(pytorch_model: nn.Module, batch, experiment_name: str, frozen_model: nn.Module = None) -> dict:
    if experiment_name == 'v8-overhead_camera_pose':
        assert frozen_model is not None
        in_rgb = batch['rgb'].cuda()
        target = batch['overhead_camera_pose_3params'].cuda()

        # (B, 48, 240, 320),  (B, 768, 15, 20)
        features, encoding = unet.get_feature_map_output_v2(frozen_model, in_rgb)

        pred = pytorch_model((features, encoding))

    else:
        raise NotImplementedError()

    return {
        'out': pred,
        'target': target,
    }


def load_checkpoint(filename, use_cpu=False) -> typing.Tuple[nn.Module, optim.Optimizer, dict, nn.Module]:
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
    pytorch_model, optimizer, frozen_model = get_pytorch_model_and_optimizer(
        model_name=metadata_dict['model_name'],
        experiment_name=metadata_dict['experiment_name'],
    )
    pytorch_model.load_state_dict(loaded_dict['model_state_dict'])
    optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

    return pytorch_model, optimizer, metadata_dict, frozen_model


def load_checkpoint_as_frozen_model(filename, use_cpu=False):
    pytorch_model, optimizer, metadata_dict, _ = load_checkpoint(filename, use_cpu=use_cpu)
    del optimizer

    for item in pytorch_model.parameters():
        item.requires_grad = False

    pytorch_model.eval()

    return pytorch_model, metadata_dict


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

    if isinstance(pytorch_model, nn.DataParallel):
        model_state_dict = pytorch_model.module.state_dict()
    else:
        model_state_dict = pytorch_model.state_dict()

    saved_dict = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata,
    }

    with open(save_filename, 'wb') as f:
        torch.save(saved_dict, f)

    return save_filename


class BottleneckDetector(object):
    """
    Used to detect IO bottleneck and show a warning message.
    """

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
            if self.delay_count > 0.5 * self.count or np.sum(self.delay_times) > 4:
                self.logger.warning('{} bottleneck detected: {} out of {}.  Total delay: {:.3f}'.format(self.name, self.delay_count, self.count, np.sum(self.delay_times)))
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
        self.load_checkpoint = path.expanduser(args.load_checkpoint)
        self.save_dir = path.expanduser(args.save_dir)
        self.num_data_workers = args.num_data_workers
        self.batch_size = args.batch_size
        self.use_cpu = args.use_cpu
        self.log_filename = path.join(self.save_dir, '{}_{}.log'.format(self.experiment_name, self.model_name))

        self.logger = log.make_logger('trainer', level=log.DEBUG)
        log.add_stream_handler(self.logger, level=log.INFO)
        log.add_file_handler(self.logger, filename=self.log_filename, level=log.DEBUG)

        if self.load_checkpoint == 'most_recent':
            saved_checkpoints = sorted(glob.glob(path.join(self.save_dir, '*.pth')))  # sorting is important
            self.logger.info('\n' + '\n'.join(saved_checkpoints))
            assert len(saved_checkpoints) > 0
            self.load_checkpoint = saved_checkpoints[-1]
            assert int(path.basename(self.load_checkpoint).split('_')[0]) != 0
            self.logger.info('Using most recent checkpoint: '.format(self.load_checkpoint))
            assert path.isfile(self.load_checkpoint)

        elif self.load_checkpoint and not self.load_checkpoint.startswith('/'):
            # If a filename is given, assume it's in the save directory.
            self.load_checkpoint = path.join(self.save_dir, self.load_checkpoint)
            assert path.isfile(self.load_checkpoint)

        self.logger.info('Initializing Trainer:\n{}'.format(args))

        if not self.use_cpu:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.logger.info('cudnn version: {}'.format(cudnn.version()))
            assert torch.cuda.is_available()

        self.dataset = get_dataset(experiment_name=self.experiment_name, split_name='train')
        self.logger.info('Number of examples: %d', len(self.dataset))

        self.use_subset = 'overfit-' in self.experiment_name

        if self.use_subset:
            sampler = torch.utils.data.SubsetRandomSampler(np.arange(7))
            shuffle = None
            self.logger.info('Debug mode. Using subset.')
        else:
            sampler = None
            shuffle = True

        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_data_workers, shuffle=shuffle, drop_last=True, pin_memory=True, sampler=sampler)
        self.logger.info('Initialized data loader.')

        self.global_step = 0  # Total number of steps. Preserved across checkpoints.
        self.epoch = 0
        self.iter = 0  # Step within the current epoch.
        self.loaded_metadata = None

        if 'OVERHEAD_OTF' in self.experiment_name:
            self.training_mode = 'multi_stage'
            raise RuntimeError('This script is not ready for this yet.')  # TODO
        else:
            self.training_mode = 'end_to_end'

        if self.load_checkpoint:
            self.model, self.optimizer, self.loaded_metadata, self.frozen_model = load_checkpoint(self.load_checkpoint, use_cpu=self.use_cpu)
            self.logger.info('Loaded metadata from {}:\n{}'.format(self.load_checkpoint, self.loaded_metadata))

            assert self.experiment_name == self.loaded_metadata['experiment_name']
            assert self.model_name == self.loaded_metadata['model_name']
            if self.batch_size != self.loaded_metadata['batch_size']:
                self.logger.info('batch_size changed from {} to {}'.format(self.loaded_metadata['batch_size'], self.batch_size))
            self.global_step = self.loaded_metadata['global_step']
            # Other attributes are "overwritten" by the values given in `args`.
        else:
            self.model, self.optimizer, self.frozen_model = get_pytorch_model_and_optimizer(model_name=self.model_name, experiment_name=self.experiment_name)
            # Immediately save a checkpoint at global step 0.

        if args.learning_rate > 0:
            torch_utils.set_optimizer_learning_rate(self.optimizer, learning_rate=args.learning_rate)
            self.logger.info('Set learning rate to {}'.format(args.learning_rate))

        device_ids = list(range(torch.cuda.device_count()))
        self.logger.info('device_ids: {}'.format(device_ids))
        self.model = nn.DataParallel(self.model, device_ids=device_ids)

        if not self.use_cpu:
            self.model = self.model.cuda()
            torch.set_default_tensor_type('torch.FloatTensor')  # Back to defaults.
        self.model.train()

        if not self.load_checkpoint:
            assert self.try_save_checkpoint()

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

        # pool = ThreadPool(20)
        # def save_example(overhead_features, i, name):
        #     ##### TODO
        #     out_filename = '/mnt/scratch2/daeyuns/overhead_features/pred/{}.bin'.format(name)
        #     io_utils.ensure_dir_exists(path.dirname(out_filename))
        #     out_arr = torch_utils.recursive_torch_to_numpy(overhead_features[i]).astype(np.float16)
        #     io_utils.save_array_compressed(out_filename, out_arr)

        self.logger.info('Training mode: {}'.format(self.training_mode))

        for i_epoch in range(max_epochs):
            self.epoch = i_epoch

            io_bottleneck_detector = BottleneckDetector(name='IO', logger=self.logger, check_every=20, threshold_seconds=0.15)

            if self.training_mode == 'end_to_end':
                for i_iter, batch in enumerate(self.data_loader):
                    io_bottleneck_detector.toc()

                    self.optimizer.zero_grad()
                    loss_all = compute_loss(pytorch_model=self.model, batch=batch, experiment_name=self.experiment_name, frozen_model=self.frozen_model)

                    if loss_all.dim() > 0:
                        # https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152/2
                        # Has the same effect as sum.
                        loss_all.backward(torch.ones_like(loss_all))
                    else:
                        loss_all.backward()

                    self.optimizer.step()

                    self.global_step += 1
                    self.iter = i_iter + 1
                    self.logger.info('%08d, %03d, %07d, %.5f', self.global_step, self.epoch, self.iter, loss_all.mean().item())

                    self.try_save_checkpoint()
                    io_bottleneck_detector.tic()
            elif self.training_mode == 'multi_stage':
                it = enumerate(self.data_loader)
                assert isinstance(self.frozen_model[0], feat.Transformer)
                i_iter, batch = next(it)

                with torch.cuda.device(unet_overhead.device_ids[0]):
                    start_end_indices = dataset_utils.divide_start_end_indices(self.batch_size, num_chunks=len(self.frozen_model))

                    for item_i, item in enumerate(self.frozen_model):
                        item.prefetch_batch_async(batch, start_end_indices=start_end_indices[item_i], target_device_id=unet_overhead.device_ids[0], options={'use_gt_geometry': False})

                    for next_i_iter, next_batch in it:
                        io_bottleneck_detector.toc()
                        # Current iteration: i_iter, batch

                        # `batch` is the current batch.
                        assert 'from_prev_stage' not in batch
                        # log.info('popping')
                        transformer_feat_list = []
                        transformer_name_list = []
                        for item in self.frozen_model:
                            transformer_feat, _, transformer_names = item.pop_batch(target_device_id=unet_overhead.device_ids[0])
                            transformer_feat_list.append(transformer_feat)
                            transformer_name_list.extend(transformer_names)

                        overhead_features = torch.cat(transformer_feat_list, dim=0)
                        batch['from_prev_stage'] = {
                            'overhead_features': overhead_features.cuda(async=True)  # blocks until available.
                        }

                        assert len(transformer_name_list) == len(batch['name'])
                        assert len(overhead_features) == len(batch['name'])
                        assert self.batch_size == len(batch['name'])
                        for bi in range(self.batch_size):
                            name0 = transformer_name_list[bi]
                            name1 = batch['name'][bi]
                            assert name0 == name1, (name0, name1)

                        # log.info('request async')
                        for item_i, item in enumerate(self.frozen_model):
                            item.prefetch_batch_async(next_batch, start_end_indices=start_end_indices[item_i], target_device_id=unet_overhead.device_ids[0], options={'use_gt_geometry': False})

                        # log.info('zero grad')
                        self.optimizer.zero_grad()
                        # log.info('computing loss')
                        loss_all = compute_loss(pytorch_model=self.model, batch=batch, experiment_name=self.experiment_name, frozen_model=None)
                        # log.info('computing backward')
                        loss_all.backward(torch.ones_like(loss_all).cuda(unet_overhead.device_ids[-1]))
                        # log.info('computing step')
                        self.optimizer.step()

                        self.global_step += 1
                        self.iter = i_iter + 1
                        self.logger.info('%08d, %03d, %07d, %.5f', self.global_step, self.epoch, self.iter, loss_all.mean().item())

                        # log.info('try save')
                        self.try_save_checkpoint()

                        i_iter = next_i_iter
                        batch = next_batch
                        io_bottleneck_detector.tic()
                        # log.info('end of iteration')


            else:
                raise RuntimeError('Unrecognized training mode: {}'.format(self.training_mode))


def surface_normal_eval(checkpoint_filename, split_name='test', num_examples=1000, random_seed=0, use_cpu=False, visualize=False):
    dataset = v8.MultiLayerDepth(split=split_name, subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=('rgb', 'normals'))

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model, _, loaded_metadata, _ = load_checkpoint(checkpoint_filename, use_cpu=use_cpu)

    print(loaded_metadata)

    # Sanity check.
    assert 'normals' in loaded_metadata['experiment_name']

    model = model.eval()
    model = model.cuda()

    indices = np.arange(len(dataset))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    all_eval_out = []

    for i in range(num_examples):
        index = indices[i]
        in_rgb = torch.Tensor(dataset[index]['rgb'][None]).cuda()
        target = torch.Tensor(dataset[index]['normals'][None]).cuda()

        pred = model(in_rgb)

        # error can be nan, if the scene has no objects
        eval_out = loss_fn.eval_mode_compute_masked_surface_normal_error(pred, target, return_pred_normalized=visualize, return_error_map=visualize)

        if visualize:
            import matplotlib.pyplot as pt
            pt.figure(figsize=(18, 3))

            pt.subplot(1, 4, 1)
            pt.imshow(v8.undo_rgb_whitening(in_rgb).squeeze().transpose(1, 2, 0))
            pt.axis('off')

            pt.subplot(1, 4, 2)
            pt.imshow(eval_out['pred_normalized'].squeeze().transpose(1, 2, 0) * 0.5 + 0.5)
            pt.axis('off')

            pt.subplot(1, 4, 3)
            target_viz = torch_utils.recursive_torch_to_numpy(target).squeeze().transpose(1, 2, 0) * 0.5 + 0.5
            target_viz[np.isnan(target_viz)] = 0
            pt.imshow(target_viz)
            pt.axis('off')

            pt.subplot(1, 4, 4)
            pt.imshow(eval_out['error_map'].squeeze())
            pt.colorbar()
            pt.clim(0, 3.1415)
            pt.set_cmap('Reds')
            pt.axis('off')

            pt.show()

        all_eval_out.append(eval_out)

    error = np.array([item['error'] for item in all_eval_out])

    # 1-d array of all error values. Can contain nan values.
    return error


def semantic_segmentation_from_raw_prediction(pred):
    pred_np = torch_utils.recursive_torch_to_numpy(pred)  # (B, 40, h, w)
    pred_np_argmax = np.argmax(pred_np, axis=1).astype(np.uint8)
    return pred_np_argmax


def segment_predicted_depth(pred_depth, pred_nyu40):
    """
    TODO: use last-exit segmentation.
    :param pred_depth: Background must be aligned.
    :param pred_nyu40: np.ndarray of shape (B, H, W) and type np.uint8
    :return:
    """
    assert pred_depth.ndim == 4
    assert pred_nyu40.ndim == 3
    assert pred_depth.shape[1] == 4
    assert pred_nyu40.dtype == np.uint8

    ret = pred_depth.copy()
    ret.transpose(1, 0, 2, 3)[:3, pred_nyu40 == 34] = np.nan

    return ret


def traditional_depth_from_aligned_multi_layer_depth(aligned_and_segmented_depth):
    assert aligned_and_segmented_depth.ndim == 4
    assert aligned_and_segmented_depth.shape[1] == 4

    ret = aligned_and_segmented_depth.copy()

    mask = ~np.isfinite(aligned_and_segmented_depth[:, 0])
    ret[:, 0][mask] = ret[:, 3][mask]

    return ret[:, 0]


def reduce_list_of_dicts_of_lists(ld):
    keys = list(ld[0].keys())
    ret = collections.defaultdict(list)
    for item in ld:
        for k in keys:
            ret[k].extend(item[k])
    ret = dict(ret)
    return ret


class Evaluation():
    def __init__(self, model_metadata):
        self.pred = None
        self.target = None
        self.model_metadata = model_metadata

    def set_input(self, pred, batch):
        self.pred = pred
        self.batch = batch

    @staticmethod
    def _mean_of_finite_per_example(arr):
        """
        :param arr: (B, .., ...)
        :return: list of size B.
        """
        arr_2d = arr.reshape(arr.shape[0], -1)
        mask = torch.isfinite(arr_2d)
        ret = (arr_2d.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1, dtype=torch.float32)).tolist()
        return ret

    @staticmethod
    def _masked_iou_per_example(a, b, ignore_mask):
        """
        :param a: (B, .., ...)
        :param b: same as a
        :return: list of size B.
        """
        assert a.shape == b.shape
        assert ignore_mask.shape == b.shape
        a_2d = torch_utils.recursive_torch_to_numpy(a.reshape(a.shape[0], -1)).astype(np.bool)
        b_2d = torch_utils.recursive_torch_to_numpy(b.reshape(b.shape[0], -1)).astype(np.bool)
        mask_2d = torch_utils.recursive_torch_to_numpy(ignore_mask.reshape(ignore_mask.shape[0], -1)).astype(np.bool)

        intersection = a_2d & b_2d
        union = a_2d | b_2d
        intersection[mask_2d] = 0
        union[mask_2d] = 0

        return (intersection.sum(axis=1) / union.sum(axis=1)).tolist()

    def multi_layer_depth_aligned_background_l1(self):
        # l1 here means L1 error
        assert 'aligned_background' in self.model_metadata['experiment_name']
        target = self.batch['multi_layer_depth_aligned_background'].cuda()

        if 'nolog' in self.model_metadata['experiment_name']:
            l1_with_nan = (target - self.pred).abs()
        else:
            l1_with_nan = (target - loss_fn.undo_log_depth(self.pred)).abs()

        mask_visible_bg = torch.isnan(target[:, 1])  # (B, H, W)

        overall = self._mean_of_finite_per_example(l1_with_nan)
        objects = self._mean_of_finite_per_example(l1_with_nan[:, :3])
        background = self._mean_of_finite_per_example(l1_with_nan[:, 3])
        visible_objects = self._mean_of_finite_per_example(l1_with_nan[:, 0])
        invisible_objects = self._mean_of_finite_per_example(l1_with_nan[:, (1, 2)])
        instance_exit = self._mean_of_finite_per_example(l1_with_nan[:, 1])
        last_exit = self._mean_of_finite_per_example(l1_with_nan[:, 2])

        background_l1_with_nan = l1_with_nan[:, 3].clone()  # (B, H, W)
        background_l1_with_nan[mask_visible_bg] = np.nan  # ignore visible background error
        invisible_background = self._mean_of_finite_per_example(background_l1_with_nan)

        # overwrite and reuse same variable name
        background_l1_with_nan = l1_with_nan[:, 3].clone()  # (B, H, W)
        background_l1_with_nan[~mask_visible_bg] = np.nan  # ignore visible background error
        visible_background = self._mean_of_finite_per_example(background_l1_with_nan)

        # Swap visible background in channels 0 and 3.
        l1_with_nan[:, 0][mask_visible_bg], l1_with_nan[:, 3][mask_visible_bg] = l1_with_nan[:, 3][mask_visible_bg], l1_with_nan[:, 0][mask_visible_bg]

        visible_surfaces = self._mean_of_finite_per_example(l1_with_nan[:, 0])
        invisible_surfaces = self._mean_of_finite_per_example(l1_with_nan[:, 1:])

        return {
            'overall': overall,
            'objects': objects,
            'background': background,
            'visible_objects': visible_objects,
            'invisible_objects': invisible_objects,
            'visible_surfaces': visible_surfaces,
            'invisible_surfaces': invisible_surfaces,
            'visible_background': visible_background,
            'invisible_background': invisible_background,
            'instance_exit': instance_exit,
            'last_exit': last_exit,
        }

    def multi_layer_depth_unaligned_background_l1(self):
        assert self.model_metadata['experiment_name'] == 'v8-multi_layer_depth'
        target = self.batch['multi_layer_depth'].cuda()

        if 'nolog' in self.model_metadata['experiment_name']:
            # This shouldn't happen. we didn't do this experiment for this model.
            raise RuntimeError()
            # l1_with_nan = (target - self.pred).abs()
        else:
            # (B, 4, H, W)
            p = loss_fn.undo_log_depth(self.pred)
            l1_with_nan = (target - p).abs()

        overall = self._mean_of_finite_per_example(l1_with_nan)
        visible_surfaces = self._mean_of_finite_per_example(l1_with_nan[:, 0])
        invisible_surfaces = self._mean_of_finite_per_example(l1_with_nan[:, 1:])

        # Swap visible background in channels 0 and 3.
        # TODO: visualize and make sure swap worked.
        mask_visible_bg = torch.isnan(target[:, 1])
        l1_with_nan[:, 0][mask_visible_bg], l1_with_nan[:, 3][mask_visible_bg] = l1_with_nan[:, 3][mask_visible_bg], l1_with_nan[:, 0][mask_visible_bg]

        objects = self._mean_of_finite_per_example(l1_with_nan[:, :3])
        background = self._mean_of_finite_per_example(l1_with_nan[:, 3])

        background_l1_with_nan = l1_with_nan[:, 3].clone()  # (B, H, W)
        background_l1_with_nan[mask_visible_bg] = np.nan  # zero out visible background error
        invisible_background = self._mean_of_finite_per_example(background_l1_with_nan)

        # overwrite and reuse same variable name
        background_l1_with_nan = l1_with_nan[:, 3].clone()  # (B, H, W)
        background_l1_with_nan[~mask_visible_bg] = np.nan  # zero out visible background error
        visible_background = self._mean_of_finite_per_example(background_l1_with_nan)

        invisible_objects = self._mean_of_finite_per_example(l1_with_nan[:, (1, 2)])
        instance_exit = self._mean_of_finite_per_example(l1_with_nan[:, 1])
        last_exit = self._mean_of_finite_per_example(l1_with_nan[:, 2])

        return {
            'overall': overall,
            'objects': objects,
            'background': background,
            'visible_objects': self._mean_of_finite_per_example(l1_with_nan[:, 0]),
            'invisible_objects': invisible_objects,
            'visible_surfaces': visible_surfaces,
            'invisible_surfaces': invisible_surfaces,
            'visible_background': visible_background,
            'invisible_background': invisible_background,
            'instance_exit': instance_exit,
            'last_exit': last_exit,
        }

    def category_nyu40_merged_background_l2(self):
        # l2 here means two layers

        assert self.pred.shape[1] == 80
        assert 'category' in self.model_metadata['experiment_name']
        argmax_l1 = semantic_segmentation_from_raw_prediction(self.pred[:, :40])
        # argmax_l2 = semantic_segmentation_from_raw_prediction(self.pred[:, 40:])

        target = self.batch['category_nyu40_merged_background']
        target_l1 = torch_utils.recursive_torch_to_numpy(target[:, 2])

        ignored = (target_l1 == 65535) | (target_l1 == 33)  # (B, H, W)
        correct = (target_l1 == argmax_l1).astype(np.float32)
        correct[ignored] = np.nan

        accuracy = self._mean_of_finite_per_example(torch_utils.recursive_numpy_to_torch(correct))

        pred_foreground = argmax_l1 != 34
        target_foreground = target_l1 != 34

        foreground_iou = self._masked_iou_per_example(pred_foreground, target_foreground, ignore_mask=ignored)

        return {
            'layer1_accuracy': accuracy,
            'layer1_foreground_iou': foreground_iou,
        }


def save_mldepth_as_meshes(pred_segmented_depth, example, force=False):
    assert pred_segmented_depth.ndim == 3
    out_pred_filenames = []
    # out_gt_filenames = []
    for i in range(4):
        out_filename = path.join(config.default_out_root, 'v8_pred_depth_mesh/{}/pred_{}.ply'.format(example['name'], i))
        if i == 0:
            dd_factor = 7
        elif i == 1:
            dd_factor = 5
        elif i == 2:
            dd_factor = 2
        else:
            dd_factor = 7

        if force or not path.isfile(out_filename):
            depth_mesh_utils_cpp.depth_to_mesh(pred_segmented_depth[i], example['camera_filename'], camera_index=0, dd_factor=dd_factor, out_ply_filename=out_filename)
        out_pred_filenames.append(out_filename)

        # out_filename = '/data3/out/scene3d/v8_depth_mesh/{}_gt_{}.ply'.format(example['name'], i)
        # if not path.isfile(out_filename):
        #     depth_mesh_utils_cpp.depth_to_mesh(example['multi_layer_depth_aligned_background'][i], example['camera_filename'], camera_index=0, dd_factor=10, out_ply_filename=out_filename)
        # out_gt_filenames.append(out_filename)

    return {
        'pred': out_pred_filenames,
        # 'gt': out_gt_filenames,
    }


def save_mldepth_as_meshes_for_visualization(pred_segmented_depth, example, force=False):
    assert pred_segmented_depth.ndim == 3
    out_pred_filenames = []
    # out_gt_filenames = []
    for i in range(4):
        out_filename = '/mnt/ramdisk/vis_mesh/single.ply'

        if i == 0:
            dd_factor = 7
        elif i == 1:
            dd_factor = 5
        elif i == 2:
            dd_factor = 2
        else:
            dd_factor = 7

        if force or not path.isfile(out_filename):
            depth_mesh_utils_cpp.depth_to_mesh(pred_segmented_depth[i], example['camera_filename'], camera_index=0, dd_factor=dd_factor, out_ply_filename=out_filename)
        out_pred_filenames.append(out_filename)

        # out_filename = '/data3/out/scene3d/v8_depth_mesh/{}_gt_{}.ply'.format(example['name'], i)
        # if not path.isfile(out_filename):
        #     depth_mesh_utils_cpp.depth_to_mesh(example['multi_layer_depth_aligned_background'][i], example['camera_filename'], camera_index=0, dd_factor=10, out_ply_filename=out_filename)
        # out_gt_filenames.append(out_filename)

    return {
        'pred': out_pred_filenames,
        # 'gt': out_gt_filenames,
    }


def save_mldepth_as_meshes_realworld(pred_segmented_depth, out_dir_name):
    assert pred_segmented_depth.ndim == 3
    out_pred_filenames = []
    # out_gt_filenames = []

    ref_cam = [
        'P',
        0.0, 0.0, 0.0,  # position
        0.0, 0.0, -1.0,  # viewing dir
        0.0, 1.0, 0.0,  # up
        -0.00617793056641, 0.00617793056641, -0.00463344946349, 0.00463344946349, 0.01, 100  # intrinsics
    ]

    tmp_camera_name = '/mnt/ramdisk/cam_view.txt'
    with open(tmp_camera_name, 'w') as f:
        f.write(' '.join([str(item) for item in ref_cam]))

    for i in range(4):
        out_filename = path.join(config.default_out_root, out_dir_name, 'pred_{}.ply'.format(i))
        # if not path.isfile(out_filename):
        depth_mesh_utils_cpp.depth_to_mesh(pred_segmented_depth[i], tmp_camera_name, camera_index=0, dd_factor=10, out_ply_filename=out_filename)
        out_pred_filenames.append(out_filename)

        # out_filename = '/data3/out/scene3d/v8_depth_mesh/{}_gt_{}.ply'.format(example['name'], i)
        # if not path.isfile(out_filename):
        #     depth_mesh_utils_cpp.depth_to_mesh(example['multi_layer_depth_aligned_background'][i], example['camera_filename'], camera_index=0, dd_factor=10, out_ply_filename=out_filename)
        # out_gt_filenames.append(out_filename)

    return {
        'pred': out_pred_filenames,
        # 'gt': out_gt_filenames,
    }


def save_height_prediction_as_meshes(height_map_model_batch_out, hm_model, original_camera_filenames, example_names):
    import uuid

    default_overhead_camera_height = 7
    out_ply_filenames = []

    assert len(original_camera_filenames) == len(height_map_model_batch_out['pred_cam'])
    for i, row in enumerate(height_map_model_batch_out['pred_cam']):
        assert row.shape[0] == 4
        random_string = uuid.uuid4().hex
        new_cam_filename = path.join(hm_model.transformer.tmp_out_root, 'eval_world_ortho_cam_{}.txt'.format(random_string))
        x, y, scale, theta = row.tolist()
        if path.isfile(new_cam_filename):
            os.remove(new_cam_filename)

        with open(original_camera_filenames[i], 'r') as f:
            lines = f.readlines()
        items = lines[0].strip().split()
        ref_cam_components = items[:1] + [float(item) for item in items[1:]]

        house_id, camera_id = pbrs_utils.parse_house_and_camera_ids_from_string(example_names[i])

        floor_height = find_gt_floor_height(house_id=house_id, camera_id=camera_id)
        camera_height = default_overhead_camera_height + floor_height

        feat.make_overhead_camera_file(new_cam_filename, x, y, scale, theta, ref_cam=ref_cam_components, camera_height=camera_height)
        assert path.isfile(new_cam_filename)

        overhead_heightmap = height_map_model_batch_out['pred_height_map'][i].squeeze()
        overhead_depth = default_overhead_camera_height - overhead_heightmap

        out_filename_bg = path.join(config.default_out_root, 'v8_pred_depth_mesh/{}/overhead_bg.ply'.format(example_names[i]))  # TODO
        out_filename_fg = path.join(config.default_out_root, 'v8_pred_depth_mesh/{}/overhead_fg.ply'.format(example_names[i]))  # TODO

        overhead_depth_bg = overhead_depth.copy()
        overhead_depth_bg[overhead_heightmap > 0.01] = np.nan

        overhead_depth_fg = overhead_depth.copy()
        overhead_depth_fg[overhead_heightmap <= 0.01] = np.nan

        depth_mesh_utils_cpp.depth_to_mesh(overhead_depth_fg, camera_filename=new_cam_filename, camera_index=1, dd_factor=6, out_ply_filename=out_filename_fg)
        depth_mesh_utils_cpp.depth_to_mesh(overhead_depth_bg, camera_filename=new_cam_filename, camera_index=1, dd_factor=6, out_ply_filename=out_filename_bg)

        out_ply_filenames.append([out_filename_bg, out_filename_fg])  # list of lists.

        if path.isfile(new_cam_filename):
            os.remove(new_cam_filename)

    assert len(out_ply_filenames) == len(example_names)

    return out_ply_filenames


def save_height_mesh(overhead_heightmap, x, y, scale, theta, original_camera_filename, example_name):
    import uuid

    default_overhead_camera_height = 7
    out_ply_filenames = []

    random_string = uuid.uuid4().hex
    tmp_out_root = '/tmp/scene3d_transformer_cam'
    new_cam_filename = path.join(tmp_out_root, 'eval_world_ortho_cam_{}.txt'.format(random_string))
    if path.isfile(new_cam_filename):
        os.remove(new_cam_filename)

    with open(original_camera_filename, 'r') as f:
        lines = f.readlines()
    items = lines[0].strip().split()
    ref_cam_components = items[:1] + [float(item) for item in items[1:]]

    house_id, camera_id = pbrs_utils.parse_house_and_camera_ids_from_string(example_name)

    floor_height = find_gt_floor_height(house_id=house_id, camera_id=camera_id)
    camera_height = default_overhead_camera_height + floor_height

    feat.make_overhead_camera_file(new_cam_filename, x, y, scale, theta, ref_cam=ref_cam_components, camera_height=camera_height)
    assert path.isfile(new_cam_filename)

    # overhead_heightmap = height_map_model_batch_out['pred_height_map'][i].squeeze()
    overhead_depth = default_overhead_camera_height - overhead_heightmap

    out_dir = '/home/daeyun/mnt/v9_gt_overhead_mesh'

    out_filename_bg = path.join(out_dir, '{}/overhead_bg.ply'.format(example_name))
    out_filename_fg = path.join(out_dir, '{}/overhead_fg.ply'.format(example_name))

    overhead_depth_bg = overhead_depth.copy()
    overhead_depth_bg[overhead_heightmap > 0.01] = np.nan

    overhead_depth_fg = overhead_depth.copy()
    overhead_depth_fg[overhead_heightmap <= 0.01] = np.nan

    depth_mesh_utils_cpp.depth_to_mesh(overhead_depth_fg, camera_filename=new_cam_filename, camera_index=1, dd_factor=6, out_ply_filename=out_filename_fg)
    depth_mesh_utils_cpp.depth_to_mesh(overhead_depth_bg, camera_filename=new_cam_filename, camera_index=1, dd_factor=6, out_ply_filename=out_filename_bg)

    out_ply_filenames.append([out_filename_bg, out_filename_fg])  # list of lists.

    if path.isfile(new_cam_filename):
        os.remove(new_cam_filename)

    assert len(out_ply_filenames) == 1

    return out_ply_filenames


def save_height_map_output_batch(height_map_model_batch_out, example_names):
    pred = height_map_model_batch_out['pred_height_map']
    assert len(pred) == len(example_names)

    ret = []
    for i, name in enumerate(example_names):
        house_id, camera_id = pbrs_utils.parse_house_and_camera_ids_from_string(name)
        out_file = path.join(config.default_out_root, 'v8_pred/{}/{}/pred_height_map.bin'.format(house_id, camera_id))
        io_utils.ensure_dir_exists(path.dirname(out_file))
        io_utils.save_array_compressed(out_file, pred[i].squeeze())  # (300, 300)
        ret.append(out_file)

    return ret


def mesh_precision_recall_parallel(gt_pred_filename_pairs, sampling_density, thresholds):
    pool = ThreadPool(5)

    params = []
    for gt_filenames, pred_filenames in gt_pred_filename_pairs:
        params.append([gt_filenames, pred_filenames, sampling_density, thresholds])

    return pool.starmap(depth_mesh_utils_cpp.mesh_precision_recall, params)


def predict_cam_params(regression_model, feature_extrator_model, rgb_batch) -> np.ndarray:
    assert feature_extrator_model.__class__.__name__ == 'Unet2'
    assert regression_model.__class__.__name__ == 'Unet2Regression'
    assert not regression_model.training
    assert not feature_extrator_model.training

    device = next(feature_extrator_model.parameters()).device
    assert device == next(regression_model.parameters()).device

    with torch.cuda.device(device.index):
        in_rgb = rgb_batch.cuda(device)
        # (B, 48, 240, 320),  (B, 768, 15, 20)
        features, encoding = unet.get_feature_map_output_v2(feature_extrator_model, in_rgb)

        assert features.device == device
        assert encoding.device == device

        pred = regression_model((features, encoding))
        ret = torch_utils.recursive_torch_to_numpy(pred)
        return ret


class HeightMapModel(object):
    def __init__(self, checkpoint_filenames, device_id=0):
        self.checkpoint_filenames = checkpoint_filenames
        self.device_id = device_id
        assert 'pose_3param' in self.checkpoint_filenames
        assert 'overhead_height_map_model' in self.checkpoint_filenames

        with torch.cuda.device(self.device_id):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.regression_model, _, loaded_metadata, self.regression_feature_extractor_model = load_checkpoint(self.checkpoint_filenames['pose_3param'], use_cpu=False)
            for item in self.regression_model.parameters():
                item.requires_grad = False
            self.regression_model.eval()
            print(loaded_metadata)
            self.regression_model_metadata = loaded_metadata

            self.height_map_model, _, loaded_metadata, _ = load_checkpoint(self.checkpoint_filenames['overhead_height_map_model'], use_cpu=False)
            for item in self.height_map_model.parameters():
                item.requires_grad = False
            self.height_map_model.eval()
            print(loaded_metadata)
            self.height_map_model_metadata = loaded_metadata

            assert self.regression_model is not None
            assert self.regression_feature_extractor_model is not None

            self.transformer = feat.Transformer(
                depth_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00700000_008_0001768.pth'),
                segmentation_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-category_nyu40_merged_background-2l/0/00800000_022_0016362.pth'),
                device_id=self.device_id,
                num_workers=5,
                cam_param_regression_model=self.regression_model,
                cam_param_feature_extractor_model=self.regression_feature_extractor_model,
            )

    def predict_height_map_single(self, batch):
        features, cam, transformer_names = self.transformer._get_transformed_features(batch, [0, 1], use_gt_geometry=False)
        pred_height_map = self.height_map_model.get_output(torch.Tensor(features).cuda())

        return torch_utils.recursive_torch_to_numpy(pred_height_map).squeeze(), features.squeeze()

    def predict_height_map(self, batch, visualize_indices=None):
        input_batch = {
            'rgb': batch['rgb'],
            'name': batch['name'],
        }

        with torch.cuda.device(self.device_id):
            assert 'camera_filename' not in input_batch

            self.transformer.prefetch_batch_async(input_batch, start_end_indices=None, target_device_id=self.device_id, options={'use_gt_geometry': False})
            overhead_features, predicted_cam, transformer_names = self.transformer.pop_batch(target_device_id=1)
            assert batch['name'] == transformer_names
            assert overhead_features.shape[0] == len(batch['name'])

            pred_height_map = self.height_map_model.get_output(overhead_features.cuda())

            if visualize_indices:
                import matplotlib.pyplot as pt
                overhead_features_np = torch_utils.recursive_torch_to_numpy(overhead_features)

                for i in visualize_indices:
                    rgb_features = overhead_features_np[i, 2:5]
                    print(batch['name'][i])

                    pt.figure()
                    pt.title('Input RGB')
                    pt.imshow(v8.undo_rgb_whitening(torch_utils.recursive_torch_to_numpy(batch['rgb'][i])).transpose(1, 2, 0))
                    pt.axis('off')

                    pt.figure()
                    pt.title('Transformed RGB')
                    pt.imshow(v8.undo_rgb_whitening(rgb_features).transpose(1, 2, 0))
                    pt.colorbar()
                    pt.axis('off')

                    pt.figure()
                    pt.title('Best guess depth')
                    pt.imshow(overhead_features_np[i, 0])
                    pt.colorbar()
                    pt.axis('off')

                    # pt.figure()
                    # pt.imshow(overhead_features_np[i, 1])
                    # pt.colorbar()
                    # pt.axis('off')

                    fv = torch_utils.recursive_torch_to_numpy(batch['overhead_features_v3'][i, 1])
                    pt.figure()
                    pt.title('Frustum visibility map diff (GT-Pred)')
                    pt.imshow(fv - overhead_features_np[i, 1])
                    pt.colorbar()
                    pt.axis('off')

                    pt.figure(figsize=(14, 4))
                    pt.subplot(1, 3, 1)
                    pt.title('Predicted height map')
                    pred_height_i = pred_height_map[i].squeeze()
                    pt.imshow(pred_height_i)
                    pt.colorbar()
                    pt.axis('off')

                    pt.subplot(1, 3, 2)
                    pt.title('GT height map')
                    gt_height_i = torch_utils.recursive_torch_to_numpy(batch['multi_layer_overhead_depth'][i][0]).squeeze()
                    pt.imshow(gt_height_i)
                    pt.colorbar()
                    pt.axis('off')

                    pt.subplot(1, 3, 3)
                    pt.title('Error map')
                    pt.imshow(np.abs(gt_height_i - pred_height_i), cmap='Reds')
                    pt.clim(0, 1.5)
                    pt.colorbar()
                    pt.axis('off')

                    pt.show()

            pred_height_map_np = torch_utils.recursive_torch_to_numpy(pred_height_map)

            return {
                'pred_height_map': pred_height_map_np,
                'pred_cam': torch_utils.recursive_torch_to_numpy(predicted_cam),
            }


def find_gt_floor_height(house_id, camera_id):
    gt_mesh_filename1 = path.join(config.default_out_root, 'v8_gt_mesh/{}/{}/gt_bg.ply'.format(house_id, camera_id))
    gt_mesh_filename2 = path.join(config.default_out_root, 'v8_gt_mesh/{}/{}/gt_objects.ply'.format(house_id, camera_id))
    floor_filename = path.join(path.dirname(gt_mesh_filename1), 'floor.txt')
    if path.isfile(floor_filename):
        with open(floor_filename, 'r') as f:
            content = f.read()
            try:
                height = float(content.strip())
                return height
            except ValueError as ex:
                pass

    fv = io_utils.read_mesh(gt_mesh_filename1)
    ycoords = sorted(fv['v'][:, 1].tolist())
    height1 = np.median(ycoords[:int(max(len(ycoords) * 0.01, 100))]).item()

    fv = io_utils.read_mesh(gt_mesh_filename2)
    ycoords = sorted(fv['v'][:, 1].tolist())
    height2 = np.median(ycoords[:int(max(len(ycoords) * 0.01, 100))]).item()

    ret = min(height1, height2)

    with open(floor_filename, 'w') as f:
        f.write('{:.8f}'.format(ret))

    return ret


def find_gt_floor_height_v9(house_id, camera_id):
    gt_mesh_filename1 = path.join(config.default_out_root, 'v9_gt_mesh/{}/{}/gt_bg.ply'.format(house_id, camera_id))
    gt_mesh_filename2 = path.join(config.default_out_root, 'v9_gt_mesh/{}/{}/gt_objects.ply'.format(house_id, camera_id))
    floor_filename = path.join(path.dirname(gt_mesh_filename1), 'floor.txt')
    if path.isfile(floor_filename):
        with open(floor_filename, 'r') as f:
            content = f.read()
            try:
                height = float(content.strip())
                return height
            except ValueError as ex:
                pass

    fv = io_utils.read_mesh(gt_mesh_filename1)
    ycoords = sorted(fv['v'][:, 1].tolist())
    height1 = np.median(ycoords[:int(max(len(ycoords) * 0.01, 100))]).item()

    fv = io_utils.read_mesh(gt_mesh_filename2)
    ycoords = sorted(fv['v'][:, 1].tolist())
    height2 = np.median(ycoords[:int(max(len(ycoords) * 0.01, 100))]).item()

    ret = min(height1, height2)

    with open(floor_filename, 'w') as f:
        f.write('{:.8f}'.format(ret))

    return ret


def symlink_all_files_in_dir(src_dir, tgt_dir):
    files = glob.glob(path.join(src_dir, '*'))
    io_utils.ensure_dir_exists(tgt_dir)
    out_files = []
    for file in files:
        out_files.append(file)
        out_file = path.join(tgt_dir, path.basename(file))
        if path.islink(out_file):
            os.remove(out_file)
        elif path.exists(out_file):
            raise RuntimeError('file exists and is not a link: {}'.format(out_file))
        os.symlink(file, out_file)
    return out_files


class PRCurveEvaluation(object):
    def __init__(self, save_filename):
        self.save_filename = save_filename
        self.load()

        step = 0.01
        self.thresholds = np.array([0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04] + np.arange(0.05, 1 + step, step).tolist() + np.arange(1 + step * 2, 2 + step, step * 2).tolist())
        self.density = 2500

    def load(self):
        try:
            with portalocker.Lock(self.save_filename, mode='rb', timeout=10) as f:
                self.results = pickle.load(f)
        except FileNotFoundError as ex:
            log.info('File not found. Initializing an empty dict. {}'.format(self.save_filename))
            self.results = {}

    def key(self, pred_or_gt, target_type, source_list):
        assert pred_or_gt in ['pred', 'gt_depth']
        assert target_type in ['obj', 'background', 'both']
        if isinstance(source_list, str):
            source_list = [source_list]
        assert isinstance(source_list, (tuple, list)), source_list
        return ','.join([pred_or_gt.strip(), target_type.strip(), '+'.join(sorted(source_list))])

    def is_mesh_empty(self, mesh_filename):
        if isinstance(mesh_filename, (list, tuple)):
            return np.all([self.is_mesh_empty(item) for item in mesh_filename])
        return path.getsize(mesh_filename) < 190 or not path.isfile(mesh_filename)

    def mesh_precision_recall(self, gt_mesh_filenames, pred_mesh_filenames):
        assert isinstance(gt_mesh_filenames, (list, tuple))
        assert isinstance(pred_mesh_filenames, (list, tuple))
        assert gt_mesh_filenames
        assert pred_mesh_filenames

        if self.is_mesh_empty(gt_mesh_filenames):
            log.error('GT mesh is empty. This should not happen.    {}'.format(gt_mesh_filenames))
            raise RuntimeError()
        if self.is_mesh_empty(pred_mesh_filenames):
            log.warn('Nothing was predicted. Precision=1, Recall=0')
            return {'p': 1.0, 'r': 0.0, }  # for all thresholds.

        precision, recall = depth_mesh_utils_cpp.mesh_precision_recall(gt_mesh_filenames, pred_mesh_filenames, self.density, thresholds=self.thresholds)
        return {'p': precision, 'r': recall}

    def run_if_not_exists(self, example_name, key, gt_mesh_filenames, pred_mesh_filenames, force=False):
        assert isinstance(gt_mesh_filenames, (list, tuple))
        assert isinstance(pred_mesh_filenames, (list, tuple))
        assert gt_mesh_filenames
        assert pred_mesh_filenames

        if example_name not in self.results:
            self.results[example_name] = {}
        if not force and self.check_already_computed(example_name, key):
            log.info('Already computed. Skipping. {}'.format(example_name, key))
            return
        pr = self.mesh_precision_recall(gt_mesh_filenames, pred_mesh_filenames)
        self.results[example_name][key] = pr

    def names(self):
        return list(self.results.keys())

    def check_already_computed(self, name, key):
        key = self.force_string_key(key)
        if key not in self.results[name]:
            return False
        elif 'p' not in self.results[name][key]:
            return False
        elif 'r' not in self.results[name][key]:
            return False
        elif (isinstance(self.results[name][key]['p'], (tuple, list)) and len(self.results[name][key]['p']) == 0) or self.results[name][key]['p'] is None:
            return False
        elif (isinstance(self.results[name][key]['r'], (tuple, list)) and len(self.results[name][key]['r']) == 0) or self.results[name][key]['r'] is None:
            return False
        return True

    def count(self, key):
        ret = 0
        for name in self.names():
            if self.check_already_computed(name, key):
                ret += 1
        return ret

    def save(self):
        log.info('Saving {}'.format(self.save_filename))
        # TODO: merge before overwriting.
        with portalocker.Lock(self.save_filename, mode='wb', timeout=10) as f:
            pickle.dump(self.results, f)

    def update(self, other):
        assert isinstance(other, PRCurveEvaluation)
        PRCurveEvaluation.dict_merge(self.results, other.results)

    def all_keys(self):
        return sorted(set(itertools.chain(*[list(item.keys()) for item in self.results.values()])))

    def run_evaluation(self, example):
        """
        This is the main function.
        """
        assert isinstance(example, dict)
        assert 'name' in example

        name = example['name']
        house_id, camera_id = pbrs_utils.parse_house_and_camera_ids_from_string(name)

        gt_bg, gt_objects = sorted(glob.glob(path.join(config.default_out_root, 'v9_gt_mesh/{}/{}/gt*.ply'.format(house_id, camera_id))))
        gt_depths = sorted(glob.glob(path.join(config.default_out_root, 'v9_gt_mesh/{}/{}/d*.ply'.format(house_id, camera_id))))
        print(gt_depths)
        assert len(gt_depths) == 5

        gt_overhead_fg = sorted(glob.glob(path.join(config.default_out_root, 'v9_gt_overhead_mesh/{}/{}/overhead_fg.ply'.format(house_id, camera_id))))
        assert len(gt_overhead_fg) == 1
        gt_overhead_fg = gt_overhead_fg[0]
        assert path.exists(gt_overhead_fg)

        pred_depths = sorted(glob.glob(path.join(config.default_out_root, 'v9_pred_depth_mesh/{}/{}/pred_*.ply'.format(house_id, camera_id))))
        assert len(pred_depths) == 5

        pred_overhead_fg = sorted(glob.glob(path.join(config.default_out_root, 'v9_pred_depth_mesh/{}/{}/overhead_fg_clipped.ply'.format(house_id, camera_id))))
        assert len(pred_overhead_fg) == 1
        pred_overhead_fg = pred_overhead_fg[0]

        # pred_files_list = sorted(glob.glob(path.join(config.default_out_root, 'v9_pred_depth_mesh/{}/{}/*.ply'.format(house_id, camera_id))))
        # pred_files = {path.basename(item).split('.')[0]: item for item in pred_files_list}
        # f3d_pred = path.join(config.default_out_root, 'factored3d_pred/{}/{}/codes_transformed_clipped.ply'.format(house_id, camera_id))

        # assert path.isfile(f3d_pred), f3d_pred

        # if 'overhead_fg_clipped' not in pred_files:
        #     print('Overhead file is not available. Skipping for now.    {}'.format(name))
        #     return

        # self.run_if_not_exists(name, self.key('pred', 'obj', ['overhead_fg']), [gt_objects], [pred_files['overhead_fg_clipped']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['0']), [gt_objects], [pred_files['pred_0']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['1']), [gt_objects], [pred_files['pred_1']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['2']), [gt_objects], [pred_files['pred_2']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['0', 'overhead_fg']), [gt_objects], [pred_files['pred_0'], pred_files['overhead_fg_clipped']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['1', 'overhead_fg']), [gt_objects], [pred_files['pred_1'], pred_files['overhead_fg_clipped']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['2', 'overhead_fg']), [gt_objects], [pred_files['pred_2'], pred_files['overhead_fg_clipped']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['0', '1']), [gt_objects], [pred_files['pred_0'], pred_files['pred_1']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['0', '2']), [gt_objects], [pred_files['pred_0'], pred_files['pred_2']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['1', '2']), [gt_objects], [pred_files['pred_1'], pred_files['pred_2']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['0', '1', '2']), [gt_objects], [pred_files['pred_0'], pred_files['pred_1'], pred_files['pred_2']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['0', '1', '2', 'overhead_fg']), [gt_objects], [pred_files['pred_0'], pred_files['pred_1'], pred_files['pred_2'], pred_files['overhead_fg_clipped']])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['0', '1', '2', 'overhead_fg', 'f3d']), [gt_objects], [pred_files['pred_0'], pred_files['pred_1'], pred_files['pred_2'], pred_files['overhead_fg_clipped'], f3d_pred])
        # self.run_if_not_exists(name, self.key('pred', 'obj', ['f3d']), [gt_objects], [f3d_pred])

        # self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['1', '2', '3']), [gt_objects], [gt_depths[1], gt_depths[2], gt_depths[3]])
        # self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['2', '3']), [gt_objects], [gt_depths[2], gt_depths[3]])
        # self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['3']), [gt_objects], [gt_depths[3]])

        self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['0']), [gt_objects], [gt_depths[0]])
        self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['1']), [gt_objects], [gt_depths[1]])
        self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['2']), [gt_objects], [gt_depths[2]])
        self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['3']), [gt_objects], [gt_depths[3]])
        self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['overhead_fg']), [gt_objects], [gt_overhead_fg])
        self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['0', '1']), [gt_objects], [gt_depths[0], gt_depths[1]])
        self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['0', '1', '2']), [gt_objects], [gt_depths[0], gt_depths[1], gt_depths[2]])
        self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['0', '1', '2', '3']), [gt_objects], [gt_depths[0], gt_depths[1], gt_depths[2], gt_depths[3]])
        self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['0', '1', '2', '3', 'overhead_fg']), [gt_objects], [gt_depths[0], gt_depths[1], gt_depths[2], gt_depths[3], gt_overhead_fg])

        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0']), [gt_bg, gt_objects], [gt_depths[0]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['1']), [gt_bg, gt_objects], [gt_depths[1]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['2']), [gt_bg, gt_objects], [gt_depths[2]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['3']), [gt_bg, gt_objects], [gt_depths[3]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['4']), [gt_bg, gt_objects], [gt_depths[4]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['overhead_fg']), [gt_bg, gt_objects], [gt_overhead_fg])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['4', '0']), [gt_bg, gt_objects], [gt_depths[4], gt_depths[0]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['4', '0', '1']), [gt_bg, gt_objects], [gt_depths[4], gt_depths[0], gt_depths[1]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['4', '0', '1', '2']), [gt_bg, gt_objects], [gt_depths[4], gt_depths[0], gt_depths[1], gt_depths[2]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '1']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[1]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '1', '2']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[1], gt_depths[2]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '1', '2', '3']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[1], gt_depths[2], gt_depths[3]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '1', '2', '3', '4']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[1], gt_depths[2], gt_depths[3], gt_depths[4]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '1', '2', '3', '4', 'overhead_fg']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[1], gt_depths[2], gt_depths[3], gt_depths[4], gt_overhead_fg])

        # LDI
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '2', '4']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[2], gt_depths[4]])
        self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '2']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[2]])

        self.run_if_not_exists(name, self.key('pred', 'obj', ['0']), [gt_objects], [pred_depths[0]])
        self.run_if_not_exists(name, self.key('pred', 'obj', ['1']), [gt_objects], [pred_depths[1]])
        self.run_if_not_exists(name, self.key('pred', 'obj', ['2']), [gt_objects], [pred_depths[2]])
        self.run_if_not_exists(name, self.key('pred', 'obj', ['3']), [gt_objects], [pred_depths[3]])
        self.run_if_not_exists(name, self.key('pred', 'obj', ['overhead_fg']), [gt_objects], [pred_overhead_fg], force=True)
        self.run_if_not_exists(name, self.key('pred', 'obj', ['0', 'overhead_fg']), [gt_objects], [pred_depths[0], pred_overhead_fg])
        self.run_if_not_exists(name, self.key('pred', 'obj', ['0', '1']), [gt_objects], [pred_depths[0], pred_depths[1]])
        self.run_if_not_exists(name, self.key('pred', 'obj', ['0', '1', '2']), [gt_objects], [pred_depths[0], pred_depths[1], pred_depths[2]])
        self.run_if_not_exists(name, self.key('pred', 'obj', ['0', '1', '2', '3']), [gt_objects], [pred_depths[0], pred_depths[1], pred_depths[2], pred_depths[3]])
        self.run_if_not_exists(name, self.key('pred', 'obj', ['0', '1', '2', '3', 'overhead_fg']), [gt_objects], [pred_depths[0], pred_depths[1], pred_depths[2], pred_depths[3], pred_overhead_fg], force=True)

        self.run_if_not_exists(name, self.key('pred', 'both', ['0']), [gt_bg, gt_objects], [pred_depths[0]])
        self.run_if_not_exists(name, self.key('pred', 'both', ['1']), [gt_bg, gt_objects], [pred_depths[1]])
        self.run_if_not_exists(name, self.key('pred', 'both', ['2']), [gt_bg, gt_objects], [pred_depths[2]])
        self.run_if_not_exists(name, self.key('pred', 'both', ['3']), [gt_bg, gt_objects], [pred_depths[3]])
        self.run_if_not_exists(name, self.key('pred', 'both', ['4']), [gt_bg, gt_objects], [pred_depths[4]])
        self.run_if_not_exists(name, self.key('pred', 'both', ['overhead_fg']), [gt_bg, gt_objects], [pred_overhead_fg], force=True)
        self.run_if_not_exists(name, self.key('pred', 'both', ['4', '0']), [gt_bg, gt_objects], [pred_depths[4], pred_depths[0]])
        self.run_if_not_exists(name, self.key('pred', 'both', ['4', '0', '1']), [gt_bg, gt_objects], [pred_depths[4], pred_depths[0], pred_depths[1]])
        self.run_if_not_exists(name, self.key('pred', 'both', ['4', '0', '1', '2']), [gt_bg, gt_objects], [pred_depths[4], pred_depths[0], pred_depths[1], pred_depths[2]])
        self.run_if_not_exists(name, self.key('pred', 'both', ['4', '0', '1', '2', '3']), [gt_bg, gt_objects], [pred_depths[4], pred_depths[0], pred_depths[1], pred_depths[2], pred_depths[3]])
        self.run_if_not_exists(name, self.key('pred', 'both', ['4', '0', '1', '2', '3', 'overhead_fg']), [gt_bg, gt_objects], [pred_depths[4], pred_depths[0], pred_depths[1], pred_depths[2], pred_depths[3], pred_overhead_fg], force=True)

        # self.run_if_not_exists(name, self.key('pred', 'both', ['overhead_fg']), [gt_bg, gt_objects], [pred_files['overhead_fg_clipped']])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['0']), [gt_bg, gt_objects], [pred_files['pred_0']])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['1']), [gt_bg, gt_objects], [pred_files['pred_1']])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['2']), [gt_bg, gt_objects], [pred_files['pred_2']])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['3']), [gt_bg, gt_objects], [pred_files['pred_3']])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['0', '3']), [gt_bg, gt_objects], [pred_files['pred_0'], pred_files['pred_3']])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['0', '1', '3']), [gt_bg, gt_objects], [pred_files['pred_0'], pred_files['pred_1'], pred_files['pred_3']])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['0', '1', '2', '3']), [gt_bg, gt_objects], [pred_files['pred_0'], pred_files['pred_1'], pred_files['pred_2'], pred_files['pred_3']])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['0', '1', '2', '3', 'overhead_fg']), [gt_bg, gt_objects], [pred_files['pred_0'], pred_files['pred_1'], pred_files['pred_2'], pred_files['pred_3'], pred_files['overhead_fg_clipped']])
        #
        # # self.run_if_not_exists(name, self.key('gt_depth', 'both', ['overhead_fg']), [gt_bg, gt_objects], [pred_files['overhead_fg_clipped']])
        # self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0']), [gt_bg, gt_objects], [gt_depths[0]])
        # self.run_if_not_exists(name, self.key('gt_depth', 'both', ['1']), [gt_bg, gt_objects], [gt_depths[1]])
        # self.run_if_not_exists(name, self.key('gt_depth', 'both', ['2']), [gt_bg, gt_objects], [gt_depths[2]])
        # self.run_if_not_exists(name, self.key('gt_depth', 'both', ['3']), [gt_bg, gt_objects], [gt_depths[3]])
        # self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '3']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[3], ])
        # self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '1', '3']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[1], gt_depths[3], ])
        # self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '1', '2', '3']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[1], gt_depths[2], gt_depths[3], ])
        # # self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '1', '2', '3', 'overhead_fg']), [gt_bg, gt_objects], [])
        #
        # # self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['2', '3']), [gt_objects], [gt_depths[2], gt_depths[3]])
        # # self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['1', '2', '3']), [gt_objects], [gt_depths[1], gt_depths[2], gt_depths[3]])
        # # self.run_if_not_exists(name, self.key('gt_depth', 'obj', ['0', '1', '2', '3']), [gt_objects], [gt_depths[0], gt_depths[1], gt_depths[2], gt_depths[3]])
        #
        # self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '1']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[1], ])
        # self.run_if_not_exists(name, self.key('gt_depth', 'both', ['0', '1', '2']), [gt_bg, gt_objects], [gt_depths[0], gt_depths[1], gt_depths[2], ])

    def as_array(self, key, precision_or_recall):
        key = self.force_string_key(key)
        assert isinstance(key, str)
        assert precision_or_recall in ['p', 'r']

        collected = []
        names = list(self.results.keys())
        for name in names:
            if key in self.results[name]:
                pr = self.results[name][key]
                if precision_or_recall in pr:
                    values = pr[precision_or_recall]
                    if isinstance(values, float):
                        values = np.full(self.thresholds.shape, values, dtype=np.float32)
                    collected.append(values)

        print('{} measurements in key {}'.format(len(collected), key))

        if len(collected) == 0:
            raise RuntimeError('not found')

        y = np.array(collected)
        return y

    def mean(self, key, precision_or_recall):
        return self.as_array(key=key, precision_or_recall=precision_or_recall).mean(axis=0)

    def std(self, key, precision_or_recall):
        return self.as_array(key=key, precision_or_recall=precision_or_recall).std(axis=0)

    def force_string_key(self, key):
        if isinstance(key, str):
            return key
        elif isinstance(key, (tuple, list)):
            pred_or_gt, target_type, source_list = key
            return self.key(pred_or_gt, target_type, source_list)
        raise RuntimeError('key error: {}'.format(key))

    def plot(self, key, precision_or_recall, max_threshold, plot_quantile=False, **kwargs):
        key = self.force_string_key(key)
        assert isinstance(key, str)
        assert precision_or_recall in ['p', 'r']
        import matplotlib.pyplot as pt
        arr = self.as_array(key, precision_or_recall)
        y = np.mean(arr, axis=0)
        x = self.thresholds

        end = (x <= max_threshold).sum() + 1
        x = x[:end]
        y = y[:end]

        pt.plot(x, y, **kwargs)

        if plot_quantile:
            color = kwargs.get('color', None)
            upper = np.quantile(arr, q=0.75, axis=0)[:end]
            lower = np.quantile(arr, q=0.25, axis=0)[:end]
            pt.fill_between(x, lower, upper, alpha=0.1, facecolor=color, antialiased=True)

    def value(self, key, precision_or_recall, x):
        key = self.force_string_key(key)
        assert isinstance(key, str)
        assert precision_or_recall in ['p', 'r']
        arr = self.as_array(key, precision_or_recall)
        y = np.mean(arr, axis=0)
        try:
            index = np.where(np.isclose(self.thresholds, x))[0][0]
        except IndexError:
            raise ValueError('x={} is not in thresholds. x must be one of the values in self.thresholds.'.format(x))
        return y[index]

    @staticmethod
    def dict_merge(target, source):
        """https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
        """
        assert isinstance(target, dict)
        assert isinstance(source, dict)
        for k, v in source.items():
            if (k in target and isinstance(target[k], dict) and isinstance(source[k], collections.Mapping)):
                PRCurveEvaluation.dict_merge(target[k], source[k])
            else:
                target[k] = source[k]


class PRCurveEvaluationScanNet(object):
    def __init__(self, save_filename):
        self.save_filename = save_filename
        self.load()

        step = 0.01
        self.thresholds = np.array([0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04] + np.arange(0.05, 1 + step, step).tolist() + np.arange(1 + step * 2, 2 + step, step * 2).tolist())
        self.density = 2500

        self.first_n = None

    def load(self):
        try:
            with portalocker.Lock(self.save_filename, mode='rb', timeout=10) as f:
                self.results = pickle.load(f)
        except FileNotFoundError as ex:
            log.info('File not found. Initializing an empty dict. {}'.format(self.save_filename))
            self.results = {}

    def key(self, pred_or_gt, target_type, source_list):
        assert pred_or_gt in ['pred', 'gt_depth']
        assert target_type in ['obj', 'background', 'both']
        if isinstance(source_list, str):
            source_list = [source_list]
        assert isinstance(source_list, (tuple, list)), source_list
        return ','.join([pred_or_gt.strip(), target_type.strip(), '+'.join(sorted(source_list))])

    def is_mesh_empty(self, mesh_filename):
        if isinstance(mesh_filename, (list, tuple)):
            return np.all([self.is_mesh_empty(item) for item in mesh_filename])
        return path.getsize(mesh_filename) < 190 or not path.isfile(mesh_filename)

    def mesh_precision_recall(self, gt_mesh_filenames, pred_mesh_filenames):
        assert isinstance(gt_mesh_filenames, (list, tuple))
        assert isinstance(pred_mesh_filenames, (list, tuple))
        assert gt_mesh_filenames
        assert pred_mesh_filenames

        if self.is_mesh_empty(gt_mesh_filenames):
            log.error('GT mesh is empty. This should not happen.    {}'.format(gt_mesh_filenames))
            raise RuntimeError()
        if self.is_mesh_empty(pred_mesh_filenames):
            log.warn('Nothing was predicted. Precision=1, Recall=0')
            return {'p': 1.0, 'r': 0.0, }  # for all thresholds.

        precision, recall = depth_mesh_utils_cpp.mesh_precision_recall(gt_mesh_filenames, pred_mesh_filenames, self.density, thresholds=self.thresholds)
        return {'p': precision, 'r': recall}

    def run_if_not_exists(self, example_name, key, gt_mesh_filenames, pred_mesh_filenames, force=False):
        assert isinstance(gt_mesh_filenames, (list, tuple))
        assert isinstance(pred_mesh_filenames, (list, tuple))
        assert gt_mesh_filenames
        assert pred_mesh_filenames

        if example_name not in self.results:
            self.results[example_name] = {}
        if not force and self.check_already_computed(example_name, key):
            log.info('Already computed. Skipping. {}'.format(example_name, key))
            return
        pr = self.mesh_precision_recall(gt_mesh_filenames, pred_mesh_filenames)
        self.results[example_name][key] = pr

    def names(self):
        return list(self.results.keys())

    def check_already_computed(self, name, key):
        key = self.force_string_key(key)
        if key not in self.results[name]:
            return False
        elif 'p' not in self.results[name][key]:
            return False
        elif 'r' not in self.results[name][key]:
            return False
        elif (isinstance(self.results[name][key]['p'], (tuple, list)) and len(self.results[name][key]['p']) == 0) or self.results[name][key]['p'] is None:
            return False
        elif (isinstance(self.results[name][key]['r'], (tuple, list)) and len(self.results[name][key]['r']) == 0) or self.results[name][key]['r'] is None:
            return False
        return True

    def count(self, key):
        ret = 0
        for name in self.names():
            if self.check_already_computed(name, key):
                ret += 1
        return ret

    def save(self):
        log.info('Saving {}'.format(self.save_filename))
        # TODO: merge before overwriting.
        with portalocker.Lock(self.save_filename, mode='wb', timeout=10) as f:
            pickle.dump(self.results, f)

    def update(self, other):
        assert isinstance(other, PRCurveEvaluation)
        PRCurveEvaluation.dict_merge(self.results, other.results)

    def all_keys(self):
        return sorted(set(itertools.chain(*[list(item.keys()) for item in self.results.values()])))

    def run_evaluation(self, example):
        """
        This is the main function.
        """
        assert isinstance(example, dict)
        assert 'name' in example

        name = example['name']
        # gt_mesh = path.join(config.scannet_frustum_clipped_root, name, 'meshes.obj')
        gt_mesh = path.join(config.scannet_frustum_clipped_highres_root, name, 'meshes.obj')

        # TODO
        # pred_depths = sorted(glob.glob('/data4/out/scene3d/v9_scannet_pred_depth_mesh/{}/pred_*.ply'.format(name)))
        # assert len(pred_depths) == 5, pred_depths
        # pred_overhead = '/data4/out/scene3d/v9_scannet_pred_depth_mesh/{}/overhead_fg_clipped.ply'.format(name)
        # assert path.isfile(pred_overhead), pred_overhead
        # assert path.isfile(gt_mesh)

        f3d_layout = '/data4/out/scene3d/factored3d_pred/scannet/{}/layout_farclipped_transformed_clipped.ply'.format(name)
        f3d_objects = '/data4/out/scene3d/factored3d_pred/scannet/{}/codes_transformed_clipped.ply'.format(name)
        assert path.isfile(f3d_objects)
        assert path.isfile(f3d_layout)
        assert path.isfile(gt_mesh)

        # self.run_if_not_exists(name, self.key('pred', 'both', ['0']), [gt_mesh], [pred_depths[0]])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['4']), [gt_mesh], [pred_depths[4]])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['overhead_fg']), [gt_mesh], [pred_overhead])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['0', '1', '2', '3']), [gt_mesh], [pred_depths[0], pred_depths[1], pred_depths[2], pred_depths[3]])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['0', '1', '2', '3', '4']), [gt_mesh], [pred_depths[0], pred_depths[1], pred_depths[2], pred_depths[3], pred_depths[4]])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['0', '1', '2', '3', '4', 'overhead_fg']), [gt_mesh], [pred_depths[0], pred_depths[1], pred_depths[2], pred_depths[3], pred_depths[4], pred_overhead])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['0', '1', '2', '3', 'overhead_fg']), [gt_mesh], [pred_depths[0], pred_depths[1], pred_depths[2], pred_depths[3], pred_overhead])
        # self.run_if_not_exists(name, self.key('pred', 'both', ['0', 'overhead_fg']), [gt_mesh], [pred_depths[0], pred_overhead])
        self.run_if_not_exists(name, self.key('pred', 'both', ['f3d_obj']), [gt_mesh], [f3d_objects])
        self.run_if_not_exists(name, self.key('pred', 'both', ['f3d_room']), [gt_mesh], [f3d_layout])
        self.run_if_not_exists(name, self.key('pred', 'both', ['f3d_both']), [gt_mesh], [f3d_objects, f3d_layout])

    def as_array(self, key, precision_or_recall):
        key = self.force_string_key(key)
        assert isinstance(key, str)
        assert precision_or_recall in ['p', 'r']

        collected = []
        names = list(self.results.keys())
        for name in names:
            if key in self.results[name]:
                pr = self.results[name][key]
                if precision_or_recall in pr:
                    values = pr[precision_or_recall]
                    if isinstance(values, float):
                        values = np.full(self.thresholds.shape, values, dtype=np.float32)
                    collected.append(values)
                    if self.first_n and len(collected) >= self.first_n:
                        break

        print('{} measurements in key {}'.format(len(collected), key))

        if len(collected) == 0:
            raise RuntimeError('not found')

        y = np.array(collected)
        return y

    def mean(self, key, precision_or_recall):
        return self.as_array(key=key, precision_or_recall=precision_or_recall).mean(axis=0)

    def std(self, key, precision_or_recall):
        return self.as_array(key=key, precision_or_recall=precision_or_recall).std(axis=0)

    def force_string_key(self, key):
        if isinstance(key, str):
            return key
        elif isinstance(key, (tuple, list)):
            pred_or_gt, target_type, source_list = key
            return self.key(pred_or_gt, target_type, source_list)
        raise RuntimeError('key error: {}'.format(key))

    def plot(self, key, precision_or_recall, max_threshold, plot_quantile=False, **kwargs):
        key = self.force_string_key(key)
        assert isinstance(key, str)
        assert precision_or_recall in ['p', 'r']
        import matplotlib.pyplot as pt
        arr = self.as_array(key, precision_or_recall)
        y = np.mean(arr, axis=0)
        x = self.thresholds

        end = (x <= max_threshold).sum() + 1
        x = x[:end]
        y = y[:end]

        pt.plot(x, y, **kwargs)

        if plot_quantile:
            color = kwargs.get('color', None)
            upper = np.quantile(arr, q=0.75, axis=0)[:end]
            lower = np.quantile(arr, q=0.25, axis=0)[:end]
            pt.fill_between(x, lower, upper, alpha=0.1, facecolor=color, antialiased=True)

    def value(self, key, precision_or_recall, x):
        key = self.force_string_key(key)
        assert isinstance(key, str)
        assert precision_or_recall in ['p', 'r']
        arr = self.as_array(key, precision_or_recall)
        y = np.mean(arr, axis=0)
        try:
            index = np.where(np.isclose(self.thresholds, x))[0][0]
        except IndexError:
            raise ValueError('x={} is not in thresholds. x must be one of the values in self.thresholds.'.format(x))
        return y[index]

    @staticmethod
    def dict_merge(target, source):
        """https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
        """
        assert isinstance(target, dict)
        assert isinstance(source, dict)
        for k, v in source.items():
            if (k in target and isinstance(target[k], dict) and isinstance(source[k], collections.Mapping)):
                PRCurveEvaluation.dict_merge(target[k], source[k])
            else:
                target[k] = source[k]


def nyu_pointcloud(name, out_filename):
    m = sio.loadmat(path.join(config.nyu_root, 'depth/{}.mat'.format(name)))

    x = np.array([np.mgrid[0:427, 0:561][0].ravel(), np.mgrid[0:427, 0:561][1].ravel()])
    d = m['depth'].ravel()
    x3d = (x[0, :] - m['K'][0, 2]) * d / m['K'][0, 0]
    y3d = (x[1, :] - m['K'][1, 2]) * d / m['K'][1, 1]
    pts = np.array([y3d, x3d, -d]).T
    io_utils.save_simple_points_ply(out_filename, pts)
    # pts2 = m['Rtilt'].T.dot(pts.T).T
    # io_utils.save_simple_points_ply('/mnt/ramdisk/hi2.ply', pts2)


def nyu_gravity_angle(name):
    m = sio.loadmat(path.join(config.nyu_root, 'depth/{}.mat'.format(name)))
    v = np.array([0, 1, 0], dtype=np.float64).reshape(3, 1)
    v2 = m['Rtilt'].dot(v)
    theta = np.arccos(np.inner(v.ravel(), v2.ravel()))
    return theta


def nyu_rgb_image(name):
    img = io_utils.read_jpg(path.join(config.nyu_root, 'images/{}.jpg'.format(name)))[3:-3]
    resized = scipy.misc.imresize(img, (240, 320))
    return resized


def convert_mesh_to_camcoord(mesh_filename, camera_filename):
    if isinstance(mesh_filename, str):
        fv = io_utils.read_mesh_assimp(mesh_filename)
    else:
        fv = mesh_filename
    cam_line = [float(item) for item in io_utils.read_lines_and_strip(camera_filename)[0].split()[1:]]
    cam_pos = np.array(cam_line[:3])
    cam_viewdir = np.array(cam_line[3:6])
    cam_up = np.array(cam_line[6:9])
    R = transforms.lookat_matrix(cam_pos, obj_xyz=cam_pos + cam_viewdir, up=cam_up)
    v_transformed = R.dot(np.concatenate([fv['v'], np.ones((fv['v'].shape[0], 1))], 1).T).T
    fv['v'] = v_transformed
    return fv


def convert_binvox_to_pcl(binvox_filename):
    from third_party import binvox_rw
    with open(binvox_filename, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    pts = (np.vstack(np.where(model.data)).T + 0.5) / model.dims * model.scale + model.translate
    voxel_pcl_file = binvox_filename.replace('.binvox', '_binvox_pcl.ply')
    io_utils.save_simple_points_ply(voxel_pcl_file, pts)


def save_binvox_model_as_pcl(model, out_filename):
    pts = (np.vstack(np.where(model.data)).T + 0.5) / model.dims * model.scale + model.translate
    io_utils.save_simple_points_ply(out_filename, pts)


def binvox_iou(filename1, filename2):
    from third_party import binvox_rw
    if isinstance(filename1, str):
        with open(filename1, 'rb') as f:
            vox1 = binvox_rw.read_as_3d_array(f).data.copy()
    else:
        vox1 = filename1

    if isinstance(filename2, str):
        with open(filename2, 'rb') as f:
            vox2 = binvox_rw.read_as_3d_array(f).data.copy()
    else:
        vox2 = filename2

    iou = float((vox1 & vox2).sum()) / (vox1 | vox2).sum()
    return iou


def voxels_to_mesh(pred_vol, thresh=0.5):
    # import mcubes
    # pred_vol_thresholded = np.pad(pred_vol, [(1, 1), (1, 1), (1, 1)], 'constant', constant_values=(0,)) > thresh
    # v_all, f_all = mcubes.marching_cubes(pred_vol_thresholded, thresh)
    # v_all = v_all - 1 + 0.5  # undo padding offset
    # return v_all, f_all

    # code from factored3d
    cube_v = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    cube_v = cube_v - 0.5
    cube_f = np.array([[1, 7, 5], [1, 3, 7], [1, 4, 3], [1, 2, 4], [3, 8, 7], [3, 4, 8], [5, 7, 8], [5, 8, 6], [1, 5, 6], [1, 6, 2], [2, 6, 8], [2, 8, 4]]).astype(np.int)
    v_counter = 0
    tot_points = np.greater(pred_vol, thresh).sum()
    v_all = np.tile(cube_v, [tot_points, 1])
    f_all = np.tile(cube_f, [tot_points, 1])
    f_offset = np.tile(np.linspace(0, 12 * tot_points - 1, 12 * tot_points), 3).reshape(3, 12 * tot_points).transpose()
    f_offset = (f_offset // 12 * 8).astype(np.int)
    f_all += f_offset
    for x in range(pred_vol.shape[0]):
        for y in range(pred_vol.shape[1]):
            for z in range(pred_vol.shape[2]):
                if pred_vol[x, y, z] > thresh:
                    radius = pred_vol[x, y, z]
                    v_all[v_counter:v_counter + 8, :] *= radius
                    v_all[v_counter:v_counter + 8, :] += (np.array([[x, y, z]]) + 0.5)
                    v_counter += 8
    return v_all, f_all - 1


def convert_binvox_to_mesh(binvox_filename):
    from third_party import binvox_rw
    with open(binvox_filename, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)

    v, f = voxels_to_mesh(model.data, thresh=0.95)
    v = v / model.dims * model.scale + model.translate
    fv = {'f': f, 'v': v, }

    out_file = binvox_filename.replace('.binvox', '_binvox_mesh.off')
    io_utils.save_off(fv, out_file)


def carve_y(overhead_only_voxel_file, all_voxel_file):
    from third_party import binvox_rw
    with open(overhead_only_voxel_file, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    ovh_vox = model.data.copy()

    with open(all_voxel_file, 'rb') as f:
        model2 = binvox_rw.read_as_3d_array(f)
    all_vox = model2.data.copy()

    dx, _, dz = ovh_vox.shape
    for xi in range(dx):
        for zi in range(dz):
            col = ovh_vox[xi, :, zi]
            nz = np.nonzero(col)[0]
            if nz.size == 0:
                continue
            all_vox[xi, nz[-1] + 1:, zi] = 0

    return all_vox


def random_string(l=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(l))


class VoxelIoUEvaluation(object):
    def __init__(self, save_filename):
        self.save_filename = save_filename
        self.load()

    def load(self):
        try:
            with portalocker.Lock(self.save_filename, mode='rb', timeout=10) as f:
                self.results = pickle.load(f)
        except FileNotFoundError as ex:
            log.info('File not found. Initializing an empty dict. {}'.format(self.save_filename))
            self.results = {}

    def key(self, pred_or_gt, target_type, source_list):
        assert pred_or_gt in ['pred', 'gt_depth']
        assert target_type in ['obj']
        if isinstance(source_list, str):
            source_list = [source_list]
        assert isinstance(source_list, (tuple, list)), source_list
        return ','.join([pred_or_gt.strip(), target_type.strip(), '+'.join(sorted(source_list))])

    def run_command(self, command):
        print(command)
        os.system(command)

    def run_evaluation(self, example):
        """
        This is the main function.
        """
        assert isinstance(example, dict)
        assert 'name' in example

        name = example['name']
        house_id, camera_id = pbrs_utils.parse_house_and_camera_ids_from_string(name)

        gt_bg, gt_objects = sorted(glob.glob(path.join(config.default_out_root, 'v9_gt_mesh/{}/{}/gt*.ply'.format(house_id, camera_id))))
        gt_depths = sorted(glob.glob(path.join(config.default_out_root, 'v9_gt_mesh/{}/{}/d*.ply'.format(house_id, camera_id))))
        assert len(gt_depths) == 5

        gt_overhead_fg = sorted(glob.glob(path.join(config.default_out_root, 'v9_gt_overhead_mesh/{}/{}/overhead_fg.ply'.format(house_id, camera_id))))
        assert len(gt_overhead_fg) == 1
        gt_overhead_fg = gt_overhead_fg[0]
        assert path.exists(gt_overhead_fg)

        pred_depths = sorted(glob.glob(path.join(config.default_out_root, 'v9_pred_depth_mesh/{}/{}/pred_*.ply'.format(house_id, camera_id))))
        assert len(pred_depths) == 5

        pred_overhead_fg = sorted(glob.glob(path.join(config.default_out_root, 'v9_pred_depth_mesh/{}/{}/overhead_fg_clipped.ply'.format(house_id, camera_id))))
        assert len(pred_overhead_fg) == 1
        pred_overhead_fg = pred_overhead_fg[0]

        print('gt_depths', gt_depths)
        print('gt_objects', gt_objects)
        print('pred_depths', pred_depths)

        voxel_related_data_out_basedir = '/mnt/ramdisk/voxels_data/tmp_{}'.format(name.replace('/', '_'))
        for fname in glob.glob(path.join(voxel_related_data_out_basedir, '*')):
            os.remove(fname)

        camera_filename = example['camera_filename']
        fv = convert_mesh_to_camcoord(gt_objects, camera_filename)
        gt_objects_camcoord = path.join(voxel_related_data_out_basedir, 'gt_objects_camcoord.off')
        io_utils.save_off(fv, gt_objects_camcoord)

        # vox_res = 64
        vox_res = 400
        vox_method_flags = '-pb'

        self.run_command('/home/daeyun/usr/bin/binvox {} -d {} -bb -5 -5 -10 5 5 0 {}'.format(vox_method_flags, vox_res, gt_objects_camcoord))
        gt_object_voxel_file = gt_objects_camcoord.replace('.off', '.binvox')
        convert_binvox_to_pcl(gt_object_voxel_file)
        # convert_binvox_to_mesh(gt_object_voxel_file)

        ret = {}

        # f3d voxelization
        f3d_pred = path.join(config.default_out_root_v8, 'factored3d_pred/{}/{}/codes_transformed_clipped.ply'.format(house_id, camera_id))
        assert path.isfile(f3d_pred), f3d_pred
        fv = convert_mesh_to_camcoord(f3d_pred, camera_filename)
        f3d_objects_camcoord = path.join(voxel_related_data_out_basedir, 'f3d_objects_camcoord.off')
        io_utils.save_off(fv, f3d_objects_camcoord)

        self.run_command('/home/daeyun/usr/bin/binvox {} -d {} -bb -5 -5 -10 5 5 0 {}'.format(vox_method_flags, vox_res, f3d_objects_camcoord))
        f3d_objects_voxel_file = f3d_objects_camcoord.replace('.off', '.binvox')
        convert_binvox_to_pcl(f3d_objects_voxel_file)
        # convert_binvox_to_mesh(f3d_objects_voxel_file)

        iou = binvox_iou(f3d_objects_voxel_file, gt_object_voxel_file)
        print('IoU_f3d', iou)
        ret['IoU_f3d'] = iou

        '''
        # depth overhead only
        depth_meshes_fv_list = [io_utils.read_mesh_assimp(pred_d) for pred_d in [pred_overhead_fg]]
        # depth_meshes_fv_list = [io_utils.read_mesh_assimp(pred_d) for pred_d in pred_depths[:4]]

        depth_meshes_merged = io_utils.merge_meshes(*depth_meshes_fv_list)
        fv = convert_mesh_to_camcoord(depth_meshes_merged, camera_filename)
        depth_objects_camcoord = path.join(voxel_related_data_out_basedir, 'depth_meshes_ovh_camcoord.off')
        io_utils.save_off(fv, depth_objects_camcoord)
        self.run_command('/home/daeyun/usr/bin/binvox {} -d {} -bb -5 -5 -10 5 5 0 {}'.format(vox_method_flags, vox_res, depth_objects_camcoord))
        depth_ovh_voxel_file = depth_objects_camcoord.replace('.off', '.binvox')
        convert_binvox_to_pcl(depth_ovh_voxel_file)
        convert_binvox_to_mesh(depth_ovh_voxel_file)
        iou = binvox_iou(depth_ovh_voxel_file, gt_object_voxel_file)
        print('IoU_overheadonly', iou)
        ret['IoU_overheadonly'] = iou
        '''

        # depth voxelization no overhead
        depth_meshes_fv_list = [io_utils.read_mesh_assimp(pred_d) for pred_d in pred_depths[:4]]
        # depth_meshes_fv_list = [io_utils.read_mesh_assimp(pred_d) for pred_d in pred_depths[:4]]

        depth_meshes_merged = io_utils.merge_meshes(*depth_meshes_fv_list)
        fv = convert_mesh_to_camcoord(depth_meshes_merged, camera_filename)
        depth_objects_camcoord = path.join(voxel_related_data_out_basedir, 'depth_meshes_frontal4_camcoord.off')
        io_utils.save_off(fv, depth_objects_camcoord)
        self.run_command('/home/daeyun/usr/bin/binvox {} -d {} -bb -5 -5 -10 5 5 0 {}'.format(vox_method_flags, vox_res, depth_objects_camcoord))
        depth_objects_voxel_file = depth_objects_camcoord.replace('.off', '.binvox')
        convert_binvox_to_pcl(depth_objects_voxel_file)
        # convert_binvox_to_mesh(depth_objects_voxel_file)

        iou = binvox_iou(depth_objects_voxel_file, gt_object_voxel_file)
        print('IoU_frontaldepthonly', iou)
        ret['IoU_frontaldepthonly'] = iou

        # depth voxelization
        depth_meshes_fv_list = [io_utils.read_mesh_assimp(pred_d) for pred_d in pred_depths[:4] + [pred_overhead_fg]]
        # depth_meshes_fv_list = [io_utils.read_mesh_assimp(pred_d) for pred_d in pred_depths[:4]]

        depth_meshes_merged = io_utils.merge_meshes(*depth_meshes_fv_list)
        fv = convert_mesh_to_camcoord(depth_meshes_merged, camera_filename)
        depth_objects_camcoord = path.join(voxel_related_data_out_basedir, 'depth_meshes_frontal4_and_ovh_camcoord.off')
        io_utils.save_off(fv, depth_objects_camcoord)
        self.run_command('/home/daeyun/usr/bin/binvox {} -d {} -bb -5 -5 -10 5 5 0 {}'.format(vox_method_flags, vox_res, depth_objects_camcoord))
        depth_objects_voxel_file = depth_objects_camcoord.replace('.off', '.binvox')
        convert_binvox_to_pcl(depth_objects_voxel_file)
        # convert_binvox_to_mesh(depth_objects_voxel_file)

        iou = binvox_iou(depth_objects_voxel_file, gt_object_voxel_file)
        print('IoU_alldepths', iou)
        ret['IoU_alldepths'] = iou

        return voxel_related_data_out_basedir, ret


def read_voxels(filename):
    from third_party import binvox_rw
    with open(filename, 'rb') as f:
        vox = binvox_rw.read_as_3d_array(f)
    return vox


def voxelize_multi_layer_depth(ground_truth_voxels, depths, two_layer_only=False):
    # TODO(daeyun): ground_truth_voxels isn't necessary
    from third_party import binvox_rw
    if isinstance(ground_truth_voxels, str):
        model = read_voxels(ground_truth_voxels)
    else:
        model = ground_truth_voxels

    new_model = model.clone()
    new_model.data.fill(1)
    proj_xy, depth_values, linear_indices = voxel.project_cam_voxels_to_image(new_model)

    assert depths.shape[0] >= 4
    assert depths.shape[0] <= 5
    assert depths.ndim == 3

    indexed = depths[:4, proj_xy[:, 1], proj_xy[:, 0]].T.copy()
    indexed = np.sort(indexed, axis=1)

    warnings.simplefilter(action="ignore", category=RuntimeWarning)

    if two_layer_only:
        is_between = ((indexed[:, 2] <= depth_values) & (indexed[:, 3] >= depth_values))
    else:
        is_between = ((indexed[:, 0] <= depth_values) & (indexed[:, 1] >= depth_values)) | ((indexed[:, 2] <= depth_values) & (indexed[:, 3] >= depth_values))

    # s = model.scale/np.mean(model.dims) * 0.5
    # is_between = ((indexed[:,0] <= depth_values - s) & (indexed[:,1] >= depth_values + s)) | ((indexed[:,2] <= depth_values - s) & (indexed[:,3] >= depth_values + s))
    # is_between = ((indexed[:,0] <= depth_values - s) & (indexed[:,1] >= depth_values)) | ((indexed[:,2] <= depth_values - s) & (indexed[:,3] >= depth_values))
    # is_between = ((indexed[:,0] <= depth_values) & (indexed[:,1] >= depth_values + s)) | ((indexed[:,2] <= depth_values) & (indexed[:,3] >= depth_values + s))
    # is_between = ((indexed[:,0] <= depth_values + s) & (indexed[:,1] >= depth_values - s)) | ((indexed[:,2] <= depth_values + s) & (indexed[:,3] >= depth_values - s))

    warnings.simplefilter(action="default", category=RuntimeWarning)

    new_model.data.fill(False)
    new_model.data.flat[linear_indices[is_between]] = True

    return new_model


def convert_height_map_to_depth(height_map, example_name):
    camera_filename = path.join(config.scene3d_root, 'v9/renderings', example_name + '_cam.txt')
    with open(camera_filename, 'r') as f:
        content = f.readlines()
    cam_o = content[1].strip().split()
    assert cam_o[0] == 'O'

    campos = np.array([float(item) for item in cam_o[1:4]])

    house_id, camera_id = example_name.split('/')
    floor_height = find_gt_floor_height_v9(house_id=house_id, camera_id=camera_id)

    ret = - height_map - (floor_height - campos[1])
    return ret


def project_point_cloud_to_orthographic_depth_image(pts, campos, viewdir, up, lrbt, im_hw):
    """
    This is not really a good implementation. Mostly used for debugging purposes.
    :return:
    """
    Rt = transforms.lookat_matrix(campos, campos + viewdir, up=up)
    transformed = Rt.dot(np.hstack((pts, np.ones([pts.shape[0], 1]))).T).T

    x = (transformed[:, 0] - lrbt[0]) / (lrbt[1] - lrbt[0]) * (im_hw[1])
    y = (transformed[:, 1] - lrbt[2]) / (lrbt[3] - lrbt[2]) * (im_hw[0])
    d = transformed[:, 2]

    ret = np.full(im_hw, fill_value=np.nan)
    for i in range(x.shape[0]):
        yi = im_hw[0] - int(round(y[i]))
        xi = int(round(x[i]))
        if yi < 0 or yi >= im_hw[0] or xi < 0 or xi >= im_hw[1]:
            continue
        if np.isnan(ret[yi, xi]):
            ret[yi, xi] = d[i]
        else:
            ret[yi, xi] = min(ret[yi, xi], d[i])

    return ret


def carve_voxels_using_height_map(voxels, overhead_height_map, example_name):
    if isinstance(voxels, str):
        model = read_voxels(voxels)
    else:
        model = voxels
    proj_xy, depth_values, linear_indices = voxel.project_cam_voxels_to_overhead_image(model, example_name)

    overhead_depth = convert_height_map_to_depth(overhead_height_map, example_name)

    indexed = overhead_depth[proj_xy[:, 1], proj_xy[:, 0]].T.copy()

    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    is_in_front = depth_values < indexed
    warnings.simplefilter(action="default", category=RuntimeWarning)
    new_model = model.clone()
    new_model.data.flat[linear_indices[is_in_front]] = False

    new_model2 = model.clone()
    new_model2.data.fill(True)
    proj_xy, depth_values, linear_indices = voxel.project_cam_voxels_to_overhead_image(new_model2, example_name)
    indexed = overhead_depth[proj_xy[:, 1], proj_xy[:, 0]].T.copy()
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    is_aligned = (depth_values >= indexed) & ((depth_values - indexed) < (10 / 400))
    warnings.simplefilter(action="default", category=RuntimeWarning)
    new_model.data.flat[linear_indices[is_aligned]] = True

    return new_model
