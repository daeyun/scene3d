import argparse
import glob
import time
import typing
from scene3d import depth_mesh_utils_cpp
import collections
import os
from os import path

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch import optim
from torch.backends import cudnn

from scene3d import config
from scene3d import feat
from scene3d import io_utils
from scene3d import log
from scene3d import pbrs_utils
from scene3d import loss_fn
from scene3d import torch_utils
from scene3d.dataset import dataset_utils
from scene3d.dataset import v1
from scene3d.dataset import v2
from scene3d.dataset import v8
from scene3d.net import unet
from scene3d.net import unet_no_bn
from scene3d.net import unet_overhead
from multiprocessing.pool import ThreadPool

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
    'OVERHEAD_OTF_01',  # all features, predicted geometry only.
    'OVERHEAD_offline_01',  # all features, predicted geometry only.
    'OVERHEAD_offline_02',  # no semantics, predicted geometry only.
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


def compute_loss(pytorch_model: nn.Module, batch, experiment_name: str, frozen_model: typing.Union[nn.Module, feat.FeatureGenerator] = None) -> torch.Tensor:
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


def save_mldepth_as_meshes(pred_segmented_depth, example):
    assert pred_segmented_depth.ndim == 3
    out_pred_filenames = []
    # out_gt_filenames = []
    for i in range(4):
        out_filename = '/data3/out/scene3d/v8_pred_depth_mesh/{}/pred_{}.ply'.format(example['name'], i)  # TODO
        if not path.isfile(out_filename):
            depth_mesh_utils_cpp.depth_to_mesh(pred_segmented_depth[i], example['camera_filename'], camera_index=0, dd_factor=10, out_ply_filename=out_filename)
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

        overhead_depth = default_overhead_camera_height - height_map_model_batch_out['pred_height_map'][i].squeeze()
        out_filename = '/data3/out/scene3d/v8_pred_depth_mesh/{}/overhead.ply'.format(example_names[i])  # TODO
        depth_mesh_utils_cpp.depth_to_mesh(overhead_depth, camera_filename=new_cam_filename, camera_index=1, dd_factor=6, out_ply_filename=out_filename)
        out_ply_filenames.append(out_filename)

        if path.isfile(new_cam_filename):
            os.remove(new_cam_filename)

    return out_ply_filenames


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

    def predict_height_map(self, batch, visualize_indices=None):
        with torch.cuda.device(self.device_id):
            assert 'camera_filename' not in batch

            self.transformer.prefetch_batch_async(batch, start_end_indices=None, target_device_id=self.device_id, options={'use_gt_geometry': False})
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
    gt_mesh_filename = '/data3/out/scene3d/v8_gt_mesh/{}/{}/gt_bg.ply'.format(house_id, camera_id)
    fv = io_utils.read_mesh(gt_mesh_filename)
    ycoords = sorted(fv['v'][:, 1].tolist())
    return np.median(ycoords[:int(max(len(ycoords) * 0.01, 100))]).item()
