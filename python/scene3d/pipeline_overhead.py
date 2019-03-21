import time
from os import path

from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import torch.utils.data
from torch.backends import cudnn

from scene3d import config
from scene3d import train_eval_pipeline
from scene3d import feat
from scene3d import io_utils
from scene3d import log
from scene3d import torch_utils
from scene3d.dataset import v9
from scene3d.dataset import v8


def get_dataset(split_name, use_gt_geometry=False):
    if use_gt_geometry:
        fields = ('rgb', 'overhead_camera_pose_4params', 'camera_filename', 'multi_layer_overhead_depth', 'multi_layer_depth_aligned_background')
    else:
        fields = ('rgb', 'overhead_camera_pose_4params', 'camera_filename', 'multi_layer_overhead_depth')

    data = v9.MultiLayerDepth(
        split=split_name,
        # split='/data4/scene3d/v9/train_s150.txt',
        subtract_mean=True, image_hw=(240, 320), rgb_scale=1.0 / 255,
        fields=fields, first_n=None)

    return data


def get_dataset_v8():
    data = v8.MultiLayerDepth(
        split='train',
        # split='/data4/scene3d/v9/train_s150.txt',
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255,
        fields=('rgb', 'overhead_camera_pose_4params', 'camera_filename', 'multi_layer_overhead_depth'))

    return data


def debug_process_one_batch():
    dataset_all = get_dataset(split_name='/data4/scene3d/v9/train_s150.txt', use_gt_geometry=False)

    depth_checkpoint_filename = '/data4/out/scene3d/v9/v9-multi_layer_depth_aligned_background_multi_branch/0/01149000_005_0003355.pth'
    segmentation_checkpoint_filename = '/data4/out/scene3d/v9/v9-category_nyu40_merged_background-2l/0/01130000_005_0001780.pth'

    transformer = feat.Transformer(
        # depth_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00700000_008_0001768.pth'),
        # segmentation_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-category_nyu40_merged_background-2l/0/00800000_022_0016362.pth'),
        depth_checkpoint_filename=depth_checkpoint_filename,
        segmentation_checkpoint_filename=segmentation_checkpoint_filename,
        device_id=1,
        num_workers=5,
    )

    loader = torch.utils.data.DataLoader(dataset_all, batch_size=10, num_workers=5, shuffle=False, drop_last=False, pin_memory=True)
    it = enumerate(loader)
    i_iter, batch = next(it)

    out = transformer.get_transformed_features(batch, [0, 10], use_gt_geometry=False)
    out = transformer.get_transformed_features(batch, [0, 10], use_gt_geometry=False)
    out = transformer.get_transformed_features(batch, [0, 10], use_gt_geometry=False)
    return out

    # transformer.prefetch_batch_async(batch, start_end_indices=None, target_device_id=1, options={'use_gt_geometry': False})
    #
    # for next_i_iter, next_batch in it:
    #     # overhead_features, transformer_names = transformer.pop_batch(target_device_id=1)
    #     overhead = transformer.pop_batch(target_device_id=1)
    #     return overhead
