from os import path

import cv2
import numpy as np
from torch.utils import data

from scene3d import config
from scene3d import io_utils
from scene3d import render_depth
from scene3d.dataset import dataset_utils


class MultiLayerDepthAndSegmentation(data.Dataset):
    """
    Includes multi-layer segmentation.
    """

    def __init__(self, train=True, first_n=None, rgb_scale=1.0, subtract_mean=True, image_hw=(240, 320)):
        self.filename_prefixes = io_utils.read_lines_and_strip(path.join(config.scene3d_root, 'v7/renderings.txt'))

        # eval_set_size = 16026
        eval_set_size = 10  # TODO(daeyun): temporary value

        if train:
            self.filename_prefixes = self.filename_prefixes[:-eval_set_size]
        else:
            self.filename_prefixes = self.filename_prefixes[-eval_set_size:]

        if first_n is not None and first_n > 0:
            self.filename_prefixes = self.filename_prefixes[:first_n]

        self.image_hw = image_hw
        self.rgb_mean = np.array([178.1781, 158.5039, 142.5141], dtype=np.float32)
        self.rgb_scale = rgb_scale
        self.subtract_mean = subtract_mean

    def __len__(self):
        return len(self.filename_prefixes)

    def __getitem__(self, index):
        example_name = self.filename_prefixes[index]
        # TODO: some images don't have background.
        bin_filenames = [
            path.join(config.scene3d_root, 'v7/renderings', example_name + '.bin'),
            path.join(config.scene3d_root, 'v7/renderings', example_name + '_bg.bin'),
            path.join(config.scene3d_root, 'v7/renderings', example_name + '_c.bin'),
            path.join(config.scene3d_root, 'v7/renderings', example_name + '_bg_c.bin'),
            path.join(config.scene3d_root, 'v7/renderings', example_name + '_td.bin'),
            path.join(config.scene3d_root, 'v7/renderings', example_name + '_td_prim.bin'),
            path.join(config.scene3d_root, 'v7/renderings', example_name + '_prim.bin'),
        ]
        txt_filenames = [
            path.join(config.scene3d_root, 'v7/renderings', example_name + '_cam.txt'),
            path.join(config.scene3d_root, 'v7/renderings', example_name + '_td_cam.txt'),
        ]

        # Prepare GT depth images.
        depths = io_utils.read_array_compressed(bin_filenames[0], dtype=np.float32)
        background_depth = io_utils.read_array_compressed(bin_filenames[1], dtype=np.float32)
        categories = io_utils.read_array_compressed(bin_filenames[2], dtype=np.uint8)
        background_categories = io_utils.read_array_compressed(bin_filenames[3], dtype=np.uint8)
        top_down_foreground = io_utils.read_array_compressed(bin_filenames[4], dtype=np.float32)
        top_down_foreground_prim = io_utils.read_array_compressed(bin_filenames[5], dtype=np.float32)
        foreground_prim = io_utils.read_array_compressed(bin_filenames[6], dtype=np.float32)



        back_of_same_category = np.full_like(categories[0], fill_value=np.nan)
        for xx in range(categories.shape[1]):
            for yy in range(categories.shape[2]):
                last_of_category = (categories[0, xx, yy] != categories[:, xx, yy]).nonzero()[0][0] - 1
                back_of_same_category[xx, yy] = last_of_category
        back_of_same_category_depth = dataset_utils.grid_indexing_2d(depths, back_of_same_category)





        count_image = (~np.isnan(depths)).sum(axis=0)
        cmax = np.maximum(count_image - 2, 0)
        d1 = dataset_utils.grid_indexing_2d(depths, cmax)
        d1_c = dataset_utils.grid_indexing_2d(categories, cmax)
        # d1[np.isnan(background_depth)] = np.nan
        # d1_c[np.isnan(background_depth)] = 255

        d0 = depths[0]
        d0_c = categories[0]
        # r0 = background_depth - d1
        # r1 = d1 - d0

        # 0: room layout.
        # 1: direct depth to empty space in front of the room layout.
        # 2: direct depth to filled space in front of the empty space.
        l1 = background_depth
        l2 = d1
        l3 = d0
        gt_ml_depth = np.stack([l1, l2, l3])

        gt_ml_category = np.stack([background_categories, d1_c, d0_c])

        _, h, w = gt_ml_depth.shape
        if (h, w) != tuple(self.image_hw):
            gt_ml_depth = cv2.resize(gt_ml_depth.transpose(1, 2, 0), dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

        # Prepare input RGB image.
        png_filename = path.join(config.pbrs_root, 'mlt_v2', example_name + '_mlt.png')
        in_rgb = cv2.cvtColor(cv2.imread(png_filename), cv2.COLOR_BGR2RGB)
        h, w, ch = in_rgb.shape
        assert ch == 3
        if (h, w) != tuple(self.image_hw):
            in_rgb = cv2.resize(in_rgb, dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_LINEAR)
        in_rgb = in_rgb.transpose((2, 0, 1)).astype(np.float32)
        if self.subtract_mean:
            in_rgb -= self.rgb_mean.reshape(3, 1, 1)
        if self.rgb_scale != 1.0:
            in_rgb *= self.rgb_scale

        # Making sure values are contiguous in memory, in case it matters.
        in_rgb = in_rgb.copy()
        gt_ml_depth = gt_ml_depth.copy()

        with open(txt_filenames[0], 'r') as f:
            cam_params = [float(item) for item in f.read().strip().split()]
        with open(txt_filenames[1], 'r') as f:
            overhead_cam_params = [float(item) for item in f.read().strip().split()]

        # NOTE: target_depth contains nan values. They need to be replaced or excluded when computing the loss function.
        return example_name, in_rgb, gt_ml_depth, count_image, d0, gt_ml_category, foreground_prim, top_down_foreground, top_down_foreground_prim, cam_params, overhead_cam_params, back_of_same_category_depth
