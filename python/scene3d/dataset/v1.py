from os import path

import cv2
import numpy as np
from torch.utils import data

from scene3d import config
from scene3d import io_utils


class MultiLayerDepth(data.Dataset):
    def __init__(self, train=True, first_n=None, rgb_scale=1.0, subtract_mean=True, image_hw=(480, 640)):
        self.filename_prefixes = io_utils.read_lines_and_strip(path.join(config.scene3d_root, 'v1/renderings.txt'))

        eval_set_size = 16000

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
        bin_filenames = [
            path.join(config.scene3d_root, 'v1/renderings', example_name + '_00.bin'),
            path.join(config.scene3d_root, 'v1/renderings', example_name + '_01.bin'),
        ]

        # Prepare GT depth images.
        depth = io_utils.read_array_compressed(bin_filenames[0], dtype=np.float32)
        background_depth = io_utils.read_array_compressed(bin_filenames[1], dtype=np.float32)
        target_depth = np.stack((depth, background_depth), axis=2)
        h, w, _ = target_depth.shape
        if (h, w) != tuple(self.image_hw):
            target_depth = cv2.resize(target_depth, dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST)

        # 0: room layout.
        # 1: diff between room layout and depth.
        target_depth = np.stack([target_depth[:, :, 1], target_depth[:, :, 1] - target_depth[:, :, 0]], axis=0)

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
        target_depth = target_depth.copy()

        # NOTE: target_depth contains nan values. They need to be replaced or excluded when computing the loss function.
        return example_name, in_rgb, target_depth


class NYU40Segmentation(data.Dataset):
    def __init__(self, train=True, first_n=None, rgb_scale=1.0, subtract_mean=True, image_hw=(480, 640)):
        self.filename_prefixes = io_utils.read_lines_and_strip(path.join(config.scene3d_root, 'v1/renderings.txt'))

        eval_set_size = 16000

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

        # Prepare label.
        label_png_filename = path.join(config.pbrs_root, 'category_v2', example_name + '_category40.png')
        target_category = cv2.imread(label_png_filename, cv2.IMREAD_UNCHANGED)
        assert 2 == target_category.ndim
        if (h, w) != tuple(self.image_hw):
            target_category = cv2.resize(target_category, dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST)

        # Making sure values are contiguous in memory, in case it matters.
        in_rgb = in_rgb.copy()
        target_category = target_category.copy()

        # `in_rgb` has shape (B, 3, h, w)
        # `target_category` has shape (B, h, w)
        return example_name, in_rgb, target_category


class MultiLayerDepthNYU40Segmentation(data.Dataset):
    def __init__(self, train=True, first_n=None, rgb_scale=1.0, subtract_mean=True, image_hw=(480, 640)):
        self.filename_prefixes = io_utils.read_lines_and_strip(path.join(config.scene3d_root, 'v1/renderings.txt'))

        eval_set_size = 16000

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
        bin_filenames = [
            path.join(config.scene3d_root, 'v1/renderings', example_name + '_00.bin'),
            path.join(config.scene3d_root, 'v1/renderings', example_name + '_01.bin'),
        ]

        # Prepare GT depth images.
        depth = io_utils.read_array_compressed(bin_filenames[0], dtype=np.float32)
        background_depth = io_utils.read_array_compressed(bin_filenames[1], dtype=np.float32)
        target_depth = np.stack((depth, background_depth), axis=2)
        h, w, _ = target_depth.shape
        if (h, w) != tuple(self.image_hw):
            target_depth = cv2.resize(target_depth, dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST)

        # 0: room layout.
        # 1: diff between room layout and depth.
        target_depth = np.stack([target_depth[:, :, 1], target_depth[:, :, 1] - target_depth[:, :, 0]], axis=0)

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

        # Prepare segmentation label.
        label_png_filename = path.join(config.pbrs_root, 'category_v2', example_name + '_category40.png')
        target_category = cv2.imread(label_png_filename, cv2.IMREAD_UNCHANGED)
        assert 2 == target_category.ndim
        if (h, w) != tuple(self.image_hw):
            target_category = cv2.resize(target_category, dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST)

        # Making sure values are contiguous in memory, in case it matters.
        in_rgb = in_rgb.copy()
        target_depth = target_depth.copy()
        target_category = target_category.copy()

        # NOTE: target_depth contains nan values. They need to be replaced or excluded when computing the loss function.
        return example_name, in_rgb, (target_depth, target_category)
