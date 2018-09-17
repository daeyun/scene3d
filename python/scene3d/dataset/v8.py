from os import path

import cv2
import numpy as np
from torch.utils import data

from scene3d import config
from scene3d import log
from scene3d import torch_utils
from scene3d import io_utils
from scene3d import render_depth
from scene3d.dataset import dataset_utils


def undo_rgb_whitening(rgb):
    rgb_mean = np.array([178.1781, 158.5039, 142.5141], dtype=np.float32)
    scale = scale = 1.0 / 255

    rgb = torch_utils.recursive_torch_to_numpy(rgb).copy()
    assert rgb.dtype == np.float32

    rgb /= scale

    if rgb.ndim == 4:
        rgb += rgb_mean[None, :, None, None]
    else:
        rgb += rgb_mean[:, None, None]

    # [0, 1]
    ret = rgb / 255.
    return np.minimum(ret, 1.0)  # force <=1.0


class MultiLayerDepth(data.Dataset):
    """
    Includes multi-layer segmentation.

    TODO(daeyun): incomplete implemetnation
    """

    def __init__(self, split='train', first_n=None, start_index=None, rgb_scale=1.0, subtract_mean=True, image_hw=(240, 320),
                 fields=('name', 'camera_filename', 'rgb')):
        if split == 'train':
            split_filename = path.join(config.scene3d_root, 'v8/train.txt')
        elif split == 'test':
            # split_filename = path.join(config.scene3d_root, 'v8/validation.txt')
            split_filename = path.join(config.scene3d_root, 'v8/test.txt')
        elif split == 'all':
            split_filename = path.join(config.scene3d_root, 'v8/all.txt')
        else:
            raise RuntimeError('Invalid split name: {}'.format(split))

        self.filename_prefixes = io_utils.read_lines_and_strip(split_filename)

        if first_n is not None and first_n > 0:
            assert start_index is None
            self.filename_prefixes = self.filename_prefixes[:first_n]
        elif start_index is not None and start_index > 0:
            assert first_n is None
            self.filename_prefixes = self.filename_prefixes[start_index:]

        self.image_hw = image_hw
        self.rgb_mean = np.array([178.1781, 158.5039, 142.5141], dtype=np.float32)
        self.rgb_scale = rgb_scale
        self.subtract_mean = subtract_mean

        valid_fields = (
            'rgb',
            'depth',
            'depth_overhead',
            'features_overhead',
            'name',
            'camera_filename',
        )

        assert isinstance(fields, (tuple, list))

        for field in fields:
            if field not in valid_fields:
                raise RuntimeError('Unknown field name: {}', field)

        self.fields = fields

    def __len__(self):
        return len(self.filename_prefixes)

    def get_rgb(self, example_name, ret):
        field_name = 'rgb'
        if field_name not in self.fields or field_name in ret:
            return
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
        in_rgb = dataset_utils.force_contiguous(in_rgb)
        ret[field_name] = in_rgb

    def get_depth(self, example_name, ret):
        field_name = 'depth'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_ldi.bin')
        ldi = io_utils.read_array_compressed(bin_filename, dtype=np.float32)
        _, h, w = ldi.shape
        if (h, w) != tuple(self.image_hw):
            # TODO(daeyun): need to double check this.
            ldi = cv2.resize(ldi.transpose(1, 2, 0), dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        ldi = dataset_utils.force_contiguous(ldi)
        ret[field_name] = ldi

    def get_depth_overhead(self, example_name, ret):
        field_name = 'depth_overhead'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_ldi-o.bin')
        ldi_overhead = io_utils.read_array_compressed(bin_filename, dtype=np.float32)
        ldi_overhead = dataset_utils.force_contiguous(ldi_overhead)
        ret[field_name] = ldi_overhead

    def get_features_overhead(self, example_name, ret, version='v1'):
        field_name = 'features_overhead'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/overhead/{}/features'.format(version), example_name + '.bin')
        ldi_overhead = io_utils.read_array_compressed(bin_filename, dtype=np.float16).astype(np.float32)
        ldi_overhead = dataset_utils.force_contiguous(ldi_overhead)
        ret[field_name] = ldi_overhead

    def __getitem__(self, index):
        example_name = self.filename_prefixes[index]
        # TODO: some images don't have background.
        bin_filenames = [
            path.join(config.scene3d_root, 'v8/renderings', example_name + '_ldi.bin'),
            path.join(config.scene3d_root, 'v8/renderings', example_name + '_ldi-o.bin'),
            path.join(config.scene3d_root, 'v8/renderings', example_name + '_model.bin'),
            path.join(config.scene3d_root, 'v8/renderings', example_name + '_model-o.bin'),
            path.join(config.scene3d_root, 'v8/renderings', example_name + '_n.bin'),
            path.join(config.scene3d_root, 'v8/renderings', example_name + '_oit.bin'),
        ]
        txt_filenames = [
            path.join(config.scene3d_root, 'v8/renderings', example_name + '_cam.txt'),
            path.join(config.scene3d_root, 'v8/renderings', example_name + '_aabb.txt'),
        ]

        camera_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_cam.txt')

        ret = {
            'name': example_name,
            'camera_filename': camera_filename,
        }

        self.get_rgb(example_name, ret)
        self.get_depth(example_name, ret)
        self.get_depth_overhead(example_name, ret)
        self.get_features_overhead(example_name, ret)

        return ret
