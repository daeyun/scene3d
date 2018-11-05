import math
import re
from os import path

import cv2
import numpy as np
import numpy.linalg as la
from torch.utils import data

from scene3d import config
from scene3d import log
from scene3d import training_utils_cpp
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


def is_filename_prefix_valid(value):
    return re.match(r'^[a-z0-9]{32}/\d{6}$', value) is not None


class MultiLayerDepth(data.Dataset):
    """
    Includes multi-layer segmentation.

    TODO(daeyun): incomplete implemetnation
    """

    def __init__(self, split='train', first_n=None, start_index=None, rgb_scale=1.0, subtract_mean=True, image_hw=(240, 320),
                 fields=('name', 'camera_filename', 'rgb')):
        if split == 'train':
            split_filename = path.join(config.scene3d_root, 'v8/train_v2.txt')
        elif split == 'test':
            # split_filename = path.join(config.scene3d_root, 'v8/validation.txt')
            split_filename = path.join(config.scene3d_root, 'v8/test_v2.txt')
        elif split == 'all':
            split_filename = path.join(config.scene3d_root, 'v8/all_v2.txt')
        elif split.startswith('/') and split.endswith('.txt') and path.isfile(split):
            split_filename = split
        else:
            raise RuntimeError('Invalid split name: {}'.format(split))

        log.info('Loading examples from file {}'.format(split_filename))

        self.filename_prefixes = io_utils.read_lines_and_strip(split_filename)

        # Make sure the text file contains valid example names.
        assert is_filename_prefix_valid(self.filename_prefixes[0]) and is_filename_prefix_valid(self.filename_prefixes[-1])

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
            'input_depth',
            'multi_layer_depth',
            'multi_layer_depth_and_input_depth',
            'multi_layer_depth_aligned_background',
            'multi_layer_depth_aligned_background_and_input_depth',
            'multi_layer_depth_replicated_background',
            'multi_layer_overhead_depth',
            'overhead_features',
            'name',
            'camera_filename',
            'normals',
            'normal_direction_volume',
            'model_id',
            'category_nyu40',
            'category_nyu40_merged_background',
            'category_nyu40_merged_background_replicated',
            'overhead_category_nyu40',
            'overhead_category_nyu40_merged_background',
            'overhead_camera_pose_3params',
            'overhead_camera_pose_4params',
        )

        assert isinstance(fields, (tuple, list))

        for field in fields:
            if field not in valid_fields:
                raise RuntimeError('Unknown field name: {}', field)

        self.fields = fields

        training_utils_cpp.initialize_category_mapping()  # Should run only once.

        self.depth_mean = 2.8452337
        self.depth_std = 1.1085573

    def __len__(self):
        return len(self.filename_prefixes)

    def get_rgb(self, example_name, ret):
        field_name = 'rgb'
        if field_name not in self.fields or field_name in ret:
            return
        png_filename = path.join(config.pbrs_root, 'mlt_v2', example_name + '_mlt.png')
        in_rgb = cv2.cvtColor(cv2.imread(png_filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
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

    def get_input_depth(self, example_name, ret):
        field_name = 'input_depth'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_ldi.bin')
        ldi = io_utils.read_array_compressed(bin_filename, dtype=np.float32)
        depth = ldi[:1]
        depth = (depth - self.depth_mean) / (self.depth_std * 3)
        depth[np.isnan(depth)] = 0  # NOTE: This is used as input. So nan values are zeroed.
        c, h, w = depth.shape
        assert c == 1
        if (h, w) != tuple(self.image_hw):
            # TODO(daeyun): need to double check this.
            depth = cv2.resize(depth.transpose(1, 2, 0), dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        depth = dataset_utils.force_contiguous(depth)
        ret[field_name] = depth

    def get_multi_layer_depth(self, example_name, ret):
        field_name = 'multi_layer_depth'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_ldi.bin')
        ldi = io_utils.read_array_compressed(bin_filename, dtype=np.float32)
        _, h, w = ldi.shape
        if (h, w) != tuple(self.image_hw):
            # TODO(daeyun): need to double check this.
            ldi = cv2.resize(ldi.transpose(1, 2, 0), dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

        # fix bug in last-exit layer
        last_exit_mask = np.isnan(ldi[2])
        ldi[2][last_exit_mask] = ldi[1][last_exit_mask]

        ldi = dataset_utils.force_contiguous(ldi)
        ret[field_name] = ldi

    def get_multi_layer_depth_and_input_depth(self, example_name, ret):
        field_name = 'multi_layer_depth_and_input_depth'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_ldi.bin')
        ldi = io_utils.read_array_compressed(bin_filename, dtype=np.float32)
        _, h, w = ldi.shape
        if (h, w) != tuple(self.image_hw):
            # TODO(daeyun): need to double check this.
            ldi = cv2.resize(ldi.transpose(1, 2, 0), dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

        depth = ldi[:1].copy()  # traditional depth
        depth = (depth - self.depth_mean) / (self.depth_std * 3)
        depth[np.isnan(depth)] = 0  # NOTE: This is used as input. So nan values are zeroed.

        # fix bug in last-exit layer
        last_exit_mask = np.isnan(ldi[2])
        ldi[2][last_exit_mask] = ldi[1][last_exit_mask]

        # Depth is in the first channel.
        depth_and_ldi = np.concatenate((depth, ldi), axis=0)
        depth_and_ldi = dataset_utils.force_contiguous(depth_and_ldi)
        ret[field_name] = depth_and_ldi

    def get_multi_layer_depth_aligned_background(self, example_name, ret):
        field_name = 'multi_layer_depth_aligned_background'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_ldi.bin')
        ldi = io_utils.read_array_compressed(bin_filename, dtype=np.float32)
        _, h, w = ldi.shape
        if (h, w) != tuple(self.image_hw):
            # TODO(daeyun): need to double check this.
            ldi = cv2.resize(ldi.transpose(1, 2, 0), dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

        # Swap visible background in channels 0 and 3.
        bg_mask = np.isnan(ldi[1])
        ldi[0][bg_mask], ldi[3][bg_mask] = ldi[3][bg_mask], ldi[0][bg_mask]

        # fix bug in last-exit layer
        last_exit_mask = np.isnan(ldi[2])
        ldi[2][last_exit_mask] = ldi[1][last_exit_mask]

        ldi = dataset_utils.force_contiguous(ldi)
        ret[field_name] = ldi

    def get_multi_layer_depth_aligned_background_and_input_depth(self, example_name, ret):
        field_name = 'multi_layer_depth_aligned_background_and_input_depth'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_ldi.bin')
        ldi = io_utils.read_array_compressed(bin_filename, dtype=np.float32)
        _, h, w = ldi.shape
        if (h, w) != tuple(self.image_hw):
            # TODO(daeyun): need to double check this.
            ldi = cv2.resize(ldi.transpose(1, 2, 0), dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

        depth = ldi[:1].copy()  # traditional depth
        depth = (depth - self.depth_mean) / (self.depth_std * 3)
        depth[np.isnan(depth)] = 0  # NOTE: This is used as input. So nan values are zeroed.

        # Swap visible background in channels 0 and 3.
        bg_mask = np.isnan(ldi[3])
        ldi[0][bg_mask], ldi[3][bg_mask] = ldi[3][bg_mask], ldi[0][bg_mask]

        # fix bug in last-exit layer
        last_exit_mask = np.isnan(ldi[2])
        ldi[2][last_exit_mask] = ldi[1][last_exit_mask]

        # Depth is in the first channel.
        depth_and_ldi = np.concatenate((depth, ldi), axis=0)
        depth_and_ldi = dataset_utils.force_contiguous(depth_and_ldi)
        ret[field_name] = depth_and_ldi

    def get_multi_layer_depth_replicated_background(self, example_name, ret):
        field_name = 'multi_layer_depth_replicated_background'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_ldi.bin')
        ldi = io_utils.read_array_compressed(bin_filename, dtype=np.float32)
        _, h, w = ldi.shape
        if (h, w) != tuple(self.image_hw):
            # TODO(daeyun): need to double check this.
            ldi = cv2.resize(ldi.transpose(1, 2, 0), dsize=(self.image_hw[1], self.image_hw[0]), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

        # Copy visible background to all channels.
        bg_mask = np.isnan(ldi[1])
        ldi[1][bg_mask] = ldi[2][bg_mask] = ldi[3][bg_mask] = ldi[0][bg_mask]

        # fix bug in last-exit layer
        last_exit_mask = np.isnan(ldi[2])
        ldi[2][last_exit_mask] = ldi[1][last_exit_mask]

        ldi = dataset_utils.force_contiguous(ldi)
        ret[field_name] = ldi

    def get_multi_layer_overhead_depth(self, example_name, ret):
        field_name = 'multi_layer_overhead_depth'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_ldi-o.bin')
        ldi_overhead = io_utils.read_array_compressed(bin_filename, dtype=np.float32)
        ldi_overhead = dataset_utils.force_contiguous(ldi_overhead)
        ret[field_name] = ldi_overhead

    def get_overhead_features(self, example_name, ret, version='v1'):
        field_name = 'overhead_features'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/overhead/{}/features'.format(version), example_name + '.bin')
        ldi_overhead = io_utils.read_array_compressed(bin_filename, dtype=np.float16).astype(np.float32)
        ldi_overhead = dataset_utils.force_contiguous(ldi_overhead)
        ret[field_name] = ldi_overhead

    def get_normals(self, example_name, ret):
        field_name = 'normals'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_n.bin')
        d = io_utils.read_array_compressed(bin_filename, dtype=np.float32)
        d = dataset_utils.force_contiguous(d)
        ret[field_name] = d

    def get_normal_direction_volume(self, example_name, ret):
        field_name = 'normal_direction_volume'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_oit.bin')
        d = io_utils.read_array_compressed(bin_filename, dtype=np.float32)
        d = dataset_utils.force_contiguous(d)
        assert d.ndim == 2
        ret[field_name] = d[None]

    def get_model_id(self, example_name, ret):
        field_name = 'model_id'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_model.bin')
        d = io_utils.read_array_compressed(bin_filename, dtype=np.uint16)
        d = dataset_utils.force_contiguous(d)
        ret[field_name] = d

    def get_model_id_overhead(self, example_name, ret):
        field_name = 'model_id_overhead'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_model-o.bin')
        d = io_utils.read_array_compressed(bin_filename, dtype=np.uint16)
        d = dataset_utils.force_contiguous(d)
        ret[field_name] = d.astype(np.int)

    def get_category_nyu40(self, example_name, ret):
        field_name = 'category_nyu40'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_model.bin')
        d = io_utils.read_array_compressed(bin_filename, dtype=np.uint16)
        d = dataset_utils.force_contiguous(d)
        training_utils_cpp.model_index_to_category(d, mapping_name="nyuv2_40class")

        # fix bug in last-exit layer
        last_exit_mask = d[2] == 65535
        d[2][last_exit_mask] = d[1][last_exit_mask]

        d = dataset_utils.force_contiguous(d.astype(np.int))
        ret[field_name] = d

    def get_category_nyu40_merged_background(self, example_name, ret):
        field_name = 'category_nyu40_merged_background'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_model.bin')
        d = io_utils.read_array_compressed(bin_filename, dtype=np.uint16)
        d = dataset_utils.force_contiguous(d)
        training_utils_cpp.model_index_to_category(d, mapping_name="nyuv2_40class_merged_background")

        bg_mask = d[0] == 34  # merged into wall category (34). Ignored in layers >=1.
        d[1][bg_mask] = d[2][bg_mask] = d[3][bg_mask] = 65535

        # fix bug in last-exit layer
        last_exit_mask = d[2] == 65535
        d[2][last_exit_mask] = d[1][last_exit_mask]

        d = dataset_utils.force_contiguous(d.astype(np.int))
        ret[field_name] = d

    def get_category_nyu40_merged_background_replicated(self, example_name, ret):
        field_name = 'category_nyu40_merged_background_replicated'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_model.bin')
        d = io_utils.read_array_compressed(bin_filename, dtype=np.uint16)
        d = dataset_utils.force_contiguous(d)
        training_utils_cpp.model_index_to_category(d, mapping_name="nyuv2_40class_merged_background")

        bg_mask = d[0] == 34  # merged into wall category (34). Ignored in layers >=1.
        d[1][bg_mask] = d[2][bg_mask] = d[3][bg_mask] = d[0][bg_mask]

        # fix bug in last-exit layer
        last_exit_mask = d[2] == 65535
        d[2][last_exit_mask] = d[1][last_exit_mask]

        d = dataset_utils.force_contiguous(d.astype(np.int))
        ret[field_name] = d

    def get_overhead_category_nyu40(self, example_name, ret):
        field_name = 'overhead_category_nyu40'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_model-o.bin')
        d = io_utils.read_array_compressed(bin_filename, dtype=np.uint16)
        d = dataset_utils.force_contiguous(d)
        training_utils_cpp.model_index_to_category(d, mapping_name="nyuv2_40class")
        ret[field_name] = d.astype(np.int)

    def get_overhead_category_nyu40_merged_background(self, example_name, ret):
        field_name = 'overhead_category_nyu40_merged_background'
        if field_name not in self.fields or field_name in ret:
            return
        bin_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_model-o.bin')
        d = io_utils.read_array_compressed(bin_filename, dtype=np.uint16)
        d = dataset_utils.force_contiguous(d)
        training_utils_cpp.model_index_to_category(d, mapping_name="nyuv2_40class_merged_background")

        bg_mask = d[0] == 34  # merged into wall category (34). Ignored in layers >=1.
        d[1][bg_mask] = d[2][bg_mask] = d[3][bg_mask] = 65535

        d = dataset_utils.force_contiguous(d)
        ret[field_name] = d.astype(np.int)

    def get_overhead_camera_pose_3params(self, example_name, ret):
        field_name = 'overhead_camera_pose_3params'
        if field_name not in self.fields or field_name in ret:
            return

        with open(ret['camera_filename'], 'r') as f:
            content = f.readlines()

        cam_p = content[0].strip().split()
        assert cam_p[0] == 'P'
        cam_o = content[1].strip().split()
        assert cam_o[0] == 'O'

        campos = np.array([float(item) for item in cam_p[1:4]])
        viewdir = np.array([float(item) for item in cam_p[4:7]])
        up = np.array([float(item) for item in cam_p[7:10]])
        right = np.cross(viewdir, up)
        assert np.abs(right[1]) < 1e-5

        # Convert to 2d unit vectors, assuming gravity direction is in the y axis.
        right[1] = 0
        right = right / la.norm(right)
        viewdir[1] = 0
        viewdir = viewdir / la.norm(viewdir)

        campos_o = np.array([float(item) for item in cam_o[1:4]])

        # up_o = np.array([float(item) for item in cam_o[7:10]])
        # assert np.allclose(up_o, viewdir, atol=1e-5, rtol=1e-3)

        dpos = campos_o - campos
        x = np.inner(dpos, right)
        y = np.inner(dpos, viewdir)

        lrbt = [float(item) for item in cam_o[-6:-2]]

        # assert abs(lrbt[0] + lrbt[1]) < 1e-7
        # assert abs(lrbt[2] + lrbt[3]) < 1e-7

        scale = math.hypot(lrbt[1], lrbt[3])

        ret[field_name] = np.array((x, y, scale), dtype=np.float32)

    def get_overhead_camera_pose_4params(self, example_name, ret):
        field_name = 'overhead_camera_pose_4params'
        if field_name not in self.fields or field_name in ret:
            return

        with open(ret['camera_filename'], 'r') as f:
            content = f.readlines()

        cam_p = content[0].strip().split()
        assert cam_p[0] == 'P'
        cam_o = content[1].strip().split()
        assert cam_o[0] == 'O'

        campos = np.array([float(item) for item in cam_p[1:4]])
        viewdir = np.array([float(item) for item in cam_p[4:7]])
        up = np.array([float(item) for item in cam_p[7:10]])
        right = np.cross(viewdir, up)
        assert np.abs(right[1]) < 1e-5

        # Convert to 2d unit vectors, assuming gravity direction is in the y axis.
        right[1] = 0
        right = right / la.norm(right)
        viewdir[1] = 0
        viewdir = viewdir / la.norm(viewdir)

        campos_o = np.array([float(item) for item in cam_o[1:4]])

        # up_o = np.array([float(item) for item in cam_o[7:10]])
        # assert np.allclose(up_o, viewdir, atol=1e-5, rtol=1e-3)

        dpos = campos_o - campos
        x = np.inner(dpos, right)
        y = np.inner(dpos, viewdir)

        lrbt = [float(item) for item in cam_o[-6:-2]]

        # assert abs(lrbt[0] + lrbt[1]) < 1e-7
        # assert abs(lrbt[2] + lrbt[3]) < 1e-7

        scale = math.hypot(lrbt[1], lrbt[3])

        viewdir3d = np.array([float(item) for item in cam_p[4:7]])
        gravity_direction = np.array([0, -1, 0], dtype=np.float64)
        theta = np.arccos(np.inner(viewdir3d, gravity_direction))

        ret[field_name] = np.array((x, y, scale, theta), dtype=np.float32)

    def __getitem__(self, index):
        example_name = self.filename_prefixes[index]
        camera_filename = path.join(config.scene3d_root, 'v8/renderings', example_name + '_cam.txt')

        ret = {
            'name': example_name,
            'camera_filename': camera_filename,
        }

        # some images don't have background.

        self.get_rgb(example_name, ret)
        self.get_input_depth(example_name, ret)
        self.get_multi_layer_depth(example_name, ret)
        self.get_multi_layer_depth_and_input_depth(example_name, ret)
        self.get_multi_layer_depth_aligned_background(example_name, ret)
        self.get_multi_layer_depth_aligned_background_and_input_depth(example_name, ret)
        self.get_multi_layer_depth_replicated_background(example_name, ret)
        self.get_multi_layer_overhead_depth(example_name, ret)
        self.get_overhead_features(example_name, ret)
        self.get_normals(example_name, ret)
        self.get_normal_direction_volume(example_name, ret)
        self.get_model_id(example_name, ret)
        self.get_category_nyu40(example_name, ret)
        self.get_category_nyu40_merged_background(example_name, ret)
        self.get_category_nyu40_merged_background_replicated(example_name, ret)
        self.get_overhead_category_nyu40(example_name, ret)
        self.get_overhead_category_nyu40_merged_background(example_name, ret)
        self.get_overhead_camera_pose_3params(example_name, ret)
        self.get_overhead_camera_pose_4params(example_name, ret)

        return ret
