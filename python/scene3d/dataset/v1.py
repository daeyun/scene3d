import numpy as np

from torch.utils import data
from scene3d import config
from scene3d import io_utils
from scene3d import log
from os import path
import imageio
import cv2


class MultiLayerDepth(data.Dataset):
    def __init__(self, first_n=None, train=True):
        self.filename_prefixes = io_utils.read_lines_and_strip(path.join(config.scene3d_root, 'v1/renderings.txt'))

        if train:
            train_first_n = int(len(self.filename_prefixes) * 0.95)
            self.filename_prefixes = self.filename_prefixes[:train_first_n]
        else:
            train_first_n = int(len(self.filename_prefixes) * 0.95)
            self.filename_prefixes = self.filename_prefixes[train_first_n:]

        if first_n is not None:
            self.filename_prefixes = self.filename_prefixes[:first_n]

        self.input_image_size = (480, 640)
        self.rgb_mean = np.array([178.17808226, 158.50391329, 142.51412396], dtype=np.float32)

    def __len__(self):
        return len(self.filename_prefixes)

    def __getitem__(self, index):
        example_name = self.filename_prefixes[index]
        bin_filenames = [
            path.join(config.scene3d_root, 'v1/renderings', example_name + '_00.bin'),
            path.join(config.scene3d_root, 'v1/renderings', example_name + '_01.bin'),
        ]

        depth = io_utils.read_array_compressed(bin_filenames[0], dtype=np.float32)
        background_depth = io_utils.read_array_compressed(bin_filenames[1], dtype=np.float32)
        target_depth = np.array([depth, background_depth])

        png_filename = path.join(config.pbrs_root, 'mlt_v2', example_name + '_mlt.png')

        in_bgr = cv2.imread(png_filename)
        in_rgb = cv2.cvtColor(in_bgr, cv2.COLOR_BGR2RGB)
        in_rgb = in_rgb.transpose([2, 0, 1]).astype(np.float32)
        in_rgb -= self.rgb_mean[:, None, None]

        # NOTE: target_depth contains nan values. They need to be replaced or excluded when computing the loss function.
        return example_name, in_rgb, target_depth
