import os
import re
import glob
import numpy as np
from os import path

from scene3d import log
from scene3d import config
from scene3d import io_utils
from scene3d import exec_utils

renderer_executable = path.abspath(path.join(path.dirname(__file__), '../../cpp/cmake-build-release/apps/render'))


def run_render(obj_filename, camera_filename, out_dir, hw=(480, 640)):
    assert obj_filename.endswith('.obj'), obj_filename
    assert camera_filename.endswith('.txt'), camera_filename
    io_utils.ensure_dir_exists(out_dir)
    io_utils.assert_file_exists(renderer_executable)
    io_utils.assert_file_exists(obj_filename)
    io_utils.assert_file_exists(camera_filename)

    _, stdout, stderr = exec_utils.run_command([
        renderer_executable,
        '--height={}'.format(int(hw[0])),
        '--width={}'.format(int(hw[1])),
        '--obj={}'.format(obj_filename),
        '--cameras={}'.format(camera_filename),
        '--out_dir={}'.format(out_dir),
    ])

    # There should be no output if it ran successfully.
    # assert stdout == '', stdout
    # assert stderr == '', stderr

    output_files = sorted(re.findall(r'Output file: (\S+?)\s', stdout))
    output_files = [item.strip() for item in output_files]

    # Sanity check
    assert len(output_files) > 0
    assert len(output_files) == len(set(output_files))
    for item in output_files:
        assert item, item

    return output_files


def grid_indexing_2d(arr_3d: np.ndarray, indices: np.ndarray):
    """
    2d grid indexing of 3d array, along the first dimension.
    :param arr_3d: 3D array of shape (C, H, W)
    :param indices: 2D array of shape (H, W) containing integer values from 0 to C-1. Negative indexing won't work.
    :return: 2D array of shape (H, W), containing values selected from `arr_3d`.
    """
    assert arr_3d.ndim == 3
    assert indices.ndim == 2
    assert arr_3d.shape[1:] == indices.shape
    sz = np.prod(indices.shape).item()  # H*W
    ind_2d = np.arange(sz, dtype=np.int).reshape(indices.shape)
    return arr_3d.ravel()[ind_2d + indices * sz].copy()
