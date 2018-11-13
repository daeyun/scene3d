import os
import re
import glob
from os import path

import shutil
import numpy as np

from scene3d import log
from scene3d import config
from scene3d import io_utils
from scene3d import exec_utils
from scene3d import suncg_utils

executable = path.abspath(path.join(path.dirname(__file__), '../../../cpp/cmake-build-release/apps/extract_frustum_mesh'))
category_mapping_file = config.category_mapping_csv_filename


def extract_mesh_inside_frustum(mesh_filename, camera_filename, camera_index, out_filename):
    io_utils.assert_file_exists(executable)
    io_utils.assert_file_exists(mesh_filename)
    io_utils.assert_file_exists(camera_filename)
    io_utils.ensure_dir_exists(path.dirname(out_filename))
    assert camera_index >= 0

    _, stdout, stderr = exec_utils.run_command([
        executable,
        '--camera_filename={}'.format(camera_filename),
        '--camera_index={}'.format(camera_index),
        '--mesh={}'.format(mesh_filename),
        '--out={}'.format(out_filename),
    ])

    io_utils.assert_file_exists(out_filename)

    return out_filename
