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
category_mapping_file = path.abspath(path.join(path.dirname(__file__), '../../resources/ModelCategoryMapping.csv'))


def parse_overhead_cam(s):
    lines = re.findall(r'Overhead camera: (.*)(?:\n|$)', s)
    if lines is None:
        log.error('Could not find "Overhead camera" line in stdout.')
    return lines


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
        '--categories={}'.format(category_mapping_file),  # TODO(daeyun): make sure this is compatible with the github release.
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
