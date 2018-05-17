import os
import re
import glob
from os import path

from scene3d import log
from scene3d import config
from scene3d import io_utils
from scene3d import exec_utils

renderer_executable = path.abspath(path.join(path.dirname(__file__), '../../cpp/cmake-build-release/apps/render'))


def run_render(obj_filename, camera_filename, out_dir):
    # TODO(daeyun): This python wrapper is not up-to-date because the c++ code changed.
    raise NotImplementedError()

    assert obj_filename.endswith('.obj'), obj_filename
    assert camera_filename.endswith('.txt'), camera_filename
    io_utils.ensure_dir_exists(out_dir)
    io_utils.assert_file_exists(renderer_executable)
    io_utils.assert_file_exists(obj_filename)
    io_utils.assert_file_exists(camera_filename)

    _, stdout, stderr = exec_utils.run_command([
        renderer_executable,
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
