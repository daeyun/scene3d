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

executable = path.abspath(path.join(path.dirname(__file__), '../../../cpp/cmake-build-release/apps/generate_gt_mesh'))
category_mapping_file = config.category_mapping_csv_filename

# sudo mount -t tmpfs -o size=2600m tmpfs /mnt/ramdisk
tmp_out_root = '/mnt/ramdisk/scene3d/generate_gt_mesh'


def generate_gt_mesh(house_id, camera_filenames, out_dir, dd_factor=15.0, tmp_out_dir=tmp_out_root):
    """
    :param house_id: e.g. '0004d52d1aeeb8ae6de39d6bd993e992'
    :param camera_filenames:  list of .txt filenames containing camera parameters in the v8 directory.
    :param out_dir: Directory where .ply files are saved.
    :param dd_factor: Depth discontinuity factor for the background mesh, relative to pixel footprint.
    :param tmp_out_dir:  This needs to be set, for thread safety. Optional if not running in parallel.
    :return:
    """
    io_utils.assert_file_exists(executable)
    assert isinstance(house_id, str)

    if isinstance(camera_filenames, str):
        camera_filenames = [camera_filenames]
    assert isinstance(camera_filenames, (tuple, list))

    camera_filenames = sorted(camera_filenames)

    camera_lines = []
    for camera_filename in camera_filenames:
        with open(camera_filename, 'r') as f:
            camera_lines.append(f.readline().strip())  # read the first line

    assert io_utils.ensure_dir_exists(out_dir)
    io_utils.ensure_dir_exists(tmp_out_dir)

    tmp_camera_filename = path.join(tmp_out_dir, 'gt_mesh_cam_params.txt')
    with open(tmp_camera_filename, 'w') as f:
        f.write('\n'.join(camera_lines))

    tmp_house_obj_file = path.join(tmp_out_root, 'house_obj_default/gt_coverage/house.obj')
    obj_filename, new_house_json_filename = suncg_utils.house_obj_from_json(house_id=house_id, out_file=tmp_house_obj_file, return_house_json_filename=True)

    _, stdout, stderr = exec_utils.run_command([
        executable,
        '--camera_filename={}'.format(tmp_camera_filename),
        '--out_dir={}'.format(tmp_out_dir),  # Output to temp dir first then rename.
        '--obj={}'.format(obj_filename),
        '--json={}'.format(new_house_json_filename),
        '--category={}'.format(category_mapping_file),
        '--resample_height={}'.format(round(240 * 1.5)),
        '--resample_width={}'.format(round(320 * 1.5)),
        '--dd_factor={:.2f}'.format(dd_factor),
        '--save_objects',
        '--save_background',
    ])

    num_outputs_per_camera = 2

    # There should be no output if it ran successfully.
    # assert stdout == '', stdout
    assert stderr == '', (stdout, stderr)

    output_files = sorted(re.findall(r'Output file: (\S+?)\s', stdout))
    output_files = [item.strip() for item in output_files]
    assert len(camera_filenames) * num_outputs_per_camera == len(output_files), output_files

    # Sanity check
    assert len(output_files) > 0
    assert len(output_files) == len(set(output_files))
    for item in output_files:
        assert item, item

    def extract_camera_id(fname):
        return re.search(r'/(\d+?)_', fname).groups()[0]

    # Rename
    renamed_output_files = []
    count = 0
    for i in range(len(camera_filenames)):
        out_names = []  # for ith camera
        for j in range(num_outputs_per_camera):
            out_names.append(output_files[count])
            count += 1
        assert len(set([extract_camera_id(item) for item in out_names])) == 1  # check all same id.

        old_id = extract_camera_id(out_names[0])
        correct_id = extract_camera_id(camera_filenames[i])
        for fname in out_names:
            new_basename = re.sub(r'^{}_'.format(old_id), '{}_'.format(correct_id), path.basename(fname), count=1)
            renamed_fname = path.join(out_dir, new_basename)
            assert path.isfile(fname), fname
            if old_id != correct_id:
                assert path.basename(fname) != path.basename(renamed_fname)
                assert correct_id in renamed_fname
            print(fname, renamed_fname)
            shutil.copy2(fname, renamed_fname)
            renamed_output_files.append(renamed_fname)

    return renamed_output_files
