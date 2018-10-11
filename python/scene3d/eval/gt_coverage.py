import os
import re
import glob
import numpy as np
from os import path

from scene3d import log
from scene3d import config
from scene3d import io_utils
from scene3d import exec_utils
from scene3d import suncg_utils

executable = path.abspath(path.join(path.dirname(__file__), '../../../cpp/cmake-build-release/apps/gt_coverage'))
category_mapping_file = config.category_mapping_csv_filename

# sudo mount -t tmpfs -o size=2600m tmpfs /mnt/ramdisk
tmp_out_root = '/mnt/ramdisk/scene3d'


def run_gt_coverage(house_id, example_name):
    io_utils.assert_file_exists(executable)

    tmp_house_obj_file = path.join(tmp_out_root, 'house_obj_default/gt_coverage/house.obj')
    obj_filename, new_house_json_filename = suncg_utils.house_obj_from_json(house_id=house_id, out_file=tmp_house_obj_file, return_house_json_filename=True)

    _, stdout, stderr = exec_utils.run_command([
        executable,
        '--obj={}'.format(obj_filename),
        '--json={}'.format(new_house_json_filename),
        '--example_name={}'.format(example_name),
        '--category={}'.format(category_mapping_file),
    ])

    # There should be no output if it ran successfully.
    # assert stdout == '', stdout
    assert stderr == '', (stdout, stderr)

    all_output = {}

    def collect_output(message_prefix):
        message_prefix_pattern = message_prefix.replace('(', r'\(').replace(')', r'\)')
        output = [float(item) for item in re.findall(r'{}: (\S+?)\s'.format(message_prefix_pattern), stdout)]
        assert len(output) == 4, (output, message_prefix)
        all_output[message_prefix] = output

    collect_output('Frontal coverage')
    collect_output('Frontal coverage (cumulative)')
    collect_output('Overhead coverage')
    collect_output('Overhead coverage (cumulative)')
    collect_output('Combined coverage (cumulative)')

    collect_output('Frontal coverage, obj only')
    collect_output('Frontal coverage, obj only (cumulative)')
    collect_output('Overhead coverage, obj only')
    collect_output('Overhead coverage, obj only (cumulative)')
    collect_output('Combined coverage, obj only (cumulative)')

    return all_output


if __name__ == '__main__':
    out = run_gt_coverage('0004d52d1aeeb8ae6de39d6bd993e992', '0004d52d1aeeb8ae6de39d6bd993e992/000003')
    print(out)
    # run_gt_coverage('0004d52d1aeeb8ae6de39d6bd993e992', '0004d52d1aeeb8ae6de39d6bd993e992/000000')
