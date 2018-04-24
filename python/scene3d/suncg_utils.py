import os
from os import path

from scene3d import log
from scene3d import config
from scene3d import io_utils
from scene3d import exec_utils

scn2scn_executable = path.abspath(path.join(path.dirname(__file__), '../../cpp/third_party/install/SUNCGtoolbox/gaps/bin/x86_64/scn2scn'))


def house_obj_from_json(house_id, out_file='/tmp/scene3d/house_obj_default/house.obj'):
    assert out_file.endswith('.obj'), out_file
    io_utils.ensure_dir_exists(path.dirname(out_file))
    io_utils.assert_file_exists(scn2scn_executable)
    house_json = path.join(config.suncg_root, 'house/{}/house.json'.format(house_id))
    io_utils.assert_file_exists(house_json)

    if path.isfile(out_file):
        os.remove(out_file)

    _, stdout, stderr = exec_utils.run_command([
        scn2scn_executable, house_json, out_file,
    ], cwd=path.dirname(house_json))

    # There should be no output if it ran successfully.
    assert stdout == '', stdout
    assert stderr == '', stderr

    io_utils.assert_file_exists(out_file)

    return out_file
