import os
from os import path

import ujson

from scene3d import log
from scene3d import config
from scene3d import io_utils
from scene3d import exec_utils

scn2scn_executable = path.join(config.cpp_third_party_root, 'install/SUNCGtoolbox/gaps/bin/x86_64/scn2scn')

__suncg_plant_and_person_ids = {"130", "237", "323", "324", "325", "333", "346", "463", "620", "623", "624", "634", "637", "717", "718", "719", "720", "721", "723", "724", "725", "726", "727", "728",
                                "731", "732", "733", "734", "735", "736", "737", "738", "739", "740", "741", "742", "786", "s__1335", "s__1541", "s__1543", "s__1779", "s__1780", "s__1781", "s__1782",
                                "s__1783", "s__1784", "s__1785", "s__1786", "s__1787", "s__1788", "s__1957", "s__1958", "s__1959", "s__1960", "s__1961", "s__1962"}


def __preprocess_house_json(house_json):
    """
    Remove plant and person meshes.
    """
    if 'levels' in house_json:
        for i in range(len(house_json['levels'])):
            if 'nodes' in house_json['levels'][i]:
                for j in range(len(house_json['levels'][i]['nodes'])):
                    if 'modelId' in house_json['levels'][i]['nodes'][j]:
                        model_id = house_json['levels'][i]['nodes'][j]['modelId']
                        if model_id in __suncg_plant_and_person_ids:
                            house_json['levels'][i]['nodes'][j]['valid'] = 0
    return house_json


def house_obj_from_json(house_id, out_file='/tmp/scene3d/house_obj_default/house.obj'):
    assert out_file.endswith('.obj'), out_file
    io_utils.ensure_dir_exists(path.dirname(out_file))
    io_utils.assert_file_exists(scn2scn_executable)
    house_json_filename = path.join(config.suncg_root, 'house/{}/house.json'.format(house_id))
    io_utils.assert_file_exists(house_json_filename)

    with open(house_json_filename, 'r') as f:
        house_json = ujson.load(f)

    house_json = __preprocess_house_json(house_json)

    new_house_json_filename = path.join(path.dirname(out_file), 'house_p.json')
    with open(new_house_json_filename, 'w') as f:
        ujson.dump(house_json, f)

    if path.isfile(out_file):
        os.remove(out_file)

    # CWD should be the directory containing the source json file
    cwd = path.dirname(house_json_filename)

    _, stdout, stderr = exec_utils.run_command([
        scn2scn_executable, new_house_json_filename, out_file,
    ], cwd=cwd)

    # There should be no output if it ran successfully.
    assert stdout == '', stdout
    assert stderr == '', stderr

    io_utils.assert_file_exists(out_file)

    return out_file
