import glob
import json
import typing
from os import path

from scene3d import config
from scene3d import io_utils

excluded_house_ids = {
    '16457772c699601ea06b99ede30e80de',  # obj import error. meshlab doesn't seem to work on this mesh either, for some reason.
    '43071b3dec29e9dcd2ea0e3703e1e020',  # scn2scn segfaults
    '4d6d17661df14cac393403e23f954b71',  # scn2scn segfaults
    'abc44a95da3f3738d4a8629e85ef6405',  # scn2scn segfaults
    'd69819a0392af9ecb62a5889eb8a53d3',  # scn2scn segfaults
}


def load_pbrs_filenames():
    # List of png filenames in pbrs.
    cache_file = path.join(config.pbrs_root, 'mlt_v2_files.json')
    if path.isfile(cache_file):
        with open(cache_file, 'r') as f:
            rel_filenames = json.load(f)
        ret = [path.join(config.pbrs_root, file) for file in rel_filenames]
    else:
        files = glob.glob(path.join(config.pbrs_root, 'mlt_v2/**/*.png'))
        files = sorted(files)
        rel_filenames = [path.relpath(file, config.pbrs_root) for file in files]
        with open(cache_file, 'w') as f:
            json.dump(rel_filenames, f)
        ret = files
    return ret


def get_camera_params_line(house_id: str, camera_id: typing.Union[str, int] = None):
    """
    `camera_id` is not always sequential. It should match the RGB filenames.
    """
    camera_filename = path.join(config.pbrs_root, 'camera_v2', house_id, 'room_camera.txt')
    lines = io_utils.read_lines_and_strip(camera_filename)
    if camera_id is None:
        return lines
    return lines[int(camera_id)]
