import glob
import re
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

"""
Special house ids

19e13fe07c37efba7a739d31cd7b130a     # Apartment with no furniture. A lot of doors.
"""

__house_id_to_camera_ids_cache = None


def _house_id_to_camera_id_mapping() -> typing.Dict[str, typing.Sequence[str]]:
    global __house_id_to_camera_ids_cache
    if __house_id_to_camera_ids_cache is None:
        filenames = load_pbrs_filenames()
        __house_id_to_camera_ids_cache = {}
        for filename in filenames:
            h_id, c_id = parse_house_and_camera_ids_from_string(filename)
            if h_id not in __house_id_to_camera_ids_cache:
                __house_id_to_camera_ids_cache[h_id] = []
            __house_id_to_camera_ids_cache[h_id].append(c_id)
        __house_id_to_camera_ids_cache = {k: tuple(v) for k, v in __house_id_to_camera_ids_cache.items()}
    return __house_id_to_camera_ids_cache


def load_pbrs_filenames(example_name_list=None) -> typing.Sequence[str]:
    if example_name_list is None:
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

    elif example_name_list.endswith('.txt'):
        example_names = io_utils.read_lines_and_strip(example_name_list)
        ret = [path.join(config.pbrs_root, 'mlt_v2', item + '_mlt.png') for item in example_names]
        io_utils.assert_file_exists(ret[0])
        return ret

    else:
        raise NotImplementedError()


def get_camera_params_line(house_id: str, camera_id: typing.Union[str, int] = None) -> typing.Union[typing.Sequence[str], str]:
    """
    `camera_id` is not always sequential. It should match the RGB filenames.
    """
    camera_filename = path.join(config.pbrs_root, 'camera_v2', house_id, 'room_camera.txt')
    lines = io_utils.read_lines_and_strip(camera_filename)
    if camera_id is None:
        return lines
    return lines[int(camera_id)]


def parse_house_id_from_string(s) -> str:
    """
    Returns a base-16 substring of length 32.
    """
    m = re.search(r'(?:[^\da-f]|^)([\da-f]{32})(?:[^\da-f]|$)', s)
    if m is None:
        raise RuntimeError('Could not find house id in {} '.format(s))
    return m.group(1)


def parse_house_and_camera_ids_from_string(s) -> typing.Tuple[str, str]:
    """
    Returns a base-16 substring of length 32, followed by a separator and the camera id.
    Example:
        Input: '/data2/pbrs/mlt_v2/0005b92a9ed6349df155a462947bfdfe/000017_mlt.png'
        Output: ('0005b92a9ed6349df155a462947bfdfe', '000017')
    """
    m = re.search(r'(?:[^\da-f]|^)([\da-f]{32})(?:[^\da-f])([\d]+)(?:[\_\.]|$)', s)
    if m is None:
        raise RuntimeError('Could not find house id in {} '.format(s))
    return m.group(1), m.group(2)


def parse_example_name_from_string(s):
    return '/'.join(parse_house_and_camera_ids_from_string(s))


def camera_ids(house_id: str) -> typing.Sequence[str]:
    """
    Example:
        Input: '0004d52d1aeeb8ae6de39d6bd993e992'
        Output: ('000000', '000001', '000002', '000003', '000004', '000005', '000007')
    """
    mapping = _house_id_to_camera_id_mapping()
    return mapping[house_id]


def save_filtered_pbrs_camera_file(out_filename: str, house_id: str):
    """
    Extracts the camera ids that match the RGB images in PBRS and generates a new camera file, to be used as input to the rendering pipeline.
    """
    assert out_filename.endswith('.txt')
    lines = get_camera_params_line(house_id)
    c_ids = camera_ids(house_id)
    new_camera_file_content = '\n'.join([lines[int(cid)] for cid in c_ids]).strip()
    with open(out_filename, 'w') as f:
        f.write(new_camera_file_content)
