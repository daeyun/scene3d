import collections
import glob
import re
from os import path

from scene3d import pbrs_utils

out_root = '/data2/scene3d/v1'

generated_files = glob.glob(path.join(out_root, 'renderings/**/*.bin'))
camera_ids_by_house_id = collections.defaultdict(lambda: collections.defaultdict(int))
for file in generated_files:
    m = re.findall(r'renderings/([^/]+?)/(\d+?)_.+?.bin', file)
    house_id = m[0][0]
    camera_id = int(m[0][1])
    camera_ids_by_house_id[house_id][camera_id] += 1

num_house_camera_pairs = 0
for house_id in camera_ids_by_house_id.keys():
    for camera_id in camera_ids_by_house_id[house_id].keys():
        # There should be two renderings for each (house, camera) pair, for now.
        assert camera_ids_by_house_id[house_id][camera_id] == 2
        num_house_camera_pairs += 1

excluded_house_ids = {
    '16457772c699601ea06b99ede30e80de',  # obj import error. meshlab doesn't seem to work on this mesh either, for some reason.
    '43071b3dec29e9dcd2ea0e3703e1e020',  # scn2scn segfaults
    '4d6d17661df14cac393403e23f954b71',  # scn2scn segfaults
    'abc44a95da3f3738d4a8629e85ef6405',  # scn2scn segfaults
    'd69819a0392af9ecb62a5889eb8a53d3',  # scn2scn segfaults
}

pbrs_files = pbrs_utils.load_pbrs_filenames()

num_pbrs_files = 0
for file in pbrs_files:
    m = re.findall(r'mlt_v2/([^/]+?)/(\d+?)_mlt.png', file)
    house_id = m[0][0]
    camera_id = int(m[0][1])
    if house_id in excluded_house_ids:
        continue
    num_pbrs_files += 1

assert num_house_camera_pairs == num_pbrs_files
