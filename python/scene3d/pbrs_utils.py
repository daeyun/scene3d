import glob
import json
from os import path

from scene3d import config


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
