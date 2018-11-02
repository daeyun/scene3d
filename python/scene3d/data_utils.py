import collections
import re


def defaultdict_to_dict(d):
    if isinstance(d, collections.defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def extract_global_step_from_filename(filename):
    if isinstance(filename, int):
        return filename
    elif isinstance(filename, str):
        return int(re.search(r'(\d{8})_\d{3}_\d{7}', filename).groups()[0])
    else:
        raise RuntimeError(str(filename))
