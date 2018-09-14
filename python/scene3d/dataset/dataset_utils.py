import numpy as np
from os import path
import re
import glob
import random
from scene3d import io_utils
from scene3d import log


def grid_indexing_2d(arr_3d: np.ndarray, indices: np.ndarray):
    """
    2d grid indexing of 3d array, along the first dimension.
    :param arr_3d: 3D array of shape (C, H, W)
    :param indices: 2D array of shape (H, W) containing integer values from 0 to C-1. Negative indexing won't work.
    :return: 2D array of shape (H, W), containing values selected from `arr_3d`.
    """
    assert arr_3d.ndim == 3
    assert indices.ndim == 2
    assert arr_3d.shape[1:] == indices.shape
    sz = np.prod(indices.shape).item()  # H*W
    ind_2d = np.arange(sz, dtype=np.int).reshape(indices.shape)
    return arr_3d.ravel()[ind_2d + indices * sz].copy()


def split_dataset(house_id_filename_txt, data_dir, seed=0, ratio=0.8):
    """
    Splits the dataset into two.

    :param house_id_filename_txt:  e.g. renderings_completed.txt file from generate_dataset.py script.
    :param data_dir: Directory where the images were generated. e.g. /data2/scene3d/v8/renderings
    :param ratio: Ratio of houses, not images, in the first split.
    Take subset of it to generate the splits; validation, test, training, etc.
    """
    house_ids = io_utils.read_lines_and_strip(house_id_filename_txt)
    house_ids = sorted(house_ids)

    log.info('{} house ids in {}'.format(len(house_ids), house_id_filename_txt))

    random.seed(seed)
    random.shuffle(house_ids)

    num_first_half = round(len(house_ids) * ratio)

    house_id_splits = [
        house_ids[:num_first_half],
        house_ids[num_first_half:],
    ]

    log.info('{} and {} houses. Ratio: {}'.format(len(house_id_splits[0]), len(house_id_splits[1]), len(house_id_splits[0]) / (len(house_id_splits[0]) + len(house_id_splits[1]))))

    ret = []

    for split in house_id_splits:
        generated_filenames = []
        for house_id in split:
            binary_filenames = glob.glob(path.join(data_dir, '{}/*.bin'.format(house_id)))
            assert len(binary_filenames) > 0, binary_filenames
            generated_filenames.extend(binary_filenames)

        entries = []
        for filename in generated_filenames:
            items = filename.split('/')
            entry = '{}/{}'.format(items[-2], re.match(r'\d+', items[-1])[0])
            entries.append(entry)
        entries = sorted(list(set(entries)))
        ret.append(entries)

    log.info('{} and {} images. Ratio: {}'.format(len(ret[0]), len(ret[1]), len(ret[0]) / (len(ret[0]) + len(ret[1]))))

    return ret


def force_contiguous(arr: np.ndarray):
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr, dtype=arr.dtype)
    assert arr.flags['C_CONTIGUOUS']
    return arr
