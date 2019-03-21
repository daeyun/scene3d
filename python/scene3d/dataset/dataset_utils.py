import numpy as np
from os import path
import typing
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


def divide_start_end_indices(size, num_chunks, offset=0):
    assert num_chunks <= size, (size, num_chunks)
    indices = np.array_split(np.arange(size) + offset, num_chunks)
    ret = [(ind[0], ind[-1] + 1) for ind in indices]

    for start, end in ret:
        assert start < end

    return ret


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    # https://stackoverflow.com/a/312464
    for i in range(0, len(l), n):
        yield l[i:i + n]


def shard_filenames(original_filename, set_name, n) -> typing.List[str]:
    """
    Shard names are 1-indexed.

    :param original_filename: Original filename.
    :param set_name: name of this set. e.g. shuffled, ordered.
    :param n: Number of shards.
    :return: List of strings.
    """
    assert n > 1
    assert isinstance(set_name, str)
    assert isinstance(original_filename, str)
    assert isinstance(n, int)
    prefix, ext = path.splitext(original_filename)

    ret = []

    for i in range(n):
        fname = '{prefix}__{set_name}_{i:04d}_of_{n:04d}{ext}'.format(
            prefix=prefix,
            set_name=set_name,
            ext=ext,
            i=i + 1,
            n=n,
        )
        ret.append(fname)

    return ret


def create_shards(split_filename, num_shards, shuffle=False):
    lines = io_utils.read_lines_and_strip(split_filename)
    assert len(lines) >= num_shards

    if shuffle:
        rand = random.Random()
        rand.seed(42)
        rand.shuffle(lines)
        set_name = 'shuffled'
    else:
        set_name = 'ordered'

    new_filenames = shard_filenames(split_filename, set_name, num_shards)

    if len(lines) % num_shards == 0:
        sharded_lines = list(chunks(lines, len(lines) // num_shards))
    else:
        sharded_lines = list(chunks(lines, len(lines) // num_shards + 1))
    assert len(sharded_lines) == num_shards
    assert len(sharded_lines) == len(new_filenames)

    for sharded_lines_i, new_filename_i in zip(sharded_lines, new_filenames):
        log.info('Writing {} lines in {}'.format(len(sharded_lines_i), new_filename_i))
        with open(new_filename_i, 'w') as f:
            f.write('\n'.join(sharded_lines_i))

    return new_filenames
