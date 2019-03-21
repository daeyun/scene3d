from scene3d import config
from scene3d import pipeline_overhead
from scene3d.dataset import dataset_utils
from scene3d import io_utils
from scene3d import log
from os import path
import os
import shutil


def overhead_features_verify_shard_exists(split_file, dirname):
    names = io_utils.read_lines_and_strip(split_file)
    for i, name in enumerate(names):
        bin_filename = path.join(dirname, '{}.bin'.format(name))
        if not path.isfile(bin_filename):
            log.error('Does not exist: {}, {}'.format(i, bin_filename))
            # raise RuntimeError()
            return False
    log.info('OK   {}  in  {}'.format(split_file, dirname))
    return True


def get_v9_train_name_to_shard_mapping():
    train_filename = path.join(config.scene3d_root, 'v9', 'train.txt')
    shard_filenames = dataset_utils.shard_filenames(train_filename, 'shuffled', 13)

    ret = {}
    for i, fname in enumerate(shard_filenames):
        names = io_utils.read_lines_and_strip(fname)
        for name in names:
            ret[name] = '{:04d}'.format(i + 1)

    return ret


def check_all_overhead_shards_v9():
    train_filename = path.join(config.scene3d_root, 'v9', 'train.txt')
    shard_filenames = dataset_utils.shard_filenames(train_filename, 'shuffled', 13)

    shard_dir = '/data4/out/scene3d/overhead_pred/'

    for fname in shard_filenames:
        log.info('checking {}'.format(fname))
        overhead_features_verify_shard_exists(fname, shard_dir)


def move_shard(split_file, source_dirname, target_dirname):
    # assert overhead_features_verify_shard_exists(split_file, source_dirname)

    names = io_utils.read_lines_and_strip(split_file)
    for i, name in enumerate(names):
        bin_filename = path.join(source_dirname, '{}.bin'.format(name))
        if path.isfile(bin_filename):
            new_bin_filename = path.join(target_dirname, '{}.bin'.format(name))
            io_utils.ensure_dir_exists(path.dirname(new_bin_filename))
            shutil.move(bin_filename, new_bin_filename)
            log.info(new_bin_filename)

    log.info('OK   {}  in  {}'.format(split_file, source_dirname))
