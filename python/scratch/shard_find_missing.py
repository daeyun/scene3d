from scene3d import io_utils
from scene3d import config
import socket
from scene3d import pipeline_overhead
from scene3d.dataset import dataset_utils
from scene3d.dataset import v9
from scene3d import io_utils
from scene3d import log
from os import path
import os
import shutil


def main2():
    train_filename = path.join(config.scene3d_root, 'v9', 'train.txt')
    shard_filenames = dataset_utils.shard_filenames(train_filename, 'shuffled', 13)

    linked_shard_dir = '/mnt/scratch0/daeyuns/data/overhead_pred_shard_links/'

    missing_count = 0
    for i, fname in enumerate(shard_filenames[:-1]):
        names = io_utils.read_lines_and_strip(fname)
        for name in names:
            linked_filename = path.join(linked_shard_dir, '{:04d}/{}.bin'.format(i + 1, name))
            if not path.isfile(linked_filename):
                missing_count += 1
                print(linked_filename)


def main4():
    hostname = socket.gethostname()
    if hostname == 'aleph0':
        s = ['0009', '0010', '0011', '0012']
    elif hostname == 'aleph1':
        s = ['0001', '0002', '0003', '0004']
    elif hostname == 'aleph2':
        s = ['0005', '0006', '0007', '0008']
    else:
        raise RuntimeError()
    print(s)

    lines = sorted(list(set(io_utils.read_lines_and_strip(path.expanduser('~/tmp/missing.txt')))))
    for jj, line in enumerate(lines):
        rightone = False
        for si in s:
            if '/{}/'.format(si) in line:
                rightone = True
                break
        if not rightone:
            continue

        assert path.isfile(line), line
        new_filename = line.replace('/mnt/scratch0/daeyuns/data/overhead_pred_shard_links/', '/home/daeyuns/out/scene3d/overhead_pred_shards/')
        io_utils.ensure_dir_exists(path.dirname(new_filename))
        print(new_filename)
        shutil.move(line, new_filename)


def main():
    d = v9.MultiLayerDepth(split='train', fields=['name', 'overhead_features_v9_v1'])
    for i in range(len(d)):
        out = d[i]['overhead_features_v9_v1']
        print(out.shape)


if __name__ == '__main__':
    main()
