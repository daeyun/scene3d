import collections
import glob
import os
import threading
import re
import json
import shutil
import time
from os import path
import traceback
import sys

import imageio
import matplotlib.cm
import matplotlib.pyplot as pt
import numpy as np

from scene3d import io_utils
from scene3d import log
from scene3d import feat
from scene3d.dataset import v8
from scene3d import config
from scene3d import render_depth
from scene3d import suncg_utils
from scene3d import pbrs_utils

"""
This script writes a lot of temporary files.
For performance reasons and to save SSD write cycles, the recommended way is to mount a memory mapped volume for those temporary files.
See https://www.jamescoyle.net/how-to/943-create-a-ram-disk-in-linux

> sudo mount -t tmpfs -o size=2600m tmpfs /mnt/ramdisk
"""

num_threads = 28
out_root = '/data3/scene3d/v8/overhead/v2'

# Can be any directory. Temporary output files are written here.
# tmp_out_root = '/tmp/scene3d'
tmp_out_root = '/mnt/ramdisk/scene3d/feat_gen'

global_lock = threading.Lock()
global_lock2 = threading.Lock()
global_start_time = time.time()
files_by_house_id = collections.defaultdict(list)
camera_ids_by_house_id = collections.defaultdict(list)
example_indices = []
num_total = 0

dataset = v8.MultiLayerDepth(split='all', subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=('multi_layer_depth_aligned_background', 'overhead_camera_pose_4params'))


def record_completed_example_index(example_index):
    completed_txt_file = path.join(out_root, 'completed.txt')
    global global_lock2
    with global_lock2:
        with open(completed_txt_file, 'a') as f:
            f.write('{}\n'.format(example_index))


def load_completed_indices():
    completed_txt_file = path.join(out_root, 'completed.txt')
    if not path.isfile(completed_txt_file):
        return []
    with open(completed_txt_file, 'r') as f:
        lines = f.readlines()
    ret = [item.strip() for item in lines if item]
    return ret


def generate_features(thread_id, example_index):
    global dataset

    example = dataset[example_index]
    depth = example['multi_layer_depth_aligned_background'][:3]
    x, y, scale, theta = dataset[example_index]['overhead_camera_pose_4params']
    new_cam_filename = path.join(tmp_out_root, '{}/ortho_cam.txt'.format(thread_id))

    feat.make_overhead_camera_file(new_cam_filename, x, y, scale, theta)
    io_utils.block_until_file_exists(new_cam_filename, sleep_seconds=0.02)

    feat_arr = feat.best_guess_depth_and_frustum_mask(depth, new_cam_filename)
    os.remove(new_cam_filename)  # clean up

    house_id, camera_id = example['name'].split('/')

    out_dir = path.join(out_root, 'features/{}'.format(house_id))

    io_utils.ensure_dir_exists(out_dir)
    out_filename = path.join(out_dir, '{}_b.bin'.format(camera_id))
    io_utils.save_array_compressed(out_filename, feat_arr)
    log.info("{}/{}  (thread {})".format(house_id, camera_id, thread_id))


def thread_worker(thread_id):
    global example_indices
    global global_lock
    global global_start_time
    global num_total

    while True:
        """
        Get a house_id from the global queue, render all images, repeat until queue is empty.
        """
        with global_lock:
            num_remaining = len(example_indices)
            if num_remaining == 0:
                break
            example_index = example_indices.pop()
        num_remaining -= 1
        num_processed = num_total - num_remaining
        remaining_seconds = (time.time() - global_start_time) / num_processed * num_remaining
        log.info('Thread id: {}. processing {}. {} out of {}. ETA: {:.1f} minutes'.format(
            thread_id, example_index, num_processed, num_total, remaining_seconds / 60))

        try:
            generate_features(thread_id, example_index)
            record_completed_example_index(example_index)
        except Exception as ex:
            # TODO(daeyun): This happens for some of the house ids. Need to check later.
            log.warn('There was an error (example index: {}). Skipped for now. Error was:\n {}'.format(example_index, ex))
            traceback.print_exc(file=sys.stderr)


def main():
    files = pbrs_utils.load_pbrs_filenames()

    global files_by_house_id
    global camera_ids_by_house_id
    global example_indices
    global num_total

    for file in files:
        m = re.findall(r'mlt_v2/([^/]+?)/(\d+?)_mlt.png', file)
        house_id = m[0][0]
        camera_id = int(m[0][1])
        files_by_house_id[house_id].append(file)
        camera_ids_by_house_id[house_id].append(camera_id)

    files_by_house_id = dict(files_by_house_id)
    camera_ids_by_house_id = dict(camera_ids_by_house_id)

    example_indices = np.arange(len(dataset)).tolist()

    completed_house_ids = set(load_completed_indices())
    remaining_house_ids = [int(house_id) for house_id in example_indices if house_id not in completed_house_ids]
    assert len(remaining_house_ids) + len(completed_house_ids) == len(example_indices)  # sanity check

    # This is the global work queue. Excludes house ids present in the completed file list.
    example_indices = remaining_house_ids[::-1]  # list is popped from the end.
    num_total = len(example_indices)

    processes = [threading.Thread(target=thread_worker, args=(tid,)) for tid in range(num_threads)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
