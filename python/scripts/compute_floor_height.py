import collections
import glob
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

num_threads = 12
out_root = '/data2/scene3d/v8_re'

# Can be any directory. Temporary output files are written here.
# tmp_out_root = '/tmp/scene3d'
tmp_out_root = '/mnt/ramdisk/scene3d'

global_lock = threading.Lock()
global_start_time = time.time()
files_by_house_id = collections.defaultdict(list)
camera_ids_by_house_id = collections.defaultdict(list)
house_ids = []
num_total = 0


def record_completed_house_id(house_id):
    completed_txt_file = path.join(out_root, 'renderings_completed.txt')
    with open(completed_txt_file, 'a') as f:
        f.write('{}\n'.format(house_id))


def load_completed_house_ids():
    completed_txt_file = path.join(out_root, 'renderings_completed.txt')
    if not path.isfile(completed_txt_file):
        return []
    with open(completed_txt_file, 'r') as f:
        lines = f.readlines()
    ret = [item.strip() for item in lines if item]
    return ret


def compute_floor_height(thread_id, house_id):
    out_dir = path.join(out_root, 'renderings/{}'.format(house_id))
    log.info("Processing house id: {}  (thread {})".format(house_id, thread_id))

    io_utils.ensure_dir_exists(out_dir)

    # build house obj
    tmp_house_obj_file = path.join(tmp_out_root, 'house_obj_default/{}/house.obj'.format(thread_id))
    obj_filename, new_house_json_filename = suncg_utils.house_obj_from_json(house_id=house_id, out_file=tmp_house_obj_file, return_house_json_filename=True)

    camera_filenames = sorted(glob.glob('/data2/scene3d/v8/renderings/{}/*_cam.txt'.format(house_id)))
    new_camera_lines = []
    for item in camera_filenames:
        with open(item) as f:
            lines = f.readlines()
        new_camera_lines.append(lines[1].strip())
    print('Source camera_files: {}'.format(camera_filenames))
    tmp_camera_file = path.join(tmp_out_root, 'house_obj_default/{}/camera.txt'.format(thread_id))
    with open(tmp_camera_file, 'w') as f:
        f.write('\n'.join(new_camera_lines))
    print('camera file: {}'.format(tmp_camera_file))

    # Saved to a tmp directory first and then renamed and moved later. Not all cameras were used so the output files needed to be renamed to match pbrs's.
    heights = render_depth.run_floor_height(obj_filename=obj_filename, json_filename=new_house_json_filename, camera_filename=tmp_camera_file)

    print(heights)

    with open(path.join(out_dir, 'floor_heights.txt'), 'w') as f:
        f.write('\n'.join(heights))




def thread_worker(thread_id):
    global house_ids
    global global_lock
    global global_start_time
    global num_total

    while True:
        """
        Get a house_id from the global queue, render all images, repeat until queue is empty.
        """
        with global_lock:
            if len(house_ids) == 0:
                break
            house_id = house_ids[0]
            # This is a bad way to implement a consumer queue but takes < 0.01 second.
            house_ids = house_ids[1:]

            num_remaining = len(house_ids)
            num_processed = num_total - num_remaining
            remaining_seconds = (time.time() - global_start_time) / num_processed * num_remaining
            log.info('Thread id: {}. processing {}. {} out of {}. ETA: {:.1f} minutes'.format(
                thread_id, house_id, num_processed, num_total, remaining_seconds / 60))

        try:
            compute_floor_height(thread_id, house_id)
            record_completed_house_id(house_id)
        except Exception as ex:
            # TODO(daeyun): This happens for some of the house ids. Need to check later.
            log.warn('There was an error (house id: {}). Skipped for now. Error was:\n {}'.format(house_id, ex))
            traceback.print_exc(file=sys.stderr)


def main():
    files = pbrs_utils.load_pbrs_filenames()

    global files_by_house_id
    global camera_ids_by_house_id
    global house_ids
    global num_total

    for file in files:
        m = re.findall(r'mlt_v2/([^/]+?)/(\d+?)_mlt.png', file)
        house_id = m[0][0]
        camera_id = int(m[0][1])
        files_by_house_id[house_id].append(file)
        camera_ids_by_house_id[house_id].append(camera_id)

    files_by_house_id = dict(files_by_house_id)
    camera_ids_by_house_id = dict(camera_ids_by_house_id)

    house_ids = [item for item in sorted(files_by_house_id.keys()) if item not in pbrs_utils.excluded_house_ids]

    completed_house_ids = set(load_completed_house_ids())
    remaining_house_ids = [house_id for house_id in house_ids if house_id not in completed_house_ids]
    assert len(remaining_house_ids) + len(completed_house_ids) == len(house_ids)  # sanity check

    # This is the global work queue. Excludes house ids present in the completed file list.
    house_ids = remaining_house_ids
    num_total = len(house_ids)

    processes = [threading.Thread(target=thread_worker, args=(tid,)) for tid in range(num_threads)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
