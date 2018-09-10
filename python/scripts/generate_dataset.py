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

num_threads = 12
out_root = '/data2/scene3d/v8'
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


def generate_depth_images(thread_id, house_id):
    out_dir = path.join(out_root, 'renderings/{}'.format(house_id))

    io_utils.ensure_dir_exists(out_dir)

    # build house obj
    tmp_house_obj_file = path.join(tmp_out_root, 'house_obj_default/{}/house.obj'.format(thread_id))
    obj_filename, new_house_json_filename = suncg_utils.house_obj_from_json(house_id=house_id, out_file=tmp_house_obj_file, return_house_json_filename=True)

    source_room_camera_file = path.join(config.pbrs_root, 'camera_v2/{}/room_camera.txt'.format(house_id))

    # make new camera file. after filtering out ones ignored provided by pbrs.
    out_room_camera_file = path.join(tmp_out_root, '{}/room_camera.txt'.format(thread_id))
    io_utils.ensure_dir_exists(path.dirname(out_room_camera_file))
    with open(source_room_camera_file, 'r') as f:
        lines = f.readlines()
    new_camera_file_content = ''.join([lines[cid] for cid in camera_ids_by_house_id[house_id]]).strip()
    with open(out_room_camera_file, 'w') as f:
        f.write(new_camera_file_content)

    # Saved to a tmp directory first and then renamed and moved later. Not all cameras were used so the output files needed to be renamed to match pbrs's.
    tmp_render_out_dir = path.join(tmp_out_root, '{}/renderings'.format(thread_id))
    output_files = render_depth.run_render(obj_filename=obj_filename, json_filename=new_house_json_filename, camera_filename=out_room_camera_file,
                                           out_dir=tmp_render_out_dir, hw=(480 // 2, 640 // 2))

    output_bin_files = [item for item in output_files if item.endswith('.bin')]
    output_txt_files = [item for item in output_files if item.endswith('.txt')]

    # sanity check. two images per camera for now. +2 more because of segmentation. +1 because of oc thickness images.
    # TODO(daeyun): some images have no background.
    # assert len(output_bin_files) == len(camera_ids_by_house_id[house_id]) * (6), (len(output_bin_files), len(camera_ids_by_house_id[house_id]))

    # copy the renderings to final output directory.
    camera_ids = camera_ids_by_house_id[house_id]
    output_files_by_camera_id = collections.defaultdict(list)
    for i, output_file in enumerate(output_bin_files):
        # TODO: some images don't have background.
        m = re.findall(r'/(\d+)(_[a-zA-Z_\-]*)?.bin$', output_file)[0]
        camera_index = int(m[0])
        suffix = m[1]  # empty if not a background file.
        camera_id = camera_ids[camera_index]
        new_bin_filename = path.join(out_dir, '{:06d}{}.bin'.format(camera_id, suffix))
        shutil.copyfile(output_file, new_bin_filename)
        output_files_by_camera_id[camera_id].append(new_bin_filename)

    for i, output_file in enumerate(output_txt_files):
        m = re.findall(r'/(\d+)(_[a-zA-Z_\-]*)?.txt$', output_file)[0]
        camera_index = int(m[0])
        suffix = m[1]  # empty if not a background file.
        camera_id = camera_ids[camera_index]
        new_bin_filename = path.join(out_dir, '{:06d}{}.txt'.format(camera_id, suffix))
        shutil.copyfile(output_file, new_bin_filename)

    # Code for generating visualization. Disabled for now.
    # TODO(daeyun): refactor
    if False:
        cmap = matplotlib.cm.get_cmap('viridis')
        cmap_array = np.array([cmap(item)[:3] for item in np.arange(0, 1, 1.0 / 256)]).astype(np.float32)

        for camera_id, camera_image_files in sorted(output_files_by_camera_id.items()):
            camera_images = np.array([io_utils.read_array_compressed(item) for item in camera_image_files])

            max_value = camera_images[~np.isnan(camera_images)].max()
            min_value = camera_images[~np.isnan(camera_images)].min()

            rescaled = ((camera_images - min_value) / (max_value - min_value) * 255).astype(np.uint8)
            colored = cmap_array[rescaled.ravel()].reshape(camera_images.shape + (3,))

            colored[np.isnan(camera_images)] = 1
            colored = (colored * 255).astype(np.uint8)

            for i, filename in enumerate(camera_image_files):
                out_png_filename = filename.split('.bin')[0] + '_vis.png'
                imageio.imwrite(out_png_filename, colored[i])

            pbrs_depth = imageio.imread(path.join(config.pbrs_root, 'depth_v2/{}/{:06d}_depth.png'.format(house_id, camera_id)))
            pbrs_depth = (pbrs_depth - pbrs_depth.min()) / (pbrs_depth.max() - pbrs_depth.min())
            pbrs_depth = (pbrs_depth * 255).astype(np.uint8)

            imageio.imwrite(path.join(out_dir, '{:06d}_depth_rescaled.png'.format(camera_id)), pbrs_depth)


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
            generate_depth_images(thread_id, house_id)
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
