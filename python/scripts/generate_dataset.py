import collections
import glob
import threading
import re
import shutil
import time
from os import path

import imageio
import matplotlib.cm
import matplotlib.pyplot as pt
import numpy as np

from scene3d import io_utils
from scene3d import log
from scene3d import render_depth
from scene3d import suncg_utils

num_threads = 11
global_lock = threading.Lock()
global_start_time = time.time()

files_by_house_id = collections.defaultdict(list)
camera_ids_by_house_id = collections.defaultdict(list)
house_ids = []

out_root = '/data2/scene3d/v1'

# Can be any directory. Temporary output files are written here.
tmp_out_root = '/tmp/scene3d'


def thread_worker(thread_id):
    global house_ids
    global global_lock
    global global_start_time

    while True:
        """
        Get a house_id from the global queue, render all images, repeat until queue is empty.
        """
        # TODO: In case the script is interrupted, continue from where it left off.
        with global_lock:
            if len(house_ids) == 0:
                break
            house_id = house_ids[0]
            # This is a bad way to implement a consumer queue but takes < 0.01 second.
            house_ids = house_ids[1:]

            num_remaining = len(house_ids)
            num_total = len(files_by_house_id)
            num_processed = num_total - num_remaining
            remaining_seconds = (time.time() - global_start_time) / num_processed * num_remaining
            log.info('Thread id: {}. processing {}. {} out of {}. ETA: {:.1f} minutes'.format(
                thread_id, house_id, num_processed, num_total, remaining_seconds / 60))

        out_dir = path.join(out_root, 'renderings/{}'.format(house_id))

        io_utils.ensure_dir_exists(out_dir)

        # build house obj
        tmp_house_obj_file = path.join(tmp_out_root, 'house_obj_default/{}/house.obj'.format(thread_id))
        obj_filename = suncg_utils.house_obj_from_json(house_id=house_id, out_file=tmp_house_obj_file)

        source_room_camera_file = '/data2/pbrs/camera_v2/{}/room_camera.txt'.format(house_id)

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
        output_files = render_depth.run_render(obj_filename=obj_filename, camera_filename=out_room_camera_file, out_dir=tmp_render_out_dir)

        # sanity check. two images per camera for now.
        assert len(output_files) == len(camera_ids_by_house_id[house_id]) * 2

        # copy the renderings to final output directory.
        camera_ids = camera_ids_by_house_id[house_id]
        output_files_by_camera_id = collections.defaultdict(list)
        for i, output_file in enumerate(output_files):
            m = re.findall(r'/(\d+)_(\d+).bin$', output_file)[0]
            camera_index = int(m[0])
            image_index = int(m[1])
            camera_id = camera_ids[camera_index]
            new_bin_filename = path.join(out_dir, '{:06d}_{:02d}.bin'.format(camera_id, image_index))
            shutil.copyfile(output_file, new_bin_filename)
            output_files_by_camera_id[camera_id].append(new_bin_filename)

        # Code for generating visualization. Disabled for now.
        # TODO(daeyun): refactor
        if True:
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

                pbrs_depth = imageio.imread('/data2/pbrs/depth_v2/{}/{:06d}_depth.png'.format(house_id, camera_id))
                pbrs_depth = (pbrs_depth - pbrs_depth.min()) / (pbrs_depth.max() - pbrs_depth.min())
                pbrs_depth = (pbrs_depth * 255).astype(np.uint8)

                imageio.imwrite(path.join(out_dir, '{:06d}_depth_rescaled.png'.format(camera_id)), pbrs_depth)


def main():
    files = glob.glob('/data2/pbrs/mlt_v2/**/*.png')
    files = sorted(files)

    global files_by_house_id
    global camera_ids_by_house_id
    global house_ids

    for file in files:
        m = re.findall(r'mlt_v2/([^/]+?)/(\d+?)_mlt.png', file)
        house_id = m[0][0]
        camera_id = int(m[0][1])
        files_by_house_id[house_id].append(file)
        camera_ids_by_house_id[house_id].append(camera_id)

    files_by_house_id = dict(files_by_house_id)
    camera_ids_by_house_id = dict(camera_ids_by_house_id)

    house_ids = sorted(files_by_house_id.keys())

    processes = [threading.Thread(target=thread_worker, args=(tid,)) for tid in range(num_threads)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
