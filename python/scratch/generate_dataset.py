import collections
import glob
import re
import shutil
from os import path

import imageio
import matplotlib.cm
import matplotlib.pyplot as pt
import numpy as np

from scene3d import io_utils
from scene3d import render_depth
from scene3d import suncg_utils
from scene3d import pbrs_utils

# %%

pbrs_filenames = pbrs_utils.load_pbrs_filenames()

# %%

house_ids = sorted(list(set([item.split('/')[-2] for item in pbrs_filenames])))

# %%

out_room_camera_file = '/tmp/scene3d/room_camera.txt'
with open(out_room_camera_file, 'w') as f:
    f.write('\n'.join(pbrs_utils.get_camera_params_line(house_ids[0])))

# %%

house_obj_filename = suncg_utils.house_obj_from_json(house_ids[0])

# %%


render_depth.run_render(obj_filename=house_obj_filename, camera_filename=out_room_camera_file, out_dir='/tmp/scene3d/rendered', hw=(480 / 2, 640 / 2))

# %%

len(pbrs_filenames)

# %%

# %%


# %%

files = glob.glob('/data2/pbrs/mlt_v2/**/*.png')
files = sorted(files)

# %%

files_by_house_id = collections.defaultdict(list)
camera_ids_by_house_id = collections.defaultdict(list)

for file in files:
    m = re.findall(r'mlt_v2/([^/]+?)/(\d+?)_mlt.png', file)
    house_id = m[0][0]
    camera_id = int(m[0][1])
    files_by_house_id[house_id].append(file)
    camera_ids_by_house_id[house_id].append(camera_id)

files_by_house_id = dict(files_by_house_id)
camera_ids_by_house_id = dict(camera_ids_by_house_id)

house_ids = sorted(files_by_house_id.keys())

# %%

first_n = 5

for house_id in house_ids[:first_n]:
    out_dir = '/data2/scene3d/tmp0/renderings/{}'.format(house_id)

    io_utils.ensure_dir_exists(out_dir)

    # build house obj
    obj_filename = suncg_utils.house_obj_from_json(house_id=house_id)
    print(house_id, obj_filename)

    # make new camera file
    source_room_camera_file = '/data2/pbrs/camera_v2/{}/room_camera.txt'.format(house_id)
    out_room_camera_file = '/tmp/scene3d/room_camera.txt'
    with open(source_room_camera_file, 'r') as f:
        lines = f.readlines()
    new_camera_file_content = ''.join([lines[cid] for cid in camera_ids_by_house_id[house_id]]).strip()
    with open(out_room_camera_file, 'w') as f:
        f.write(new_camera_file_content)

    tmp_render_out_dir = '/tmp/scene3d/renderings'
    output_files = render_depth.run_render(obj_filename=obj_filename, camera_filename=out_room_camera_file, out_dir=tmp_render_out_dir)

    # sanity check. two images per camera for now.
    assert len(output_files) == len(camera_ids_by_house_id[house_id]) * 2

    # Code for generating visualization. Disabled for now.
    # TODO(daeyun): refactor
    if True:
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
