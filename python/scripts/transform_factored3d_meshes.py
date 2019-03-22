from scene3d.eval import f3d_utils
import glob
import shutil
import uuid
from scene3d import io_utils
import os
import pickle
import time
from os import path
import numpy as np

from scene3d import depth_mesh_utils_cpp
import numpy.linalg as la
from scene3d import scannet_utils
from scene3d import camera
from scene3d import config
from scene3d import scannet_utils
from scene3d.eval import generate_gt_mesh
from scene3d.dataset import v8


def main_v8():
    dataset = v8.MultiLayerDepth(
        split='/data2/scene3d/v8/test_v2_subset_factored3d.txt',
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=('rgb',))
    indices = np.arange(len(dataset))

    start_time = time.time()
    count = 0

    for i in indices:
        example = dataset[i]
        example_name = example['name']
        mesh_dir = '/data3/out/scene3d/factored3d_pred/{}'.format(example_name)
        source_mesh = path.join(mesh_dir, 'codes.obj')
        if path.isfile(path.join(mesh_dir, 'codes_transformed_clipped.ply')):
            print('{} exists. skipping'.format(example_name))
            continue

        while not (path.isfile(source_mesh) and path.getsize(source_mesh) > 1024):
            print('{} does not exist. waiting.'.format(source_mesh))
            time.sleep(1)

        out_filenames = f3d_utils.align_factored3d_mesh_with_our_gt(source_mesh, example_name)

        print(out_filenames)

        count += 1
        remaining = len(indices) - count
        eta = (time.time() - start_time) / count * remaining
        print('ETA: {:.2f} minutes'.format(eta / 60))


def main_scannet():
    names = io_utils.read_lines_and_strip('/data4/scannet_frustum_clipped/test_2000__shuffled_0001_of_0005.txt')

    start_time = time.time()
    count = 0

    for ii, name in enumerate(names):
        if ii < 42:
            continue

        filename_meshes = path.join(config.scannet_frustum_clipped_root, '{}/meshes.obj'.format(name))
        filename_cam_info = path.join(config.scannet_frustum_clipped_root, '{}/cam_info.pkl'.format(name))
        filename_img = path.join(config.scannet_frustum_clipped_root, '{}/img.jpg'.format(name))
        filename_proposals = path.join(config.scannet_frustum_clipped_root, '{}/proposals.mat'.format(name))

        mesh_dir = '/data4/out/scene3d/factored3d_pred/scannet/{}'.format(name)

        assert filename_cam_info.endswith('.pkl')
        assert path.isfile(filename_cam_info)

        with open(filename_cam_info, 'rb') as f:
            c = pickle.load(f)

        cam = camera.OrthographicCamera.from_Rt(Rt=la.inv(c['pose'])[:3], is_world_to_cam=True)
        cam.viewdir *= -1
        cam.up_vector *= -1
        line = scannet_utils.to_our_camera_line(cam.Rt(), c['K'])

        assert isinstance(line, str)
        tempdir = scannet_utils.find_temp_dir()
        random_string = uuid.uuid4().hex[:10]
        camera_filename = path.join(tempdir, 'scene3d_temp_camera', 'cam_{}.txt'.format(random_string))
        io_utils.ensure_dir_exists(path.dirname(camera_filename))
        with open(camera_filename, 'w') as f:
            f.write('P {}'.format(line))

        source_mesh1 = path.join(mesh_dir, 'codes.obj')
        out_filenames = f3d_utils.align_factored3d_mesh_with_our_gt_scannet(source_mesh1, cam, camera_filename)
        print(out_filenames)

        source_mesh2 = path.join(mesh_dir, 'layout.obj')
        out_filenames = f3d_utils.align_factored3d_mesh_with_our_gt_scannet(source_mesh2, cam, camera_filename)
        print(out_filenames)

        count += 1
        remaining = len(names) - count
        eta = (time.time() - start_time) / count * remaining
        print('ETA: {:.2f} minutes'.format(eta / 60))

        #################3

        mesh_filename = path.join(config.scannet_frustum_clipped_root, name, 'meshes.obj')

        new_gt_filename = path.join('/home/daeyun/mnt/v9_visualization_scannet', 'gt.obj')
        shutil.copy(mesh_filename, new_gt_filename)
        print(new_gt_filename)

        files = glob.glob('/data4/out/scene3d/v9_scannet_pred_depth_mesh/{}/*'.format(name))

        for fname in files:
            new_fname = fname.replace('/data4/out/scene3d/v9_scannet_pred_depth_mesh/{}'.format(name), '/home/daeyun/mnt/v9_visualization_scannet')
            io_utils.ensure_dir_exists(path.dirname(new_fname))
            shutil.copy(fname, new_fname)

        files = glob.glob('/data4/out/scene3d/factored3d_pred/scannet/{}/*'.format(name))
        print('/data4/out/scene3d/factored3d_pred')
        print(files)

        for fname in files:
            new_fname = fname.replace('/data4/out/scene3d/factored3d_pred/scannet/{}'.format(name), '/home/daeyun/mnt/v9_visualization_scannet')
            io_utils.ensure_dir_exists(path.dirname(new_fname))
            shutil.copy(fname, new_fname)

        return


if __name__ == '__main__':
    main_scannet()
