import os
import time
from os import path
import numpy as np

from scene3d import depth_mesh_utils_cpp
from scene3d.eval import generate_gt_mesh
from scene3d.dataset import v9


def is_mesh_empty(mesh_filename):
    if isinstance(mesh_filename, (list, tuple)):
        return np.any([is_mesh_empty(item) for item in mesh_filename])
    return not path.isfile(mesh_filename) or path.getsize(mesh_filename) < 200


def main():
    dataset = v9.MultiLayerDepth(
        # split='test',
        # split='/data2/scene3d/v9/validation_s168.txt',
        split='/data2/scene3d/v9/test_v2_subset_factored3d.txt',
        # split='/data3/factored3d/filename.txt',
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_aligned_background'))
    indices = np.arange(len(dataset))

    start_time = time.time()
    count = 0

    for i in indices:
        example = dataset[i]

        house_id, camera_id = example['name'].split('/')
        camera_filename = example['camera_filename']
        out_dir = '/data3/out/scene3d/v9_gt_mesh/{}/{}'.format(house_id, camera_id)

        files_to_check_for_skip = [
            path.join(out_dir, 'd0.ply'),
            path.join(out_dir, 'd1.ply'),
            path.join(out_dir, 'd2.ply'),
            path.join(out_dir, 'd3.ply'),
            path.join(out_dir, 'd4.ply'),
            path.join(out_dir, 'gt_bg.ply'),
            path.join(out_dir, 'gt_objects.ply'),
        ]

        if not is_mesh_empty(files_to_check_for_skip):
            print(i, 'Skipping. already generated.')
            continue

        print(i, example['name'])

        out_filenames = generate_gt_mesh.generate_gt_mesh(house_id, [camera_filename], out_dir=out_dir)

        assert len(out_filenames) == 2
        gt_background_mesh_filename = out_filenames[0]
        gt_object_mesh_filename = out_filenames[1]
        assert '_bg.ply' in gt_background_mesh_filename
        assert '_objects.ply' in gt_object_mesh_filename

        # Renames files in each directory.
        os.rename(gt_background_mesh_filename, path.join(path.dirname(gt_background_mesh_filename), 'gt_bg.ply'))
        os.rename(gt_object_mesh_filename, path.join(path.dirname(gt_object_mesh_filename), 'gt_objects.ply'))

        mld = example['multi_layer_depth_aligned_background']

        depth_mesh_utils_cpp.depth_to_mesh(mld[0], camera_filename, camera_index=0, dd_factor=10, out_ply_filename=path.join(out_dir, 'd0.ply'))
        depth_mesh_utils_cpp.depth_to_mesh(mld[1], camera_filename, camera_index=0, dd_factor=10, out_ply_filename=path.join(out_dir, 'd1.ply'))
        depth_mesh_utils_cpp.depth_to_mesh(mld[2], camera_filename, camera_index=0, dd_factor=10, out_ply_filename=path.join(out_dir, 'd2.ply'))
        depth_mesh_utils_cpp.depth_to_mesh(mld[3], camera_filename, camera_index=0, dd_factor=10, out_ply_filename=path.join(out_dir, 'd3.ply'))
        depth_mesh_utils_cpp.depth_to_mesh(mld[4], camera_filename, camera_index=0, dd_factor=15, out_ply_filename=path.join(out_dir, 'd4.ply'))  # background has dd_factor 15

        count += 1
        remaining = len(indices) - count
        eta = (time.time() - start_time) / count * remaining
        print('ETA: {:.2f} minutes'.format(eta / 60))


if __name__ == '__main__':
    main()
