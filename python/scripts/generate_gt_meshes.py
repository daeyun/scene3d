import os
from os import path
import numpy as np

from scene3d import depth_mesh_utils_cpp
from scene3d.eval import generate_gt_mesh
from scene3d.dataset import v8


def main():
    dataset = v8.MultiLayerDepth(split='test', subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_aligned_background'))
    indices = np.arange(len(dataset))

    # TODO(daeyun): read this from a text file.
    np.random.seed(42)
    np.random.shuffle(indices)

    for j in range(500):
        i = indices[j]
        example = dataset[i]

        house_id, camera_id = example['name'].split('/')
        camera_filename = example['camera_filename']
        out_dir = '/data3/out/scene3d/v8_gt_mesh/{}/{}'.format(house_id, camera_id)

        print(j, i, example['name'])

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
        depth_mesh_utils_cpp.depth_to_mesh(mld[3], camera_filename, camera_index=0, dd_factor=15, out_ply_filename=path.join(out_dir, 'd3.ply'))


if __name__ == '__main__':
    main()
