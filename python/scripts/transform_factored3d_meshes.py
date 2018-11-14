from scene3d.eval import f3d_utils
import os
import time
from os import path
import numpy as np

from scene3d import depth_mesh_utils_cpp
from scene3d.eval import generate_gt_mesh
from scene3d.dataset import v8


def main():
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


if __name__ == '__main__':
    main()
