import os
import time
from os import path
import pyperclip
import numpy as np

from scene3d import depth_mesh_utils_cpp
from scene3d import io_utils
from scene3d import train_eval_pipeline
from scene3d.eval import generate_gt_mesh
from scene3d.dataset import v8

# Some examples of scenes that were removed. and why
excluded = [
    'b90df6ea1a5b5be27712b6680cff0091/000006',
    '7ab013e1a9f291b791575f341e711e6a/000018',
    '96bbb47095e684d64ef2f9f0411bbd4a/000032',  # a very small portion of object is present in ground truth but invisible.
    'b90df6ea1a5b5be27712b6680cff0091/000011',  # large scene, almost invisible small object
    'b90df6ea1a5b5be27712b6680cff0091/000016',  # large scene, almost invisible small object
    'b90df6ea1a5b5be27712b6680cff0091/000019',  # large scene, almost invisible small object
    '2bbeaa3ef4ddfefcf4ea018d55371a2c/000027',  # large scene, almost invisible small object
    'b90df6ea1a5b5be27712b6680cff0091/000013',  # no object
    'b90df6ea1a5b5be27712b6680cff0091/000023',
    'b90df6ea1a5b5be27712b6680cff0091/000024',
    '2bbeaa3ef4ddfefcf4ea018d55371a2c/000033',  # viewing angle makes object invisible.
    '2fec16e24fd9576ae07bae1da7ea2f74/000028',  # viewing angle makes object invisible.
    '2bbeaa3ef4ddfefcf4ea018d55371a2c/000035',
    '9be54388338028f7e0f6c7cf150a6c03/000012',  # tiny portion object visible through gap
    'b5acedae2f34ee429d8e326a6ed3f06c/000018',  # a single thin hanger object in a corner of a large hallway
    'b5acedae2f34ee429d8e326a6ed3f06c/000020',
    'd114369b48f6e20143cdb7314d9ae61f/000033',  # tiny portion object visible through gap
    'bbad0863de5d5ff81885c160e121066f/000021',  # only small portion visible. rest is truncated.
    '35211d4794d529e0237f50b1d630d807/000004',  # only small portion visible. rest is truncated.
    'e5b6b0c3fe6e693870f0c94f033ac9f7/000040',  # a single thin object in a corner of a large hallway. invisible
    '3850311c754352231a115d60ca8143f4/000012',
    '5755bb54b2f007432a1b08d31cb64b6a/000022',
    '5e5141fb68452bfbc809c3bb34539d77/000033',
    'a0953e0223d4def8495b56dbe04030e6/000020',
    'a229ecffead50c8ce6b502d7011277ee/000038',
    'b90df6ea1a5b5be27712b6680cff0091/000007',
    'd114369b48f6e20143cdb7314d9ae61f/000013',
    'd114369b48f6e20143cdb7314d9ae61f/000017',  # invisible
    'd114369b48f6e20143cdb7314d9ae61f/000023',
    'e5b6b0c3fe6e693870f0c94f033ac9f7/000039',  # viewing angle of thin object
    'e5b6b0c3fe6e693870f0c94f033ac9f7/000041',
    '1708f1ee0578bd1ae4981ca689d3aae7/000022',
    '8259a98607f4c6c79a6a5a3edc33b862/000030',
    '8259a98607f4c6c79a6a5a3edc33b862/000031',
    'a27d935ef978d4181bf5b35483e3028b/000031',
]


def is_mesh_empty(mesh_filename, size_bytes):
    if isinstance(mesh_filename, (list, tuple)):
        return np.any([is_mesh_empty(item, size_bytes=size_bytes) for item in mesh_filename])
    return not path.isfile(mesh_filename) or path.getsize(mesh_filename) < size_bytes


def main():
    dataset = v8.MultiLayerDepth(
        # split='test',
        # split='/data2/scene3d/v8/validation_s168.txt',
        split='/data2/scene3d/v8/test_v2_subset_factored3d.txt',
        # split='/data3/factored3d/filename.txt',
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=[])
    indices = np.arange(len(dataset))

    start_time = time.time()

    count = 0
    count_too_few_pixels = 0
    count_no_background = 0
    count_no_object = 0

    ok_examples = []

    for i in indices:
        example = dataset[i]
        if example['name'] in excluded:
            print(example['name'], 'excluded')
            continue

        house_id, camera_id = example['name'].split('/')
        out_dir = '/data3/out/scene3d/v8_gt_mesh/{}/{}'.format(house_id, camera_id)

        files_to_check_for_skip = [
            path.join(out_dir, 'd0.ply'),
            path.join(out_dir, 'd1.ply'),
            path.join(out_dir, 'd2.ply'),
            path.join(out_dir, 'd3.ply'),
            path.join(out_dir, 'gt_bg.ply'),
            path.join(out_dir, 'gt_objects.ply'),
        ]

        if is_mesh_empty(path.join(out_dir, 'gt_objects.ply'), size_bytes=1024):
            print(example['name'], 'No objects', i)
            count_no_object += 1
            train_eval_pipeline.symlink_all_files_in_dir(out_dir, '/mnt/ramdisk/scene3d/inspection')
            continue
        if is_mesh_empty(path.join(out_dir, 'gt_bg.ply'), size_bytes=2 ** 20):
            print(example['name'], 'No background', 1)
            count_no_background += 1
            train_eval_pipeline.symlink_all_files_in_dir(out_dir, '/mnt/ramdisk/scene3d/inspection')
            continue

        # object mesh is big enough and there is background too. but for somereason object is not rendered. happens when objects are outside the viewing frustum, etc.
        if is_mesh_empty(files_to_check_for_skip, size_bytes=4096 * 20 + 450):
            print(example['name'], 'No rendered depth', 1)
            count_too_few_pixels += 1
            train_eval_pipeline.symlink_all_files_in_dir(out_dir, '/mnt/ramdisk/scene3d/inspection')
            # pyperclip.copy(example['name'])
            # raise RuntimeError(example['name'])
            continue

        ok_examples.append(example['name'])

    print('no objects', count_no_object)
    print('no background', count_no_background)
    print('excluded', len(excluded))
    print(len(dataset) - count_no_object - count_no_background - len(excluded) - count_too_few_pixels)
    print(len(ok_examples))
    io_utils.write_lines('/mnt/ramdisk/new_names.txt', ok_examples)


if __name__ == '__main__':
    main()
