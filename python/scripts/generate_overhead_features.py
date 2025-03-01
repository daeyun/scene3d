import time
import os
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import numpy as np
import matplotlib.pyplot as pt
from os import path
import torch.utils.data
from torch.backends import cudnn

from scene3d import pbrs_utils
from scene3d import suncg_utils
from scene3d import render_depth
from scene3d import io_utils
from scene3d import config
from scene3d import camera
from scene3d import train_eval_pipeline
from scene3d import feat
from scene3d import epipolar
from scene3d.net import unet
from scene3d import log
from scene3d import torch_utils
from scene3d.dataset import dataset_utils
from scene3d.dataset import v8

model_filename1 = '/data3/out/scene3d/v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00600000_003_0014413.pth'
model_filename2 = '/data3/out/scene3d/v8/v8-category_nyu40_merged_background-2l/0/00600000_013_0019123.pth'
batch_size = 12
num_data_workers = 8

out_dir = '/data3/scene3d/v8/overhead/v2'

pool = ThreadPool(4)


def print_eta(start_time, num_processed, num_total):
    elapsed = time.time() - start_time
    spo = elapsed / num_processed  # seconds per operation
    eta_seconds = (num_total - num_processed) * spo
    log.info('{}/{}   ETA: {:d} minutes'.format(num_processed, num_total, round(eta_seconds / 60)))


def make_output_filename(example_name):
    return path.join(out_dir, 'features', example_name + '.bin')


def make_dataset_object():
    dataset_all = v8.MultiLayerDepth(split='all', subtract_mean=True, image_hw=(240, 320), rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth'))

    # Pick up from where we left off.
    filename_index_offset = 0
    while filename_index_offset < len(dataset_all.filename_prefixes):
        example_name = dataset_all.filename_prefixes[filename_index_offset]
        out_filename = make_output_filename(example_name)
        if not path.isfile(out_filename):
            break
        filename_index_offset += 1
    if filename_index_offset > 0:
        dataset_all = v8.MultiLayerDepth(split='all', subtract_mean=True, start_index=filename_index_offset, image_hw=(240, 320), rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth'))
    log.info('filename_index_offset: {}'.format(filename_index_offset))

    return dataset_all


def main():
    cudnn.enabled = True
    cudnn.benchmark = True

    dataset_all = make_dataset_object()

    log.info('Loading model {}'.format(model_filename1))
    model1, _, _, _ = train_eval_pipeline.load_checkpoint(model_filename1, use_cpu=False)
    model1.train(False)
    model1.cuda()

    log.info('Loading model {}'.format(model_filename2))
    model2, _, _, _ = train_eval_pipeline.load_checkpoint(model_filename2, use_cpu=False)
    model2.train(False)
    model2.cuda()

    # in_rgb = torch.Tensor(in_rgb_np[None]).cuda()

    loader = torch.utils.data.DataLoader(dataset_all, batch_size=batch_size, num_workers=num_data_workers, shuffle=False, drop_last=False, pin_memory=True)

    start_time = time.time()
    num_processed = 0
    num_total = len(dataset_all)

    for i_iter, batch in enumerate(loader):
        log.info(i_iter)

        # (B, 48, 240, 320)
        feature_map1, _ = unet.get_feature_map_output_v2(model1, batch['rgb'].cuda())
        feature_map1_np = torch_utils.recursive_torch_to_numpy(feature_map1)
        assert feature_map1_np.shape[1] == 48

        # (B, 64, 240, 320)
        feature_map2 = unet.get_feature_map_output_v1(model2, batch['rgb'].cuda())
        feature_map2_np = torch_utils.recursive_torch_to_numpy(feature_map2)
        assert feature_map2_np.shape[1] == 64

        rgb_np = v8.undo_rgb_whitening(batch['rgb'])

        concatenated_features = np.concatenate([rgb_np, feature_map1_np, feature_map2_np], axis=1)

        num_examples = len(concatenated_features)
        assert num_examples == len(batch['camera_filename'])

        feat = dataset_utils.force_contiguous(concatenated_features.transpose(0, 2, 3, 1))  # (B, H, W, C)
        front = dataset_utils.force_contiguous(torch_utils.recursive_torch_to_numpy(batch['multi_layer_depth'][:, 0]))  # (B, H, W)
        back = dataset_utils.force_contiguous(torch_utils.recursive_torch_to_numpy(batch['multi_layer_depth'][:, 1]))  # (B, H, W)
        camera_filenames = batch['camera_filename']

        # (B, 300, 300, 67)
        tranformed_batch = epipolar.feature_transform_parallel(feat, front_depth_data=front, back_depth_data=back, camera_filenames=camera_filenames, target_height=300, target_width=300)

        # (B, 67, 300, 300)
        tranformed_batch = tranformed_batch.transpose(0, 3, 1, 2)
        tranformed_batch = tranformed_batch.astype(np.float16)
        tranformed_batch = dataset_utils.force_contiguous(tranformed_batch)

        out_filenames = []
        out_transformed_batch_list = []
        for j in range(num_examples):
            example_name = batch['name'][j]
            assert example_name in camera_filenames[j], (example_name, camera_filenames[j])
            out_filename = make_output_filename(example_name)
            io_utils.ensure_dir_exists(path.dirname(out_filename))
            out_filenames.append(out_filename)
            out_transformed_batch_list.append(tranformed_batch[j])

        pool.starmap_async(io_utils.save_array_compressed, zip(out_filenames, out_transformed_batch_list))
        # for out_filename, out_transformed_batch in zip(out_filenames, out_transformed_batch_list):
        #     io_utils.save_array_compressed(out_filename, out_transformed_batch)

        log.info(out_filenames)
        num_processed += num_examples
        print_eta(start_time, num_processed, num_total)

    print('DONE')


if __name__ == '__main__':
    main()
