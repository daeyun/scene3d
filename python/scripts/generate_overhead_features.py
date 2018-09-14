import time
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
from scene3d import feat
from scene3d import epipolar
from scene3d.net import unet
from scene3d import log
from scene3d import torch_utils
from scene3d.dataset import dataset_utils
from scene3d.dataset import v8

model_filename = '/data2/out/scene3d/v2/multi-layer-d-3/0/v1_030_0000850_00664000.pth'
batch_size = 12
num_data_workers = 2

out_dir = '/data3/scene3d/v8/overhead/v1'


def print_eta(start_time, num_processed, num_total):
    elapsed = time.time() - start_time
    spo = elapsed / num_processed  # seconds per operation
    eta_seconds = (num_total - num_processed) * spo
    log.info('{}/{}   ETA: {:d} minutes'.format(num_processed, num_total, round(eta_seconds / 60)))


def main():
    cudnn.enabled = True
    cudnn.benchmark = True

    dataset_all = v8.MultiLayerDepth(split='all', subtract_mean=True, image_hw=(240, 320), rgb_scale=1.0 / 255)

    log.info('Loading model {}'.format(model_filename))
    model = torch_utils.load_torch_model(model_filename, use_cpu=False)
    model.train(False)
    model.cuda()

    # in_rgb = torch.Tensor(in_rgb_np[None]).cuda()

    loader = torch.utils.data.DataLoader(dataset_all, batch_size=batch_size, num_workers=num_data_workers, shuffle=False, drop_last=False, pin_memory=True)

    start_time = time.time()
    num_processed = 0
    num_total = len(dataset_all)

    for i_iter, batch in enumerate(loader):
        log.info(i_iter)

        # (B, 64, 240, 320)
        feature_map = unet.get_feature_map_output(model, batch['rgb'].cuda())

        feature_map_np = torch_utils.recursive_torch_to_numpy(feature_map)
        rgb_np = v8.undo_rgb_whitening(batch['rgb'])

        concatenated_features = np.concatenate([rgb_np, feature_map_np], axis=1)

        num_examples = len(concatenated_features)
        assert num_examples == len(batch['camera_filename'])

        feat = dataset_utils.force_contiguous(concatenated_features.transpose(0, 2, 3, 1))  # (B, H, W, C)
        front = dataset_utils.force_contiguous(torch_utils.recursive_torch_to_numpy(batch['depth'][:, 0]))  # (B, H, W)
        back = dataset_utils.force_contiguous(torch_utils.recursive_torch_to_numpy(batch['depth'][:, 1]))  # (B, H, W)
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
            out_filename = path.join(out_dir, 'features', example_name + '.bin')
            io_utils.ensure_dir_exists(path.dirname(out_filename))
            out_filenames.append(out_filename)
            out_transformed_batch_list.append(tranformed_batch[j])

        pool = ThreadPool(9)
        pool.starmap(io_utils.save_array_compressed, zip(out_filenames, out_transformed_batch_list))

        # io_utils.save_array_compressed(out_filename, tranformed_batch[j])
        log.info(out_filenames)
        num_processed += num_examples
        print_eta(start_time, num_processed, num_total)

    print('DONE')


if __name__ == '__main__':
    main()
