import time
from os import path

from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import torch.utils.data
from torch.backends import cudnn

from scene3d import config
from scene3d import train_eval_pipeline
from scene3d import feat
from scene3d import io_utils
from scene3d import log
from scene3d import torch_utils
from scene3d.dataset import v8

batch_size = 50
num_data_workers = 4
pool = ThreadPool(20)

_num_processed = 0


def print_eta(start_time, num_processed, num_total):
    elapsed = time.time() - start_time
    spo = elapsed / num_processed  # seconds per operation
    eta_seconds = (num_total - num_processed) * spo
    log.info('{}/{}   ETA: {:d} minutes'.format(num_processed, num_total, round(eta_seconds / 60)))


def save_example(overhead_features, i, name):
    # out_filename = '/mnt/scratch2/daeyuns/overhead_features/pred/{}.bin'.format(name)
    out_filename = '/data3/out/scene3d/overhead_pred/{}.bin'.format(name)
    io_utils.ensure_dir_exists(path.dirname(out_filename))
    out_arr = torch_utils.recursive_torch_to_numpy(overhead_features[i]).astype(np.float16)
    io_utils.save_array_compressed(out_filename, out_arr)
    print(out_filename)


def process_batch(i_iter, batch, overhead_features, transformer_names):
    assert len(transformer_names) == len(batch['name'])
    assert len(overhead_features) == len(batch['name'])
    for bi in range(len(transformer_names)):
        name0 = transformer_names[bi]
        name1 = batch['name'][bi]
        assert name0 == name1, (name0, name1)

    pool.starmap_async(save_example, [[overhead_features, i, name] for i, name in enumerate(batch['name'])])

    global _num_processed
    _num_processed += len(transformer_names)


def main():
    dataset_all = v8.MultiLayerDepth(
        # split='all',
        split='/mnt/ramdisk/remaining_features.txt',
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255,
        fields=('rgb', 'overhead_camera_pose_4params', 'camera_filename', 'multi_layer_overhead_depth'))

    transformer = feat.Transformer(
        depth_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00700000_008_0001768.pth'),
        segmentation_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-category_nyu40_merged_background-2l/0/00800000_022_0016362.pth'),
        device_id=1,
        num_workers=5,
    )

    loader = torch.utils.data.DataLoader(dataset_all, batch_size=batch_size, num_workers=num_data_workers, shuffle=False, drop_last=False, pin_memory=True)
    it = enumerate(loader)
    i_iter, batch = next(it)

    transformer.prefetch_batch_async(batch, start_end_indices=None, target_device_id=1, options={'use_gt_geometry': False})

    start_time = time.time()

    for next_i_iter, next_batch in it:
        overhead_features, transformer_names = transformer.pop_batch(target_device_id=1)
        transformer.prefetch_batch_async(next_batch, start_end_indices=None, target_device_id=1, options={'use_gt_geometry': False})
        process_batch(i_iter, batch, overhead_features, transformer_names)
        i_iter = next_i_iter
        batch = next_batch
        print_eta(start_time, _num_processed, len(dataset_all))
    overhead_features, transformer_names = transformer.pop_batch(target_device_id=1)
    process_batch(i_iter, batch, overhead_features, transformer_names)


if __name__ == '__main__':
    with torch.cuda.device(1):
        main()
