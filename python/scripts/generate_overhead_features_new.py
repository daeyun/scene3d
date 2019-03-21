import time
from os import path

from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import torch.utils.data
from torch.backends import cudnn

from scene3d import config
from scene3d import train_eval_pipeline
from scene3d import pipeline_overhead
from scene3d import feat
from scene3d import io_utils
from scene3d import log
from scene3d import torch_utils
import argparse

parser = argparse.ArgumentParser(description='generate overhead features')
parser.add_argument('--device_id', type=int)
parser.add_argument('--save_dir', type=str)  # '/mnt/scratch2/daeyuns/data/out/scene3d/overhead_pred'
parser.add_argument('--split_name', type=str)
parser.add_argument('--use_gt_geometry', type=bool, default=False)
parser.add_argument('--gating_function_index', type=int, default=0)
parser.add_argument('--num_data_workers', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=25)
parser.add_argument('--num_transformer_workers', type=int, default=4)
args = parser.parse_args()

batch_size = args.batch_size
num_data_workers = args.num_data_workers
pool = ThreadPool(20)

_num_processed = 0


def print_eta(start_time, num_processed, num_total):
    elapsed = time.time() - start_time
    spo = elapsed / num_processed  # seconds per operation
    eta_seconds = (num_total - num_processed) * spo
    log.info('{}/{}   ETA: {:d} minutes'.format(num_processed, num_total, round(eta_seconds / 60)))


def save_example(overhead_features, i, name):
    # out_filename = '/data4/out/scene3d/overhead_pred/{}.bin'.format(name)
    out_filename = path.join(args.save_dir, '{}.bin'.format(name))
    io_utils.ensure_dir_exists(path.dirname(out_filename))
    out_arr = torch_utils.recursive_torch_to_numpy(overhead_features[i]).astype(np.float16)
    io_utils.save_array_compressed(out_filename, out_arr)
    print(out_filename)


def process_batch(i_iter, batch, overhead_features, transformer_names, async=True):
    assert len(transformer_names) == len(batch['name'])
    assert len(overhead_features) == len(batch['name'])
    for bi in range(len(transformer_names)):
        name0 = transformer_names[bi]
        name1 = batch['name'][bi]
        assert name0 == name1, (name0, name1)

    if async:
        pool.starmap_async(save_example, [[overhead_features, i, name] for i, name in enumerate(batch['name'])])
    else:
        pool.starmap(save_example, [[overhead_features, i, name] for i, name in enumerate(batch['name'])])

    global _num_processed
    _num_processed += len(transformer_names)


def can_skip_batch(batch):
    names = batch['name']
    for name in names:
        assert isinstance(name, str)
        bin_filename = path.join(args.save_dir, '{}.bin'.format(name))
        if not path.isfile(bin_filename):
            return False
    return True


def main():
    dataset_all = pipeline_overhead.get_dataset(args.split_name, use_gt_geometry=args.use_gt_geometry)

    depth_checkpoint_filename = path.join(config.default_out_root, 'v9/v9-multi_layer_depth_aligned_background_multi_branch/0/01149000_005_0003355.pth')
    segmentation_checkpoint_filename = path.join(config.default_out_root, 'v9/v9-category_nyu40_merged_background-2l/0/01130000_005_0001780.pth')

    transformer = feat.Transformer(
        # depth_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00700000_008_0001768.pth'),
        # segmentation_checkpoint_filename=path.join(config.default_out_root, 'v8/v8-category_nyu40_merged_background-2l/0/00800000_022_0016362.pth'),
        depth_checkpoint_filename=depth_checkpoint_filename,
        segmentation_checkpoint_filename=segmentation_checkpoint_filename,
        device_id=args.device_id,
        num_workers=args.num_transformer_workers,
        gating_function_index=args.gating_function_index,
    )

    loader = torch.utils.data.DataLoader(dataset_all, batch_size=batch_size, num_workers=num_data_workers, shuffle=False, drop_last=False, pin_memory=True)
    it = enumerate(loader)

    while True:
        i_iter, batch = next(it)
        if can_skip_batch(batch):
            log.info('Already exists. Skipping batch {}'.format(i_iter))
        else:
            log.info('New batch {}'.format(i_iter))
            break

    transformer.prefetch_batch_async(batch, start_end_indices=None, target_device_id=args.device_id, options={'use_gt_geometry': args.use_gt_geometry})

    start_time = time.time()

    for next_i_iter, next_batch in it:
        overhead_features, cameras, transformer_names = transformer.pop_batch(target_device_id=args.device_id)
        log.info('Prefetching batch {}'.format(next_i_iter))
        transformer.prefetch_batch_async(next_batch, start_end_indices=None, target_device_id=args.device_id, options={'use_gt_geometry': args.use_gt_geometry})
        process_batch(i_iter, batch, overhead_features, transformer_names, async=True)
        i_iter = next_i_iter
        batch = next_batch
        print_eta(start_time, _num_processed, len(dataset_all))
    overhead_features, cameras, transformer_names = transformer.pop_batch(target_device_id=args.device_id)
    process_batch(i_iter, batch, overhead_features, transformer_names, async=False)  # This will wait until all files are saved.


if __name__ == '__main__':
    with torch.cuda.device(args.device_id):
        log.info(args)
        main()
