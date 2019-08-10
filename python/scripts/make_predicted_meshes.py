import torch
from os import path
from os import path
import pyassimp
import numpy as np
from pprint import pprint
from scene3d import geom2d
from scene3d import pbrs_utils
from scene3d import suncg_utils
from scene3d import torch_utils
from scene3d import render_depth
from scene3d import train_eval_pipeline
from scene3d import transforms
from scene3d import io_utils
from scene3d import depth_mesh_utils_cpp
from scene3d import config
import glob
import collections
from scene3d import camera
from scene3d import loss_fn
from scene3d import category_mapping
from scene3d import feat
from scene3d import evaluation
from scene3d import log
from scene3d import epipolar
from scene3d.eval import generate_gt_mesh
from scene3d.eval import post_processing
from scene3d import data_utils
from scene3d.dataset import dataset_utils
# from scene3d.dataset import v8
from scene3d.dataset import v9
from scene3d.dataset import v2
from scene3d.net import unet
from scene3d.net import unet_overhead
import cv2
import torch
from torch.backends import cudnn
from torch import nn

from scene3d import category_mapping
import pickle
from scene3d import train_eval_pipeline


def main(force_indices=tuple()):
    """
    :param force_indices: re-generate those indices even if it's already done
    :return:
    """
    depth_checkpoint = path.join(config.default_out_root, 'v9/v9-multi_layer_depth_aligned_background_multi_branch/0/01149000_005_0003355.pth')
    seg_checkpoint = path.join(config.default_out_root, 'v9/v9-category_nyu40_merged_background-2l/0/01130000_005_0001780.pth')

    # depth_checkpoint = path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00906000_010_0000080.pth')
    # seg_checkpoint = path.join(config.default_out_root, 'v8/v8-category_nyu40_merged_background-2l/0/00966000_009_0005272.pth')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    depth_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(depth_checkpoint)
    print(metadata)
    seg_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(seg_checkpoint)
    print(metadata)

    dataset = v9.MultiLayerDepth(
        split=[
            path.join(config.scene3d_root, 'v9/validation_s159.txt'),
            path.join(config.scene3d_root, 'v9/test_subset_factored3d.txt'),
        ],
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=('rgb', 'camera_filename'))

    count = 0
    for i in range(len(dataset)):
        if i in force_indices:
            force = True
        else:
            force = False

        example = dataset[i]
        print(count, i, example['name'])

        house_id, camera_id = pbrs_utils.parse_house_and_camera_ids_from_string(example['name'])
        pred_meshes = sorted(glob.glob(path.join(config.default_out_root, 'v9_pred_depth_mesh/{}/{}/pred_*.ply'.format(house_id, camera_id))))
        if not force:
            if len(pred_meshes) == 5:
                print('Already computed. Skipping this example.')
                continue
            elif len(pred_meshes) > 5:
                print(pred_meshes)
                raise RuntimeError()

        depth_pred = depth_model(torch.Tensor(example['rgb'][None]).cuda())
        seg_pred = seg_model(torch.Tensor(example['rgb'][None]).cuda())
        assert depth_pred.shape[0] == 1
        assert seg_pred.shape[0] == 1

        seg_pred_l1 = torch_utils.recursive_torch_to_numpy(seg_pred[:, :40])
        seg_pred_l2 = torch_utils.recursive_torch_to_numpy(seg_pred[:, 40:])
        seg_argmax_l1 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(seg_pred_l1)
        seg_argmax_l2 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(seg_pred_l2)

        segmented_depth = train_eval_pipeline.segment_predicted_depth(torch_utils.recursive_torch_to_numpy(depth_pred), seg_argmax_l1, seg_argmax_l2)
        assert segmented_depth.shape[0] == 1
        segmented_depth = np.squeeze(segmented_depth)

        out = train_eval_pipeline.save_mldepth_as_meshes_v9(segmented_depth, example, force=True)
        print(out)

        # Sanity check to make sure corresponding ground truths exist. not necessary
        real_gt_meshes = sorted(glob.glob(path.join(config.default_out_root, 'v9_gt_mesh/{}/{}/gt_*.ply'.format(house_id, camera_id))))
        depth_gt_meshes = sorted(glob.glob(path.join(config.default_out_root, 'v9_gt_mesh/{}/{}/d*.ply'.format(house_id, camera_id))))
        assert len(real_gt_meshes) == 2, real_gt_meshes
        assert len(depth_gt_meshes) == 5, depth_gt_meshes

        count += 1


def main_runtime(force_indices=tuple()):
    """
    :param force_indices: re-generate those indices even if it's already done
    :return:
    """
    depth_checkpoint = path.join(config.default_out_root, 'v9/v9-multi_layer_depth_aligned_background_multi_branch/0/01149000_005_0003355.pth')
    seg_checkpoint = path.join(config.default_out_root, 'v9/v9-category_nyu40_merged_background-2l/0/01130000_005_0001780.pth')

    # depth_checkpoint = path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00906000_010_0000080.pth')
    # seg_checkpoint = path.join(config.default_out_root, 'v8/v8-category_nyu40_merged_background-2l/0/00966000_009_0005272.pth')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    depth_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(depth_checkpoint)
    print(metadata)
    seg_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(seg_checkpoint)
    print(metadata)

    dataset = v9.MultiLayerDepth(
        split=[
            # path.join(config.scene3d_root, 'v9/validation_s159.txt'),
            path.join(config.scene3d_root, 'v9/test_subset_factored3d.txt'),
        ],
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=('rgb', 'camera_filename'))

    count = 0
    for i in range(len(dataset)):
        if i in force_indices:
            force = True
        else:
            force = False

        example = dataset[i]
        print(count, i, example['name'])

        house_id, camera_id = pbrs_utils.parse_house_and_camera_ids_from_string(example['name'])
        pred_meshes = sorted(glob.glob(path.join('/tmp/scene3dtmp', 'v9_pred_depth_mesh/{}/{}/pred_*.ply'.format(house_id, camera_id))))

        depth_pred = depth_model(torch.Tensor(example['rgb'][None]).cuda())
        seg_pred = seg_model(torch.Tensor(example['rgb'][None]).cuda())
        assert depth_pred.shape[0] == 1
        assert seg_pred.shape[0] == 1

        seg_pred_l1 = torch_utils.recursive_torch_to_numpy(seg_pred[:, :40])
        seg_pred_l2 = torch_utils.recursive_torch_to_numpy(seg_pred[:, 40:])
        seg_argmax_l1 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(seg_pred_l1)
        seg_argmax_l2 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(seg_pred_l2)

        segmented_depth = train_eval_pipeline.segment_predicted_depth(torch_utils.recursive_torch_to_numpy(depth_pred), seg_argmax_l1, seg_argmax_l2)
        assert segmented_depth.shape[0] == 1
        segmented_depth = np.squeeze(segmented_depth)

        out = train_eval_pipeline.save_mldepth_as_meshes_v9(segmented_depth, example, force=True)
        print(out)

        # Sanity check to make sure corresponding ground truths exist. not necessary
        real_gt_meshes = sorted(glob.glob(path.join(config.default_out_root, 'v9_gt_mesh/{}/{}/gt_*.ply'.format(house_id, camera_id))))
        depth_gt_meshes = sorted(glob.glob(path.join(config.default_out_root, 'v9_gt_mesh/{}/{}/d*.ply'.format(house_id, camera_id))))
        assert len(real_gt_meshes) == 2, real_gt_meshes
        assert len(depth_gt_meshes) == 5, depth_gt_meshes

        count += 1


def save_frontal():
    depth_checkpoint = path.join(config.default_out_root, 'v9/v9-multi_layer_depth_aligned_background_multi_branch/0/01149000_005_0003355.pth')
    seg_checkpoint = path.join(config.default_out_root, 'v9/v9-category_nyu40_merged_background-2l/0/01130000_005_0001780.pth')

    # depth_checkpoint = path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00906000_010_0000080.pth')
    # seg_checkpoint = path.join(config.default_out_root, 'v8/v8-category_nyu40_merged_background-2l/0/00966000_009_0005272.pth')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    depth_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(depth_checkpoint)
    print(metadata)
    seg_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(seg_checkpoint)
    print(metadata)

    dataset = v9.MultiLayerDepth(
        split=[
            # path.join(config.scene3d_root, 'v9/validation_s159.txt'),
            path.join(config.scene3d_root, 'v9/test_subset_factored3d.txt'),
            # path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0001_of_0009.txt'),  # sharded for running on multiple machines
            # path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0002_of_0009.txt'),  # sharded for running on multiple machines
        ],
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=('rgb', 'camera_filename', 'multi_layer_depth_aligned_background'))

    count = 0
    for i in range(len(dataset)):
        example = dataset[i]
        print(count, i, example['name'])

        house_id, camera_id = pbrs_utils.parse_house_and_camera_ids_from_string(example['name'])
        pred_meshes = sorted(glob.glob(path.join(config.default_out_root, 'v9_pred_depth_mesh/{}/{}/pred_*.ply'.format(house_id, camera_id))))

        depth_pred = depth_model(torch.Tensor(example['rgb'][None]).cuda())
        seg_pred = seg_model(torch.Tensor(example['rgb'][None]).cuda())
        assert depth_pred.shape[0] == 1
        assert seg_pred.shape[0] == 1

        seg_pred_l1 = torch_utils.recursive_torch_to_numpy(seg_pred[:, :40])
        seg_pred_l2 = torch_utils.recursive_torch_to_numpy(seg_pred[:, 40:])
        seg_argmax_l1 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(seg_pred_l1)
        seg_argmax_l2 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(seg_pred_l2)

        segmented_depth = train_eval_pipeline.segment_predicted_depth(torch_utils.recursive_torch_to_numpy(depth_pred), seg_argmax_l1, seg_argmax_l2)
        assert segmented_depth.shape[0] == 1
        segmented_depth = np.squeeze(segmented_depth)

        # TODO: temporary hardcoded paths
        io_utils.ensure_dir_exists('/data4/out/scene3d/pred_segmented_depth/{}'.format(example['name']))
        with open('/data4/out/scene3d/pred_segmented_depth/{}/segmented_depth.pkl'.format(example['name']), 'wb') as f:
            pickle.dump(segmented_depth, f)
        with open('/data4/out/scene3d/pred_segmented_depth/{}/raw_depth.pkl'.format(example['name']), 'wb') as f:
            pickle.dump(torch_utils.recursive_torch_to_numpy(depth_pred), f)
        with open('/data4/out/scene3d/pred_segmented_depth/{}/example.pkl'.format(example['name']), 'wb') as f:
            pickle.dump(example, f)

        count += 1
        print(count)


def overhead():
    # need v9 updates

    checkpoint_filenames = {
        'pose_3param': path.join(config.default_out_root_v8, 'v8/v8-overhead_camera_pose/0/00420000_018_0014478.pth'),
        # 'overhead_height_map_model': path.join(config.default_out_root_v8, 'v8/OVERHEAD_offline_01/0/00050000_001_0004046.pth'),
        'overhead_height_map_model': path.join(config.default_out_root, 'v9/v9_OVERHEAD_v1_heightmap_01/0/00056000_003_0011546.pth'),
        'overhead_segmentation_model': path.join(config.default_out_root, 'v9/v9_OVERHEAD_v1_segmentation_01/0/00016000_000_0002000.pth'),
    }

    dataset = v9.MultiLayerDepth(
        # split='/data2/scene3d/v8/validation_s168.txt',
        # split=path.join(config.scene3d_root, 'v9/test_subset_factored3d.txt'),
        [
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0001_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0002_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0003_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0004_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0005_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0006_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0007_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0007_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0008_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0009_of_0009.txt'),  # sharded for running on multiple machines
        ],
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=['rgb', ])

    batch_size = 5
    num_data_workers = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_data_workers, shuffle=False, drop_last=False, pin_memory=True)

    hm_model = train_eval_pipeline.HeightMapModel(checkpoint_filenames, device_id=1)

    for i_iter, batch in enumerate(loader):
        print(i_iter)
        names = batch['name']

        skip = True
        for name in names:
            if not path.isfile(path.join(config.default_out_root, 'v9_pred_depth_mesh/{}/overhead_bg_clipped.ply'.format(name))) \
                    or not path.isfile(path.join(config.default_out_root, 'v9_pred_depth_mesh/{}/overhead_fg_clipped.ply'.format(name))):
                skip = False
                break
        if skip:
            print('Already computed. Skipping this batch.')
            continue

        out = hm_model.predict_height_map(batch)
        train_eval_pipeline.save_height_map_output_batch(out, names)

        try:
            out_ply_filenames = train_eval_pipeline.save_height_prediction_as_meshes(out, hm_model, batch['camera_filename'], batch['name'])
        except (Exception, pyassimp.errors.AssimpError) as ex:
            print('ERROR')
            print(ex)
            continue

        for fnames, cam_file in zip(out_ply_filenames, batch['camera_filename']):
            assert isinstance(fnames, (list, tuple))
            for fname in fnames:
                try:
                    out_filename = fname.replace('.ply', '_clipped.ply')
                    post_processing.extract_mesh_inside_frustum(fname, cam_file, 0, out_filename=out_filename)
                except Exception as ex:
                    print('ERROR')
                    print(ex)
                    continue


if __name__ == '__main__':
    cudnn.benchmark = True
    # main(force_indices=[426, 440])
    # main()
    main_runtime()
    # save_frontal()
    # overhead()
