import glob
import multiprocessing as mp
import pickle
from os import path

import numpy as np
from scene3d import train_eval_pipeline_v9
from scene3d import io_utils
from scene3d.dataset import v9
from third_party import binvox_rw


def read_voxels(filename):
    with open(filename, 'rb') as f:
        vox = binvox_rw.read_as_3d_array(f)
    return vox


def generate_gt_voxels(i, vox_dir, dataset, binvox_reference_model):
    example_name = path.basename(vox_dir).replace('_', '/')
    out_filename = path.join(vox_dir, 'proj_depth4_gt_voxels.npz')

    if path.isfile(out_filename) and path.getsize(out_filename) > 256:
        print('Already exists. Skipping. {}'.format(out_filename))
    else:
        print(i, example_name)
        ret = {}
        # dataset.get_rgb(example_name, ret)
        dataset.get_multi_layer_depth_aligned_background(example_name, ret)
        new_model = train_eval_pipeline_v9.voxelize_multi_layer_depth(binvox_reference_model, ret['multi_layer_depth_aligned_background'])
        new_model_data = new_model.data
        train_eval_pipeline_v9.save_binvox_model_as_pcl(new_model, out_filename.replace('.npz', '_pcl.ply'))
        # Compression reduces size by 1/100.
        np.savez_compressed(out_filename, voxels=new_model_data)


def generate_gt_voxels_in_parallel():
    """
    This GT is mostly used for sanity check purposes. Predictions are actually evaluated against binvox-generated voxels.
    """
    # Assume all voxels in this directory have the same resolution. We'll read the first one as the reference one from which a copy is made for each voxel we generate.
    vox_dirs = sorted(glob.glob('/data5/out/scene3d/voxelization_experiment_res400_cv/*'))
    binvox_reference_model = read_voxels(path.join(vox_dirs[0], 'gt_objects_camcoord.binvox'))

    # Needed for access to ground truth.
    dataset = v9.MultiLayerDepth(split='test', subtract_mean=False, image_hw=(240, 320), rgb_scale=1.0 / 255, fields=('rgb', 'multi_layer_depth_aligned_background', 'category_nyu40_merged_background'))
    with mp.pool.Pool(processes=9) as pool:
        args = []
        for i in range(len(vox_dirs)):
            vox_dir = vox_dirs[i]
            args.append((i, vox_dir, dataset, binvox_reference_model))
        r = pool.starmap_async(generate_gt_voxels, args)
        r.wait()


def generate_pred_voxels(i, vox_dir, binvox_reference_model):
    example_name = path.basename(vox_dir).replace('_', '/')
    out_filename = path.join(vox_dir, 'proj_depth4_pred_voxels.npz')

    if path.isfile(out_filename) and path.getsize(out_filename) > 256:
        print('Already exists. Skipping. {}'.format(out_filename))
    else:
        print(i, example_name)

        segmented_depth_filename = '/data4/out/scene3d/pred_segmented_depth/{}/segmented_depth.pkl'.format(example_name)
        with open(segmented_depth_filename, 'rb') as f:
            segmented_depth = pickle.load(f)

        new_model = train_eval_pipeline_v9.voxelize_multi_layer_depth(binvox_reference_model, segmented_depth)
        new_model_data = new_model.data
        train_eval_pipeline_v9.save_binvox_model_as_pcl(new_model, out_filename.replace('.npz', '_pcl.ply'))
        # Compression reduces size by 1/100.
        np.savez_compressed(out_filename, voxels=new_model_data)


def generate_pred_voxels_in_parallel():
    # Assume all voxels in this directory have the same resolution. We'll read the first one as the reference one from which a copy is made for each voxel we generate.
    vox_dirs = sorted(glob.glob('/data5/out/scene3d/voxelization_experiment_res400_cv/*'))
    binvox_reference_model = read_voxels(path.join(vox_dirs[0], 'gt_objects_camcoord.binvox'))
    with mp.pool.Pool(processes=9) as pool:
        args = []
        for i in range(len(vox_dirs)):
            vox_dir = vox_dirs[i]
            args.append((i, vox_dir, binvox_reference_model))
        r = pool.starmap_async(generate_pred_voxels, args)
        r.wait()


def generate_pred_voxels_two_layer_only(i, vox_dir, binvox_reference_model):
    example_name = path.basename(vox_dir).replace('_', '/')
    out_filename = path.join(vox_dir, 'proj_depth2_pred_voxels2.npz')

    print(i, example_name)

    segmented_depth_filename = '/data4/out/scene3d/pred_segmented_depth/{}/segmented_depth.pkl'.format(example_name)
    with open(segmented_depth_filename, 'rb') as f:
        segmented_depth = pickle.load(f)

    new_model = train_eval_pipeline_v9.voxelize_multi_layer_depth(binvox_reference_model, segmented_depth, two_layer_only=True)
    new_model_data = new_model.data
    train_eval_pipeline_v9.save_binvox_model_as_pcl(new_model, out_filename.replace('.npz', '_pcl.ply'))
    # Compression reduces size by 1/100.
    np.savez_compressed(out_filename, voxels=new_model_data)


def generate_pred_voxels_two_layer_only_in_parallel():
    # Assume all voxels in this directory have the same resolution. We'll read the first one as the reference one from which a copy is made for each voxel we generate.
    vox_dirs = sorted(glob.glob('/data5/out/scene3d/voxelization_experiment_res400_cv/*'))
    binvox_reference_model = read_voxels(path.join(vox_dirs[0], 'gt_objects_camcoord.binvox'))
    with mp.pool.Pool(processes=9) as pool:
        args = []
        for i in range(len(vox_dirs)):
            vox_dir = vox_dirs[i]
            args.append((i, vox_dir, binvox_reference_model))
        r = pool.starmap_async(generate_pred_voxels_two_layer_only, args)
        r.wait()


def carve_pred_voxels(i, vox_dir):
    example_name = path.basename(vox_dir).replace('_', '/')
    out_filename = path.join(vox_dir, 'proj_depth4_pred_voxels_carved2.npz')

    if path.isfile(out_filename) and path.getsize(out_filename) > 256:
        print('Already exists. Skipping. {}'.format(out_filename))
    else:
        print(i, example_name, vox_dir)

        binvoxGT = read_voxels(path.join(vox_dir, 'gt_objects_camcoord.binvox'))
        voxprojOursFrontal = binvoxGT.clone()
        voxprojOursFrontal.data = np.load(path.join(vox_dir, 'proj_depth4_pred_voxels.npz'))['voxels']

        # this says v8 and our v9 predictions were actually overwritten here, i think
        pred_height_map = io_utils.read_array_compressed('/data4/out/scene3d/v8_pred/{}/pred_height_map.bin'.format(example_name))
        pred_height_map[pred_height_map == 0] = np.nan

        voxprojOursFrontal_carved = train_eval_pipeline_v9.carve_voxels_using_height_map(voxprojOursFrontal, pred_height_map, example_name)

        np.savez_compressed(out_filename, voxels=voxprojOursFrontal_carved.data)


def carve_pred_voxels_in_parallel():
    # Assume all voxels in this directory have the same resolution. We'll read the first one as the reference one from which a copy is made for each voxel we generate.
    vox_dirs = sorted(glob.glob('/data5/out/scene3d/voxelization_experiment_res400_cv/*'))
    with mp.pool.Pool(processes=9) as pool:
        args = []
        for i in range(len(vox_dirs)):
            vox_dir = vox_dirs[i]
            args.append((i, vox_dir))
        r = pool.starmap_async(carve_pred_voxels, args)
        r.wait()


if __name__ == '__main__':
    # generate_gt_voxels_in_parallel()
    # generate_pred_voxels_in_parallel()
    generate_pred_voxels_two_layer_only_in_parallel()
    # carve_pred_voxels_in_parallel()
