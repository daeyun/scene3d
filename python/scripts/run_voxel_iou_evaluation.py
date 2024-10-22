from scene3d import train_eval_pipeline_v9  # TODO
import numpy as np
from scene3d import io_utils
import multiprocessing as mp
import glob
from scene3d import config
from scene3d import camera
from scene3d.dataset import v9
from scene3d import visualizer
from scene3d import train_eval_pipeline
from third_party import binvox_rw
from scene3d import train_eval_pipeline_v9
from scene3d import transforms
import pickle
import multiprocessing as mp
from os import path
from scene3d.dataset import v9
from scene3d import config
import os
import sys
from scene3d import io_utils
import shutil

skipped = [
    'e5e5e9fb2f46af947a72644a9c3fff51/000008',
    'cb1ab3737e580172555b6318c5c3fc62/000037',
    '3d7a0d0b5f35afaf8fa9f1e70cd5a0db/000015',
    '6b264902885b77cfca22ef95b59086d0/000011',
    '14d5e1fc4eef5ff65c3cb1e7536e5460/000008',
    '050236b2dcaa484aad506207089cab5d/000021',
    '7689aa62d5ae001a73afd0ae7b78b98d/000020',
    '7ab013e1a9f291b791575f341e711e6a/000022',
    'eeb725e88e257a39abaafeb0dff1753b/000003',
    '76911d7f0e62db7fc33de4b4f6e1a17c/000056',
    '3099b4289757325d3ca9f267035a15f0/000013',
]


def generate_voxels(i, dataset, pr_eval):
    example = dataset[i]
    print(i, example['name'])
    if example['name'] in skipped:
        print('skipped')
        return None

    import pyassimp
    out = None
    target_directory = '/data5/out/scene3d/voxelization_experiment_res400_cv/{}'.format(example['name'].replace('/', '_'))
    if path.isdir(target_directory):
        return None
    try:
        outdir, out = pr_eval.run_evaluation(example)

        shutil.move(outdir, target_directory)
    except pyassimp.AssimpError as ex:
        print('mesh io error. skipping..', ex)

    print('Done: Index {}'.format(i))

    if out is not None:
        for k, val in out.items():
            print('OUT {}: {}'.format(k, val))
    return out


def generate_binvox_voxels():
    dataset = v9.MultiLayerDepth(
        split=[
            # path.join(config.scene3d_root, 'v9/test_subset_factored3d.txt'),
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0001_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0002_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0003_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0004_of_0009.txt'),  # sharded for running on multiple machines
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0005_of_0009.txt'),  # sharded for running on multiple machines
            # path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0006_of_0009.txt'),  # sharded for running on multiple machines
            # path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0007_of_0009.txt'),  # sharded for running on multiple machines
            # path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0008_of_0009.txt'),  # sharded for running on multiple machines
            # path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0009_of_0009.txt'),  # sharded for running on multiple machines
        ],
        subtract_mean=True, image_hw=(240, 320), rgb_scale=1.0 / 255, fields=[])
    print('# {} TOTAL'.format(len(dataset)))

    pkl_filename = path.join(config.default_out_root, 'v9_voxel_iou/voxel_iou_0.pkl')
    pr_eval = train_eval_pipeline_v9.VoxelIoUEvaluation(save_filename=pkl_filename)

    with mp.pool.Pool(processes=12) as pool:
        args = []
        for i in range(len(dataset)):
            args.append((i, dataset, pr_eval))

        pool.starmap(generate_voxels, args)


def read_voxels(filename):
    with open(filename, 'rb') as f:
        vox = binvox_rw.read_as_3d_array(f)
    return vox


def compute_iou(vox1, vox2):
    if isinstance(vox1, np.ndarray):
        a = vox1
    else:
        a = vox1.data
    if isinstance(vox2, np.ndarray):
        b = vox2
    else:
        b = vox2.data
    return float((a & b).sum()) / (a | b).sum()


def do_evaluation(vox_dir):
    example_name = path.basename(vox_dir).replace('_', '/')
    print(example_name)
    try:
        binvoxGT = read_voxels(path.join(vox_dir, 'gt_objects_camcoord.binvox'))
        binvoxFactored3d = read_voxels(path.join(vox_dir, 'f3d_objects_camcoord.binvox'))
        binvoxOursFrontal = read_voxels(path.join(vox_dir, 'depth_meshes_frontal4_camcoord.binvox'))
        binvoxOursFrontalAndOverhead = read_voxels(path.join(vox_dir, 'depth_meshes_frontal4_and_ovh_camcoord.binvox'))
        # voxprojGT = np.load(path.join(vox_dir, 'proj_depth4_gt_voxels.npz'))['voxels']
        voxprojOursFrontal = np.load(path.join(vox_dir, 'proj_depth4_pred_voxels.npz'))['voxels']

        ret = {
            'name': example_name,
            # 'binvoxGT_voxprojGT': compute_iou(binvoxGT, voxprojGT),
            'binvoxGT_binvoxFactored3d': compute_iou(binvoxGT, binvoxFactored3d),
            'binvoxGT_binvoxOursFrontal': compute_iou(binvoxGT, binvoxOursFrontal),
            'binvoxGT_binvoxOursFrontalAndOvh': compute_iou(binvoxGT, binvoxOursFrontalAndOverhead),
            'binvoxGT_voxprojOursFrontal': compute_iou(binvoxGT, voxprojOursFrontal),
            # 'voxprojGT_binvoxFactored3d': compute_iou(voxprojGT, binvoxFactored3d),
            # 'voxprojGT_binvoxOursFrontal': compute_iou(voxprojGT, binvoxOursFrontal),
            # 'voxprojGT_binvoxOursFrontalAndOvh': compute_iou(voxprojGT, binvoxOursFrontalAndOverhead),
            # 'voxprojGT_voxprojOursFrontal': compute_iou(voxprojGT, voxprojOursFrontal),

            'metadata': {
                'count': {
                    'binvoxGT': binvoxGT.data.sum(),
                    'binvoxFactored3d': binvoxFactored3d.data.sum(),
                    'voxprojOursFrontal': voxprojOursFrontal.sum(),
                },
            },
        }
    except Exception as ex:
        print('################# Ejror: {}'.format(example_name))
        raise ex

    return ret


def do_evaluation_parallel():
    vox_dirs = sorted(glob.glob('/data5/out/scene3d/voxelization_experiment_res400_cv/*'))
    print(len(vox_dirs))
    results = []
    with mp.Pool(processes=10) as pool:
        futures = []
        for vox_dir in vox_dirs:
            futures.append(pool.apply_async(func=do_evaluation, args=(vox_dir,)))
        for future in futures:
            res = future.get()
            results.append(res)

    print('saving..')
    with open('/data5/out/scene3d/voxel_iou_01.pkl', 'wb') as f:
        pickle.dump(results, f)


def do_evaluation_extra(vox_dir):
    """
    Evalaute the carved version
    :param vox_dir:
    :return:
    """
    example_name = path.basename(vox_dir).replace('_', '/')
    print(example_name)
    try:
        if not path.isfile(path.join(vox_dir, 'proj_depth2_pred_voxels.npz')):
            return None

        binvoxGT = read_voxels(path.join(vox_dir, 'gt_objects_camcoord.binvox'))
        voxprojOursFrontal = np.load(path.join(vox_dir, 'proj_depth4_pred_voxels.npz'))['voxels']
        voxprojOursFrontal2 = np.load(path.join(vox_dir, 'proj_depth2_pred_voxels2.npz'))['voxels']  # failed experiment. ignore these (proj_depth2*) files.

        ret = {
            'name': example_name,
            'binvoxGT_voxprojOursFrontal': compute_iou(binvoxGT, voxprojOursFrontal),
            'binvoxGT_voxprojOursFrontal2': compute_iou(binvoxGT, voxprojOursFrontal2),
        }
    except Exception as ex:
        print('################# Ejror: {}'.format(example_name))
        raise ex

    return ret


def do_evaluation_extra_parallel():
    vox_dirs = sorted(glob.glob('/data5/out/scene3d/voxelization_experiment_res400_cv/*'))
    print(len(vox_dirs))
    results = []
    with mp.Pool(processes=10) as pool:
        futures = []
        for vox_dir in vox_dirs:
            futures.append(pool.apply_async(func=do_evaluation_extra, args=(vox_dir,)))
        for future in futures:
            res = future.get()
            if res is not None:
                results.append(res)

    print('saving..')
    with open('/data5/out/scene3d/voxel_iou_02.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    # generate_binvox_voxels()
    # do_evaluation_parallel()
    do_evaluation_extra_parallel()
