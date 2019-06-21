from scene3d.eval import f3d_utils
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
import glob
import shutil
import uuid
from scene3d import io_utils
import os
import pickle
import time
from os import path
import numpy as np

from scene3d import depth_mesh_utils_cpp
import numpy.linalg as la
from scene3d import scannet_utils
from scene3d import camera
from scene3d import config
from scene3d import visualizer
from scene3d import scannet_utils
from scene3d.eval import generate_gt_mesh
from scene3d.dataset import v8
import matplotlib.pyplot as pt


class ScanNet:
    def __init__(self):
        self.scan = visualizer.ScannetMeshGenerator()
        self.scan.visualize = False

    def save_depth_meshes(self, name, force=False):
        self.scan.load_scannet(name)
        out_filenames = self.scan.generate_mesh(force=force)
        return out_filenames

    def save_transformed_factored3d_meshes(self, name):
        self.scan.load_scannet(name)

        with open(self.scan.filename_cam_info, 'rb') as f:
            c = pickle.load(f)
        cam = camera.OrthographicCamera.from_Rt(Rt=la.inv(c['pose'])[:3], is_world_to_cam=True)
        cam.viewdir *= -1
        cam.up_vector *= -1
        line = scannet_utils.to_our_camera_line(cam.Rt(), c['K'])

        assert isinstance(line, str)
        tempdir = scannet_utils.find_temp_dir()
        random_string = uuid.uuid4().hex[:10]
        camera_filename = path.join(tempdir, 'scene3d_temp_camera', 'cam_{}.txt'.format(random_string))
        io_utils.ensure_dir_exists(path.dirname(camera_filename))
        with open(camera_filename, 'w') as f:
            f.write('P {}'.format(line))

        mesh_dir = path.join(config.factored3d_mesh_dir, 'scannet', name)

        source_mesh1 = path.join(mesh_dir, 'codes.obj')
        source_mesh2 = path.join(mesh_dir, 'layout.obj')
        source_mesh2_farclipped = f3d_utils.clip_infinity_vertices(source_mesh2, threshold=7)

        with scannet_utils.temporary_camera_file_context(self.scan.filename_cam_info) as cam_filename:
            out_filenames1 = f3d_utils.align_factored3d_mesh_with_our_gt_scannet(source_mesh1, cam, cam_filename)
        with scannet_utils.temporary_camera_file_context(self.scan.filename_cam_info) as cam_filename:
            out_filenames2 = f3d_utils.align_factored3d_mesh_with_our_gt_scannet(source_mesh2_farclipped, cam, cam_filename)

        return out_filenames1 + out_filenames2


# frontal images and factored3d
def main():
    # names = io_utils.read_lines_and_strip('/data4/scannet_frustum_clipped/test_2000__shuffled_0001_of_0005.txt')
    # names = io_utils.read_lines_and_strip('/data4/scannet_frustum_clipped/test_2000__shuffled_0002_of_0005.txt')
    names = io_utils.read_lines_and_strip('/data4/scannet_frustum_clipped/test_2000__shuffled_0003_of_0005.txt')

    for i in range(len(names)):
        name = names[i]
        print(i, name)

        sn = ScanNet()
        sn.save_depth_meshes(name, force=True)
        sn.save_transformed_factored3d_meshes(name)

        # sn.scan.set_symlinks()
        # pt.imshow(sn.scan.input_image)
        # pt.show()


def overhead():
    # need v9 updates

    checkpoint_filenames = {
        'pose_3param': path.join(config.default_out_root_v8, 'v8/v8-overhead_camera_pose/0/00420000_018_0014478.pth'),
        # 'overhead_height_map_model': path.join(config.default_out_root_v8, 'v8/OVERHEAD_offline_01/0/00050000_001_0004046.pth'),
        'overhead_height_map_model': path.join(config.default_out_root, 'v9/v9_OVERHEAD_v1_heightmap_01/0/00056000_003_0011546.pth'),
        'overhead_segmentation_model': path.join(config.default_out_root, 'v9/v9_OVERHEAD_v1_segmentation_01/0/00016000_000_0002000.pth'),
    }

    names = io_utils.read_lines_and_strip('/data4/scannet_frustum_clipped/test_2000__shuffled_0001_of_0005.txt')
    # names = io_utils.read_lines_and_strip('/data4/scannet_frustum_clipped/test_2000__shuffled_0003_of_0005.txt')

    hm_model = train_eval_pipeline.HeightMapModel(checkpoint_filenames, device_id=1, num_transformer_workers=1)

    for i in range(len(names)):
        name = names[i]
        sn = ScanNet()
        sn.scan.load_scannet(name, image_wh=(240, 320))

        with scannet_utils.temporary_camera_file_context(sn.scan.filename_cam_info) as camfile:
            rgb = sn.scan.input_image_meansub_cuda
            assert len(rgb.shape) == 4

            batch = {
                'name': [name],
                'rgb': rgb,
                'camera_filename': [camfile],
            }

            print(i)
            print(i, name)

            out_filename = path.join(config.default_out_root, 'v9_scannet_pred_depth_mesh/{}/overhead_fg.ply'.format(name))
            out_filename_clipped = path.join(config.default_out_root, 'v9_scannet_pred_depth_mesh/{}/overhead_fg_clipped.ply'.format(name))

            # if not path.isfile(out_filename) or not path.isfile(out_filename_clipped):
            #     print('Already computed. Skipping this batch.')
            #     continue

            assert len(batch['name']) == 1
            assert len(batch['rgb']) == 1
            heights = [sn.scan.camera_height]
            thetas = [sn.scan.gravity_angle]
            print(sn.scan.gravity_angle)

            hm_model.transformer.set_theta(thetas)
            out = hm_model.predict_height_map(batch)
            # train_eval_pipeline.save_height_map_output_batch_of_size_one_scannet(out, batch['name'])

            # try:
            #     out_ply_filenames = train_eval_pipeline.save_height_prediction_as_meshes_scannet(out, hm_model, batch['camera_filename'], batch['name'], heights=heights, thetas=thetas)
            #
            # except (Exception, pyassimp.errors.AssimpError) as ex:
            #     print('ERROR')
            #     print(ex)
            #     continue
            #
            # assert len(out_ply_filenames) == 1
            #
            # try:
            #     post_processing.extract_mesh_inside_frustum(out_filename, batch['camera_filename'][0], 0, out_filename=out_filename_clipped)
            # except Exception as ex:
            #     print('ERROR')
            #     print(ex)
            #     continue

            pt.figure()
            sn.scan.set_symlinks()
            pt.imshow(sn.scan.input_image)
            pt.show()
            print('#######################33')


if __name__ == '__main__':
    # main()
    overhead()
