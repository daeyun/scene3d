from scene3d.eval import f3d_utils
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


def main():
    names = io_utils.read_lines_and_strip('/data4/scannet_frustum_clipped/test_2000__shuffled_0001_of_0005.txt')
    # names = io_utils.read_lines_and_strip('/data4/scannet_frustum_clipped/test_2000__shuffled_0002_of_0005.txt')

    for i in range(len(names)):
        name = names[i]
        print(i, name)

        sn = ScanNet()
        # sn.save_depth_meshes(name, force=True)
        # sn.save_transformed_factored3d_meshes(name)

        # sn.scan.set_symlinks()
        # pt.imshow(sn.scan.input_image)
        # pt.show()


if __name__ == '__main__':
    main()
