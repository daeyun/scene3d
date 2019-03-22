import numpy.linalg as la
import tempfile
import os
import uuid
from contextlib import contextmanager
from os import path
import numpy as np
import math
from scene3d import io_utils
import pickle
from scene3d import camera


def to_our_camera_line(R_cam_to_world, K):
    if R_cam_to_world.shape != (4, 4):
        R = np.eye(4)
        R[:3] = R_cam_to_world
        R_cam_to_world = R

    cam = camera.OrthographicCamera.from_Rt(Rt=la.inv(R_cam_to_world)[:3], is_world_to_cam=True)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # assume camera center is at the center of the image
    # the center is actually slightly off (by 0.1~0.2 pixels). but doesn't seem to make a difference.
    h = 480.0
    w = 640.0

    fov_y = math.atan2(h, (2 * fy))
    fov_x = math.atan2(w, (2 * fx))

    assert np.allclose(1.0, la.norm(cam.viewdir))
    assert np.allclose(1.0, la.norm(cam.up_vector))

    hw_ratio = h / float(w)
    near = 0.1
    far = 20

    left, right, bottom, top, near, far = camera.make_perspective_frustum_params(hw_ratio=hw_ratio, x_fov=fov_x, near=near, far=far)

    line = '{:.10f} {:.10f} {:.10f}  {:.10f} {:.10f} {:.10f}  {:.10f} {:.10f} {:.10f}  {:.10f} {:.10f} {:.10f} {:.10f} {:.10f} {:.10f}'.format(
        cam.pos[0],
        cam.pos[1],
        cam.pos[2],
        -cam.viewdir[0],
        -cam.viewdir[1],
        -cam.viewdir[2],
        -cam.up_vector[0],
        -cam.up_vector[1],
        -cam.up_vector[2],
        left,
        right,
        bottom,
        top,
        near,
        far,
    )

    return line


def find_temp_dir():
    if path.isdir('/mnt/ramdisk'):  # specific to our use case
        ret = '/mnt/ramdisk/tmp'
        io_utils.ensure_dir_exists(ret)
        return ret
    return tempfile.gettempdir()


@contextmanager
def temporary_camera_file_context(cam_info_pkl_file):
    assert cam_info_pkl_file.endswith('.pkl')
    assert path.isfile(cam_info_pkl_file)

    with open(cam_info_pkl_file, 'rb') as f:
        c = pickle.load(f)

    line = to_our_camera_line(c['pose'], c['K'])
    assert isinstance(line, str)

    tempdir = find_temp_dir()
    random_string = uuid.uuid4().hex[:10]
    camera_filename = path.join(tempdir, 'scene3d_temp_camera', 'cam_{}.txt'.format(random_string))
    io_utils.ensure_dir_exists(path.dirname(camera_filename))

    with open(camera_filename, 'w') as f:
        f.write('P {}'.format(line))

    try:
        yield camera_filename
    finally:
        os.remove(camera_filename)
