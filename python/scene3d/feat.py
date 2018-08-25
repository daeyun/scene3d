import numpy as np
import numpy.linalg as la
import time
import matplotlib.pyplot as pt

from scene3d import transforms
from scene3d import camera


def epipolar_line(xy, cam_params, td_cam_params, depth_image, back_depth_image, td_depth_image, plot=True):
    depth_image = depth_image.squeeze()
    back_depth_image = back_depth_image.squeeze()
    td_depth_image = td_depth_image.squeeze()

    fx = 0.5 * 320 / np.tan(cam_params[9])

    eye_xyz = np.array(cam_params[:3])
    obj_xyz = eye_xyz + np.array(cam_params[3:6])
    up = np.array(cam_params[6:9])
    Rt = transforms.lookat_matrix(eye_xyz, obj_xyz, up)
    cam = camera.OrthographicCamera.from_Rt(Rt, wh=(320, 240))

    eye_xyz = np.array(td_cam_params[:3])
    obj_xyz = eye_xyz + np.array(td_cam_params[3:6])
    up = np.array(td_cam_params[6:9])
    Rt = transforms.lookat_matrix(eye_xyz, obj_xyz, up)
    l = td_cam_params[9]
    r = td_cam_params[10]
    t = td_cam_params[11]
    b = td_cam_params[12]
    cam2 = camera.OrthographicCamera.from_Rt(Rt, wh=(320, 240), trbl=(t, r, b, l))

    x, y = xy


    d1 = depth_image[y, x]
    d2 = back_depth_image[y, x]

    if np.isnan(d1):
        return None, None, None

    image_optical_center = np.array([320 / 2, 240 / 2, 0])
    image_plane_coord = np.array([x + 0.5, 240 - (y + 0.5), -fx])
    ray_dir = image_plane_coord - image_optical_center
    # ray_dir /= -ray_dir[2]
    ray_dir /= la.norm(ray_dir)

    ray_dir = cam.cam_to_world_normal(ray_dir[None]).ravel()

    p = cam.pos + ray_dir * d1
    xy1, _ = cam2.world_to_image(p[None])

    p = cam.pos + ray_dir * d2
    xy2, _ = cam2.world_to_image(p[None])

    if plot:
        pt.figure(figsize=(11,10))
        pt.subplot(1, 2, 1)
        pt.imshow(depth_image)
        pt.scatter([x], [y], c='red')


    xy1 = xy1.astype(np.int32).ravel()
    xy2 = xy2.astype(np.int32).ravel()

    step = 1/400

    coords = (xy1[None].T.dot(np.arange(0, 1, step)[None]).T + xy2[None].T.dot(np.arange(1, 0, -step)[None]).T).round().astype(np.int32)
    coords = np.unique(coords, axis=0)
    inds = np.logical_and(coords >= [0, 0], coords < [320, 240]).all(axis=1)
    coords = coords[inds]

    if plot:
        pt.subplot(1, 2, 2)
        pt.scatter(coords[:,0], coords[:,1], c='lime', s=1)
        # pt.plot([xy1[0, 0], xy2[0, 0]], [xy1[0, 1], xy2[0, 1]], color='red')
        pt.scatter(xy1[0], xy1[1], c='red', s=60)
        pt.imshow(td_depth_image)


    return xy1, xy2, coords
