import numpy as np
from os import path
from scene3d import config
from scene3d import transforms
from scene3d import camera


def voxel2mesh(voxels, surface_view, scale=0.01, cube_dist_scale=1.1):
    """
    Taken from 3D-R2N2 (Choy et al.)
    Source: https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py
    """
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    verts = []
    faces = []
    curr_vert = 0

    positions = np.where(voxels > 0.3)
    voxels[positions] = 1
    for i, j, k in zip(*positions):
        # identifies if current voxel has an exposed face
        if not surface_view or np.sum(voxels[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]) < 27:
            verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
            faces.extend(cube_faces + curr_vert)
            curr_vert += len(cube_verts)

    return np.array(verts), np.array(faces)


def binvox_model_to_pts(model):
    pts = (np.vstack(np.where(model.data)).T + 0.5) / model.dims * model.scale + model.translate
    return pts


def project_cam_voxels_to_image(binvox_model):
    pts = binvox_model_to_pts(binvox_model)

    # specific to pbrs for now
    height = 240
    width = 320

    xfov = 0.5534
    yfov = 0.43389587

    fx = width / np.tan(xfov) * 0.5
    fy = height / np.tan(yfov) * 0.5

    K = np.array([
        [-fx, 0, width / 2],
        [0, fy, height / 2],
        [0, 0, 1],
    ], dtype=np.float32)

    proj_xy = K.dot(pts.T)
    proj_xy = np.rint(proj_xy[:2, :] / proj_xy[2, :]).astype(np.int64).T

    in_image = ((proj_xy >= np.array([0, 0])) & (proj_xy < np.array([width, height]))).all(axis=1)
    linear_indices = np.where(in_image)[0]  # TODO(daeyun): this only works when binvox_model has all ones.
    depth_values = -pts[:, 2]

    return proj_xy[in_image], depth_values[in_image], linear_indices


def project_cam_voxels_to_overhead_image(binvox_model, example_name):
    pts = binvox_model_to_pts(binvox_model)

    camera_filename = path.join(config.scene3d_root, 'v9/renderings', example_name + '_cam.txt')
    with open(camera_filename, 'r') as f:
        content = f.readlines()
    cam_p = content[0].strip().split()
    assert cam_p[0] == 'P'
    cam_o = content[1].strip().split()
    assert cam_o[0] == 'O'

    campos = np.array([float(item) for item in cam_p[1:4]])
    viewdir = np.array([float(item) for item in cam_p[4:7]])
    up = np.array([float(item) for item in cam_p[7:10]])
    ref_world_to_cam_Rt = transforms.lookat_matrix(campos, campos + viewdir, up=up)
    ref_cam = camera.OrthographicCamera.from_Rt(Rt=ref_world_to_cam_Rt)
    pts_world = ref_cam.cam_to_world(pts)

    # overhead camera
    overhead_campos = np.array([float(item) for item in cam_o[1:4]])
    overhead_viewdir = np.array([float(item) for item in cam_o[4:7]])
    overhead_up = np.array([float(item) for item in cam_o[7:10]])
    lrbt = [float(item) for item in cam_o[10:14]]
    overhead_Rt = transforms.lookat_matrix(overhead_campos, overhead_campos + overhead_viewdir, up=overhead_up)
    overhead_pts = overhead_Rt.dot(np.hstack((pts_world, np.ones([pts_world.shape[0], 1]))).T).T

    im_hw = (300, 300)  # specific to our overhead depth resolution.
    x = (overhead_pts[:, 0] - lrbt[0]) / (lrbt[1] - lrbt[0]) * (im_hw[1])
    y = (overhead_pts[:, 1] - lrbt[2]) / (lrbt[3] - lrbt[2]) * (im_hw[0])

    proj_xy = np.rint(np.array([x, y]).T).astype(np.int64)
    proj_xy[:, 1] = im_hw[0] - proj_xy[:, 1]

    in_image = ((proj_xy >= np.array([0, 0])) & (proj_xy < np.array(im_hw[::-1]))).all(axis=1)

    linear_indices = np.where(binvox_model.data.ravel())[0][np.where(in_image)[0]]
    depth_values = -overhead_pts[:, 2]

    return proj_xy[in_image], depth_values[in_image], linear_indices
