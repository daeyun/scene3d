import numpy as np


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
        [-fx, 0, width/2],
        [0, fy, height/2],
        [0, 0, 1],
    ], dtype=np.float32)

    proj_xy = K.dot(pts.T)
    proj_xy = np.rint(proj_xy[:2, :]/proj_xy[2, :]).astype(np.int64).T

    in_image = ((proj_xy >= np.array([0,0])) & (proj_xy < np.array([width,height]))).all(axis=1)
    linear_indices = np.where(in_image)[0]
    depth_values = -pts[:,2]

    return proj_xy[in_image], depth_values[in_image], linear_indices
