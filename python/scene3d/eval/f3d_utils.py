import numpy as np
import numpy.linalg as la
from os import path

from scene3d import camera
from scene3d import io_utils
from scene3d import pbrs_utils
from scene3d import transforms
from scene3d.eval import post_processing


def align_factored3d_mesh_with_our_gt(mesh_filename, example_name):
    assert mesh_filename.endswith('.obj')
    house_id, camera_id = example_name.split('/')

    cam_params = pbrs_utils.get_camera_params_line(house_id=house_id, camera_id=camera_id)
    cam_params = cam_params.split()

    pos = np.array([float(item) for item in cam_params[:3]])
    view = np.array([float(item) for item in cam_params[3:6]])
    up = np.array([float(item) for item in cam_params[6:9]])
    obj_pos = pos + view

    t = transforms.lookat_matrix(pos, obj_pos, up)

    # we only care about the extrinsics, so the frustum doesn't matter.
    cam = camera.OrthographicCamera.from_Rt(Rt=t)

    fv = io_utils.read_mesh(mesh_filename)
    v_orig = fv['v'].copy()

    v_orig[:, 1] *= -1
    v_orig[:, 2] *= -1
    v_transformed = cam.cam_to_world(v_orig)
    new_fv = {'f': fv['f'], 'v': v_transformed}

    out_filename = mesh_filename.replace('.obj', '_transformed.off')
    io_utils.save_off(new_fv, out_filename)

    out_filename2 = mesh_filename.replace('.obj', '_transformed_clipped.ply')
    camera_filename = '/data2/scene3d/v8/renderings/{}_cam.txt'.format(example_name)
    post_processing.extract_mesh_inside_frustum(out_filename, camera_filename=camera_filename, camera_index=0, out_filename=out_filename2)

    return [out_filename, out_filename2]


def align_factored3d_mesh_with_our_gt_scannet(mesh_filename, cam: camera.OrthographicCamera, camera_filename):
    assert mesh_filename.endswith('.obj') or mesh_filename.endswith('.off')

    pos = cam.pos
    view = cam.viewdir
    up = cam.up_vector
    obj_pos = pos + view

    t = transforms.lookat_matrix(pos, obj_pos, up)

    # we only care about the extrinsics, so the frustum doesn't matter.
    cam = camera.OrthographicCamera.from_Rt(Rt=t)

    fv = io_utils.read_mesh(mesh_filename)
    v_orig = fv['v'].copy()

    v_orig[:, 1] *= -1
    v_orig[:, 2] *= -1
    v_transformed = cam.cam_to_world(v_orig)
    new_fv = {'f': fv['f'], 'v': v_transformed}

    if mesh_filename.endswith('.obj'):
        out_filename = mesh_filename.replace('.obj', '_transformed.off')
        out_filename2 = mesh_filename.replace('.obj', '_transformed_clipped.ply')
    else:
        out_filename = mesh_filename.replace('.off', '_transformed.off')
        out_filename2 = mesh_filename.replace('.off', '_transformed_clipped.ply')

    io_utils.save_off(new_fv, out_filename)
    print('#####', io_utils.read_lines_and_strip(camera_filename))
    post_processing.extract_mesh_inside_frustum(out_filename, camera_filename=camera_filename, camera_index=0, out_filename=out_filename2)

    return [out_filename, out_filename2]


def align_factored3d_mesh_with_meshlab_cam_coords(mesh_filename, out_filename):
    assert mesh_filename.endswith('.obj')

    pos = np.array([0, 0, 0], dtype=np.float32)
    view = np.array([0, 0, -1], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    obj_pos = pos + view

    t = transforms.lookat_matrix(pos, obj_pos, up)

    # we only care about the extrinsics, so the frustum doesn't matter.
    cam = camera.OrthographicCamera.from_Rt(Rt=t)

    fv = io_utils.read_mesh(mesh_filename)
    v_orig = fv['v'].copy()

    v_orig[:, 1] *= -1
    v_orig[:, 2] *= -1
    v_transformed = cam.cam_to_world(v_orig)
    new_fv = {'f': fv['f'], 'v': v_transformed}

    # out_filename = mesh_filename.replace('.obj', '_transformed.off')
    io_utils.save_stl(new_fv, out_filename)

    # out_filename2 = mesh_filename.replace('.obj', '_transformed_clipped.ply')
    # camera_filename = '/data2/scene3d/v8/renderings/{}_cam.txt'.format(example_name)
    # post_processing.extract_mesh_inside_frustum(out_filename, camera_filename=camera_filename, camera_index=0, out_filename=out_filename2)

    return [out_filename]
    # return [out_filename, out_filename2]


def clip_infinity_vertices(filename, threshold):
    fv = io_utils.read_mesh(filename)
    sel = fv['v'][:, 2] > threshold

    # limit the distance of all vertices to 7, from the origin.
    fv['v'][sel] /= la.norm(fv['v'][sel], axis=1).reshape(-1, 1) / threshold

    affected_faces = np.isin(fv['f'], np.where(sel)[0])
    fv['f'] = fv['f'][~np.all(affected_faces, axis=1)]

    prefix, ext = path.splitext(filename)

    new_filename = prefix + '_farclipped' + '.off'
    io_utils.save_off(fv, new_filename)
    return new_filename
