import os
import threading
import time
import uuid
from multiprocessing.pool import ThreadPool
from os import path

import numpy as np
import numpy.linalg as la
import torch

from scene3d import camera
from scene3d import epipolar
from scene3d import io_utils
from scene3d import log
from scene3d import loss_fn
from scene3d import torch_utils
from scene3d import train_eval_pipeline
from scene3d import transforms
from scene3d.dataset import dataset_utils
from scene3d.net import unet


def epipolar_line(xy, cam_params, td_cam_params, depth_image, back_depth_image, td_depth_image, plot=True):
    """
    Deprecated. Do not use.
    """
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
        import matplotlib.pyplot as pt
        pt.figure(figsize=(11, 10))
        pt.subplot(1, 2, 1)
        pt.imshow(depth_image)
        pt.scatter([x], [y], c='red')

    xy1 = xy1.astype(np.int32).ravel()
    xy2 = xy2.astype(np.int32).ravel()

    step = 1 / 400

    coords = (xy1[None].T.dot(np.arange(0, 1, step)[None]).T + xy2[None].T.dot(np.arange(1, 0, -step)[None]).T).round().astype(np.int32)
    coords = np.unique(coords, axis=0)
    inds = np.logical_and(coords >= [0, 0], coords < [320, 240]).all(axis=1)
    coords = coords[inds]

    if plot:
        import matplotlib.pyplot as pt
        pt.subplot(1, 2, 2)
        pt.scatter(coords[:, 0], coords[:, 1], c='lime', s=1)
        # pt.plot([xy1[0, 0], xy2[0, 0]], [xy1[0, 1], xy2[0, 1]], color='red')
        pt.scatter(xy1[0], xy1[1], c='red', s=60)
        pt.imshow(td_depth_image)

    return xy1, xy2, coords


def fix_ray_displacement(depth):
    """
    There was a bug in some examples in the v2 dataset. Use this to fix it, if you're still using v2. New users do not have to worry about this.
    """
    distance_to_image_plane = float(np.mean([160 / np.tan(0.5534), 120 / np.tan(0.433896)]))

    h = 240
    w = 320
    xy = np.mgrid[:h, :w][::-1, :, :]
    z = np.full_like(xy[:1], fill_value=-distance_to_image_plane)
    xyz = np.concatenate((xy + 0.5, z), axis=0)

    cam_ray_direction = xyz - np.array([w * 0.5, h * 0.5, 0])[:, None, None]
    cam_ray_direction = cam_ray_direction.transpose([1, 2, 0]).reshape(-1, 3).copy()
    cam_ray_direction = cam_ray_direction / la.norm(cam_ray_direction, axis=1)[:, None]

    new_z = -depth.reshape(-1) * cam_ray_direction[:, 2]

    fixed = new_z.reshape(depth.shape)
    return fixed.astype(depth.dtype)


def overhead_features_from_trained_model(i, dataset, model):
    """
    For preliminary experiment. Probably not needed long term.
    """

    def enforce_depth_order(closer_depth, further_depth):
        with np.errstate(invalid='ignore'):
            mask = closer_depth > further_depth
        ret_further = further_depth.copy()
        ret_further[mask] = closer_depth[mask]
        return closer_depth, ret_further

    def add_background(masked_foreground, layer_containing_background):
        mask = np.isnan(masked_foreground)
        ret = masked_foreground.copy()
        ret[mask] = layer_containing_background[mask]
        return ret

    example = dataset[i]

    input_rgb = torch.Tensor(example['rgb'][None]).cuda()  # (1, 3, 240, 320)
    out_features_torch = unet.get_feature_map_output(model, input_rgb)  # (1, 64, 240, 320)
    out_ldi_torch = model(input_rgb)  # (1, 3, 240, 320)

    out_features = torch_utils.recursive_torch_to_numpy(out_features_torch)[0]  # (64, 240, 320)
    out_ldi = torch_utils.recursive_torch_to_numpy(out_ldi_torch)[0]  # (3, 240, 320)
    rgb_nosub = (example['rgb'] / dataset.rgb_scale + dataset.rgb_mean[:, None, None]) * dataset.rgb_scale  # (3, 240, 320)

    all_features = np.concatenate((rgb_nosub, out_features), axis=0).transpose(1, 2, 0).copy()  # (240, 320, 67)

    camera_filename = dataset[i]['camera_filename']

    p_front = fix_ray_displacement(np.power(2, out_ldi[2]) - 0.5)  # predicted frontal depth

    t_front = example['multi_layer_depth'][0].copy()
    t_back = add_background(example['multi_layer_depth'][1].copy(), t_front)  # instance exit
    _, t_back_ordering_enforced = enforce_depth_order(p_front, t_back)

    overhead_all_features = epipolar.feature_transform(all_features, p_front, t_back_ordering_enforced, camera_filename, 300, 300)  # (300, 300, 67)
    overhead_all_features = overhead_all_features.transpose(2, 0, 1).copy()

    return overhead_all_features, p_front, t_back_ordering_enforced, t_front, t_back


def finite_mean(arr):
    m = np.isfinite(arr)
    return arr[m].mean()


def tight_ax():
    import matplotlib.pyplot as pt
    ax = pt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    pt.axis('off')


def compute_error(overhead_pred, overhead_gt, overhead_features, plot=True):
    l1 = np.abs(overhead_pred - overhead_gt)
    error1 = finite_mean(l1)

    mask = np.isnan(overhead_gt) | np.isclose(overhead_gt, 0, rtol=0.001, atol=0.001)

    mask1 = mask | np.isclose(overhead_features[0], 0, rtol=0.001, atol=0.001)
    l1_copy = l1.copy()
    l1_copy[mask1] = np.nan
    error2 = finite_mean(l1_copy)

    mask2 = mask | ~np.isclose(overhead_features[0], 0, rtol=0.001, atol=0.001)
    l1_copy2 = l1.copy()
    l1_copy2[mask2] = np.nan
    error3 = finite_mean(l1_copy2)

    if plot:
        import matplotlib.pyplot as pt
        pt.figure(figsize=(19, 4))
        pt.subplot(1, 3, 1)

        tight_ax()
        pt.imshow(l1, cmap='Reds')
        pt.title('F\nL1 Error: {:.3f}'.format(error1))
        pt.clim(0, 1)
        pt.colorbar()

        pt.subplot(1, 3, 2)
        tight_ax()
        pt.imshow(l1_copy, cmap='Reds')
        pt.title('F\nL1 Error (interpolation): {:.3f}'.format(error2))
        pt.clim(0, 1)
        pt.colorbar()

        pt.subplot(1, 3, 3)
        tight_ax()
        pt.imshow(l1_copy2, cmap='Reds')
        pt.title('F\nL1 Error (extrapolation): {:.3f}'.format(error3))
        pt.clim(0, 1)
        pt.colorbar()

    return (error1, error2, error3)


def overhead_height_map_from_trained_models(i, dataset, model_overhead, model_ldi, use_gt_front=False, plot=True, is_log_depth=False, return_output=False):
    """
    This probably doesn't belong in this file. Should be moved later.
    """

    example = dataset[i]

    overhead_features, front, back, t_front, t_back = overhead_features_from_trained_model(i, dataset, model_ldi)
    if use_gt_front:
        overhead_features = example['overhead_features']

    out = model_overhead(torch.Tensor(overhead_features[None, 3:]).cuda())

    if is_log_depth:
        out = (2 ** out) - 0.5

    out_np = torch_utils.recursive_torch_to_numpy(out)

    rgb_nosub = (example['rgb'] / dataset.rgb_scale + dataset.rgb_mean[:, None, None]) * dataset.rgb_scale  # (3, 240, 320)

    if plot:
        import matplotlib.pyplot as pt
        from scene3d import geom2d
        pt.figure(figsize=(18, 10))
        pt.subplot(1, 3, 1)
        pt.title('A\nInput RGB')
        pt.imshow(rgb_nosub.transpose(1, 2, 0))
        tight_ax()

        if use_gt_front:
            pt.subplot(1, 3, 2)
            pt.title('B\nFront (GT)')
            pt.imshow(t_front)
            tight_ax()

            pt.subplot(1, 3, 3)
            pt.title('C\nBack (GT Instance-exit)')
            pt.imshow(t_back)
            tight_ax()

        else:
            pt.subplot(1, 3, 2)
            pt.title('B\nFront (predicted)')
            pt.imshow(front)
            tight_ax()

            pt.subplot(1, 3, 3)
            pt.title('C\nBack (GT Instance-exit)')
            pt.imshow(back)
            tight_ax()

        pt.figure(figsize=(20, 20))
        pt.title('D\nTransformed overhead feature map\n(64, 300, 300)')
        geom2d.display_montage(overhead_features[3:], gridwidth=15)

        pt.figure(figsize=(19, 5))
        pt.subplot(1, 3, 1)
        pt.title('E\nTransformed overhead RGB\n(3, 300, 300)')
        pt.imshow(overhead_features[:3].transpose(1, 2, 0))
        tight_ax()

        gt_finite_mask = np.isfinite(example['multi_layer_overhead_depth'][0])
        cmax = example['multi_layer_overhead_depth'][0][gt_finite_mask].max()
        cmin = example['multi_layer_overhead_depth'][0][gt_finite_mask].min()

        pt.subplot(1, 3, 2)
        pt.title('F\nPredicted height map\n(1, 300, 300)')
        pt.imshow(out_np[0, 0])
        tight_ax()
        pt.clim(cmin, cmax)
        pt.colorbar()

        pt.subplot(1, 3, 3)
        pt.title('G\nGT height map\n(1, 300, 300)')
        pt.imshow(example['multi_layer_overhead_depth'][0])
        tight_ax()
        pt.clim(cmin, cmax)
        pt.colorbar()

    errors = compute_error(out_np[0, 0], example['multi_layer_overhead_depth'][0], overhead_features, plot=plot)

    if plot:
        pt.show()

    if return_output:
        return errors, out_np
    return errors


def compute_overhead_camera(ref_position, ref_viewdir, ref_up, x, y, scale, theta):
    """
    :param ref_position: Position of the reference camera
    :param ref_viewdir:  Viewing direction of the reference camera. Must be a unit vector.
    :param ref_up: Up direction of the reference camera. Must be a unit vector.
    :param x:
    :param y:
    :param scale: Radius of square image frame.
    :param theta: Angle between viewing direction and gravity direction in radians.
    :return:
    """
    assert np.isclose(la.norm(ref_viewdir), 1.0)
    assert np.isclose(la.norm(ref_up), 1.0)

    ref_right = np.cross(ref_viewdir, ref_up)  # x axis
    xrot = transforms.rotation_matrix(angle=-theta, direction=ref_right, deg=False)[:3, :3]

    view_dir = xrot.dot(ref_viewdir)
    right = xrot.dot(ref_right)  # x axis
    up = xrot.dot(ref_up)  # y axis
    cam_pos = ref_position + right * x + up * y

    w = scale / np.sqrt(2.0)

    # Assume square frame. Constant near and far values.
    frustum = [-w, w, -w, w, 0.005, 50]  # left, right, bottom, top, near, far

    return ['O'] + cam_pos.tolist() + view_dir.tolist() + up.tolist() + frustum


def best_guess_depth_and_frustum_mask(frontal_depth_3layers, camera_filename):
    assert frontal_depth_3layers.shape[0] == 3
    assert frontal_depth_3layers.ndim == 3

    height = 300
    width = 300

    best_guess_ml = epipolar.render_depth_from_another_view(frontal_depth_3layers, camera_filename, target_height=height, target_width=width)

    fill_value = 0

    best_guess = best_guess_ml.copy()
    mask = np.isnan(best_guess)
    best_guess[mask] = np.inf
    best_guess = np.min(best_guess, axis=0)
    mask = ~np.isfinite(best_guess)
    best_guess[mask] = fill_value

    frustum_mask = epipolar.frustum_visibility_map_from_overhead_view(camera_filename, 300, 300)

    ret = np.stack([best_guess, frustum_mask])

    return ret


def make_overhead_camera_file(out_filename, x, y, scale, theta):
    ref_cam = [
        'P',
        0.0, 0.0, 0.0,  # position
        0.0, 0.0, -1.0,  # viewing dir
        0.0, 1.0, 0.0,  # up
        -0.00617793056641, 0.00617793056641, -0.00463344946349, 0.00463344946349, 0.01, 100  # intrinsics
    ]

    ref_pos = np.array(ref_cam[1:4], dtype=np.float64)
    ref_viewdir = np.array(ref_cam[4:7], dtype=np.float64)  # -z axis
    ref_up = np.array(ref_cam[7:10], dtype=np.float64)  # y axis

    target_cam = compute_overhead_camera(ref_pos, ref_viewdir, ref_up, x=x, y=y, scale=scale, theta=theta)

    out_camera_file_content = '\n'.join([' '.join([str(item) for item in ref_cam]), ' '.join([str(item) for item in target_cam])])

    io_utils.ensure_dir_exists(path.dirname(out_filename))

    with open(out_filename, 'w') as f:
        f.write(out_camera_file_content)

    return ref_cam, target_cam


def make_extra_features_batch(depth, overhead_camera_pose_4params, tmp_out_dir):
    """
    :param depth: torch.Tensor or np.ndarray of shape (B, 4, 240, 320). Should be 'multi_layer_depth_aligned_background' if GT.
    :param overhead_camera_pose_4params: torch.Tensor of shape (B, 4).
    :param tmp_out_dir:
    :return: (B, 2, 300, 300)
    """
    assert depth.shape[1] == 4
    assert overhead_camera_pose_4params.dim() == 2
    batch_size = len(depth)
    ret = np.empty((batch_size, 2, 300, 300), dtype=np.float32)

    io_utils.ensure_dir_exists(tmp_out_dir)

    for i in range(batch_size):
        depth_i = torch_utils.recursive_torch_to_numpy(depth[i, :3])
        assert depth_i.shape[0] == 3
        x, y, scale, theta = overhead_camera_pose_4params[i]
        x, y, scale, theta = x.item(), y.item(), scale.item(), theta.item()

        random_string = uuid.uuid4().hex
        new_cam_filename = path.join(tmp_out_dir, 'ortho_cam_{}.txt'.format(random_string))

        assert not path.exists(new_cam_filename)
        make_overhead_camera_file(new_cam_filename, x, y, scale, theta)
        assert path.exists(new_cam_filename)

        feat_arr = best_guess_depth_and_frustum_mask(depth_i, new_cam_filename)
        ret[i] = feat_arr
        os.remove(new_cam_filename)  # clean up

    return ret


class FeatureGenerator(object):
    def __init__(self):
        pass
        self.batches = {}  # stores prefetched batches, by target_device_id.

    def prefetch_batch_async(self, value, start_end_indices, target_device_id, options):
        raise NotImplementedError('Abstract method.')

    def pop_batch(self, target_device_id):
        raise NotImplementedError('Abstract method.')


class Transformer(FeatureGenerator):
    def __init__(self, depth_checkpoint_filename, segmentation_checkpoint_filename, device_id, num_workers=5):
        super().__init__()
        self.device_id = device_id
        log.info('[Transformer] Device id: {}'.format(self.device_id))

        with torch.cuda.device(self.device_id):
            log.info('Loading model {}'.format(depth_checkpoint_filename))
            self.depth_model, _ = train_eval_pipeline.load_checkpoint_as_frozen_model(depth_checkpoint_filename, use_cpu=False)
            self.depth_model.cuda()

            log.info('Loading model {}'.format(segmentation_checkpoint_filename))
            self.segmentation_model, _ = train_eval_pipeline.load_checkpoint_as_frozen_model(segmentation_checkpoint_filename, use_cpu=False)
            self.segmentation_model.cuda()

        if path.isdir('/mnt/ramdisk'):
            self.tmp_out_root = '/mnt/ramdisk/scene3d_transformer_cam'
        else:
            self.tmp_out_root = '/tmp/scene3d_transformer_cam'
        io_utils.ensure_dir_exists(self.tmp_out_root)
        self.num_workers = num_workers
        self.pool = ThreadPool(self.num_workers)

    @staticmethod
    def _get_batch_subset(batch, key, start_end):
        item = batch[key]
        if start_end is None:
            return item
        assert 0 <= start_end[0]
        assert start_end[0] < start_end[1]
        assert start_end[1] <= len(item)
        return item[start_end[0]:start_end[1]]

    def _get_transformed_features(self, batch, start_end_indices, use_gt_geometry=True) -> (np.ndarray, list):
        """
        :param batch:
        :param use_gt_geometry:
        :return:
        """

        # NOTE: `batch` should not be accessed directly. call self._get_batch_subset.

        stime0 = time.time()
        with torch.cuda.device(self.device_id):
            stime = time.time()
            assert isinstance(batch, dict)
            if use_gt_geometry:
                required_fields = ['rgb', 'multi_layer_depth', 'multi_layer_depth_aligned_background', 'overhead_camera_pose_4params', 'camera_filename']
            else:
                required_fields = ['rgb', 'overhead_camera_pose_4params', 'camera_filename']

            for field in required_fields:
                assert field in batch, field

            in_rgb_np = self._get_batch_subset(batch, 'rgb', start_end_indices)
            in_rgb = in_rgb_np.cuda()

            batch_size = len(in_rgb_np)
            # log.info('reading input rgb took {}'.format(time.time() - stime))

            if use_gt_geometry:
                # (B, 48, 240, 320)
                feature_map1 = unet.get_feature_map_output_v2(self.depth_model, in_rgb, return_encoding=False, return_final_output=False)
                feature_map1_np = torch_utils.recursive_torch_to_numpy(feature_map1)
                del feature_map1
                assert feature_map1_np.shape[1] == 48

                # (B, 64, 240, 320)
                feature_map2 = unet.get_feature_map_output_v1(self.segmentation_model, in_rgb, return_final_output=False)
                feature_map2_np = torch_utils.recursive_torch_to_numpy(feature_map2)
                del feature_map2
                assert feature_map2_np.shape[1] == 64

                multi_layer_depth = torch_utils.recursive_torch_to_numpy(self._get_batch_subset(batch, 'multi_layer_depth', start_end_indices)[:, :2])
                feat = dataset_utils.force_contiguous(np.concatenate([in_rgb_np, feature_map1_np, feature_map2_np], axis=1).transpose(0, 2, 3, 1))  # (B, H, W, C)
                front = dataset_utils.force_contiguous(multi_layer_depth[:, 0])  # (B, H, W)
                back = dataset_utils.force_contiguous(multi_layer_depth[:, 1])  # (B, H, W)

                depth_aligned = self._get_batch_subset(batch, 'multi_layer_depth_aligned_background', start_end_indices)
            else:
                # (B, 48, 240, 320)
                stime = time.time()
                feature_map1, predicted_depth = unet.get_feature_map_output_v2(self.depth_model, in_rgb, return_encoding=False, return_final_output=True)
                predicted_depth = torch_utils.recursive_torch_to_numpy(loss_fn.undo_log_depth(predicted_depth))
                feature_map1_np = torch_utils.recursive_torch_to_numpy(feature_map1)
                del feature_map1
                assert feature_map1_np.shape[1] == 48

                # (B, 64, 240, 320)
                feature_map2, predicted_ss = unet.get_feature_map_output_v1(self.segmentation_model, in_rgb, return_final_output=True)
                assert predicted_ss.shape[1] == 80
                predicted_ss = train_eval_pipeline.semantic_segmentation_from_raw_prediction(predicted_ss[:, :40])
                feature_map2_np = torch_utils.recursive_torch_to_numpy(feature_map2)
                del feature_map2
                assert feature_map2_np.shape[1] == 64
                # log.info('feature map output took {}'.format(time.time() - stime))

                stime = time.time()
                predicted_depth_segmented = train_eval_pipeline.segment_predicted_depth(predicted_depth, predicted_ss)
                depth_aligned = predicted_depth_segmented

                feat = dataset_utils.force_contiguous(np.concatenate([in_rgb_np, feature_map1_np, feature_map2_np], axis=1).transpose(0, 2, 3, 1))  # (B, H, W, C)

                front = dataset_utils.force_contiguous(train_eval_pipeline.traditional_depth_from_aligned_multi_layer_depth(predicted_depth_segmented))  # (B, H, W)
                back = dataset_utils.force_contiguous(predicted_depth_segmented[:, 1])  # (B, H, W)
                # log.info('memory stuff took {}'.format(time.time() - stime))

            # (B, 300, 300, 115)
            camera_filenames = self._get_batch_subset(batch, 'camera_filename', start_end_indices)

            stime = time.time()
            tranformed_batch = epipolar.feature_transform_parallel(feat, front_depth_data=front, back_depth_data=back, camera_filenames=camera_filenames, target_height=300, target_width=300)

            ret = np.empty((batch_size, 117, 300, 300), dtype=np.float32)

            # (B, 115, 300, 300)
            ret[:, 2:] = tranformed_batch.transpose(0, 3, 1, 2)
            # log.info('feature_transform_parallel took {}'.format(time.time() - stime))

            # (B, 2, 300, 300)

            stime = time.time()
            camparams = self._get_batch_subset(batch, 'overhead_camera_pose_4params', start_end_indices)
            ret[:, :2] = make_extra_features_batch(depth_aligned, camparams, self.tmp_out_root)
            # log.info('make_extra_features_batch took {}'.format(time.time() - stime))
            # log.info('TOTAL prep took {}'.format(time.time() - stime0))

            return ret, self._get_batch_subset(batch, 'name', start_end_indices)

    def get_transformed_features(self, batch, start_end_indices, use_gt_geometry=True) -> (np.ndarray, list):
        if start_end_indices is None:
            start_end_indices = [0, len(batch['name'])]
        start, end = start_end_indices
        size = end - start
        assert size > 0
        all_indices = dataset_utils.divide_start_end_indices(size, num_chunks=self.num_workers, offset=start)

        args = []
        for item in all_indices:
            args.append([
                batch, item, use_gt_geometry
            ])

        out = self.pool.starmap(self._get_transformed_features, args)

        out_feat = []
        out_names = []
        for f, names in out:
            out_feat.append(f)
            out_names.extend(names)

        return np.concatenate(out_feat, axis=0), out_names

    def _prefetch_worker(self, batch, start_end_indices, target_device_id, use_gt_geometry, out_dict):
        assert out_dict[target_device_id] is None
        out_feat, out_names = self.get_transformed_features(batch, start_end_indices, use_gt_geometry=use_gt_geometry)
        out_dict[target_device_id] = (torch.Tensor(out_feat), out_names)

    def prefetch_batch_async(self, batch, start_end_indices, target_device_id, options=None):
        assert target_device_id not in self.batches
        self.batches[target_device_id] = None
        use_gt_geometry = options['use_gt_geometry']
        thread = threading.Thread(target=self._prefetch_worker, args=(batch, start_end_indices, target_device_id, use_gt_geometry, self.batches), kwargs={})
        thread.start()

    def pop_batch(self, target_device_id):
        stime = time.time()
        print_time = time.time()
        while target_device_id not in self.batches or self.batches[target_device_id] is None:
            if time.time() - print_time > 3:
                print('.', end='', flush=True)
                print_time = time.time()
            time.sleep(0.05)
        print('Waited {:.2f} seconds for feature transformation.'.format(time.time() - stime))
        batch = self.batches.pop(target_device_id)
        return batch
