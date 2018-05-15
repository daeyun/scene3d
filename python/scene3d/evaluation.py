import torch
import matplotlib
import matplotlib.pyplot as pt
import torch.utils.data
import torch.nn
from scene3d import torch_utils
from scene3d.dataset import v1
import numpy as np


def eval_single_depth_model(model: torch.nn.Module, depth_dataset: v1.MultiLayerDepth, indices, visualize=True):
    if model.training:
        model.eval()
    assert not model.training

    loss_list = []

    for ind in indices:
        example_name, in_rgb_np, target_np = depth_dataset[ind]
        in_rgb = torch.Tensor(in_rgb_np[None]).cuda()

        pred = model(in_rgb)
        assert pred.shape[1] == 1, 'Channel dimension must be 1.'

        rgb_np = (in_rgb_np.transpose(1, 2, 0) * 255 + depth_dataset.rgb_mean).round().astype(np.uint8)
        pred_log_np = torch_utils.recursive_torch_to_numpy(pred)[0, 0]  # (h, w)
        pred_np = np.power(2, pred_log_np) - 0.5
        target_np = target_np.transpose(1, 2, 0)
        single_target_np = target_np[:, :, 0] - target_np[:, :, 1]
        single_target_log_np = np.log2(single_target_np + 0.5)

        del pred

        mask = ~np.isnan(target_np[:, :, 0])

        loss_map = np.abs(single_target_np - pred_np)
        loss = loss_map[mask].mean()
        assert ~np.isnan(loss), 'nan: index {}'.format(ind)
        loss_map_log = np.abs(single_target_log_np - pred_log_np)

        loss_list.append(loss)

        if visualize:
            # too small or too large values are clipped for visualization.
            # tmax = single_target_np.max()
            # tmin = single_target_np.min()
            # pred_np[tmax<pred_np] = tmax
            # pred_np[tmin>pred_np] = tmin

            print('Example ID:', example_name)

            pt.figure()
            pt.imshow(rgb_np)
            pt.axis('off')
            pt.title('$Input$', fontsize=16)

            pt.figure(figsize=(22, 4))
            pt.subplot(1, 3, 1)  # 1
            pt.imshow(pred_np)
            pt.axis('off')
            pt.colorbar()
            pt.title('$Pred$', fontsize=16)
            pt.subplot(1, 3, 2)  # 2
            pt.imshow(single_target_np)
            pt.axis('off')
            pt.colorbar()
            pt.title('$GT$', fontsize=16)
            pt.subplot(1, 3, 3)  # 3
            pt.imshow(loss_map, cmap='Reds')
            pt.axis('off')
            pt.clim(0, 1)
            pt.colorbar()
            pt.title('L1 error:  {:.3f}   (log scale: {:.3f})'.format(loss_map[mask].mean(), loss_map_log[mask].mean()), fontsize=11)

            pt.show()

    return np.mean(loss_list)


def eval_multi_depth_model(model: torch.nn.Module, depth_dataset: v1.MultiLayerDepth, indices, visualize=True):
    if model.training:
        model.eval()
    assert not model.training

    loss_list = []
    single_loss_list = []

    for ind in indices:
        example_name, in_rgb_np, target_np = depth_dataset[ind]
        in_rgb = torch.Tensor(in_rgb_np[None]).cuda()

        pred = model(in_rgb)
        assert pred.shape[1] == 2, 'Channel dimension must be 2.'

        rgb_np = (in_rgb_np.transpose(1, 2, 0) * 255 + depth_dataset.rgb_mean).round().astype(np.uint8)
        pred_log_np = torch_utils.recursive_torch_to_numpy(pred)[0].transpose(1, 2, 0)  # (h, w, 2)
        pred_np = np.power(2, pred_log_np) - 0.5
        target_np = target_np.transpose(1, 2, 0)  # (h, w, 2)
        target_log_np = np.log2(target_np + 0.5)
        single_target_np = target_np[:, :, 0] - target_np[:, :, 1]
        single_pred_np = pred_np[:, :, 0] - pred_np[:, :, 1]

        del pred

        mask = ~np.isnan(target_np[:, :, 0])

        loss_map = np.abs(target_np - pred_np)
        loss_map_log = np.abs(target_log_np - pred_log_np)
        single_loss_map = np.abs(single_target_np - single_pred_np)
        loss = loss_map[mask].mean()
        assert ~np.isnan(loss), 'nan: index {}'.format(ind)
        single_loss = single_loss_map[mask].mean()

        loss_list.append(loss)
        single_loss_list.append(single_loss)

        if visualize:
            # too small or too large values are clipped for visualization.
            # tmax = single_target_np.max()
            # tmin = single_target_np.min()
            # pred_np[tmax<pred_np] = tmax
            # pred_np[tmin>pred_np] = tmin

            print('Example ID:', example_name)

            pt.figure()
            pt.imshow(rgb_np)
            pt.axis('off')
            pt.title('$Input$', fontsize=16)

            pt.figure(figsize=(22, 4))
            pt.subplot(1, 3, 1)  # 1
            pt.imshow(pred_np[:, :, 0])
            pt.axis('off')
            pt.colorbar()
            pt.title('$Pred_{bg}$', fontsize=16)
            pt.subplot(1, 3, 2)  # 2
            pt.imshow(target_np[:, :, 0])
            pt.axis('off')
            pt.colorbar()
            pt.title('$GT_{bg}$', fontsize=16)
            pt.subplot(1, 3, 3)  # 3
            pt.imshow(loss_map[:, :, 0], cmap='Reds')
            pt.axis('off')
            pt.clim(0, 1)
            pt.colorbar()
            pt.title('L1 error:  {:.3f}   (log scale: {:.3f})'.format(loss_map[:, :, 0][mask].mean(), loss_map_log[:, :, 0][mask].mean()), fontsize=11)

            pt.figure(figsize=(22, 4))
            pt.subplot(1, 3, 1)  # 1
            pt.imshow(pred_np[:, :, 1])
            pt.axis('off')
            pt.colorbar()
            pt.title('$Pred_{fg}$', fontsize=16)
            pt.subplot(1, 3, 2)  # 2
            pt.imshow(target_np[:, :, 1])
            pt.axis('off')
            pt.colorbar()
            pt.title('$GT_{fg}$', fontsize=16)
            pt.subplot(1, 3, 3)  # 3
            pt.imshow(loss_map[:, :, 1], cmap='Reds')
            pt.axis('off')
            pt.clim(0, 1)
            pt.colorbar()
            pt.title('L1 error:  {:.3f}   (log scale: {:.3f})'.format(loss_map[:, :, 1][mask].mean(), loss_map_log[:, :, 1][mask].mean()), fontsize=11)

            pt.figure(figsize=(22, 4))
            pt.subplot(1, 3, 1)  # 1
            pt.imshow(single_pred_np)
            pt.axis('off')
            pt.colorbar()
            pt.title('$Pred_{bg} - Pred_{fg}$', fontsize=16)
            pt.subplot(1, 3, 2)  # 2
            pt.imshow(single_target_np)
            pt.axis('off')
            pt.colorbar()
            pt.title('$GT_{bg} - GT_{fg}$', fontsize=16)
            pt.subplot(1, 3, 3)  # 3
            pt.imshow(single_loss_map, cmap='Reds')
            pt.axis('off')
            pt.clim(0, 1)
            pt.colorbar()
            pt.title('L1 error:  {:.3f}'.format(single_loss_map[mask].mean()), fontsize=11)

            pt.show()

    return np.mean(loss_list), np.mean(single_loss_list)


colormap40 = np.array([[0, 0, 143],
                       [182, 0, 0],
                       [0, 140, 0],
                       [195, 79, 255],
                       [1, 165, 202],
                       [236, 157, 0],
                       [118, 255, 0],
                       [89, 83, 84],
                       [255, 117, 152],
                       [148, 0, 115],
                       [0, 243, 204],
                       [72, 83, 255],
                       [166, 161, 154],
                       [0, 67, 1],
                       [237, 183, 255],
                       [138, 104, 0],
                       [97, 0, 163],
                       [92, 0, 17],
                       [255, 245, 133],
                       [0, 123, 105],
                       [146, 184, 83],
                       [171, 212, 255],
                       [126, 121, 163],
                       [255, 84, 1],
                       [10, 87, 125],
                       [168, 97, 92],
                       [231, 0, 185],
                       [255, 195, 166],
                       [91, 53, 0],
                       [0, 180, 133],
                       [126, 158, 255],
                       [231, 2, 92],
                       [184, 216, 183],
                       [192, 130, 183],
                       [111, 137, 91],
                       [138, 72, 162],
                       [91, 50, 90],
                       [220, 138, 103],
                       [79, 92, 44],
                       [0, 225, 115],
                       [0, 0, 0],  # 41st black color for uncategorized.
                       ]) / 255


def colorize_segmentation(seg40_image: np.ndarray):
    assert seg40_image.dtype == np.uint8

    seg40_image_flat = seg40_image.ravel()
    seg40_image_flat[seg40_image_flat == 255] = 40  # black color

    h, w = seg40_image.shape
    ret = colormap40[seg40_image_flat].reshape(h, w, 3)
    return ret


def eval_nyu40_segmentation_model(model: torch.nn.Module, depth_dataset: v1.NYU40Segmentation, indices, visualize=True):
    if model.training:
        model.eval()
    assert not model.training

    for ind in indices:
        example_name, in_rgb_np, target_category_np = depth_dataset[ind]

        in_rgb = torch.Tensor(in_rgb_np[None]).cuda()

        pred = model(in_rgb)
        assert pred.shape[1] == 40, 'Channel dimension must be 40.'

        rgb_np = (in_rgb_np.transpose(1, 2, 0) * 255 + depth_dataset.rgb_mean).round().astype(np.uint8)
        pred_np = torch_utils.recursive_torch_to_numpy(pred)[0]  # (40, h, w)

        target_category_np_flat = target_category_np.ravel()
        target_category_np_flat[target_category_np_flat == 255] = 40  # black color

        pred_np_argmax = np.argmax(pred_np, axis=0).astype(np.uint8)

        target_category_np_colorized = colorize_segmentation(target_category_np)
        pred_np_colorized = colorize_segmentation(pred_np_argmax)

        if visualize:
            print('Example ID:', example_name)

            pt.figure(figsize=(18, 4))
            pt.subplot(1, 3, 1)  # 1
            pt.imshow(rgb_np)
            pt.axis('off')
            pt.title('$GT$', fontsize=16)
            pt.subplot(1, 3, 2)  # 2
            pt.imshow(pred_np_colorized)
            pt.axis('off')
            pt.title('$Pred$', fontsize=16)
            pt.subplot(1, 3, 3)  # 3
            pt.imshow(target_category_np_colorized)
            pt.axis('off')
            pt.title('$GT$', fontsize=16)

            pt.show()
