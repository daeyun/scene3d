from pprint import pprint
import glob
import numpy.linalg as la
import math
import os
import shutil
import shutil
from scene3d import geom2d
from os import path
import numpy as np
import scipy as sp
import scipy.misc
from scene3d import pbrs_utils
from scene3d import scannet_utils
from scene3d import suncg_utils
from scene3d import torch_utils
from scene3d import render_depth
from scene3d import train_eval_pipeline
from scene3d.eval import f3d_utils
from scene3d import transforms
from scene3d import io_utils
from scene3d import depth_mesh_utils_cpp
from scene3d import config
from scene3d import camera
from scene3d import loss_fn
from scene3d import category_mapping
from scene3d import feat
from scene3d import evaluation
from scene3d import log
from scene3d import visualizer
from scene3d import epipolar
from scene3d.eval import generate_gt_mesh
from scene3d.eval import post_processing
from scene3d import data_utils
from scene3d.dataset import dataset_utils
from scene3d.dataset import v8
from scene3d.dataset import v2
from scene3d.net import unet
from scene3d.net import unet_overhead
from scene3d import notebook_utils
import cv2
import matplotlib.pyplot as pt
import torch
from torch import nn

from scene3d import category_mapping
import pickle


class Visualizer(object):
    def __init__(self):
        self.depth_checkpoint = path.join(config.default_out_root, 'v8/v8-multi_layer_depth_aligned_background_multi_branch/1/00906000_010_0000080.pth')
        self.seg_checkpoint = path.join(config.default_out_root, 'v8/v8-category_nyu40_merged_background-2l/0/00966000_009_0005272.pth')

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.depth_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(self.depth_checkpoint)
        print(metadata)
        self.seg_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(self.seg_checkpoint)
        print(metadata)

        checkpoint_filenames = {
            'pose_3param': path.join(config.default_out_root, 'v8/v8-overhead_camera_pose/0/00420000_018_0014478.pth'),
            'overhead_height_map_model': path.join(config.default_out_root, 'v8/OVERHEAD_offline_01/0/00050000_001_0004046.pth'),
        }

        self.hm_model = train_eval_pipeline.HeightMapModel(checkpoint_filenames, device_id=0)

        self.input_image_meansub_cuda = None
        self.input_image_name = None

        self.nyu_names = io_utils.read_lines_and_strip('/data3/nyu/eval.txt')
        np.random.RandomState(seed=0).shuffle(self.nyu_names)

    def minsub_rgb(self, img):
        rgb_mean = np.array([178.1781, 158.5039, 142.5141], dtype=np.float32) * 1
        # rgb_mean = self.input_image.mean(0, keepdims=True).mean(1, keepdims=True).squeeze()
        rgb_scale = 1 / 255.0
        in_rgb = img.astype(np.float32).transpose(2, 0, 1)
        in_rgb -= rgb_mean.reshape(3, 1, 1)
        in_rgb *= rgb_scale
        return in_rgb

    def show_input_image(self):
        print('nyu_name: {}'.format(self.input_image_name))
        notebook_utils.quick_show_rgb(self.input_image)

    def load_nyu(self, name_index=None, nyu_name=None):
        if nyu_name is not None:
            name = nyu_name
            if not isinstance(name, str):
                name = str(name)
        else:
            name = self.nyu_names[name_index]

        img = train_eval_pipeline.nyu_rgb_image(name)
        self.input_image = img
        self.input_image_meansub_cuda = torch.Tensor(self.minsub_rgb(img)[None]).cuda()
        self.input_image_name = name

    def generate_mesh(self):
        depth_pred = self.depth_model(self.input_image_meansub_cuda)
        seg_pred = self.seg_model(self.input_image_meansub_cuda)
        name = self.input_image_name

        seg_pred_argmax = train_eval_pipeline.semantic_segmentation_from_raw_prediction(seg_pred[:, :40])

        segmented_depth = loss_fn.undo_log_depth(train_eval_pipeline.segment_predicted_depth(torch_utils.recursive_torch_to_numpy(depth_pred), seg_pred_argmax))
        assert segmented_depth.shape[0] == 1
        segmented_depth = np.squeeze(segmented_depth)

        # out = train_eval_pipeline.save_mldepth_as_meshes(segmented_depth, example)

        train_eval_pipeline.save_mldepth_as_meshes_realworld(segmented_depth, '/mnt/ramdisk/nyu_mld')

        # Factored3d
        rgb = io_utils.read_jpg('/data3/nyu/images/{}.jpg'.format(name))[3:-3]
        f3d_utils.align_factored3d_mesh_with_meshlab_cam_coords('/data3/out/scene3d/factored3d_pred/nyu/{}/codes.obj'.format(name), '/mnt/ramdisk/nyu_mld/f3d_objets.stl')
        f3d_utils.align_factored3d_mesh_with_meshlab_cam_coords('/data3/out/scene3d/factored3d_pred/nyu/{}/layout.obj'.format(name), '/mnt/ramdisk/nyu_mld/f3d_layout.stl')

        fig = pt.figure(figsize=(20, 8))

        ax = fig.add_subplot(141)
        ax.imshow(segmented_depth[0])
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        ax = fig.add_subplot(142)
        ax.imshow(segmented_depth[1])
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        ax = fig.add_subplot(143)
        ax.imshow(segmented_depth[2])
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        ax = fig.add_subplot(144)
        ax.imshow(segmented_depth[3])
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        seg_img = evaluation.colorize_segmentation(seg_pred_argmax[0])
        fig = pt.figure(figsize=(20, 8))

        ax = fig.add_subplot(141)
        ax.imshow(seg_img)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        seg_pred_argmax2 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(seg_pred[:, 40:])
        seg_pred_argmax2[seg_pred_argmax == 34] = 34
        seg_img2 = evaluation.colorize_segmentation(seg_pred_argmax2[0])
        ax = fig.add_subplot(142)
        ax.imshow(seg_img2)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        ##########

        input_batch = {
            'rgb': self.input_image_meansub_cuda,
            'name': 'nyu_0',
        }

        out, features = torch_utils.recursive_torch_to_numpy(self.hm_model.predict_height_map_single(input_batch))

        f, axarr = pt.subplots(1, 4, figsize=(25, 4))

        ax = axarr[0]
        ax.imshow(out)
        notebook_utils.remove_ticks(ax)

        ax = axarr[1]
        ax.imshow(v8.undo_rgb_whitening(features[2:5]).transpose(1, 2, 0))
        notebook_utils.remove_ticks(ax)

        ax = axarr[2]
        plot = ax.imshow(features[0])
        pt.colorbar(plot, ax=ax)
        notebook_utils.remove_ticks(ax)

        ax = axarr[3]
        ax.imshow(features[1], cmap='gray')
        notebook_utils.remove_ticks(ax)

        pt.show()


class Visualizer2(object):
    def __init__(self):
        self.depth_checkpoint = path.join(config.default_out_root, 'v9/v9-multi_layer_depth_aligned_background_multi_branch/0/01149000_005_0003355.pth')
        self.seg_checkpoint = path.join(config.default_out_root, 'v9/v9-category_nyu40_merged_background-2l/0/01130000_005_0001780.pth')

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.depth_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(self.depth_checkpoint)
        print(metadata)
        self.seg_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(self.seg_checkpoint)
        print(metadata)

        # checkpoint_filenames = {
        #     'pose_3param': path.join(config.default_out_root_v8, 'v8/v8-overhead_camera_pose/0/00420000_018_0014478.pth'),
        #     # 'overhead_height_map_model': path.join(config.default_out_root_v8, 'v8/OVERHEAD_offline_01/0/00050000_001_0004046.pth'),
        # }
        checkpoint_filenames = {
            'pose_3param': path.join(config.default_out_root_v8, 'v8/v8-overhead_camera_pose/0/00420000_018_0014478.pth'),
            # 'overhead_height_map_model': path.join(config.default_out_root_v8, 'v8/OVERHEAD_offline_01/0/00050000_001_0004046.pth'),
            'overhead_height_map_model': path.join(config.default_out_root, 'v9/v9_OVERHEAD_v1_heightmap_01/0/00056000_003_0011546.pth'),
            'overhead_segmentation_model': path.join(config.default_out_root, 'v9/v9_OVERHEAD_v1_segmentation_01/0/00016000_000_0002000.pth'),
        }

        # self.hm_model = train_eval_pipeline.HeightMapModel(checkpoint_filenames, device_id=0)
        self.hm_model = train_eval_pipeline.HeightMapModel(checkpoint_filenames, device_id=0, num_transformer_workers=1)

        self.input_image_meansub_cuda = None
        self.input_image_name = None
        self.example = None

    def minsub_rgb(self, img):
        rgb_mean = np.array([178.1781, 158.5039, 142.5141], dtype=np.float32) * 1
        # rgb_mean = self.input_image.mean(0, keepdims=True).mean(1, keepdims=True).squeeze()
        rgb_scale = 1 / 255.0
        in_rgb = img.astype(np.float32)
        in_rgb -= rgb_mean.reshape(3, 1, 1)
        in_rgb *= rgb_scale
        return in_rgb

    def show_input_image(self, scale=1.0):
        print('example_name: {}'.format(self.input_image_name))
        notebook_utils.quick_show_rgb(cv2.resize(self.input_image, (int(self.input_image.shape[1] * scale), int(self.input_image.shape[0] * scale))))

    def load_pbrs(self, dataset, index):
        self.example = dataset[index]
        img = self.example['rgb'].squeeze()
        self.input_image = v8.undo_rgb_whitening(img).transpose(1, 2, 0)
        self.input_image_meansub_cuda = torch.Tensor(img[None]).cuda()
        self.input_image_name = self.example['name']

    def generate_mesh(self, visualize_gt=True):
        """
        TODO: this function is mostly used to generate visualization. name is misleading
        :return:
        """
        depth_pred = self.depth_model(self.input_image_meansub_cuda)
        seg_pred = self.seg_model(self.input_image_meansub_cuda)
        name = self.input_image_name

        assert seg_pred.shape[0] == 1
        pred_l1 = seg_pred[:, :40]
        pred_l2 = seg_pred[:, 40:]
        argmax_l1 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(pred_l1)
        argmax_l2 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(pred_l2)

        depth_pred_np = torch_utils.recursive_torch_to_numpy(depth_pred)

        segmented_depth = train_eval_pipeline.segment_predicted_depth(depth_pred_np, argmax_l1, argmax_l2)
        segmented_depth = np.squeeze(segmented_depth)
        assert segmented_depth.shape[0] == 5, segmented_depth.shape  # making sure this is v9

        # out_filename = '/mnt/ramdisk/vis_mesh/single0.ply'
        # depth_mesh_utils_cpp.depth_to_mesh(segmented_depth_single[0], self.example['camera_filename'], camera_index=0, dd_factor=10, out_ply_filename=out_filename)
        # out_filename = '/mnt/ramdisk/vis_mesh/single1.ply'
        # depth_mesh_utils_cpp.depth_to_mesh(segmented_depth_single[3], self.example['camera_filename'], camera_index=0, dd_factor=10, out_ply_filename=out_filename)

        # Factored3d
        # aligned_filenames = f3d_utils.align_factored3d_mesh_with_our_gt('/data3/out/scene3d/factored3d_pred/{}/layout.obj'.format(name), self.example['name'])

        gt_depth = self.example['multi_layer_depth_aligned_background']

        gt_pixel_values = sorted(gt_depth[np.isfinite(gt_depth)].tolist())
        # minimum and maximum after removing outliers
        cmin = gt_pixel_values[50]
        cmax = gt_pixel_values[-50]
        clim = [cmin, cmax]

        figsize = [20, 8]

        ### PREDICTED DEPTH
        fig = pt.figure(figsize=figsize)
        fig.suptitle('Prediction', fontsize=17, y=0.75, x=0.54, fontweight=500)

        ax = fig.add_subplot(151)
        im = ax.imshow(segmented_depth[0], clim=clim)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_title('$D_1 \odot M_1$', {'fontsize': 14})

        ax = fig.add_subplot(152)
        ax.imshow(segmented_depth[1], clim=clim)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_title('$D_2 \odot M_2$', {'fontsize': 14})

        ax = fig.add_subplot(153)
        ax.imshow(segmented_depth[2], clim=clim)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_title('$D_3 \odot M_3$', {'fontsize': 14})

        ax = fig.add_subplot(154)
        ax.imshow(segmented_depth[3], clim=clim)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_title('$D_4 \odot M_4$', {'fontsize': 14})

        ax = fig.add_subplot(155)
        ax.imshow(segmented_depth[4], clim=clim)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_title('$D_5 \odot M_5$', {'fontsize': 14})

        fig.subplots_adjust(right=0.96)
        cbar_ax = fig.add_axes([0.99, 0.365, 0.01, 0.273])
        fig.colorbar(im, cax=cbar_ax)

        seg_img = evaluation.colorize_segmentation(argmax_l1[0])

        # ### PREDICTED SEGMENTATION
        # fig = pt.figure(figsize=(40, 3.33))
        # ax = fig.add_subplot(141)
        # ax.imshow(seg_img)
        # ax.axes.get_xaxis().set_ticks([])
        # ax.axes.get_yaxis().set_ticks([])
        # ax.set_title('$D_1$ Segmentation  ($M_1$)', {'fontsize': 14})
        #
        # seg_img2 = evaluation.colorize_segmentation(argmax_l2[0])
        # ax = fig.add_subplot(142)
        # ax.imshow(seg_img2)
        # ax.axes.get_xaxis().set_ticks([])
        # ax.axes.get_yaxis().set_ticks([])
        # ax.set_title('$D_3$ Segmentation  ($M_3$)', {'fontsize': 14})

        pt.show()
        print('')

        if visualize_gt:
            ### GT DEPTH
            fig = pt.figure(figsize=figsize)
            fig.suptitle('Ground Truth', fontsize=17, y=0.75, x=0.54, fontweight=500)

            ax = fig.add_subplot(151)
            im = ax.imshow(gt_depth[0], clim=clim)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_1  \odot \\bar M_1$', {'fontsize': 14})

            ax = fig.add_subplot(152)
            ax.imshow(gt_depth[1], clim=clim)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_2  \odot \\bar M_1$', {'fontsize': 14})

            ax = fig.add_subplot(153)
            ax.imshow(gt_depth[2], clim=clim)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_3  \odot \\bar M_1$', {'fontsize': 14})

            ax = fig.add_subplot(154)
            ax.imshow(gt_depth[3], clim=clim)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_4  \odot \\bar M_1$', {'fontsize': 14})

            ax = fig.add_subplot(155)
            ax.imshow(gt_depth[4], clim=clim)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_5  \odot \\bar M_1$', {'fontsize': 14})

            gt_seg = self.example['category_nyu40_merged_background'].copy()
            gt_seg[gt_seg == 65535] = 34
            gt_seg = gt_seg.astype(np.uint8)

            fig.subplots_adjust(right=0.96)
            cbar_ax = fig.add_axes([0.99, 0.365, 0.01, 0.273])
            fig.colorbar(im, cax=cbar_ax)

            seg_img = evaluation.colorize_segmentation(gt_seg[0])

            ### GT SEGMENTATION
            fig = pt.figure(figsize=(40, 3.33))
            ax = fig.add_subplot(141)
            ax.imshow(seg_img)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_1$ Segmentation  ($\\bar M_1$)', {'fontsize': 14})

            seg_img2 = evaluation.colorize_segmentation(gt_seg[1])
            ax = fig.add_subplot(142)
            ax.imshow(seg_img2)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_3$ Segmentation  ($\\bar M_3$)', {'fontsize': 14})

            pt.show()
            print('')

            assert depth_pred_np.shape[0] == 1
            assert depth_pred_np.shape[1] == 5
            mae_map = np.abs(depth_pred_np[0] - gt_depth)

            error_clim = [0, 1.5]

            ### Error map
            fig = pt.figure(figsize=figsize)
            fig.suptitle('L1 Error Map', fontsize=17, y=0.75, x=0.54, fontweight=500)

            ax = fig.add_subplot(151)
            im = ax.imshow(mae_map[0], clim=error_clim, cmap='Reds')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$| D_1 - \\bar D_1 |  \odot \\bar M_1$', {'fontsize': 14})

            ax = fig.add_subplot(152)
            ax.imshow(mae_map[1], clim=error_clim, cmap='Reds')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$| D_2 - \\bar D_2 |  \odot \\bar M_2$', {'fontsize': 14})

            ax = fig.add_subplot(153)
            ax.imshow(mae_map[2], clim=error_clim, cmap='Reds')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$| D_3 - \\bar D_3 |  \odot \\bar M_3$', {'fontsize': 14})

            ax = fig.add_subplot(154)
            ax.imshow(mae_map[3], clim=error_clim, cmap='Reds')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$| D_4 - \\bar D_4 |  \odot \\bar M_4$', {'fontsize': 14})

            ax = fig.add_subplot(155)
            ax.imshow(mae_map[4], clim=error_clim, cmap='Reds')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$| D_5 - \\bar D_5 |  \odot \\bar M_5$', {'fontsize': 14})

            gt_seg = self.example['category_nyu40_merged_background'].copy()
            gt_seg[gt_seg == 65535] = 34
            gt_seg = gt_seg.astype(np.uint8)

            fig.subplots_adjust(right=0.96)
            cbar_ax = fig.add_axes([0.99, 0.365, 0.01, 0.273])
            fig.colorbar(im, cax=cbar_ax)

            seg_img = evaluation.colorize_segmentation(gt_seg[0])

            ### GT SEGMENTATION
            fig = pt.figure(figsize=(40, 3.33))
            ax = fig.add_subplot(141)
            ax.imshow(seg_img)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_1$ Segmentation  ($\\bar M_1$)', {'fontsize': 14})

            seg_img2 = evaluation.colorize_segmentation(gt_seg[1])
            ax = fig.add_subplot(142)
            ax.imshow(seg_img2)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_3$ Segmentation  ($\\bar M_3$)', {'fontsize': 14})
            pt.show()
            print('')

        ##########

        input_batch = {
            'rgb': self.input_image_meansub_cuda,
            'name': [self.example['name']],
        }

        out, features, overhead_argmax_l1 = torch_utils.recursive_torch_to_numpy(self.hm_model.predict_height_map_single(input_batch))

        # f, axarr = pt.subplots(1, 5, figsize=(28, 4))
        #
        # ax = axarr[0]
        # ax.imshow(v8.undo_rgb_whitening(features[2:5]).transpose(1, 2, 0))
        # notebook_utils.remove_ticks(ax)
        # ax.set_title('Transformed Virtual View\nRGB Features', {'fontsize': 14})
        #
        # ax = axarr[1]
        # plot = ax.imshow(features[0])
        # pt.colorbar(plot, ax=ax)
        # notebook_utils.remove_ticks(ax)
        # ax.set_title('Virtual View Best Guess Depth', {'fontsize': 14})
        #
        # ax = axarr[2]
        # ax.imshow(features[1], cmap='gray')
        # notebook_utils.remove_ticks(ax)
        # ax.set_title('Input View Frustum Mask Relative to\nProposed Virtual Camera Pose', {'fontsize': 14})
        #
        # ax = axarr[3]
        # feat_montage = geom2d.montage(features[5:5 + 25], gridwidth=None, empty_value=np.nan)
        # plot = ax.imshow(feat_montage)
        # ax.set_xticks(np.arange(0, feat_montage.shape[1], features.shape[2]))
        # ax.set_yticks(np.arange(0, feat_montage.shape[0], features.shape[1]))
        # ax.grid(which='major', color='w', linestyle='-', linewidth=1)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # pt.colorbar(plot, ax=ax)
        # ax.set_title('Transformed Virtual View\nDepth Feature Map (first 25 channels)', {'fontsize': 14})
        #
        # ax = axarr[4]
        # feat_montage = geom2d.montage(features[53:53 + 25], gridwidth=None, empty_value=np.nan)
        # plot = ax.imshow(feat_montage)
        # ax.set_xticks(np.arange(0, feat_montage.shape[1], features.shape[2]))
        # ax.set_yticks(np.arange(0, feat_montage.shape[0], features.shape[1]))
        # ax.grid(which='major', color='w', linestyle='-', linewidth=1)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # pt.colorbar(plot, ax=ax)
        # ax.set_title('Transformed Virtual VIew\nSegmentation Feature Map (first 25 channels)', {'fontsize': 14})
        #
        # f.suptitle('Epipolar Feature Transformers', fontsize=19, y=1.13, fontweight=500, x=0.427)

        gt_height_map = self.example['multi_layer_overhead_depth'][0]
        height_map_values = sorted(gt_height_map[np.isfinite(gt_height_map)].tolist())
        h_clim = [0, height_map_values[-20]]

        f, axarr = pt.subplots(1, 3, figsize=(15, 3))
        ax = axarr[0]
        out_segmented = out.copy()
        out_segmented[overhead_argmax_l1 == 34] = 0
        plot = ax.imshow(out_segmented, clim=h_clim)
        notebook_utils.remove_ticks(ax)
        pt.colorbar(plot, ax=ax)
        ax.set_title('Virtual View\nPredicted Surface', {'fontsize': 14})

        ax = axarr[1]
        gt_height_map_copy = gt_height_map.copy()
        gt_height_map_copy[np.isnan(gt_height_map_copy)] = 0
        plot = ax.imshow(gt_height_map_copy, clim=h_clim)
        notebook_utils.remove_ticks(ax)
        pt.colorbar(plot, ax=ax)
        ax.set_title('Virtual View\nGround Truth Surface', {'fontsize': 14})

        ax = axarr[2]
        overhead_error_map = np.abs(gt_height_map - out)
        plot = ax.imshow(overhead_error_map, clim=[0, 1.5], cmap='Reds')
        notebook_utils.remove_ticks(ax)
        pt.colorbar(plot, ax=ax)
        ax.set_title('Virtual View\nL1 Error Map', {'fontsize': 14})

        pt.show()


class ScannetMeshGenerator(object):
    def __init__(self):
        self.depth_checkpoint = path.join(config.default_out_root, 'v9/v9-multi_layer_depth_aligned_background_multi_branch/0/01149000_005_0003355.pth')
        self.seg_checkpoint = path.join(config.default_out_root, 'v9/v9-category_nyu40_merged_background-2l/0/01130000_005_0001780.pth')

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.depth_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(self.depth_checkpoint)
        print(metadata)
        self.seg_model, metadata = train_eval_pipeline.load_checkpoint_as_frozen_model(self.seg_checkpoint)
        print(metadata)

        self.input_image_meansub_cuda = None
        self.input_image_name = None
        self.example = None

        self.visualize = False
        self.camera_height = None
        self.cam = None
        self.gravity_angle = None

    def minsub_rgb(self, img):
        rgb_mean = np.array([178.1781, 158.5039, 142.5141], dtype=np.float32)  # same value as training set
        # rgb_mean = np.array([150, 150, 150], dtype=np.float32)  # doesn't seem to matter
        # rgb_mean = self.input_image.mean(0, keepdims=True).mean(1, keepdims=True).squeeze()
        rgb_scale = 1 / 255.0
        in_rgb = img.astype(np.float32)
        in_rgb -= rgb_mean.reshape(3, 1, 1)
        in_rgb *= rgb_scale
        return in_rgb

    def show_input_image(self, scale=1.0):
        print('example_name: {}'.format(self.input_image_name))
        notebook_utils.quick_show_rgb(cv2.resize(self.input_image, (int(self.input_image.shape[1] * scale), int(self.input_image.shape[0] * scale))))

    def load_scannet(self, name, image_wh=(240, 320)):
        self.filename_meshes = path.join(config.scannet_frustum_clipped_root, '{}/meshes.obj'.format(name))
        self.filename_cam_info = path.join(config.scannet_frustum_clipped_root, '{}/cam_info.pkl'.format(name))
        self.filename_img = path.join(config.scannet_frustum_clipped_root, '{}/img.jpg'.format(name))
        self.filename_proposals = path.join(config.scannet_frustum_clipped_root, '{}/proposals.mat'.format(name))

        img = io_utils.read_jpg(self.filename_img)
        self.input_image_original = img
        # remove undistortion artifacts
        img = cv2.copyMakeBorder(img[5:-5, 6:-6], top=5, bottom=5, left=6, right=6, borderType=cv2.BORDER_REFLECT101)
        if image_wh:
            img = scipy.misc.imresize(img, image_wh)

        self.input_image = img
        self.input_image_meansub_cuda = torch.Tensor(self.minsub_rgb(img.transpose(2, 0, 1))[None]).cuda()
        self.input_image_name = name

        with open(self.filename_cam_info, 'rb') as f:
            c = pickle.load(f)

        self.camera_height = c['height']
        self.cam = camera.OrthographicCamera.from_Rt(Rt=la.inv(c['pose'])[:3], is_world_to_cam=True)
        self.cam.viewdir *= -1
        self.cam.up_vector *= -1
        self.gravity_angle = float(math.acos(np.inner(self.cam.viewdir, np.array([0, 0, -1], dtype=np.float64))))

        print(c.keys())

    def generate_mesh(self, force=True):
        """
        TODO: this function is mostly used to generate visualization. name is misleading
        :return:
        """
        depth_pred = self.depth_model(self.input_image_meansub_cuda)
        seg_pred = self.seg_model(self.input_image_meansub_cuda)
        name = self.input_image_name

        assert seg_pred.shape[0] == 1
        pred_l1 = seg_pred[:, :40]
        pred_l2 = seg_pred[:, 40:]
        argmax_l1 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(pred_l1)
        argmax_l2 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(pred_l2)

        depth_pred_np = torch_utils.recursive_torch_to_numpy(depth_pred)

        segmented_depth = train_eval_pipeline.segment_predicted_depth(depth_pred_np, argmax_l1, argmax_l2)
        segmented_depth = np.squeeze(segmented_depth)
        assert segmented_depth.shape[0] == 5, segmented_depth.shape  # making sure this is v9

        if self.visualize:
            ### PREDICTED DEPTH
            fig = pt.figure(figsize=(20, 8))
            fig.suptitle('Multi-layer Surface Prediction', fontsize=19, y=0.75, fontweight=500)

            ax = fig.add_subplot(151)
            ax.imshow(segmented_depth[0])
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$D_1$', {'fontsize': 14})

            ax = fig.add_subplot(152)
            ax.imshow(segmented_depth[1])
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$D_2$', {'fontsize': 14})

            ax = fig.add_subplot(153)
            ax.imshow(segmented_depth[2])
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$D_3$', {'fontsize': 14})

            ax = fig.add_subplot(154)
            ax.imshow(segmented_depth[3])
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$D_4$', {'fontsize': 14})

            ax = fig.add_subplot(155)
            ax.imshow(segmented_depth[4])
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$D_5$', {'fontsize': 14})

            ### PREDICTED SEGMENTATION
            seg_img = evaluation.colorize_segmentation(argmax_l1[0])
            fig = pt.figure(figsize=(40, 3.33))
            ax = fig.add_subplot(141)
            ax.imshow(seg_img)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$D_1$ Segmentation  ($M_1$)', {'fontsize': 14})

            seg_img2 = evaluation.colorize_segmentation(argmax_l2[0])
            ax = fig.add_subplot(142)
            ax.imshow(seg_img2)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$D_3$ Segmentation  ($M_3$)', {'fontsize': 14})

        with scannet_utils.temporary_camera_file_context(self.filename_cam_info) as cam_filename:
            example = {
                'name': name,
                'camera_filename': cam_filename,
            }
            ret = train_eval_pipeline.save_mldepth_as_meshes_v9_scannet(segmented_depth, example, force=force)

        return ret

    def set_symlinks(self):
        name = self.input_image_name
        for old_item in glob.glob('/home/daeyun/mnt/v9_visualization_scannet/*'):
            if path.exists(old_item):
                os.remove(old_item)
                print('rm {}'.format(old_item))

        mesh_filename = path.join(config.scannet_frustum_clipped_root, name, 'meshes.obj')

        new_gt_filename = path.join('/home/daeyun/mnt/v9_visualization_scannet', 'gt.obj')
        shutil.copy(mesh_filename, new_gt_filename)
        print(new_gt_filename)

        files = glob.glob('/data4/out/scene3d/v9_scannet_pred_depth_mesh/{}/*'.format(name))

        for fname in files:
            new_fname = fname.replace('/data4/out/scene3d/v9_scannet_pred_depth_mesh/{}'.format(name), '/home/daeyun/mnt/v9_visualization_scannet')
            io_utils.ensure_dir_exists(path.dirname(new_fname))
            shutil.copy(fname, new_fname)

        files = glob.glob('/data4/out/scene3d/factored3d_pred/scannet/{}/*'.format(name))
        print('/data4/out/scene3d/factored3d_pred')
        print(files)

        for fname in files:
            new_fname = fname.replace('/data4/out/scene3d/factored3d_pred/scannet/{}'.format(name), '/home/daeyun/mnt/v9_visualization_scannet')
            io_utils.ensure_dir_exists(path.dirname(new_fname))
            shutil.copy(fname, new_fname)
