from pprint import pprint
from scene3d import geom2d
from os import path
import numpy as np
import scipy as sp
import scipy.misc
from scene3d import pbrs_utils
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
        self.example = None

    def minsub_rgb(self, img):
        rgb_mean = np.array([178.1781, 158.5039, 142.5141], dtype=np.float32) * 1
        # rgb_mean = self.input_image.mean(0, keepdims=True).mean(1, keepdims=True).squeeze()
        rgb_scale = 1 / 255.0
        in_rgb = img.astype(np.float32)
        in_rgb -= rgb_mean.reshape(3, 1, 1)
        in_rgb *= rgb_scale
        return in_rgb

    def show_input_image(self):
        print('example_name: {}'.format(self.input_image_name))
        notebook_utils.quick_show_rgb(self.input_image)

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

        seg_pred_argmax = train_eval_pipeline.semantic_segmentation_from_raw_prediction(seg_pred[:, :40])

        segmented_depth = loss_fn.undo_log_depth(train_eval_pipeline.segment_predicted_depth(torch_utils.recursive_torch_to_numpy(depth_pred), seg_pred_argmax))
        assert segmented_depth.shape[0] == 1
        segmented_depth = np.squeeze(segmented_depth)

        segmented_depth_single = segmented_depth.copy()

        nan_mask = np.isnan(segmented_depth_single[0])
        segmented_depth_single[3][~nan_mask] = np.nan

        out_filename = '/mnt/ramdisk/vis_mesh/single0.ply'
        depth_mesh_utils_cpp.depth_to_mesh(segmented_depth_single[0], self.example['camera_filename'], camera_index=0, dd_factor=10, out_ply_filename=out_filename)
        out_filename = '/mnt/ramdisk/vis_mesh/single1.ply'
        depth_mesh_utils_cpp.depth_to_mesh(segmented_depth_single[3], self.example['camera_filename'], camera_index=0, dd_factor=10, out_ply_filename=out_filename)

        # Factored3d
        # f3d_utils.align_factored3d_mesh_with_meshlab_cam_coords('/data3/out/scene3d/factored3d_pred/{}/codes.obj'.format(name), '/mnt/ramdisk/nyu_mld/f3d_objets.stl')
        # f3d_utils.align_factored3d_mesh_with_meshlab_cam_coords('/data3/out/scene3d/factored3d_pred/{}/layout.obj'.format(name), '/mnt/ramdisk/nyu_mld/f3d_layout.stl')

        ### PREDICTED DEPTH
        fig = pt.figure(figsize=(20, 8))
        fig.suptitle('Multi-layer Surface Prediction', fontsize=19, y=0.75, fontweight=500)

        ax = fig.add_subplot(141)
        ax.imshow(segmented_depth[0])
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_title('$D_1$', {'fontsize': 14})

        ax = fig.add_subplot(142)
        ax.imshow(segmented_depth[1])
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_title('$D_2$', {'fontsize': 14})

        ax = fig.add_subplot(143)
        ax.imshow(segmented_depth[2])
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_title('$D_3$', {'fontsize': 14})

        ax = fig.add_subplot(144)
        ax.imshow(segmented_depth[3])
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_title('$D_4$', {'fontsize': 14})

        seg_img = evaluation.colorize_segmentation(seg_pred_argmax[0])

        ### PREDICTED SEGMENTATION
        fig = pt.figure(figsize=(40, 3.33))
        ax = fig.add_subplot(141)
        ax.imshow(seg_img)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_title('$D_1$ Segmentation  ($M_1$)', {'fontsize': 14})

        seg_pred_argmax2 = train_eval_pipeline.semantic_segmentation_from_raw_prediction(seg_pred[:, 40:])
        seg_pred_argmax2[seg_pred_argmax == 34] = 34
        seg_img2 = evaluation.colorize_segmentation(seg_pred_argmax2[0])
        ax = fig.add_subplot(142)
        ax.imshow(seg_img2)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_title('$D_3$ Segmentation  ($M_3$)', {'fontsize': 14})

        pt.show()
        print('')


        if visualize_gt:
            gt_depth = self.example['multi_layer_depth_aligned_background']
            ### GT DEPTH
            fig = pt.figure(figsize=(20, 8))
            fig.suptitle('Multi-layer Surface Ground Truth', fontsize=19, y=0.75, fontweight=500)

            ax = fig.add_subplot(141)
            ax.imshow(gt_depth[0])
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_1$', {'fontsize': 14})

            ax = fig.add_subplot(142)
            ax.imshow(gt_depth[1])
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_2$', {'fontsize': 14})

            ax = fig.add_subplot(143)
            ax.imshow(gt_depth[2])
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_3$', {'fontsize': 14})

            ax = fig.add_subplot(144)
            ax.imshow(gt_depth[3])
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_4$', {'fontsize': 14})


            gt_seg = self.example['category_nyu40_merged_background'].copy()
            gt_seg[gt_seg == 65535] = 34
            gt_seg = gt_seg.astype(np.uint8)
            seg_img = evaluation.colorize_segmentation(gt_seg[0])

            ### GT SEGMENTATION
            fig = pt.figure(figsize=(40, 3.33))
            ax = fig.add_subplot(141)
            ax.imshow(seg_img)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_title('$\\bar D_1$ Segmentation  ($\\bar M_1$)', {'fontsize': 14})

            seg_img2 = evaluation.colorize_segmentation(gt_seg[2])
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
            'name': 'nyu_0',
        }

        out, features = torch_utils.recursive_torch_to_numpy(self.hm_model.predict_height_map_single(input_batch))

        f, axarr = pt.subplots(1, 5, figsize=(28, 4))

        ax = axarr[0]
        ax.imshow(v8.undo_rgb_whitening(features[2:5]).transpose(1, 2, 0))
        notebook_utils.remove_ticks(ax)
        ax.set_title('Transformed Virtual View\nRGB Features', {'fontsize': 14})

        ax = axarr[1]
        plot = ax.imshow(features[0])
        pt.colorbar(plot, ax=ax)
        notebook_utils.remove_ticks(ax)
        ax.set_title('Virtual View Best Guess Depth', {'fontsize': 14})

        ax = axarr[2]
        ax.imshow(features[1], cmap='gray')
        notebook_utils.remove_ticks(ax)
        ax.set_title('Input View Frustum Mask Relative to\nProposed Virtual Camera Pose', {'fontsize': 14})

        ax = axarr[3]
        feat_montage = geom2d.montage(features[5:5 + 25], gridwidth=None, empty_value=np.nan)
        plot = ax.imshow(feat_montage)
        ax.set_xticks(np.arange(0, feat_montage.shape[1], features.shape[2]))
        ax.set_yticks(np.arange(0, feat_montage.shape[0], features.shape[1]))
        ax.grid(which='major', color='w', linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        pt.colorbar(plot, ax=ax)
        ax.set_title('Transformed Virtual View\nDepth Feature Map (first 25 channels)', {'fontsize': 14})

        ax = axarr[4]
        feat_montage = geom2d.montage(features[53:53 + 25], gridwidth=None, empty_value=np.nan)
        plot = ax.imshow(feat_montage)
        ax.set_xticks(np.arange(0, feat_montage.shape[1], features.shape[2]))
        ax.set_yticks(np.arange(0, feat_montage.shape[0], features.shape[1]))
        ax.grid(which='major', color='w', linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        pt.colorbar(plot, ax=ax)
        ax.set_title('Transformed Virtual VIew\nSegmentation Feature Map (first 25 channels)', {'fontsize': 14})

        f.suptitle('Epipolar Feature Transformers', fontsize=19, y=1.13, fontweight=500, x=0.427)

        f, axarr = pt.subplots(1, 2, figsize=(10, 4))
        ax = axarr[0]
        plot = ax.imshow(out)
        notebook_utils.remove_ticks(ax)
        pt.colorbar(plot, ax=ax)
        ax.set_title('Predicted\nVirtual View Height Map', {'fontsize': 14})

        ax = axarr[1]
        gt_height_map = self.example['multi_layer_overhead_depth'][0]
        plot = ax.imshow(gt_height_map)
        notebook_utils.remove_ticks(ax)
        pt.colorbar(plot, ax=ax)
        ax.set_title('"Ground Truth" Height Map\n(based on viewpoint selection\nheuristics on GT mesh)', {'fontsize': 14})

        pt.show()
