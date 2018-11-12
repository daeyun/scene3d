import glob
import pickle
from os import path

import numpy as np
import torch

from scene3d import data_utils
from scene3d import io_utils
from scene3d import torch_utils
from scene3d import train_eval_pipeline
from scene3d.net import unet
from scene3d.dataset import v8


def validation():
    print('evaluating on validation set')
    dataset = v8.MultiLayerDepth(
        split='/data2/scene3d/v8/validation_s168.txt',
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=('rgb', 'overhead_camera_pose_3params'))

    batch_size = 10
    num_data_workers = 4

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_data_workers, shuffle=False, drop_last=False, pin_memory=True)

    filenames = sorted(glob.glob('/data3/out/scene3d/v8/v8-overhead_camera_pose/0/*.pth'))

    output_all = {}
    for i in range(len(filenames)):
        filename = filenames[i]
        print(i, filename)

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model, _, loaded_metadata, frozen_model = train_eval_pipeline.load_checkpoint(filename, use_cpu=False)
        for item in model.parameters():
            item.requires_grad = False
        model.eval()
        experiment_name = loaded_metadata['experiment_name']

        output_list = []
        for i_iter, batch in enumerate(loader):
            loss = train_eval_pipeline.compute_loss(model, batch, experiment_name, frozen_model)

            in_rgb = batch['rgb'].cuda()
            target = batch['overhead_camera_pose_3params'].cuda()
            # (B, 48, 240, 320),  (B, 768, 15, 20)
            features, encoding = unet.get_feature_map_output_v2(frozen_model, in_rgb)
            pred = model((features, encoding))

            bsize = len(batch['name'])
            print(i_iter, bsize, loss)
            output_list.append(torch_utils.recursive_torch_to_numpy(pred))
        output_all[data_utils.extract_global_step_from_filename(filename)] = output_list
        print(output_all[data_utils.extract_global_step_from_filename(filename)])

    out_file = '/data3/out/scene3d/overhead_param3_pred/01_validation.pkl'
    io_utils.ensure_dir_exists(path.dirname(out_file))
    with open(out_file, 'wb') as f:
        pickle.dump(output_all, f)


def training():
    print('evaluating on training set')

    dataset = v8.MultiLayerDepth(
        split='/data2/scene3d/v8/train_v2.txt',
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=('rgb', 'overhead_camera_pose_3params'))

    batch_size = 10
    num_data_workers = 4

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_data_workers, shuffle=False, drop_last=False, pin_memory=True)

    filenames = sorted(glob.glob('/data3/out/scene3d/v8/v8-overhead_camera_pose/0/*.pth'))

    output_all = {}
    for i in range(len(filenames)):
        filename = filenames[i]
        print(i, filename)

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model, _, loaded_metadata, frozen_model = train_eval_pipeline.load_checkpoint(filename, use_cpu=False)
        for item in model.parameters():
            item.requires_grad = False
        model.eval()
        experiment_name = loaded_metadata['experiment_name']

        output_list = []
        for i_iter, batch in enumerate(loader):
            loss = train_eval_pipeline.compute_loss(model, batch, experiment_name, frozen_model)

            in_rgb = batch['rgb'].cuda()
            target = batch['overhead_camera_pose_3params'].cuda()
            # (B, 48, 240, 320),  (B, 768, 15, 20)
            features, encoding = unet.get_feature_map_output_v2(frozen_model, in_rgb)
            pred = model((features, encoding))

            bsize = len(batch['name'])
            print(i_iter, bsize, loss)
            output_list.append(torch_utils.recursive_torch_to_numpy(pred))

            if i_iter > 25:
                break
        output_all[data_utils.extract_global_step_from_filename(filename)] = output_list
        print(output_all[data_utils.extract_global_step_from_filename(filename)])

    out_file = '/data3/out/scene3d/overhead_param3_pred/01_train.pkl'
    io_utils.ensure_dir_exists(path.dirname(out_file))
    with open(out_file, 'wb') as f:
        pickle.dump(output_all, f)


if __name__ == '__main__':
    validation()
    training()
