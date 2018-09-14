import torch
import torch._C
import numpy as np


def recursive_module_apply(models, func):
    if isinstance(models, (list, tuple)):
        return [recursive_module_apply(item, func=func) for item in models]
    elif isinstance(models, dict):
        return {
            key: recursive_module_apply(value, func=func) for key, value in models.items()
        }
    elif isinstance(models, torch.nn.Module):
        return func(models)
    else:
        # raise RuntimeError('unknown model type {}'.format(models))
        return models


def to_python_scalar(gpu_var):
    d = gpu_var.cpu().data.numpy()
    return d.item()


def recursive_numpy_to_torch(data, cuda: bool = False):
    if isinstance(data, (list, tuple)):
        return [recursive_numpy_to_torch(item, cuda=cuda) for item in data]
    elif isinstance(data, dict):
        return {k: recursive_numpy_to_torch(v, cuda=cuda) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        if cuda:
            ret = data.cuda()
        else:
            ret = data.cpu()
        return ret
    elif isinstance(data, np.ndarray):
        return recursive_numpy_to_torch(torch.from_numpy(data), cuda=cuda)
    else:
        raise RuntimeError('unknown data type {}'.format(data))


def recursive_torch_to_numpy(data):
    if isinstance(data, (list, tuple)):
        return [recursive_torch_to_numpy(item) for item in data]
    elif isinstance(data, dict):
        return {k: recursive_torch_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, torch.autograd.Variable):
        raise NotImplementedError()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise RuntimeError('unknown data type {}'.format(data))


def load_torch_model(filename, use_cpu=True):
    """
    Returns a dict of pytorch modules.
    :param filename: e.g. /data/mvshape/out/pytorch/mv6_vpo/models5_0050.pth
    :param use_cpu: If True, the model is loaded in cpu mode.
    :return: A dict of pytorch models that make up a neural net.  TODO(daeyun)
    """
    with open(filename, 'rb') as f:
        model = torch.load(f, map_location=lambda storage, loc: storage)

    if use_cpu:
        def set_device(module):
            return module.cpu()
    else:
        def set_device(module):
            return module.cuda()
    model = recursive_module_apply(model, func=set_device)

    return model
