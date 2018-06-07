import numpy as np


def grid_indexing_2d(arr_3d: np.ndarray, indices: np.ndarray):
    """
    2d grid indexing of 3d array, along the first dimension.
    :param arr_3d: 3D array of shape (C, H, W)
    :param indices: 2D array of shape (H, W) containing integer values from 0 to C-1. Negative indexing won't work.
    :return: 2D array of shape (H, W), containing values selected from `arr_3d`.
    """
    assert arr_3d.ndim == 3
    assert indices.ndim == 2
    assert arr_3d.shape[1:] == indices.shape
    sz = np.prod(indices.shape).item()  # H*W
    ind_2d = np.arange(sz, dtype=np.int).reshape(indices.shape)
    return arr_3d.ravel()[ind_2d + indices * sz].copy()
