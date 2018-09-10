import ctypes
from os import path
from sys import platform
from ctypes import cdll

import numpy as np
from numpy.ctypeslib import ndpointer

from scene3d import log

ctypes_lib_dirname = path.realpath(path.join(path.dirname(__file__), '../../cpp/cmake-build-release/ctypes'))

lib = None
if platform == "linux" or platform == "linux2":
    lib_filename = path.join(ctypes_lib_dirname, 'libepipolar_transform_ctypes.so')
    assert path.isfile(lib_filename), 'file does not exist: {}'.format(lib_filename)
    lib = cdll.LoadLibrary(lib_filename)
else:
    raise NotImplemented(platform)

if lib:
    c_func = getattr(lib, 'epipolar_feature_transform')
    c_func.restype = None
    c_func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ]
    log.info('Loaded shared library %s', lib._name)


def feature_transform(
        feature_map_data: np.ndarray,
        front_depth_data: np.ndarray,
        back_depth_data: np.ndarray,
        camera_filename: str,
        target_height,
        target_width):
    c_func_name = 'epipolar_feature_transform'

    assert feature_map_data.ndim == 3
    assert front_depth_data.ndim == 2
    assert back_depth_data.ndim == 2
    assert front_depth_data.shape == back_depth_data.shape
    assert feature_map_data.shape[:2] == back_depth_data.shape
    assert feature_map_data.dtype == np.float32
    assert back_depth_data.dtype == np.float32
    assert front_depth_data.dtype == np.float32

    source_channels = feature_map_data.shape[-1]
    source_height = feature_map_data.shape[0]
    source_width = feature_map_data.shape[1]

    out_data = np.empty((target_height, target_width, source_channels), dtype=np.float32)
    assert out_data.flags['C_CONTIGUOUS']

    assert feature_map_data.flags['C_CONTIGUOUS']
    assert front_depth_data.flags['C_CONTIGUOUS']
    assert back_depth_data.flags['C_CONTIGUOUS']

    c_func = getattr(lib, c_func_name)

    # Releases GIL.
    c_func(
        feature_map_data,
        front_depth_data,
        back_depth_data,
        source_channels,
        source_height,
        source_width,
        ctypes.c_char_p(camera_filename.encode()),
        target_height,
        target_width,
        out_data)

    return out_data
