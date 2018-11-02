import ctypes
import typing
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
        ctypes.c_uint32,  # H
        ctypes.c_uint32,  # W
        ctypes.c_uint32,  # C
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ]

    c_func2 = getattr(lib, 'epipolar_feature_transform_parallel')
    c_func2.restype = None
    c_func2.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ctypes.c_uint32,  # B
        ctypes.c_uint32,  # H
        ctypes.c_uint32,  # W
        ctypes.c_uint32,  # C
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ]

    c_func3 = getattr(lib, 'render_depth_from_another_view')
    c_func3.restype = None
    c_func3.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # (N, H, W)
        ctypes.c_uint32,  # H
        ctypes.c_uint32,  # W
        ctypes.c_uint32,  # N
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ]
    log.info('Loaded shared library %s', lib._name)

    c_func4 = getattr(lib, 'frustum_visibility_map_from_overhead_view')
    c_func4.restype = None
    c_func4.argtypes = [
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

    source_height = feature_map_data.shape[0]
    source_width = feature_map_data.shape[1]
    source_channels = feature_map_data.shape[2]

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
        source_height,
        source_width,
        source_channels,
        ctypes.c_char_p(camera_filename.encode()),
        target_height,
        target_width,
        out_data)

    return out_data


def feature_transform_parallel(
        feature_map_data: np.ndarray,
        front_depth_data: np.ndarray,
        back_depth_data: np.ndarray,
        camera_filenames: typing.Sequence[str],
        target_height: int,
        target_width: int):
    c_func_name = 'epipolar_feature_transform_parallel'

    assert feature_map_data.ndim == 3 + 1  # (B, H, W, C)
    assert front_depth_data.ndim == 2 + 1  # (B, H, W)
    assert back_depth_data.ndim == 2 + 1
    assert front_depth_data.shape == back_depth_data.shape
    assert feature_map_data.shape[:3] == back_depth_data.shape
    assert feature_map_data.dtype == np.float32
    assert back_depth_data.dtype == np.float32
    assert front_depth_data.dtype == np.float32
    assert feature_map_data.shape[0] == len(camera_filenames)
    assert len(camera_filenames) > 0
    assert isinstance(camera_filenames[0], str)

    batch_size = feature_map_data.shape[0]
    source_height = feature_map_data.shape[1]
    source_width = feature_map_data.shape[2]
    source_channels = feature_map_data.shape[3]

    out_data = np.empty((batch_size, target_height, target_width, source_channels), dtype=np.float32)
    assert out_data.flags['C_CONTIGUOUS']

    assert feature_map_data.flags['C_CONTIGUOUS']
    assert front_depth_data.flags['C_CONTIGUOUS']
    assert back_depth_data.flags['C_CONTIGUOUS']

    filenames_p = (ctypes.c_char_p * len(camera_filenames))()
    filenames_p[:] = [item.encode() for item in camera_filenames]

    c_func = getattr(lib, c_func_name)

    # Releases GIL.
    c_func(
        feature_map_data,
        front_depth_data,
        back_depth_data,
        batch_size,
        source_height,
        source_width,
        source_channels,
        filenames_p,
        target_height,
        target_width,
        out_data)

    return out_data


def render_depth_from_another_view(
        depth: np.ndarray,
        camera_filename: str,
        target_height: int,
        target_width: int,
        depth_disc_pixels=5.0):
    c_func_name = 'render_depth_from_another_view'

    assert depth.ndim == 3 or depth.ndim == 2
    assert depth.dtype == np.float32

    if depth.ndim == 3:
        source_num_images = depth.shape[0]
        source_height = depth.shape[1]
        source_width = depth.shape[2]
    else:
        source_num_images = 1
        source_height = depth.shape[0]
        source_width = depth.shape[1]

    out_data = np.empty((source_num_images, target_height, target_width), dtype=np.float32)
    assert out_data.flags['C_CONTIGUOUS']

    assert depth.flags['C_CONTIGUOUS']

    c_func = getattr(lib, c_func_name)

    # Releases GIL.
    c_func(
        depth,
        source_height,
        source_width,
        source_num_images,
        ctypes.c_char_p(camera_filename.encode()),
        target_height,
        target_width,
        depth_disc_pixels,
        out_data
    )

    if depth.ndim == 2:
        out_data = np.squeeze(out_data)

    return out_data


def frustum_visibility_map_from_overhead_view(
        camera_filename: str,
        target_height: int,
        target_width: int):
    c_func_name = 'frustum_visibility_map_from_overhead_view'

    out_data = np.empty((target_height, target_width), dtype=np.float32)
    assert out_data.flags['C_CONTIGUOUS']

    c_func = getattr(lib, c_func_name)

    c_func(
        ctypes.c_char_p(camera_filename.encode()),
        target_height,
        target_width,
        out_data
    )

    return out_data
