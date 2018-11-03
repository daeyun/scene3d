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
    lib_filename = path.join(ctypes_lib_dirname, 'libdepth_mesh_ctypes.so')
    assert path.isfile(lib_filename), 'file does not exist: {}'.format(lib_filename)
    lib = cdll.LoadLibrary(lib_filename)
else:
    raise NotImplemented(platform)

if lib:
    c_func = getattr(lib, 'depth_to_mesh')
    c_func.restype = None
    c_func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # depth
        ctypes.c_uint32,  # H
        ctypes.c_uint32,  # W
        ctypes.c_char_p,  # camera filename
        ctypes.c_uint32,  # camera index
        ctypes.c_float,  # depth discontinuity factor
        ctypes.c_char_p,  # output .ply filename.
    ]

    c_func2 = getattr(lib, 'mesh_precision_recall')
    c_func2.restype = None
    c_func2.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),  # GT mesh filenames.
        ctypes.c_uint32,  # Number of GT mesh filenames.
        ctypes.POINTER(ctypes.c_char_p),  # Pred mesh filenames.
        ctypes.c_uint32,  # Number of Pred mesh filenames.
        ctypes.c_float,  # Sampling density.
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # Thresholds.
        ctypes.c_uint32,  # Number of threshold values
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # Out precision values.
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # Out recall values.
    ]

    log.info('Loaded shared library %s', lib._name)


def depth_to_mesh(
        depth: np.ndarray,
        camera_filename: str,
        camera_index: int,
        dd_factor: float,
        out_ply_filename: str):
    c_func_name = 'depth_to_mesh'

    assert depth.ndim == 2
    assert path.isfile(camera_filename)
    assert camera_index >= 0
    assert dd_factor > 0.0
    assert out_ply_filename.endswith('.ply')
    assert depth.flags['C_CONTIGUOUS']

    c_func = getattr(lib, c_func_name)

    # Releases GIL.
    c_func(
        depth,
        depth.shape[0],
        depth.shape[1],
        ctypes.c_char_p(camera_filename.encode()),
        camera_index,
        dd_factor,
        ctypes.c_char_p(out_ply_filename.encode()),
    )


def mesh_precision_recall(gt_mesh_filenames,
                          pred_mesh_filenames,
                          sampling_density,
                          thresholds):
    c_func_name = 'mesh_precision_recall'

    def check_filenames(filenames) -> typing.Sequence[str]:
        if isinstance(filenames, str):
            filenames = [filenames]
        assert isinstance(filenames, (list, tuple)), filenames
        assert len(filenames) > 0
        assert isinstance(filenames[0], str), filenames
        for fname in filenames:
            assert path.isfile(fname), fname
        return filenames

    gt_mesh_filenames = check_filenames(gt_mesh_filenames)
    pred_mesh_filenames = check_filenames(pred_mesh_filenames)

    assert sampling_density > 0.0
    assert len(thresholds) > 0
    assert thresholds[0] > 0.0

    gt_filenames_p = (ctypes.c_char_p * len(gt_mesh_filenames))()
    gt_filenames_p[:] = [item.encode() for item in gt_mesh_filenames]

    pred_filenames_p = (ctypes.c_char_p * len(pred_mesh_filenames))()
    pred_filenames_p[:] = [item.encode() for item in pred_mesh_filenames]

    in_thresholds_buffer = np.array(thresholds, dtype=np.float32)
    assert in_thresholds_buffer.flags['C_CONTIGUOUS']

    out_precisions = np.full(len(thresholds), fill_value=np.nan, dtype=np.float32)
    assert out_precisions.flags['C_CONTIGUOUS']

    out_recalls = np.full(len(thresholds), fill_value=np.nan, dtype=np.float32)
    assert out_recalls.flags['C_CONTIGUOUS']

    c_func = getattr(lib, c_func_name)

    # Releases GIL.
    c_func(
        gt_filenames_p,
        len(gt_filenames_p),
        pred_filenames_p,
        len(pred_filenames_p),
        sampling_density,
        in_thresholds_buffer,
        len(thresholds),
        out_precisions,
        out_recalls,
    )

    assert np.isfinite(out_recalls).all()
    assert np.isfinite(out_precisions).all()

    return out_precisions.tolist(), out_recalls.tolist()
