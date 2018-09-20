import ctypes
import typing
from os import path
from sys import platform
from ctypes import cdll

import numpy as np
from numpy.ctypeslib import ndpointer

from scene3d import log
from scene3d import config

ctypes_lib_dirname = path.realpath(path.join(path.dirname(__file__), '../../cpp/cmake-build-release/ctypes'))

lib = None
if platform == "linux" or platform == "linux2":
    lib_filename = path.join(ctypes_lib_dirname, 'libtraining_utils.so')
    assert path.isfile(lib_filename), 'file does not exist: {}'.format(lib_filename)
    lib = cdll.LoadLibrary(lib_filename)
else:
    raise NotImplemented(platform)

if lib:
    c_func = getattr(lib, 'model_index_to_category')
    c_func.restype = None
    c_func.argtypes = [
        ctypes.c_char_p,  # std::string
        ndpointer(ctypes.c_uint16, flags="C_CONTIGUOUS"),
        ctypes.c_uint32,
    ]

    c_func2 = getattr(lib, 'initialize_category_mapping')
    c_func2.restype = None
    c_func2.argtypes = [
        ctypes.c_char_p,  # std::string
    ]
    log.info('Loaded shared library %s', lib._name)

_is_initialized = False


def model_index_to_category(model_indices: np.ndarray, mapping_name):
    assert mapping_name in [
        "nyuv2_40class_merged_background",
        "nyuv2_40class",
    ]

    c_func_name = 'model_index_to_category'

    assert model_indices.dtype == np.uint16
    assert model_indices.flags['C_CONTIGUOUS']

    c_func = getattr(lib, c_func_name)

    # Releases GIL.
    c_func(
        ctypes.c_char_p(mapping_name.encode()),
        model_indices,
        model_indices.size)


def initialize_category_mapping():
    global _is_initialized
    if _is_initialized:
        return

    c_func_name = 'initialize_category_mapping'

    c_func = getattr(lib, c_func_name)

    csv_filename = config.category_mapping_csv_filename
    assert path.isfile(csv_filename)

    # Releases GIL.
    c_func(ctypes.c_char_p(csv_filename.encode()))

    _is_initialized = True
