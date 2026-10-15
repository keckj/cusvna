"""
cusvna, CUDA Self-Validated Numerical Analysis: header-only C++17 library for rigorous affine and
interval arithmetic in CUDA kernels. The Python package only ships the headers, add get_include()
to the include paths of nvcc or NVRTC and include <cusvna/aa.cuh>, <cusvna/af1.cuh> or <cusvna/interval.cuh>.
"""

import os

__version__ = "1.0.3"

_package_dir = os.path.dirname(os.path.abspath(__file__))


def get_include():
    """Include directory containing the cusvna/ headers, to pass as -I to nvcc or NVRTC."""
    include_dir = os.path.join(_package_dir, "include")
    if os.path.isdir(os.path.join(include_dir, "cusvna")):
        return include_dir
    # source checkout or editable install: the headers are in the cusvna/ directory of the repository
    return os.path.dirname(os.path.dirname(_package_dir))
