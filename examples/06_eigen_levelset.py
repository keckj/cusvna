"""Level sets written with Eigen matrices, ray traced by interval marching with OptiX.

Same renderer as 05_levelset.py, the shapes of cuda/eigen_levelset.cu are written with Eigen vectors and
matrices (rotations, quaternions, Householder reflections, determinants, quadratic forms) of affine forms
to march the rays and of autodiff numbers to compute the normals. Eigen is compiled by NVRTC with the
eigen-nvrtc submodule.

Only AF1 forms can be Eigen scalars: the type of an AA form records its noise symbols and changes with each
nonlinear operation, while the coefficients of an Eigen matrix share the same type."""
import os, sys, enum, importlib

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
levelset = importlib.import_module('05_levelset')

eigen_include_path = os.path.realpath(os.path.join(script_dir, '..', 'ext', 'eigen-nvrtc'))

class Shape(enum.IntEnum):
    """Must match the Shape enum of cuda/eigen_levelset.cu."""
    FLOOR         = 0
    MENGER        = 1
    SPECTRAHEDRON = 2
    TREFOIL       = 3

def make_scene():
    """A Menger sponge between a spectrahedron and a trefoil knot, standing above a checkered floor."""
    floor_y = levelset.floor_y
    objects = np.zeros(4, dtype=levelset.object_dtype)
    objects[0] = ((0.0, floor_y, 0.0), 0.0, 0.0, Shape.FLOOR, 0.0, 0.0)
    objects[1] = ((0.0, floor_y + 0.97, 0.0), 1.4, 0.2, Shape.MENGER, 0.0, 0.0)
    objects[2] = ((-3.2, floor_y + 0.67, 0.6), 0.95, -0.4, Shape.SPECTRAHEDRON, 0.0, 0.0)
    objects[3] = ((3.2, floor_y + 0.9, 0.6), 0.95, 0.3, Shape.TREFOIL, 0.0, 0.0)
    return objects

if __name__ == '__main__':
    if not os.path.isdir(os.path.join(eigen_include_path, 'Eigen')):
        msg = "Cannot find eigen-nvrtc submodule, did you clone the repository recursively ?"
        levelset.cusvna_logger.fatal(msg)
        raise RuntimeError(msg)

    levelset.main(description="Interval marching of level sets written with Eigen matrices, with OptiX.",
                  cuda_source='eigen_levelset.cu', scene=make_scene, extra_include_paths=(eigen_include_path,),
                  camera=dict(eye=(0, 3.2, -8.5), look_at=(0, -0.7, 0.3)), window_name="Cusvna Eigen Level Sets",
                  forms=('af1',))
