import numpy as np
import cupy as cp
import open3d as o3d
from skimage import measure

from utils import *

T = np.float64
I = np.uint64

kernel_src = f"""
#include <cusvna/aa.cuh>

using T = {dtype_to_ctype[T]};
using I = {dtype_to_ctype[I]};
using G = cusvna::LocalThreadIndex<I>;
constexpr I N = 21;
using AA = cusvna::AA<T,I,G,N>;
using Interval = cusvna::Interval<T>;

#include "examples/cuda/octtree.cuh"
"""

if __name__ == "__main__":
    fname = r'f(x,y,z) = (x-2)^2(x+2)^2+(y-2)^2(y+2)^2+(z-2)^2(z+2)^2+(x^2y^2+x^2z^2+y^2z^2)+6*x*y*z-10(x^2+y^2+z^2)+22'
    logger.info(f"Generating octtree for function {fname}...")

    do_plot = False
    nlevels = 3 if do_plot else 10

    start_level = 4
    end_level = start_level + nlevels

    xmin, xmax = -5, 5
    ymin, ymax = -5, 5
    zmin, zmax = -5, 5
    
    if do_plot:
        Nm = 128
        logger.info(f"Running marching cubes on a {Nm}^3 grid...")
        def f(x,y,z):
            return (
                   (x-2)**2 * (x+2)**2
                 + (y-2)**2 * (y+2)**2
                 + (z-2)**2 * (z+2)**2
                 + (x**2 * y**2 + x**2 * z**2 + y**2 * z**2)
                 + 6*x*y*z
                 - 10*(x**2+y**2+z**2)
                 + 22
            )
        x = cp.linspace(xmin, xmax, Nm)
        y = cp.linspace(ymin, ymax, Nm)
        z = cp.linspace(zmin, zmax, Nm)
        x, y, z = np.ix_(x, y, z)
        V, F, _, _ = measure.marching_cubes(f(x,y,z).get(), 0)
        V[:,0] = xmin + (xmax-xmin)/Nm*V[:,0]
        V[:,1] = ymin + (ymax-ymin)/Nm*V[:,1]
        V[:,2] = zmin + (zmax-zmin)/Nm*V[:,2]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.triangles = o3d.utility.Vector3iVector(F)

    logger.info(f"Computing octtree with start_level={start_level} and end_level={end_level-1}...")

    N = 2**start_level
    assert(2**(3*end_level) < np.iinfo(I).max)

    nallocated = 10_000_000
    active_cell_indices_in  = cp.empty(nallocated, dtype=I)  # index buffer, should be large enough to handle end_level active cells count
    active_cell_indices_out = cp.empty(nallocated, dtype=I)
    active_cell_counter = cp.empty((), dtype=I)

    threads_per_block = 512
    warpsize = int(cp.cuda.Device(0).attributes['WarpSize'])
    warps_per_block = threads_per_block // warpsize;
    threads_per_block = warps_per_block * warpsize
    kname = f'kernel<{warps_per_block},{warpsize}>'
    kernel, = build_module_and_extract_kernels_from_source(logger, kernel_src, (kname,))
    assert warps_per_block <= warpsize, (warps_per_block, warpsize)
    assert threads_per_block <= kernel.max_threads_per_block, (threads_per_block, kernel.max_threads_per_block)

    start = cp.cuda.Event(disable_timing=False)
    end = cp.cuda.Event(disable_timing=False)

    nactive = (N//2)**3  # correponds to active cells of previous level = start_level - 1.
    for current_level in range(start_level, end_level+1):
        B = threads_per_block
        G = (nactive+B-1)//B

        active_cell_counter[...] = 0   # reset cell counter for next level

        start.record()
        kernel((G,), (B,),
                (T(xmin), T(xmax), T(ymin), T(ymax), T(zmin), T(zmax),
                   I(start_level), I(current_level), 
                   I(nactive), I(nallocated), 
                   active_cell_indices_in, active_cell_indices_out, active_cell_counter))
        end.record()
        end.synchronize()
        time_ms = cp.cuda.get_elapsed_time(start, end);

        nactive = int(active_cell_counter.get())
        Ntot = 2**current_level
        logger.info(f"  {time_ms:8.1f}ms -- level {current_level:2d} -- grid {Ntot:5d}^3 = 2^{3*current_level:2d}"
                    f" -- active cells {nactive}/{Ntot**3} ({100.0*nactive/(Ntot**3):2.2f}%)")

        if nactive == 0:
            break
        
        if nactive > nallocated:
            logger.warning(f"You need to allocate more indices to compute level {current_level}: nactive={nactive:2.2e} > nallocated={nallocated:2.2e}.")
            break

        if do_plot:
            indices = active_cell_indices_out[:nactive].get()

            dz = (zmax - zmin) / Ntot
            dy = (ymax - ymin) / Ntot
            dx = (xmax - xmin) / Ntot

            i = indices // (Ntot*Ntot)
            j = (indices // Ntot) % Ntot
            k = indices % Ntot

            zi = zmin + (i+0.5)*dz
            yi = ymin + (j+0.5)*dy
            xi = xmin + (k+0.5)*dx

            pts = o3d.geometry.PointCloud()
            pts.points = o3d.utility.Vector3dVector(np.column_stack([xi, yi, zi]))
            pts.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(nactive, 3)))

            assert dx==dy==dz, (dx,dy,dz)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pts, dx)

            o3d.visualization.draw_geometries([voxel_grid, mesh])

        active_cell_indices_in, active_cell_indices_out = active_cell_indices_out, active_cell_indices_in

    logger.info("All done.")
