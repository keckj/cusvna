import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.patches

from utils import *

T = np.float32
I = np.uint64

kernel_src = f"""
#include <cusvna/aa.cuh>

using T = {dtype_to_ctype[T]};
using I = {dtype_to_ctype[I]};
using G = cusvna::LocalThreadIndex<I>;
constexpr I N = 21;
using AA = cusvna::AA<T,I,G,N>;
using Interval = cusvna::Interval<T>;

#include "examples/octtree.cuh"
"""

if __name__ == "__main__":
    fname = r'f(x,y,z) = (x-2)^2(x+2)^2+(y-2)^2(y+2)^2+(z-2)^2(z+2)^2+(x^2y^2+x^2z^2+y^2z^2)+6*x*y*z-10(x^2+y^2+z^2)+22'
    logger.info(f"Generating octtree for function {fname}...")

    do_plot = False
    nlevels = 10

    start_level = 4
    end_level = start_level + nlevels
    xmin, xmax = -4, 4
    ymin, ymax = -4, 4
    zmin, zmax = -4, 4
    def f(x,y):
        return (
               (x-2)**2 * (x+2)**2
             + (y-2)**2 * (y+2)**2
             + (z-2)**2 * (z+2)**2
             + (x**2 * y**2 + x**2 * z**2 + y**2 * z**2)
             + 6*x*y*z
             - 10*(x**2+y**2+z**2)
             + 22
        )

    logger.info(f"Computing octtree with start_level={start_level} and end_level={end_level-1}...")

    N = 2**start_level
    active_cell_indices = cp.empty(100000, dtype=I)  # index buffer, should be large enough to handle end_level active cell count
    active_cell_counter = cp.empty((), dtype=I)

    threads_per_block = 512
    warpsize = cp.cuda.Device(0).attributes['WarpSize']
    warps_per_block = threads_per_block // warpsize;
    kname = f'kernel<{warps_per_block},{warpsize}>'
    kernel, = build_module_and_extract_kernels_from_source(logger, kernel_src, (kname,))
    assert threads_per_block <= kernel.max_threads_per_block, (threads_per_block, kernel.max_threads_per_block)

    start = cp.cuda.Event(disable_timing=False)
    end = cp.cuda.Event(disable_timing=False)

    nactive = (N//2)**3  # correponds to active cells of previous level = start_level - 1.
    for current_level in range(start_level, end_level+1):
        B = threads_per_block
        G = (8*nactive+B-1)//B

        active_cell_counter[...] = 0   # reset cell counter for next level

        start.record()
        kernel((G,), (B,),
                (T(xmin), T(xmax), T(ymin), T(ymax), T(zmin), T(zmax),
                   I(start_level), I(end_level),
                   I(current_level), I(nactive),
                   active_cell_indices, active_cell_counter))
        end.record()
        end.synchronize()
        time_ms = cp.cuda.get_elapsed_time(start, end);

        nactive = int(active_cell_counter.get())
        Ntot = 2**current_level
        logger.info(f"  {time_ms:6.1f}ms -- level {current_level:2d} -- grid {Ntot:5d}^3 = 2^{3*current_level:2d}"
                    f" -- active cells {nactive}/{Ntot**3} ({100.0*nactive/(Ntot**3):2.2f}%)")

        if nactive == 0:
            break

    logger.info("All done.")
