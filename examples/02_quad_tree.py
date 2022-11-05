import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.patches

from utils import *

T = np.float32
I = np.uint32

kernel_src = f"""
#include <cusvna/aa.cuh>

using T = {dtype_to_ctype[T]};
using I = {dtype_to_ctype[I]};
using G = cusvna::LocalThreadIndex<I>;
constexpr I N = 8;
using AA = cusvna::AA<T,I,G,N>;
using Interval = cusvna::Interval<T>;

#include "examples/example2.cuh"
"""

if __name__ == "__main__":
    fname = 'f(x,y) = x^2 + y^2 + x*y - 0.5*(x*y)^2 - 0.25'
    logger.info(f"Generating grids for function {fname}...")

    start_level = 2
    end_level = 6
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    def f(x,y):
        return x*x + y*y + x*y - (x*y)**2/2 - 0.25

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(14,10))
    fig.suptitle(fname)

    x = cp.linspace(xmin, xmax, 1024)[None,:]
    y = cp.linspace(xmin, xmax, 1024)[:,None]
    z = f(x,y).get()

    ax = axes
    ax.set_title(f'f(x,y)')
    ax.imshow(z, zorder=0, vmin=-4, vmax=+4, cmap='seismic',
            extent=[xmin, xmax, ymin, ymax], origin='lower')
    ax.contour(z, levels=[0], zorder=1, colors=['k'],
            extent=[xmin, xmax, ymin, ymax], origin='lower')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    logger.info(f"  computing quad_tree with start_level={start_level} and end_level={end_level}...")

    N = 2**start_level
    active_cell_indices = cp.empty((N,N), dtype=I)
    active_cell_counter = cp.empty((), dtype=I)

    threads_per_block = 128
    warpsize = cp.cuda.Device(0).attributes['WarpSize']
    warps_per_block = threads_per_block // warpsize;
    kname = f'kernel<{warps_per_block},{warpsize}>'
    kernel, = build_module_and_extract_kernels_from_source(logger, kernel_src, (kname,))
    assert threads_per_block <= kernel.max_threads_per_block, (B, kernel.max_threads_per_block)

    nactive = (N/2)*(N/2)  # correponds to active cells of previous level = start_level - 1.
    for current_level in range(start_level, end_level):
        B = threads_per_block
        G = (nactive+B-1)//B

        active_cell_counter[...] = 0   # reset cell counter for next level

        kernel((G,), (B,),
                (T(xmin), T(xmax), T(ymin), T(ymax),
                   I(start_level), I(end_level),
                   I(current_level), I(nactive),
                   active_cell_indices, active_cell_counter))

        Ntot = 2**current_level

        nactive = int(active_cell_counter.get())
        logger.info(f"  active cells at level {current_level} is {nactive}/{Ntot*Ntot}.")
        if nactive == 0:
            break

    #ax.imshow(s + 10*(s==0), zorder=0, vmin=0, vmax=10, cmap='Set1',
            #extent=[xmin, xmax, ymin, ymax], origin='lower')

    logger.info("All done.")
    #plt.show()
