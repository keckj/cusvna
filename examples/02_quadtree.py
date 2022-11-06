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
constexpr I N = 6;
using AA = cusvna::AA<T,I,G,N>;
using Interval = cusvna::Interval<T>;

#include "examples/quadtree.cuh"
"""

if __name__ == "__main__":
    fname = 'f(x,y) = x^2 + y^2 + x*y - 0.5*(x*y)^2 - 0.25'
    logger.info(f"Generating quadtree for function {fname}...")

    do_plot = True
    nlevels = 6 if do_plot else 15

    start_level = 2
    end_level = start_level + nlevels
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    def f(x,y):
        return x*x + y*y + x*y - (x*y)**2/2 - 0.25

    if do_plot:
        fig, axes = plt.subplots(nrows=2, ncols=nlevels//2, figsize=(14,10))
        fig.suptitle(fname)

        x = cp.linspace(xmin, xmax, 1024)[None,:]
        y = cp.linspace(ymin, ymax, 1024)[:,None]
        z = f(x,y).get()
    else:
        axes = np.full(nlevels, fill_value=None)

    logger.info(f"Computing quad_tree with start_level={start_level} and end_level={end_level-1}...")

    N = 2**start_level
    active_cell_indices = cp.empty(100000, dtype=I)  # index buffer, should be large enough to handle end_level active cell count
    active_cell_counter = cp.empty((), dtype=I)

    threads_per_block = 128
    warpsize = cp.cuda.Device(0).attributes['WarpSize']
    warps_per_block = threads_per_block // warpsize;
    kname = f'kernel<{warps_per_block},{warpsize}>'
    kernel, = build_module_and_extract_kernels_from_source(logger, kernel_src, (kname,))
    assert threads_per_block <= kernel.max_threads_per_block, (threads_per_block, kernel.max_threads_per_block)

    start = cp.cuda.Event(disable_timing=False)
    end = cp.cuda.Event(disable_timing=False)

    nactive = (N//2)**2  # correponds to active cells of previous level = start_level - 1.
    for current_level, ax in zip(range(start_level, end_level), axes.ravel()):
        B = threads_per_block
        G = (4*nactive+B-1)//B

        active_cell_counter[...] = 0   # reset cell counter for next level

        start.record()
        kernel((G,), (B,),
                (T(xmin), T(xmax), T(ymin), T(ymax),
                   I(start_level), I(end_level),
                   I(current_level), I(nactive),
                   active_cell_indices, active_cell_counter))
        end.record()
        end.synchronize()
        time_ms = cp.cuda.get_elapsed_time(start, end);

        nactive = int(active_cell_counter.get())
        Ntot = 2**current_level
        logger.info(f"  {time_ms*1000:6.1f}us -- level {current_level:2d} -- grid {Ntot:5d}^2 = 2^{2*current_level:2d}"
                    f" -- active cells {nactive}/{Ntot**2} ({100.0*nactive/(Ntot**2):2.2f}%)")

        if nactive == 0:
            break

        if do_plot:
            ax.set_title(f'f(x,y)')
            ax.imshow(z, zorder=0, vmin=-4, vmax=+4, cmap='seismic',
                    extent=[xmin, xmax, ymin, ymax], origin='lower')
            ax.contour(z, levels=[0], zorder=1, colors=['k'],
                    extent=[xmin, xmax, ymin, ymax], origin='lower')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            idx = active_cell_indices[:nactive].get()
            for k in idx:
                i = k // Ntot;
                j = k % Ntot;
                dy = (ymax - ymin) / Ntot
                dx = (xmax - xmin) / Ntot
                yi = ymin + i*dy;
                xi = xmin + j*dx;

                rectangle = matplotlib.patches.Rectangle((xi,yi), dx, dy,
                        zorder=2, alpha=1.0, facecolor='b', edgecolor='k')
                ax.add_patch(rectangle)

            fig.show()


    logger.info("All done.")
    if do_plot:
        plt.show()
