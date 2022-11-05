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

__global__ void kernel(const T xmin, const T xmax,
                       const T ymin, const T ymax,
                       const I level, unsigned char* sign) {{
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    const size_t nsplit = 1 << level;
    const size_t k = i*nsplit + j;

    if ((i >= nsplit) || (j >= nsplit)) {{ return; }}

    const T Lx = xmax - xmin;
    const T Ly = ymax - ymin;
    const T dx = Lx / nsplit;
    const T dy = Ly / nsplit;

    const Interval yi(ymin + i*dy, ymin + (i+1)*dy);
    const Interval xi(xmin + j*dx, xmin + (j+1)*dx);

    G index_generator;
    const AA x(xi, index_generator);
    const AA y(yi, index_generator);
    const AA xy = x*y;
    const AA z = x*x + y*y + xy - xy*xy/T(2) - T(1)/T(4);

    const Interval zi(z);
    sign[k] = zi.sign();
}}
"""

if __name__ == "__main__":
    kernel, = build_module_and_extract_kernels_from_source(logger, kernel_src, ('kernel',))

    fname = 'f(x,y) = x^2 + y^2 + x*y - 0.5*(x*y)^2 - 0.25'
    logger.info(f"Generating grids for function {fname}...")

    levels = [-1,2,3,4,5,6]
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    def f(x,y):
        return x*x + y*y + x*y - (x*y)**2/2 - 0.25

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14,10))
    fig.suptitle(fname)

    x = cp.linspace(xmin, xmax, 1024)[None,:]
    y = cp.linspace(xmin, xmax, 1024)[:,None]
    z = f(x,y).get()

    for level, ax in zip(levels, axes.ravel()):

        ax.contour(z, levels=[0], zorder=1, colors=['k'],
                extent=[xmin, xmax, ymin, ymax], origin='lower')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        if level >= 0:
            ax.set_title(f'level={level}')
            N = 2**level
            logger.info(f"  computing level={level} with {N} intervals...")
            s = cp.empty((N,N), dtype=np.uint8)

            B = 16
            G = (N+B-1)//B
            assert B*B < kernel.max_threads_per_block
            kernel((G,G), (B,B), (T(xmin), T(xmax), T(ymin), T(ymax), I(level), s))
            s = s.get()

            ax.imshow(s + 10*(s==0), zorder=0, vmin=0, vmax=10, cmap='Set1',
                    extent=[xmin, xmax, ymin, ymax], origin='lower')
        else:
            ax.set_title(f'f(x,y)')
            ax.imshow(z, zorder=0, vmin=-4, vmax=+4, cmap='seismic',
                    extent=[xmin, xmax, ymin, ymax], origin='lower')

        fig.show()

    logger.info("All done.")
    plt.show()
