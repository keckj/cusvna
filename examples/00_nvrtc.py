import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.patches

from utils import *

T = np.float32
I = np.int32

kernel_src = f"""
#include <cusvna/aa.cuh>

using T = {dtype_to_ctype[T]};
using I = {dtype_to_ctype[I]};
using G = cusvna::LocalThreadIndex<I>;
constexpr I N = 16;
using AA = cusvna::AA<T,I,G,N>;
using Interval = cusvna::Interval<T>;

__global__ void kernel(const T xmin, const T xmax, const I nintervals,
                       const I k, T* a, T* b) {{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nintervals) {{ return; }}

    const T L  = xmax - xmin;
    const T dx = L / nintervals;

    G index_generator;
    Interval xi(xmin + i*dx, xmin + (i+1)*dx);

    AA x(xi, index_generator);
    AA y = sqrt(x*x - x + T(1)/T(2)) / sqrt(x*x + T(1)/T(2));
    if (k==1) {{
        y = sqrt(y*y - y + T(1)/T(2)) / sqrt(y*y + T(1)/T(2));
    }}

    Interval yi(y);
    a[i] = yi.a;
    b[i] = yi.b;
}}
"""

if __name__ == "__main__":
    kernel, = build_module_and_extract_kernels_from_source(logger, kernel_src, ('kernel',))

    xmin, xmax = -2, 2
    Ns = [8,16,32,64]

    for k in range(2):
        fname = 'g(x) = sqrt(x^2 - x + 0.5) / sqrt(x^2 + 0.5)' if k==0 else 'g(g(x))'
        logger.info(f"Generating plots for function {fname}...")

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
        fig.suptitle(fname)

        x = cp.linspace(xmin, xmax, 1024)
        y = cp.sqrt(x*x - x + 0.5) / cp.sqrt(x*x + 0.5)
        if k == 1:
            y = cp.sqrt(y*y - y + 0.5) / cp.sqrt(y*y + 0.5)
        x, y = x.get(), y.get()

        for ax, N in zip(axes.ravel(), Ns):
            logger.info(f"  using {N} intervals.")
            a = cp.empty(N, dtype=T)
            b = cp.empty(N, dtype=T)

            block_size = 128
            grid_size  = (N+block_size-1)//block_size
            kernel((block_size,), (grid_size,), (T(xmin), T(xmax), I(N), I(k), a, b))
            a, b = a.get(), b.get()

            dx = (xmax - xmin) / N
            for i in range(N):
                xstart = xmin+i*dx
                xend = xstart + dx

                # check corectness of returned results
                mask = np.where(np.logical_and(x >= xstart, x <= xend))
                assert (y[mask] >= a[i]).all()
                assert (y[mask] <= b[i]).all()

                rectangle = matplotlib.patches.Rectangle((xstart,a[i]), dx, b[i]-a[i],
                        zorder=0, alpha=0.5, facecolor='b', edgecolor='k')
                ax.add_patch(rectangle)

            ax.set_title(f'N={N}')
            ax.plot(x, y, zorder=1, c='r')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, 2)

            fig.show()

    logger.info("All done.")
    plt.show()
