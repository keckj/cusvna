/* this file is part of example 03_octtree.py */

template <unsigned int warps_per_block, unsigned int warpsize>
__device__ I prefix_sum(I prefix,
        I* const __restrict volatile warp_offset,
        I* const __restrict volatile global_block_offset) {

    const I Z(0);
    const I laneid = threadIdx.x % warpsize;
    const I warpid = threadIdx.x / warpsize;

    /* intra warp prefix scan, all warp shuffle scan their data */
    I local_prefix(prefix);
    {
        #pragma unroll
        for (unsigned int offset=1u; offset<warpsize; offset<<=1u) {
            const I other_prefix = __shfl_up_sync(0xffffffff, local_prefix, offset);
            if (laneid >= offset) {
                local_prefix += other_prefix;
            }
        }
        __syncthreads();
        if ((laneid == warpsize-1u) && (warpid<warps_per_block)) {
            warp_offset[warpid] = local_prefix;
        }
        __syncthreads();
    }

    /* intra block prefix scan into shared memory, only first warp participates */
    {
        if (warpid == 0) {
            I warp_prefix = (laneid >= warps_per_block ? Z : warp_offset[laneid]);
            #pragma unroll
            for (unsigned int offset=1; offset<warpsize; offset<<=1u) {
                const I other_prefix = __shfl_up_sync(0xffffffff, warp_prefix, offset);
                if (laneid >= offset) {
                    warp_prefix += other_prefix;
                }
            }
            if (laneid < warps_per_block) {
                warp_offset[laneid] = warp_prefix;
            }
        }
        __syncthreads();
        local_prefix += (warpid==0 ? Z : warp_offset[warpid-1]);
    }

    /* determine an inter block offset by using atomic operation in global memory (order does not matter) */
    {
        if ((warpid == 0) && (laneid == 0)) {
            warp_offset[warps_per_block] = atomicAdd(global_block_offset, warp_offset[warps_per_block-1]);
        }
        __syncthreads();
    }

    /* inter block prefix scan */
    local_prefix += warp_offset[warps_per_block];
    __syncthreads();

    return local_prefix - prefix;
}


template <unsigned int warps_per_block, unsigned int warpsize>
__launch_bounds__(warps_per_block * warpsize)
__global__ void kernel(const T xmin, const T xmax,
                       const T ymin, const T ymax,
                       const T zmin, const T zmax,
                       const I start_level, const I end_level,
                       const I current_level, const I current_active_cells,
                       I* active_cells, I* active_cells_counter) {
    
    __shared__ I active_offsets[warps_per_block];
    
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t k = tid >> 3;

    size_t cid;
    bool active;

    if (k < current_active_cells) {
        const size_t oid = ((current_level == start_level) ? k : active_cells[k]);
        
        const size_t osplit = 1 << (current_level - 1);
        const size_t old_i = oid / (osplit*osplit);
        const size_t old_j = (oid - old_i*osplit*osplit) / osplit;
        const size_t old_k = oid % osplit;

        const size_t i_offset = (threadIdx.x & 0x04) >> 2;
        const size_t j_offset = (threadIdx.x & 0x02) >> 1;
        const size_t k_offset = (threadIdx.x & 0x00) >> 0;

        const size_t i = 2 * old_i + i_offset;
        const size_t j = 2 * old_j + j_offset;
        const size_t k = 2 * old_k + k_offset;
        const size_t nsplit = 1 << current_level;
        cid = (i*nsplit + j)*nsplit + k;

        const T Lz = zmax - zmin;
        const T Ly = ymax - ymin;
        const T Lx = xmax - xmin;

        const T dz = Lz / nsplit;
        const T dy = Ly / nsplit;
        const T dx = Lx / nsplit;

        const Interval zi(zmin + i*dy, zmin + (i+1)*dz);
        const Interval yi(ymin + j*dy, ymin + (j+1)*dy);
        const Interval xi(xmin + k*dx, xmin + (k+1)*dx);

        G index_generator;
        const AA x(xi, index_generator);
        const AA y(yi, index_generator);
        const AA z(zi, index_generator);
        const AA xx(x*x), yy(y*y), zz(z*z);

        const AA w = (
               (x-T(2))*(x-T(2)) * (x+T(2))*(x+T(2))
             + (y-T(2))*(y-T(2)) * (y+T(2))*(y+T(2))
             + (z-T(2))*(z-T(2)) * (z+T(2))*(z+T(2))
             + (xx * yy + xx * zz + yy * zz)
             + T(6)*x*y*z
             - T(10)*(xx+yy+zz)
             + T(22));

        active = (w.sign() == cusvna::Sign::UNKNOWN);
    }
    else {
        cid = I(-1);
        active = false;
    }

    const I output_index = prefix_sum<warps_per_block, warpsize>(active, active_offsets, active_cells_counter);

    if (active) {
        active_cells[output_index] = cid;
    }
}
