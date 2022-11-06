/* this file is part of example 02_quadtree.py */

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
                       const I start_level, const I end_level,
                       const I current_level, const I current_active_cells,
                       I* active_cells, I* active_cells_counter) {
    
    __shared__ I active_offsets[warps_per_block];
    
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t k = tid >> 2;

    size_t cid;
    bool active;

    if (k < current_active_cells) {
        const size_t oid = ((current_level == start_level) ? k : active_cells[k]);
        
        const size_t osplit = 1 << (current_level - 1);
        const size_t old_i = (oid / osplit);
        const size_t old_j = (oid % osplit);
        const size_t i_offset = (threadIdx.x & 0x02) >> 1;
        const size_t j_offset = (threadIdx.x & 0x01) >> 0;

        const size_t i = 2 * old_i + i_offset;
        const size_t j = 2 * old_j + j_offset;
        const size_t nsplit = 1 << current_level;
        cid = i*nsplit + j;

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

        active = (z.sign() == cusvna::Sign::UNKNOWN);
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
