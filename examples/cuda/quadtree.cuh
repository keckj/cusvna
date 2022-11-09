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
    
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    I nactive = 0;
    size_t cid[4];

    if (tid < current_active_cells) {
        const size_t oid = ((current_level == start_level) ? tid : active_cells[tid]);
        
        const size_t osplit = 1 << (current_level - 1);
        const size_t old_i = (oid / osplit);
        const size_t old_j = (oid % osplit);

        const size_t nsplit = 1 << current_level;
        const T dx = (xmax - xmin) / nsplit;
        const T dy = (ymax - ymin) / nsplit;
        
        #pragma unroll 4
        for(size_t offset=0; offset<4; offset++) {
            const size_t i = 2 * old_i + ((offset & 0x2) >> 1);
            const size_t j = 2 * old_j + ((offset & 0x1) >> 0);

            const Interval yi(ymin + i*dy, ymin + (i+1)*dy);
            const Interval xi(xmin + j*dx, xmin + (j+1)*dx);
        
            G index_generator;
            const AA x(xi, index_generator);
            const AA y(yi, index_generator);
            const AA xy = x*y;
            const AA z = x*x + y*y + xy - xy*xy/T(2) - T(1)/T(4);
        
            bool active = (z.sign() == cusvna::Sign::UNKNOWN);
            
            if (active) {
                cid[nactive] = i*nsplit + j;
                nactive++;
            }
        }
    }

    __shared__ I active_offsets[warps_per_block + 1];
    const I output_index = prefix_sum<warps_per_block, warpsize>(nactive, active_offsets, active_cells_counter);

    for(I offset=0; offset<nactive; offset++) {
        active_cells[output_index + offset] = cid[offset];
    }
}
