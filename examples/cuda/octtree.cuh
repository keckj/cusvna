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
                       const T zmin, const T zmax,
                       const I start_level, const I end_level,
                       const I current_level, const I current_active_cells,
                       I* active_cells, I* active_cells_counter) {
    
    __shared__ I active_offsets[warps_per_block];
    
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    I nactive = 0;
    size_t cid[8];

    if (tid < current_active_cells) {
        const size_t oid = ((current_level == start_level) ? tid : active_cells[tid]);
        
        const size_t osplit = 1 << (current_level - 1);
        const size_t old_i = (oid / osplit / osplit);
        const size_t old_j = ((oid / osplit) % osplit);
        const size_t old_k = (oid % osplit);

        const size_t nsplit = 1 << current_level;
        const T dz = (zmax - zmin) / nsplit;
        const T dy = (ymax - ymin) / nsplit;
        const T dx = (xmax - xmin) / nsplit;
        
        for(size_t offset=0; offset<8; offset++) {
            const size_t i = 2 * old_i + ((offset & 0x4) >> 2);
            const size_t j = 2 * old_j + ((offset & 0x2) >> 1);
            const size_t k = 2 * old_k + ((offset & 0x1) >> 0);

            const Interval zi(zmin + i*dz, zmin + (i+1)*dz);
            const Interval yi(ymin + j*dy, ymin + (j+1)*dy);
            const Interval xi(xmin + k*dx, xmin + (k+1)*dx);
        
            G index_generator;
            const AA z(zi, index_generator);
            const AA y(yi, index_generator);
            const AA x(xi, index_generator);

            const AA zz = z*z;
            const AA yy = y*y;
            const AA xx = x*x;

            const AA w = (
                   (x-T(2))*(x-T(2)) * (x+T(2))*(x+T(2))
                 + (y-T(2))*(y-T(2)) * (y+T(2))*(y+T(2))
                 + (z-T(2))*(z-T(2)) * (z+T(2))*(z+T(2))
                 + (xx * yy + xx * zz + yy * zz)
                 + T(6)*x*y*z
                 - T(10)*(xx+yy+zz)
                 + T(22));
        
            bool active = (w.sign() == cusvna::Sign::UNKNOWN);
            
            if (active) {
                cid[nactive] = (i*nsplit + j)*nsplit + k;
                nactive++;
            }
        }
    }

    const I output_index = prefix_sum<warps_per_block, warpsize>(nactive, active_offsets, active_cells_counter);

    for(I offset=0; offset<nactive; offset++) {
        active_cells[output_index + offset] = cid[offset];
    }
}
