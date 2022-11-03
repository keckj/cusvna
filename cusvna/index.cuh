#ifndef INDEX_CUH_WLNSDOV7
#define INDEX_CUH_WLNSDOV7

#include <cuda/std/cassert>
#include <cuda/std/limits>

namespace cusvna {

template <typename I, I lsi = cuda::std::numeric_limits<I>::max()>
class LocalThreadIndex {
    public:
        __device__ LocalThreadIndex(): idx(0) {}
        __device__ I operator()() { return this->new_symbol(); };
        static constexpr I last_symbol_index = lsi;

    private:
        I idx;
        LocalThreadIndex(const LocalThreadIndex& other) = delete;
        LocalThreadIndex(LocalThreadIndex&& other) = delete;
        LocalThreadIndex& operator=(const LocalThreadIndex& other) = delete;
        LocalThreadIndex& operator=(LocalThreadIndex&& other) = delete;
        __device__ I new_symbol() {
            assert(idx != last_symbol_index);  /* last index is reserved for implementation detail */
            return idx++;
        }
};

} /* end of namespace cusvna */

#endif /* end of include guard: INDEX_CUH_WLNSDOV7 */
