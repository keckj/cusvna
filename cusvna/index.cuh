#ifndef INDEX_CUH_WLNSDOV7
#define INDEX_CUH_WLNSDOV7

namespace cusvna {

template <typename I>
class LocalThreadIndex {
    public:
        __device__ LocalThreadIndex(): idx(0) {}
        __device__ I operator()() { return this->new_symbol(); };

    private:
        I idx;
        LocalThreadIndex(const LocalThreadIndex& other) = delete;
        LocalThreadIndex(LocalThreadIndex&& other) = delete;
        LocalThreadIndex& operator=(const LocalThreadIndex& other) = delete;
        LocalThreadIndex& operator=(LocalThreadIndex&& other) = delete;
        __device__ I new_symbol() {
            return idx++;
        }
};

} /* end of namespace cusvna */

#endif /* end of include guard: INDEX_CUH_WLNSDOV7 */
