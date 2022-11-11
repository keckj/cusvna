
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <cuda/std/tuple>

namespace cusvna {
namespace detail {
    
    template <int... Is>
    using integer_sequence = cuda::std::integer_sequence<int, Is...>;
    
    template <size_t N>
    using make_integer_sequence = cuda::std::make_integer_sequence<int, N>;
   
    /* generate unique type_id from a typename, type_id_t are just equal-comparable so no sorting is possible */
    using type_id_t = void(*)();

    template <typename T>
    struct IdGen {
        static constexpr type_id_t get_unique_id() {
            return IdGen::dummy;
        }
        private:
            static void dummy() {};
    };
    
    template <typename T>
    static constexpr type_id_t type_id = IdGen<T>::get_unique_id(); 

    static_assert(type_id<int> == type_id<int>);
    static_assert(type_id<float> == type_id<float>);

} /* end of namespace detail */
} /* end of namespace cusvna */
