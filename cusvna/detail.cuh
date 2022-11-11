
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <cuda/std/tuple>

namespace cusvna {
namespace detail {
    
    template <int... Is>
    using integer_sequence = cuda::std::integer_sequence<int, Is...>;
    
    template <size_t N>
    using make_integer_sequence = cuda::std::make_integer_sequence<int, N>;
    
    template <int I0=0>
    constexpr bool is_ascending_seq() { return true; }

    template <int I0, int I1, int... Is>
    constexpr bool is_ascending_seq() { return (I0 < I1) && is_ascending_seq<I1, Is...>(); }
   

    /* is in sequence */
    template <int I0, int... Is>
    struct is_in_sequence {
        static constexpr bool value = ((I0 == Is) || ...);
    };
    
    static_assert(is_in_sequence<0,0,1,2,3,4>::value, "is_in_sequence error");
    static_assert(is_in_sequence<2,0,1,2,3,4>::value, "is_in_sequence error");
    static_assert(is_in_sequence<4,0,1,2,3,4>::value, "is_in_sequence error");
    static_assert(!is_in_sequence<42,0,1,2,3,4>::value, "is_in_sequence error");

    
    /* append to integer sequence */
    template <int, typename> 
    struct append_to_sequence {};
    
    template <int I0, int... Is>
    struct append_to_sequence<I0, integer_sequence<Is...>> {
        using type = integer_sequence<Is..., I0>;
    };
    
    static_assert(
        cuda::std::is_same<
            append_to_sequence<
                42,
                integer_sequence<0,1,2,3,4>
                >::type,
            integer_sequence<0,1,2,3,4,42>
            >::value,
        "append_to_sequence error"
        );
    

    /* prepend to integer sequence */
    template <int, typename> 
    struct prepend_to_sequence {};
    
    template <int I0, int... Is>
    struct prepend_to_sequence<I0, integer_sequence<Is...>> {
        using type = integer_sequence<I0, Is...>;
    };
    
    static_assert(
        cuda::std::is_same<
            prepend_to_sequence<
                42,
                integer_sequence<0,1,2,3,4>
                >::type,
            integer_sequence<42,0,1,2,3,4>
            >::value,
        "prepend_to_sequence error"
        );

    
    /* concatenate integer sequences */
    template <typename, typename>
    struct concat_sequences {};

    template <int... Is, int... Js>
    struct concat_sequences<integer_sequence<Is...>, integer_sequence<Js...>> {
        using type = integer_sequence<Is..., Js...>;
    };
    
    static_assert(
        cuda::std::is_same<
            concat_sequences<
                integer_sequence<0,1,2,3,4>,
                integer_sequence<42,8,-12>
                >::type,
            integer_sequence<0,1,2,3,4,42,8,-12>
            >::value,
        "concat_sequences error"
        );


    /* intersect integer sequences */
    template <typename, typename>
    struct intersect_sequences {
        using type = integer_sequence<>;
    };

    template <int I0, int... Is, int... Js>
    struct intersect_sequences<integer_sequence<I0, Is...>, integer_sequence<Js...>> {
        using intersection = typename intersect_sequences<integer_sequence<Is...>, integer_sequence<Js...>>::type;
        using type = typename cuda::std::conditional<
            is_in_sequence<I0, Js...>::value,
            typename prepend_to_sequence<I0, intersection>::type,
            intersection 
        >::type;
    };

    static_assert(
        std::is_same<
            intersect_sequences<
                integer_sequence<0,1,2,3,4,6,0>,
                integer_sequence<1,5,3,4,0,0>
                >::type,
            integer_sequence<0,1,3,4,0>
            >::value,
            "intersect_sequences error"
            );


    /* merge sorted integer sequences */
    template <typename, typename>
    struct merge_sorted_sequences {
    };
    
    template <int I0, int... Is, int J0, int... Js>
    struct merge_sorted_sequences<integer_sequence<I0, Is...>, integer_sequence<J0, Js...>> {
        static_assert(is_ascending_seq<I0, Is...>(), "First sequence is not sorted.");
        static_assert(is_ascending_seq<J0, Js...>(), "Second sequence is not sorted.");
        using eq = typename merge_sorted_sequences<integer_sequence<Is...>, integer_sequence<Js...>>::type;
        using le = typename merge_sorted_sequences<integer_sequence<Is...>, integer_sequence<J0, Js...>>::type;
        using ge = typename merge_sorted_sequences<integer_sequence<I0, Is...>, integer_sequence<Js...>>::type;
        using type = typename cuda::std::conditional<
            I0 <= J0, 
            typename prepend_to_sequence<I0, typename cuda::std::conditional<I0 == J0, eq, le>::type>::type,
            typename prepend_to_sequence<J0, ge>::type
        >::type;
    };
    
    template <int... Is>
    struct merge_sorted_sequences<integer_sequence<Is...>, integer_sequence<>> {
        using type = integer_sequence<Is...>;
    };
    
    template <int... Js>
    struct merge_sorted_sequences<integer_sequence<>, integer_sequence<Js...>> {
        using type = integer_sequence<Js...>;
    };
    
    template <>
    struct merge_sorted_sequences<integer_sequence<>, integer_sequence<>> {
        using type = integer_sequence<>;
    };
    
    static_assert(
        std::is_same<
            merge_sorted_sequences<
                integer_sequence<1,3,7>,
                integer_sequence<0,2,4,6,7>
                >::type,
            integer_sequence<0,1,2,3,4,6,7>
            >::value,
            "merge_sorted_sequences error"
            );


    template <template<typename, int...> class, typename, typename>
    struct TypeFromSequence {};

    template <template<typename, int...> class cls, typename T, int... Is>
    struct TypeFromSequence<cls, T, integer_sequence<Is...>> {
        using type = cls<T,Is...>;
    };


    template <typename, typename>
    struct MergedType {};

    template <template<typename, int...> class cls, typename T, int... Is, int... Js>
    struct MergedType<cls<T,Is...>, cls<T,Js...>> {
        using type = typename TypeFromSequence<
            cls,
            T,
            typename merge_sorted_sequences<
                integer_sequence<Is...>,
                integer_sequence<Js...>
            >::type
        >::type;
    };

    
    template <typename lhs, typename rhs>
    using merged_t = typename MergedType<lhs, rhs>::type;

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
