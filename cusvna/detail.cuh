
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <cuda/std/tuple>
#include <cuda/std/array>

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
        static void dummy() {};
    };

    template <typename T>
    static constexpr type_id_t type_id = IdGen<T>::get_unique_id(); 
    
    template <int I>
    static constexpr type_id_t id = type_id<integer_sequence<I>>;

    static_assert(type_id<int> == type_id<int>);
    static_assert(type_id<float> == type_id<float>);
    static_assert(id<0> == id<0>);
    static_assert(id<1> == id<1>);
    
    template <type_id_t... Is>
    using type_id_sequence = cuda::std::integer_sequence<type_id_t, Is...>;

    
    /* is in sequence */
    template <type_id_t I0, type_id_t... Is>
    struct is_in_sequence {
        static constexpr bool value = (std::is_same<type_id_sequence<I0>, type_id_sequence<Is>>::value || ...);
    };

    static_assert(is_in_sequence<id<0>,id<0>>::value, "is_in_sequence error");
    static_assert(!is_in_sequence<id<0>,id<1>>::value, "is_in_sequence error");
    static_assert(is_in_sequence<id<0>,id<0>,id<1>,id<2>,id<3>,id<4>>::value, "is_in_sequence error");
    static_assert(is_in_sequence<id<2>,id<0>,id<1>,id<2>,id<3>,id<4>>::value, "is_in_sequence error");
    static_assert(is_in_sequence<id<4>,id<0>,id<1>,id<2>,id<3>,id<4>>::value, "is_in_sequence error");
    static_assert(!is_in_sequence<id<42>,id<0>,id<1>,id<2>,id<3>,id<4>>::value, "is_in_sequence error");
    static_assert(!is_in_sequence<id<0>>::value, "is_in_sequence error");
    
    
    /* offset in sequence */
    template <type_id_t, typename, int K=0>
    struct offset_in_sequence { 
        static constexpr int value = -1;
    };

    template <type_id_t I, type_id_t I0, type_id_t... Is, int K>
    struct offset_in_sequence<I, type_id_sequence<I0,Is...>, K>{
        static constexpr int value = (std::is_same<type_id_sequence<I>,type_id_sequence<I0>>::value ? K : 
                                      offset_in_sequence<I,type_id_sequence<Is...>,K+1>::value); 
    };

    static_assert(offset_in_sequence<id<0>,type_id_sequence<id<0>>>::value == 0, "offset_in_sequence error");
    static_assert(offset_in_sequence<id<1>,type_id_sequence<id<0>,id<1>>>::value == 1, "offset_in_sequence error");
    static_assert(offset_in_sequence<id<0>,type_id_sequence<id<1>>>::value == -1, "offset_in_sequence error");
    static_assert(offset_in_sequence<id<0>,type_id_sequence<>>::value == -1, "offset_in_sequence error");
    static_assert(offset_in_sequence<id<0>,type_id_sequence<id<1>,id<2>,id<0>>>::value == 2, "offset_in_sequence error");
    static_assert(offset_in_sequence<id<2>,type_id_sequence<id<1>,id<2>,id<0>>>::value == 1, "offset_in_sequence error");
    
    /* is set */
    template <typename>
    struct is_set {};

    template <type_id_t I0, type_id_t... Is>
    struct is_set<type_id_sequence<I0, Is...>> {
        static constexpr bool value = (!is_in_sequence<I0, Is...>::value) && is_set<type_id_sequence<Is...>>::value;
    };

    template <type_id_t I0>
    struct is_set<type_id_sequence<I0>> {
        static constexpr bool value = true;
    };
    
    template <>
    struct is_set<type_id_sequence<>> {
        static constexpr bool value = true;
    };
    
    static_assert(is_set<type_id_sequence<>>::value, "is_set error");
    static_assert(is_set<type_id_sequence<id<3>>>::value, "is_set error");
    static_assert(is_set<type_id_sequence<id<0>,id<1>,id<2>,id<3>,id<4>,id<5>>>::value, "is_set error");
    static_assert(!is_set<type_id_sequence<id<0>,id<3>,id<3>>>::value, "is_set error");
    static_assert(!is_set<type_id_sequence<id<1>,id<1>,id<2>,id<3>,id<4>,id<5>>>::value, "is_set error");
    static_assert(!is_set<type_id_sequence<id<4>,id<1>,id<2>,id<3>,id<4>,id<5>>>::value, "is_set error");
    static_assert(!is_set<type_id_sequence<id<3>,id<1>,id<2>,id<8>,id<7>,id<1>>>::value, "is_set error");

    /* append to sequence */
    template <type_id_t, typename>
    struct append_to_sequence {};

    template <type_id_t I0, type_id_t... Is>
    struct append_to_sequence<I0, type_id_sequence<Is...>> {
        using type = type_id_sequence<Is..., I0>;
    };
    static_assert(
        cuda::std::is_same<
            append_to_sequence<
                id<42>,
                type_id_sequence<id<0>,id<1>,id<2>,id<3>,id<4>>
                >::type,
            type_id_sequence<id<0>,id<1>,id<2>,id<3>,id<4>,id<42>>
            >::value,
        "append_to_sequence error"
        );
    

    /* prepend to sequence */
    template <type_id_t, typename>
    struct prepend_to_sequence {};

    template <type_id_t I0, type_id_t... Is>
    struct prepend_to_sequence<I0, type_id_sequence<Is...>> {
        using type = type_id_sequence<I0, Is...>;
    };
    static_assert(
        cuda::std::is_same<
            prepend_to_sequence<
                id<42>,
                type_id_sequence<id<0>,id<1>,id<2>,id<3>,id<4>>
                >::type,
            type_id_sequence<id<42>,id<0>,id<1>,id<2>,id<3>,id<4>>
            >::value,
        "prepend_to_sequence error"
        );


    /* extract unique type ids in sequence */
    template <typename>
    struct unique_type_ids {};

    template <type_id_t I0, type_id_t... Is>
    struct unique_type_ids<type_id_sequence<I0, Is...>> {
        using subtype = typename unique_type_ids<type_id_sequence<Is...>>::type;
        using type = typename cuda::std::conditional<
               is_in_sequence<I0,Is...>::value,
               subtype,
               typename prepend_to_sequence<I0, subtype>::type
        >::type;
    };

    template <type_id_t I0>
    struct unique_type_ids<type_id_sequence<I0>> {
        using type = type_id_sequence<I0>;
    };
    
    template <>
    struct unique_type_ids<type_id_sequence<>> {
        using type = type_id_sequence<>;
    };

    static_assert(
            std::is_same<
                unique_type_ids<type_id_sequence<>>::type,
                type_id_sequence<>
            >::value,
            "unique_type_ids error"
            );
    static_assert(
            std::is_same<
                unique_type_ids<type_id_sequence<id<3>,id<0>,id<1>,id<1>,id<0>,id<4>>>::type,
                type_id_sequence<id<3>,id<1>,id<0>,id<4>>
            >::value,
            "unique_type_ids error"
            );
    

    /* merge sequences and extract set of type ids */
    template <typename, typename>
    struct merge_type_ids {};
    
    template <type_id_t... Is, type_id_t... Js>
    struct merge_type_ids<type_id_sequence<Is...>, type_id_sequence<Js...>> {
        using type = typename unique_type_ids<type_id_sequence<Is..., Js...>>::type;
    };
    
    static_assert(
            std::is_same<
                merge_type_ids<
                    type_id_sequence<>,
                    type_id_sequence<>
                >::type,
                type_id_sequence<>
            >::value,
            "merge_type_ids error"
            );
    static_assert(
            std::is_same<
                merge_type_ids<
                    type_id_sequence<id<0>>,
                    type_id_sequence<id<1>>
                >::type,
                type_id_sequence<id<0>,id<1>>
            >::value,
            "merge_type_ids error"
            );
    static_assert(
            std::is_same<
                merge_type_ids<
                    type_id_sequence<id<0>,id<2>,id<1>>,
                    type_id_sequence<id<1>,id<3>,id<0>>
                >::type,
                type_id_sequence<id<2>,id<1>,id<3>,id<0>>
            >::value,
            "merge_type_ids error"
            );

    
    /* merge two AAs */
    template <template<typename, type_id_t...> class, typename, typename>
    struct TypeFromSequence {};

    template <template<typename, type_id_t...> class cls, typename T, type_id_t... Is>
    struct TypeFromSequence<cls, T, type_id_sequence<Is...>> {
        using type = cls<T,Is...>;
    };

    template <typename, typename>
    struct MergedType {};

    template <template<typename, type_id_t...> class cls, typename T, type_id_t... Is, type_id_t... Js>
    struct MergedType<cls<T,Is...>, cls<T,Js...>> {
        using type = typename TypeFromSequence<
            cls,
            T,
            typename merge_type_ids<
                type_id_sequence<Is...>,
                type_id_sequence<Js...>
            >::type
        >::type;
    };

    template <typename lhs, typename rhs>
    using merged_t = typename MergedType<lhs, rhs>::type;

} /* end of namespace detail */
} /* end of namespace cusvna */
