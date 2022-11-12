#ifndef DETAIL_CUH_6K1DTOM7
#define DETAIL_CUH_6K1DTOM7

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
   
    template <int I>
    struct id {};
    
    template <typename... Is>
    struct type_id_sequence {};

    template <typename I0, typename I1>
    static constexpr bool same_id = cuda::std::is_same<I0, I1>::value;

    static_assert(same_id<id<0>,id<0>>);
    static_assert(same_id<id<1>,id<1>>);
    static_assert(!same_id<id<0>,id<1>>);
    static_assert(!same_id<int,id<1>>);
    static_assert(!same_id<id<0>,int>);

    /* is in sequence */
    template <typename I0, typename... Is>
    struct IdInSequence {
        static constexpr bool value = (same_id<I0,Is> || ...);
    };
    
    template <typename I0, typename... Is>
    static constexpr bool id_in_sequence = IdInSequence<I0, Is...>::value;

    static_assert(id_in_sequence<id<0>,id<0>>, "id_in_sequence error");
    static_assert(!id_in_sequence<id<0>,id<1>>, "id_in_sequence error");
    static_assert(id_in_sequence<id<0>,id<0>,id<1>,id<2>,id<3>,id<4>>, "id_in_sequence error");
    static_assert(id_in_sequence<id<2>,id<0>,id<1>,id<2>,id<3>,id<4>>, "id_in_sequence error");
    static_assert(id_in_sequence<id<4>,id<0>,id<1>,id<2>,id<3>,id<4>>, "id_in_sequence error");
    static_assert(!id_in_sequence<id<42>,id<0>,id<1>,id<2>,id<3>,id<4>>, "id_in_sequence error");
    static_assert(!id_in_sequence<id<0>>, "id_in_sequence error");
    
    /* offset in sequence */
    template <typename, typename, int K=0>
    struct OffsetInSequence { 
    };

    template <typename I, typename I0, typename... Is, int K>
    struct OffsetInSequence<I, type_id_sequence<I0,Is...>, K>{
        static constexpr int value = (same_id<I,I0> ? K : OffsetInSequence<I,type_id_sequence<Is...>,K+1>::value); 
    };
    
    template <typename I, int K>
    struct OffsetInSequence<I, type_id_sequence<>, K>{
        static constexpr int value = -1;
    };
    
    static_assert(OffsetInSequence<id<0>,type_id_sequence<id<0>>>::value == 0, "OffsetInSequence error");
    static_assert(OffsetInSequence<id<1>,type_id_sequence<id<0>,id<1>>>::value == 1, "OffsetInSequence error");
    static_assert(OffsetInSequence<id<0>,type_id_sequence<id<1>>>::value == -1, "OffsetInSequence error");
    static_assert(OffsetInSequence<id<0>,type_id_sequence<>>::value == -1, "OffsetInSequence error");
    static_assert(OffsetInSequence<id<0>,type_id_sequence<id<1>,id<2>,id<0>>>::value == 2, "OffsetInSequence error");
    static_assert(OffsetInSequence<id<2>,type_id_sequence<id<1>,id<2>,id<0>>>::value == 1, "OffsetInSequence error");
    

    /* extract from sequence */
    template <int N, typename, int K=0>
    struct ExtractFromSequence { 
    };

    template <int N, typename I0, typename... Is, int K>
    struct ExtractFromSequence<N, type_id_sequence<I0,Is...>, K>{
        using type = typename cuda::std::conditional<
                N==K,
                I0,
                typename ExtractFromSequence<N, type_id_sequence<Is...>, K+1>::type
            >::type;
    };
    
    template <int N, int K>
    struct ExtractFromSequence<N, type_id_sequence<>, K>{
        using type = void;
    };
    
    static_assert(cuda::std::is_same<
            typename ExtractFromSequence<0, type_id_sequence<id<0>>>::type,
            id<0>
        >::value, "ExtractFromSequence error");
    static_assert(cuda::std::is_same<
            typename ExtractFromSequence<0, type_id_sequence<id<0>,id<1>>>::type,
            id<0>
        >::value, "ExtractFromSequence error");
    static_assert(cuda::std::is_same<
            typename ExtractFromSequence<1, type_id_sequence<id<0>,id<1>>>::type,
            id<1>
        >::value, "ExtractFromSequence error");
    static_assert(cuda::std::is_same<
            typename ExtractFromSequence<2, type_id_sequence<id<0>,id<1>,id<2>,id<3>>>::type,
            id<2>
        >::value, "ExtractFromSequence error");

    /* is set */
    template <typename>
    struct IsSet {};

    template <typename I0, typename... Is>
    struct IsSet<type_id_sequence<I0, Is...>> {
        static constexpr bool value = (!id_in_sequence<I0, Is...>) && IsSet<type_id_sequence<Is...>>::value;
    };

    template <typename I0>
    struct IsSet<type_id_sequence<I0>> {
        static constexpr bool value = true;
    };
    
    template <>
    struct IsSet<type_id_sequence<>> {
        static constexpr bool value = true;
    };
    
    static_assert(IsSet<type_id_sequence<>>::value, "is_set error");
    static_assert(IsSet<type_id_sequence<id<3>>>::value, "is_set error");
    static_assert(IsSet<type_id_sequence<id<0>,id<1>,id<2>,id<3>,id<4>,id<5>>>::value, "is_set error");
    static_assert(!IsSet<type_id_sequence<id<0>,id<3>,id<3>>>::value, "is_set error");
    static_assert(!IsSet<type_id_sequence<id<1>,id<1>,id<2>,id<3>,id<4>,id<5>>>::value, "is_set error");
    static_assert(!IsSet<type_id_sequence<id<4>,id<1>,id<2>,id<3>,id<4>,id<5>>>::value, "is_set error");
    static_assert(!IsSet<type_id_sequence<id<3>,id<1>,id<2>,id<8>,id<7>,id<1>>>::value, "is_set error");


    /* append to sequence */
    template <typename, typename>
    struct AppendToSequence {};

    template <typename I0, typename... Is>
    struct AppendToSequence<I0, type_id_sequence<Is...>> {
        using type = type_id_sequence<Is..., I0>;
    };
    static_assert(
        cuda::std::is_same<
            AppendToSequence<
                id<42>,
                type_id_sequence<id<0>,id<1>,id<2>,id<3>,id<4>>
                >::type,
            type_id_sequence<id<0>,id<1>,id<2>,id<3>,id<4>,id<42>>
            >::value,
        "append_to_sequence error"
        );
    

    /* prepend to sequence */
    template <typename, typename>
    struct PrependToSequence {};

    template <typename I0, typename... Is>
    struct PrependToSequence<I0, type_id_sequence<Is...>> {
        using type = type_id_sequence<I0, Is...>;
    };
    static_assert(
        cuda::std::is_same<
            PrependToSequence<
                id<42>,
                type_id_sequence<id<0>,id<1>,id<2>,id<3>,id<4>>
                >::type,
            type_id_sequence<id<42>,id<0>,id<1>,id<2>,id<3>,id<4>>
            >::value,
        "PrependToSequence error"
        );


    /* extract unique type ids in sequence */
    template <typename>
    struct UniqueTypeIds {};

    template <typename I0, typename... Is>
    struct UniqueTypeIds<type_id_sequence<I0, Is...>> {
        using subtype = typename UniqueTypeIds<type_id_sequence<Is...>>::type;
        using type = typename cuda::std::conditional<
               id_in_sequence<I0,Is...>,
               subtype,
               typename PrependToSequence<I0, subtype>::type
        >::type;
    };

    template <typename I0>
    struct UniqueTypeIds<type_id_sequence<I0>> {
        using type = type_id_sequence<I0>;
    };
    
    template <>
    struct UniqueTypeIds<type_id_sequence<>> {
        using type = type_id_sequence<>;
    };

    static_assert(
            cuda::std::is_same<
                UniqueTypeIds<type_id_sequence<>>::type,
                type_id_sequence<>
            >::value,
            "UniqueTypeIds error"
            );
    static_assert(
            cuda::std::is_same<
                UniqueTypeIds<type_id_sequence<id<3>,id<0>,id<1>,id<1>,id<0>,id<4>>>::type,
                type_id_sequence<id<3>,id<1>,id<0>,id<4>>
            >::value,
            "UniqueTypeIds error"
            );
    

    /* merge sequences and extract set of type ids */
    template <typename, typename>
    struct MergeTypeIds {};
    
    template <typename... Is, typename... Js>
    struct MergeTypeIds<type_id_sequence<Is...>, type_id_sequence<Js...>> {
        using type = typename UniqueTypeIds<type_id_sequence<Is..., Js...>>::type;
    };
    
    static_assert(
            cuda::std::is_same<
                MergeTypeIds<
                    type_id_sequence<>,
                    type_id_sequence<>
                >::type,
                type_id_sequence<>
            >::value,
            "MergeTypeIds error"
            );
    static_assert(
            cuda::std::is_same<
                MergeTypeIds<
                    type_id_sequence<id<0>>,
                    type_id_sequence<id<1>>
                >::type,
                type_id_sequence<id<0>,id<1>>
            >::value,
            "MergeTypeIds error"
            );
    static_assert(
            cuda::std::is_same<
                MergeTypeIds<
                    type_id_sequence<id<0>,id<2>,id<1>>,
                    type_id_sequence<id<1>,id<3>,id<0>>
                >::type,
                type_id_sequence<id<2>,id<1>,id<3>,id<0>>
            >::value,
            "MergeTypeIds error"
            );

    
    /* merge two AAs */
    template <template<typename, typename...> class, typename, typename>
    struct TypeFromSequence {};

    template <template<typename, typename...> class cls, typename T, typename... Is>
    struct TypeFromSequence<cls, T, type_id_sequence<Is...>> {
        using type = cls<T,Is...>;
    };

    template <typename, typename>
    struct MergedType {};

    template <template<typename, typename...> class cls, typename T, typename... Is, typename... Js>
    struct MergedType<cls<T,Is...>, cls<T,Js...>> {
        using type = typename TypeFromSequence<
            cls,
            T,
            typename MergeTypeIds<
                type_id_sequence<Is...>,
                type_id_sequence<Js...>
            >::type
        >::type;
    };
    
    template <typename lhs, typename rhs>
    using merged_t = typename MergedType<lhs, rhs>::type;

        
    /* compute new symbol type ids */
    template <typename...> struct OpId {};

    template <typename val, int opcode>
    using unop_type_id = OpId<val,id<opcode>>;
    
    template <typename lhs, typename rhs, int opcode>
    using binop_type_id = OpId<lhs,rhs,id<opcode>>;


    /* insert new symbol into an existing AA type */
    template <typename, typename>
    struct InsertNewSymbolType {};

    template <template<typename, typename...> class cls, typename T, typename... Is, typename new_symbol>
    struct InsertNewSymbolType<cls<T,Is...>, new_symbol> {
        static_assert(!id_in_sequence<new_symbol, Is...>, "Symbol is already contained in base type.");
        using type = typename TypeFromSequence<
            cls,
            T,
            typename MergeTypeIds<
                type_id_sequence<Is...>,
                type_id_sequence<new_symbol>
            >::type
        >::type;
    };
    
    template <typename cls, typename new_symbol>
    using insert_new_symbol_t = typename InsertNewSymbolType<cls, new_symbol>::type;

} /* end of namespace detail */
} /* end of namespace cusvna */

#endif /* end of include guard: DETAIL_CUH_6K1DTOM7 */
