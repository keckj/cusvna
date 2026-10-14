#ifndef COMPOSITE_CUH_H5RX2LQV
#define COMPOSITE_CUH_H5RX2LQV

/* Functions built on top of the elementary operations of affine forms (included by aa.cuh).
   Functions taking several arguments accept affine forms and scalars, at least one argument must be
   an affine form. */

#include <cusvna/aa.cuh>

namespace cusvna {
namespace detail {

    template <typename X>
    struct IsAA : cuda::std::false_type {};

    template <typename T, typename... Is>
    struct IsAA<AA<T,Is...>> : cuda::std::true_type {};

    template <typename... Xs>
    static constexpr bool any_aa = (IsAA<Xs>::value || ...);

    template <typename... Xs>
    static constexpr bool all_aa = (IsAA<Xs>::value && ...);

    template <typename X>
    struct Scalar { using type = X; };

    template <typename T, typename... Is>
    struct Scalar<AA<T,Is...>> { using type = T; };

    /* scalar type of the affine forms among Xs */
    template <typename... Xs>
    using scalar_t = typename cuda::std::common_type<typename Scalar<Xs>::type...>::type;

    template <bool B>
    using enable_if_t = typename cuda::std::enable_if<B, int>::type;

} /* end of namespace detail */


/* min(x, y) = (x + y - |x - y|)/2 and max(x, y) = (x + y + |x - y|)/2, so that min(x, y) + max(x, y)
   cancels to x + y. When x - y has a known sign, the result is exactly x or y (up to rounding). */
template <typename X, typename Y, detail::enable_if_t<detail::any_aa<X,Y>> = 0>
__host__ __device__ auto min(const X& x, const Y& y) {
    using T = detail::scalar_t<X,Y>;
    return ((x + y) - abs(x - y)) * T(0.5);
}

template <typename X, typename Y, detail::enable_if_t<detail::any_aa<X,Y>> = 0>
__host__ __device__ auto max(const X& x, const Y& y) {
    using T = detail::scalar_t<X,Y>;
    return ((x + y) + abs(x - y)) * T(0.5);
}

/* overloads for forms of the same type, more specialized than std::min and std::max */
template <typename T, typename... Is>
__host__ __device__ auto min(const AA<T,Is...>& x, const AA<T,Is...>& y) { return ((x + y) - abs(x - y)) * T(0.5); }

template <typename T, typename... Is>
__host__ __device__ auto max(const AA<T,Is...>& x, const AA<T,Is...>& y) { return ((x + y) + abs(x - y)) * T(0.5); }

template <typename X, typename Y, detail::enable_if_t<detail::any_aa<X,Y>> = 0>
__host__ __device__ auto fmin(const X& x, const Y& y) { return min(x, y); }

template <typename X, typename Y, detail::enable_if_t<detail::any_aa<X,Y>> = 0>
__host__ __device__ auto fmax(const X& x, const Y& y) { return max(x, y); }

/* min(max(x, lo), hi) */
template <typename X, typename L, typename H, detail::enable_if_t<detail::any_aa<X,L,H>> = 0>
__host__ __device__ auto clamp(const X& x, const L& lo, const H& hi) { return min(max(x, lo), hi); }

template <typename T, typename... Is>
__host__ __device__ auto clamp(const AA<T,Is...>& x, const AA<T,Is...>& lo, const AA<T,Is...>& hi) { return min(max(x, lo), hi); }


/* euclidean norms */
template <typename X, typename Y, detail::enable_if_t<detail::all_aa<X,Y>> = 0>
__host__ __device__ auto hypot(const X& x, const Y& y) { return sqrt(sqr(x) + sqr(y)); }

template <typename X, typename Y, typename Z, detail::enable_if_t<detail::all_aa<X,Y,Z>> = 0>
__host__ __device__ auto hypot(const X& x, const Y& y, const Z& z) { return sqrt(sqr(x) + sqr(y) + sqr(z)); }

template <typename X, typename Y, typename Z, detail::enable_if_t<detail::all_aa<X,Y,Z>> = 0>
__host__ __device__ auto norm3d(const X& x, const Y& y, const Z& z) { return hypot(x, y, z); }

/* reciprocals of euclidean norms */
template <typename X, typename Y, detail::enable_if_t<detail::all_aa<X,Y>> = 0>
__host__ __device__ auto rhypot(const X& x, const Y& y) { return rsqrt(sqr(x) + sqr(y)); }

template <typename X, typename Y, typename Z, detail::enable_if_t<detail::all_aa<X,Y,Z>> = 0>
__host__ __device__ auto rnorm3d(const X& x, const Y& y, const Z& z) { return rsqrt(sqr(x) + sqr(y) + sqr(z)); }


/* x*y + z */
template <typename X, typename Y, typename Z, detail::enable_if_t<detail::any_aa<X,Y,Z>> = 0>
__host__ __device__ auto fma(const X& x, const Y& y, const Z& z) { return x*y + z; }

/* a + t*(b - a) */
template <typename A, typename B, typename C, detail::enable_if_t<detail::any_aa<A,B,C>> = 0>
__host__ __device__ auto lerp(const A& a, const B& b, const C& t) { return a + t*(b - a); }

/* hermite interpolation t^2 (3 - 2t) with t = clamp((x - e0)/(e1 - e0), 0, 1) */
template <typename T, typename... Is>
__host__ __device__ auto smoothstep(const detail::identity_t<T>& e0, const detail::identity_t<T>& e1, const AA<T,Is...>& x) {
    const auto t = clamp((x - T(e0)) / (T(e1) - T(e0)), T(0), T(1));
    return sqr(t) * (T(3) - T(2)*t);
}

/* polynomial smooth minimum of signed distances with a blending radius k > 0:
   min(a, b) - h^2 k/4 with h = max(k - |a - b|, 0)/k */
template <typename A, typename B, typename K, detail::enable_if_t<detail::any_aa<A,B> && !detail::any_aa<K>> = 0>
__host__ __device__ auto smooth_min(const A& a, const B& b, const K& k) {
    using T = detail::scalar_t<A,B>;
    const auto h = max(T(k) - abs(a - b), T(0)) / T(k);
    return min(a, b) - sqr(h) * (T(k) * T(0.25));
}

template <typename A, typename B, typename K, detail::enable_if_t<detail::any_aa<A,B> && !detail::any_aa<K>> = 0>
__host__ __device__ auto smooth_max(const A& a, const B& b, const K& k) { return -smooth_min(-a, -b, k); }


/* Predicates: certainly_* are true only if the relation holds for all values represented by the forms,
   so !certainly_lt(x, y) does not mean x >= y. possibly_zero(x) is false only if x is certainly nonzero. */
template <typename X, typename Y, detail::enable_if_t<detail::any_aa<X,Y>> = 0>
__host__ __device__ bool certainly_lt(const X& x, const Y& y) { return (x - y).interval().b < 0; }

template <typename X, typename Y, detail::enable_if_t<detail::any_aa<X,Y>> = 0>
__host__ __device__ bool certainly_le(const X& x, const Y& y) { return (x - y).interval().b <= 0; }

template <typename X, typename Y, detail::enable_if_t<detail::any_aa<X,Y>> = 0>
__host__ __device__ bool certainly_gt(const X& x, const Y& y) { return (x - y).interval().a > 0; }

template <typename X, typename Y, detail::enable_if_t<detail::any_aa<X,Y>> = 0>
__host__ __device__ bool certainly_ge(const X& x, const Y& y) { return (x - y).interval().a >= 0; }

template <typename T, typename... Is>
__host__ __device__ bool possibly_zero(const AA<T,Is...>& x) {
    const Interval<T> i = x.interval();
    return (i.a <= T(0)) && (i.b >= T(0));
}

} /* end of namespace cusvna */

#endif /* end of include guard: COMPOSITE_CUH_H5RX2LQV */
