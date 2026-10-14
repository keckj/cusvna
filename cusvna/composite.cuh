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


namespace impl {
    /* affine form without noise symbol enclosing [c.a, c.b] through its error term */
    template <typename T>
    __host__ __device__ AA<T> constant_form(const Interval<T>& c) {
        AA<T> out;
        if (!detail::is_finite(c.a) || !detail::is_finite(c.b)) {
            out.set_entire();
            return out;
        }
        const T m = detail::add_rn(detail::mul_rn(c.a, T(0.5)), detail::mul_rn(c.b, T(0.5)));
        out.b() = m;
        out.err() = detail::fmax(detail::sub_up(c.b, m), detail::sub_up(m, c.a));
        return out;
    }
    
    template <int K>
    struct collapse_copy_functor {
        template <typename T, typename... Is, typename... Js>
        __host__ __device__ void operator()(AA<T,Is...>& out, const AA<T,Js...>& src) const {
            using I = typename AA<T,Is...>::template idx<K>;
            if constexpr (AA<T,Js...>::template has_symbol<I>()) {
                out.template x<K>() = src.template x<I>();
                out.template tag<K>() = src.template tag<I>();
            }
        }
    };
    
    template <int K>
    struct collapse_sum_functor {
        template <typename T, typename... Js, typename Target>
        __host__ __device__ void operator()(const AA<T,Js...>& src, T& sum, const Target& target) const {
            using I = typename AA<T,Js...>::template idx<K>;
            if constexpr (!Target::template has_symbol<I>()) {
                sum = detail::add_up(sum, T(::fabs(src.template x<K>())));
            }
        }
    };

    /* Convert src to the type Target: symbols of src that are not in Target are collapsed into the
       symbol NewSymbol of Target. The collapsed quantity only depends on the content of the inputs
       that determined src, new_tag must identify them. */
    template <typename Target, typename NewSymbol, typename T, typename... Js>
    __host__ __device__ Target collapse(const AA<T,Js...>& src, const detail::tag_t new_tag) {
        static_assert(!AA<T,Js...>::template has_symbol<NewSymbol>(), "The new symbol already exists.");
        Target out(src.b());
        out.err() = src.err();
        out.template constexpr_apply<collapse_copy_functor>(src);
        T sum(0);
        src.template constexpr_apply<collapse_sum_functor>(sum, out);
        out.template x<NewSymbol>() = sum;
        out.template tag<NewSymbol>() = new_tag;
        return out;
    }
}

/* Two-argument arctangent in [-pi, pi]. The input box is handled by one of the formulas
   atan(y/x) for x > 0, pi/2 - atan(x/y) for y > 0, -pi/2 - atan(x/y) for y < 0, the result is the
   whole range [-pi, pi] when the box meets the branch cut (y = 0, x <= 0). */
template <typename T, typename... Js, typename... Is>
__host__ __device__ detail::binop_result_t<AA<T,Js...>, AA<T,Is...>, op::ATAN2> atan2(const AA<T,Js...>& y, const AA<T,Is...>& x) {
    using Result = detail::binop_result_t<AA<T,Js...>, AA<T,Is...>, op::ATAN2>;
    using NewSymbol = detail::binop_type_id<AA<T,Js...>, AA<T,Is...>, op::ATAN2>;
    const detail::tag_t new_tag = x.content_hash(y.content_hash(impl::seed(op::ATAN2)));
    const Interval<T> X = x.interval();
    const Interval<T> Y = y.interval();
    if (X.a > T(0)) {
        return impl::collapse<Result, NewSymbol>(atan(y / x), new_tag);
    }
    const AA<T> half_pi = impl::constant_form(Interval<T>(detail::mul_rn(detail::fp<T>::pi_lo(), T(0.5)), 
                                                          detail::mul_rn(detail::fp<T>::pi_hi(), T(0.5))));
    if (Y.a > T(0)) {
        return impl::collapse<Result, NewSymbol>(half_pi - atan(x / y), new_tag);
    }
    if (Y.b < T(0)) {
        return impl::collapse<Result, NewSymbol>(-half_pi - atan(x / y), new_tag);
    }
    Result out(T(0));
    out.template x<NewSymbol>() = detail::fp<T>::pi_hi();
    out.template tag<NewSymbol>() = new_tag;
    return out;
}

template <typename T, typename... Js>
__host__ __device__ auto atan2(const AA<T,Js...>& y, const detail::identity_t<T>& x) { return atan2(y, AA<T>(T(x))); }

template <typename T, typename... Is>
__host__ __device__ auto atan2(const detail::identity_t<T>& y, const AA<T,Is...>& x) { return atan2(AA<T>(T(y)), x); }


/* fmod(x, y) = x - trunc(x/y)*y with an affine divisor y: exactly x - n*y (correlated with x and y) when
   trunc(x/y) = n is constant over the input box, else the range of the result. Entire if y may be 0. */
template <typename T, typename... Is, typename... Js>
__host__ __device__ detail::binop_result_t<AA<T,Is...>, AA<T,Js...>, op::FMOD_AA> fmod(const AA<T,Is...>& x, const AA<T,Js...>& y) {
    using Result = detail::binop_result_t<AA<T,Is...>, AA<T,Js...>, op::FMOD_AA>;
    using NewSymbol = detail::binop_type_id<AA<T,Is...>, AA<T,Js...>, op::FMOD_AA>;
    const detail::tag_t new_tag = y.content_hash(x.content_hash(impl::seed(op::FMOD_AA)));
    const Interval<T> X = x.interval();
    const Interval<T> Y = y.interval();
    Result out(T(0));
    if (!detail::is_finite(X) || !detail::is_finite(Y) || (Y.a <= T(0) && Y.b >= T(0))) {
        out.set_entire();
        return out;
    }
    const Interval<T> Q = detail::operator/(X, Y);
    const T n = ::trunc(Q.a);
    if (n == ::trunc(Q.b)) {
        return impl::collapse<Result, NewSymbol>(x - y * n, new_tag);
    }
    /* the result has the sign of x and is smaller than |y| in magnitude */
    const T M = detail::fmax(T(::fabs(Y.a)), T(::fabs(Y.b)));
    const AA<T> range = impl::constant_form(Interval<T>(X.a >= T(0) ? T(0) : -M, X.b <= T(0) ? T(0) : M));
    out.b() = range.b();
    out.template x<NewSymbol>() = range.err();
    out.template tag<NewSymbol>() = new_tag;
    return out;
}

template <typename T, typename... Js>
__host__ __device__ auto fmod(const detail::identity_t<T>& x, const AA<T,Js...>& y) { return fmod(AA<T>(T(x)), y); }

/* x^y = exp(y*log(x)) for x > 0 (entire otherwise) */
template <typename T, typename... Is, typename... Js>
__host__ __device__ auto pow(const AA<T,Is...>& x, const AA<T,Js...>& y) { return exp(y * log(x)); }

/* c^y = exp(y*log(c)) for a scalar c > 0 (entire otherwise) */
template <typename T, typename... Js>
__host__ __device__ auto pow(const detail::identity_t<T>& c, const AA<T,Js...>& y) {
    const T base(c);
    const Interval<T> L = (base > T(0) ? detail::log_fn().F(base) 
                                       : Interval<T>(-detail::infinity<T>(), detail::infinity<T>()));
    return exp(y * impl::constant_form(L));
}

} /* end of namespace cusvna */

#endif /* end of include guard: COMPOSITE_CUH_H5RX2LQV */
