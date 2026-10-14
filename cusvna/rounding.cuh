#ifndef ROUNDING_CUH_Q3M8ZT1C
#define ROUNDING_CUH_Q3M8ZT1C

/* Rigorous floating point bounds computed in the default round-to-nearest mode.
 *
 * Device code uses the directed rounding instructions of the hardware (__fadd_ru, __dmul_rd, ...).
 * In host code, or when CUSVNA_EMULATED_ROUNDING is defined, directed rounding is emulated:
 * error-free transformations (TwoSum, TwoProd) detect exact operations and the sign of the
 * rounding error, inexact results are moved outward by at least one ulp. Results of math
 * library functions are enclosed by a given number of ulps.
 *
 * When CUSVNA_IGNORE_ROUNDING is defined, rounding errors are ignored: directed operations are plain
 * operations, math library results are taken as exact and operations may be contracted into fused
 * multiply-adds. The results are no longer rigorous. Fast math remains forbidden: its approximate
 * math functions are far less accurate than the CUDA math library and break the enclosures.
 *
 * Assumptions of the rigorous modes (see README):
 *   - IEEE-754 binary32/binary64 arithmetic in round-to-nearest mode,
 *   - no fast-math (no flush to zero, no approximate division/sqrt/transcendentals),
 *   - no floating point contraction in host code (default with -std=c++17 on gcc),
 *     device code uses non-contractible intrinsics.
 */

#include <cuda/std/limits>
#include <cusvna/interval.cuh>

namespace cusvna {
namespace detail {

    template <typename T>
    struct fp {};

    template <>
    struct fp<float> {
        /* unit roundoff 2^-24 */
        __host__ __device__ static constexpr float u()      { return 5.9604644775390625e-08f; }
        /* smallest normal number, bounds absolute errors due to underflow (even with ftz) */
        __host__ __device__ static constexpr float eta()    { return 1.17549435082228750797e-38f; }
        /* below this threshold products may underflow and error-free transformations are not exact */
        __host__ __device__ static constexpr float tiny()   { return 1.97215226305252218e-31f; } /* 2^-102 */
        __host__ __device__ static constexpr float max()    { return 3.40282346638528859812e+38f; }
        /* pi_lo < pi < pi_hi */
        __host__ __device__ static constexpr float pi_lo()  { return 3.14159250259399414062f; }
        __host__ __device__ static constexpr float pi_hi()  { return 3.14159274101257324219f; }
    };

    template <>
    struct fp<double> {
        __host__ __device__ static constexpr double u()     { return 1.11022302462515654042e-16; } /* 2^-53 */
        __host__ __device__ static constexpr double eta()   { return 2.22507385850720138309e-308; }
        __host__ __device__ static constexpr double tiny()  { return 2.00416836000897277800e-292; } /* 2^-969 */
        __host__ __device__ static constexpr double max()   { return 1.79769313486231570815e+308; }
        __host__ __device__ static constexpr double pi_lo() { return 3.141592653589793115997963468544185161590576171875; }
        __host__ __device__ static constexpr double pi_hi() { return 3.141592653589793560087173318606801331043243408203125; }
    };

    template <typename T>
    __host__ __device__ inline T infinity() { return cuda::std::numeric_limits<T>::infinity(); }

    template <typename T>
    __host__ __device__ inline T quiet_nan() { return cuda::std::numeric_limits<T>::quiet_NaN(); }

    template <typename T>
    __host__ __device__ inline bool is_finite(const T x) { return (x - x) == T(0); }

    template <typename T>
    __host__ __device__ inline bool is_nan(const T x) { return x != x; }

    /* round-to-nearest operations that are never contracted into fused multiply-adds */
    __host__ __device__ inline float add_rn(const float a, const float b) {
#if defined(__CUDA_ARCH__) && !defined(CUSVNA_IGNORE_ROUNDING)
        return __fadd_rn(a, b);
#else
        return a + b;
#endif
    }

    __host__ __device__ inline double add_rn(const double a, const double b) {
#if defined(__CUDA_ARCH__) && !defined(CUSVNA_IGNORE_ROUNDING)
        return __dadd_rn(a, b);
#else
        return a + b;
#endif
    }

    __host__ __device__ inline float mul_rn(const float a, const float b) {
#if defined(__CUDA_ARCH__) && !defined(CUSVNA_IGNORE_ROUNDING)
        return __fmul_rn(a, b);
#else
        return a * b;
#endif
    }

    __host__ __device__ inline double mul_rn(const double a, const double b) {
#if defined(__CUDA_ARCH__) && !defined(CUSVNA_IGNORE_ROUNDING)
        return __dmul_rn(a, b);
#else
        return a * b;
#endif
    }

    template <typename T>
    __host__ __device__ inline T sub_rn(const T a, const T b) { return add_rn(a, -b); }

    template <typename T>
    __host__ __device__ inline T div_rn(const T a, const T b) { return a / b; }

    /* unlike ::fmin and ::fmax, nans propagate: they mean that a bound could not be computed */
    template <typename T>
    __host__ __device__ inline T fmin(const T a, const T b) { return (a < b || a != a ? a : b); }

    template <typename T>
    __host__ __device__ inline T fmax(const T a, const T b) { return (a > b || a != a ? a : b); }

    /* upper bound of |c - exact| where c is a result of an operation with an error of at most ulps ulp(s) */
    template <typename T>
    __host__ __device__ inline T error_bound(const T c, const int ulps=1) {
#ifdef CUSVNA_IGNORE_ROUNDING
        return T(0);
#else
        return add_rn(mul_rn(::fabs(c), T(2*ulps+1)*fp<T>::u()), fp<T>::eta());
#endif
    }

    /* up(c, k) >= c + k ulp(s), down(c, k) <= c - k ulp(s). Overflowed results are mapped to the
       largest finite value on the other side, nans and infinities pointing outward are kept. */
    template <typename T>
    __host__ __device__ inline T up(const T c, const int ulps=1) {
#ifdef CUSVNA_IGNORE_ROUNDING
        return c;
#else
        if (is_finite(c)) {
            return add_rn(c, error_bound(c, ulps));
        }
        return (c == -infinity<T>() ? -fp<T>::max() : c);
#endif
    }

    template <typename T>
    __host__ __device__ inline T down(const T c, const int ulps=1) {
#ifdef CUSVNA_IGNORE_ROUNDING
        return c;
#else
        if (is_finite(c)) {
            return sub_rn(c, error_bound(c, ulps));
        }
        return (c == infinity<T>() ? fp<T>::max() : c);
#endif
    }

    /* exact rounding error of s = add_rn(a,b), that is a + b = s + e (TwoSum, Knuth) */
    template <typename T>
    __host__ __device__ inline T two_sum_error(const T a, const T b, const T s) {
#ifdef CUSVNA_IGNORE_ROUNDING
        return T(0);
#else
        const T bb = sub_rn(s, a);
        return add_rn(sub_rn(a, sub_rn(s, bb)), sub_rn(b, bb));
#endif
    }

    /* exact rounding error of p = mul_rn(a,b), that is a * b = p + e, when no underflow occurs */
    template <typename T>
    __host__ __device__ inline T two_prod_error(const T a, const T b, const T p) {
#ifdef CUSVNA_IGNORE_ROUNDING
        return T(0);
#else
        return ::fma(a, b, -p);
#endif
    }

#if defined(CUSVNA_IGNORE_ROUNDING)
    /* rounding errors are ignored, directed rounding is round-to-nearest */
    template <typename T> __host__ __device__ inline T add_ru(const T a, const T b) { return a + b; }
    template <typename T> __host__ __device__ inline T add_rd(const T a, const T b) { return a + b; }
    template <typename T> __host__ __device__ inline T mul_ru(const T a, const T b) { return a * b; }
    template <typename T> __host__ __device__ inline T mul_rd(const T a, const T b) { return a * b; }
    template <typename T> __host__ __device__ inline T div_ru(const T a, const T b) { return a / b; }
    template <typename T> __host__ __device__ inline T div_rd(const T a, const T b) { return a / b; }
    template <typename T> __host__ __device__ inline T sqrt_ru(const T a) { return ::sqrt(a); }
    template <typename T> __host__ __device__ inline T sqrt_rd(const T a) { return ::sqrt(a); }
#elif defined(__CUDA_ARCH__) && !defined(CUSVNA_EMULATED_ROUNDING)
    /* device code uses the directed rounding modes of the hardware instructions, they are exact and
       as fast as round-to-nearest ones */
    #define CUSVNA_NATIVE_ROUNDING
    __device__ inline float  add_ru(const float a, const float b)   { return __fadd_ru(a, b); }
    __device__ inline double add_ru(const double a, const double b) { return __dadd_ru(a, b); }
    __device__ inline float  add_rd(const float a, const float b)   { return __fadd_rd(a, b); }
    __device__ inline double add_rd(const double a, const double b) { return __dadd_rd(a, b); }
    __device__ inline float  mul_ru(const float a, const float b)   { return __fmul_ru(a, b); }
    __device__ inline double mul_ru(const double a, const double b) { return __dmul_ru(a, b); }
    __device__ inline float  mul_rd(const float a, const float b)   { return __fmul_rd(a, b); }
    __device__ inline double mul_rd(const double a, const double b) { return __dmul_rd(a, b); }
    __device__ inline float  div_ru(const float a, const float b)   { return __fdiv_ru(a, b); }
    __device__ inline double div_ru(const double a, const double b) { return __ddiv_ru(a, b); }
    __device__ inline float  div_rd(const float a, const float b)   { return __fdiv_rd(a, b); }
    __device__ inline double div_rd(const double a, const double b) { return __ddiv_rd(a, b); }
    __device__ inline float  sqrt_ru(const float a)  { return __fsqrt_ru(a); }
    __device__ inline double sqrt_ru(const double a) { return __dsqrt_ru(a); }
    __device__ inline float  sqrt_rd(const float a)  { return __fsqrt_rd(a); }
    __device__ inline double sqrt_rd(const double a) { return __dsqrt_rd(a); }
#endif

    template <typename T>
    __host__ __device__ inline T add_up(const T a, const T b) {
#if defined(CUSVNA_NATIVE_ROUNDING) || defined(CUSVNA_IGNORE_ROUNDING)
        return add_ru(a, b);
#else
        const T s = add_rn(a, b);
        if (!is_finite(s)) { return up(s); }
        return (two_sum_error(a, b, s) > T(0) ? up(s) : s);
#endif
    }

    template <typename T>
    __host__ __device__ inline T add_down(const T a, const T b) {
#if defined(CUSVNA_NATIVE_ROUNDING) || defined(CUSVNA_IGNORE_ROUNDING)
        return add_rd(a, b);
#else
        const T s = add_rn(a, b);
        if (!is_finite(s)) { return down(s); }
        return (two_sum_error(a, b, s) < T(0) ? down(s) : s);
#endif
    }

    template <typename T>
    __host__ __device__ inline T sub_up(const T a, const T b) { return add_up(a, -b); }

    template <typename T>
    __host__ __device__ inline T sub_down(const T a, const T b) { return add_down(a, -b); }

    template <typename T>
    __host__ __device__ inline T mul_up(const T a, const T b) {
        if (a == T(0) || b == T(0)) { return T(0); }
#if defined(CUSVNA_NATIVE_ROUNDING) || defined(CUSVNA_IGNORE_ROUNDING)
        return mul_ru(a, b);
#else
        const T p = mul_rn(a, b);
        if (!is_finite(p) || ::fabs(p) < fp<T>::tiny()) { return up(p); }
        return (two_prod_error(a, b, p) > T(0) ? up(p) : p);
#endif
    }

    template <typename T>
    __host__ __device__ inline T mul_down(const T a, const T b) {
        if (a == T(0) || b == T(0)) { return T(0); }
#if defined(CUSVNA_NATIVE_ROUNDING) || defined(CUSVNA_IGNORE_ROUNDING)
        return mul_rd(a, b);
#else
        const T p = mul_rn(a, b);
        if (!is_finite(p) || ::fabs(p) < fp<T>::tiny()) { return down(p); }
        return (two_prod_error(a, b, p) < T(0) ? down(p) : p);
#endif
    }

    template <typename T>
    __host__ __device__ inline T div_up(const T a, const T b) {
        if (a == T(0) && b != T(0)) { return T(0); }
#if defined(CUSVNA_NATIVE_ROUNDING) || defined(CUSVNA_IGNORE_ROUNDING)
        return div_ru(a, b);
#else
        const T q = div_rn(a, b);
        if (!is_finite(q) || ::fabs(q) < fp<T>::tiny() || ::fabs(a) < fp<T>::tiny()) { return up(q); }
        /* a = q*b + r exactly, a/b - q = r/b */
        const T r = ::fma(-q, b, a);
        return ((r > T(0)) == (b > T(0)) && r != T(0) ? up(q) : q);
#endif
    }

    template <typename T>
    __host__ __device__ inline T div_down(const T a, const T b) {
        if (a == T(0) && b != T(0)) { return T(0); }
#if defined(CUSVNA_NATIVE_ROUNDING) || defined(CUSVNA_IGNORE_ROUNDING)
        return div_rd(a, b);
#else
        const T q = div_rn(a, b);
        if (!is_finite(q) || ::fabs(q) < fp<T>::tiny() || ::fabs(a) < fp<T>::tiny()) { return down(q); }
        const T r = ::fma(-q, b, a);
        return ((r > T(0)) != (b > T(0)) && r != T(0) ? down(q) : q);
#endif
    }

    template <typename T>
    __host__ __device__ inline T sqrt_up(const T a) {
        if (a <= T(0)) { return T(0); }
#if defined(CUSVNA_NATIVE_ROUNDING) || defined(CUSVNA_IGNORE_ROUNDING)
        return sqrt_ru(a);
#else
        const T s = ::sqrt(a);
        if (!is_finite(s) || a < fp<T>::tiny()) { return up(s); }
        return (::fma(-s, s, a) > T(0) ? up(s) : s);
#endif
    }

    template <typename T>
    __host__ __device__ inline T sqrt_down(const T a) {
        if (a <= T(0)) { return T(0); }
#if defined(CUSVNA_NATIVE_ROUNDING) || defined(CUSVNA_IGNORE_ROUNDING)
        return sqrt_rd(a);
#else
        const T s = ::sqrt(a);
        if (!is_finite(s) || a < fp<T>::tiny()) { return fmax(down(s), T(0)); }
        return (::fma(-s, s, a) < T(0) ? fmax(down(s), T(0)) : s);
#endif
    }

    /* Rigorous interval operations, R = [a,b] encloses the exact result. */
    template <typename T>
    using R = Interval<T>;

    template <typename T>
    __host__ __device__ inline R<T> point(const T x) { return R<T>(x, x); }

    /* enclosure of an exact value f(x) from a math library result fx with an error of at most ulps ulp(s) */
    template <typename T>
    __host__ __device__ inline R<T> enclose(const T fx, const int ulps) { return R<T>(down(fx, ulps), up(fx, ulps)); }

    template <typename T>
    __host__ __device__ inline R<T> join(const R<T>& x, const R<T>& y) { 
        /* nans propagate, they mean that an enclosure could not be computed */
        if (is_nan(x.a) || is_nan(x.b) || is_nan(y.a) || is_nan(y.b)) { return R<T>(quiet_nan<T>(), quiet_nan<T>()); }
        return R<T>(fmin(x.a, y.a), fmax(x.b, y.b)); 
    }

    template <typename T>
    __host__ __device__ inline R<T> operator+(const R<T>& x, const R<T>& y) { return R<T>(add_down(x.a, y.a), add_up(x.b, y.b)); }

    template <typename T>
    __host__ __device__ inline R<T> operator-(const R<T>& x, const R<T>& y) { return R<T>(sub_down(x.a, y.b), sub_up(x.b, y.a)); }

    template <typename T>
    __host__ __device__ inline R<T> operator-(const R<T>& x) { return R<T>(-x.b, -x.a); }

    template <typename T>
    __host__ __device__ inline R<T> operator*(const R<T>& x, const R<T>& y) {
        const T lo = fmin(fmin(mul_down(x.a, y.a), mul_down(x.a, y.b)), fmin(mul_down(x.b, y.a), mul_down(x.b, y.b)));
        const T hi = fmax(fmax(mul_up(x.a, y.a),   mul_up(x.a, y.b)),   fmax(mul_up(x.b, y.a),   mul_up(x.b, y.b)));
        return R<T>(lo, hi);
    }

    /* division by an interval that does not contain zero */
    template <typename T>
    __host__ __device__ inline R<T> operator/(const R<T>& x, const R<T>& y) {
        const T lo = fmin(fmin(div_down(x.a, y.a), div_down(x.a, y.b)), fmin(div_down(x.b, y.a), div_down(x.b, y.b)));
        const T hi = fmax(fmax(div_up(x.a, y.a),   div_up(x.a, y.b)),   fmax(div_up(x.b, y.a),   div_up(x.b, y.b)));
        return R<T>(lo, hi);
    }

    /* square root of an interval, negative parts are clamped to zero */
    template <typename T>
    __host__ __device__ inline R<T> sqrt(const R<T>& x) { return R<T>(sqrt_down(x.a), sqrt_up(x.b)); }

    template <typename T>
    __host__ __device__ inline R<T> sqr(const R<T>& x) {
        const T ma = ::fabs(x.a);
        const T mb = ::fabs(x.b);
        const T lo = (x.a <= T(0) && x.b >= T(0)) ? T(0) : mul_down(fmin(ma, mb), fmin(ma, mb));
        return R<T>(lo, mul_up(fmax(ma, mb), fmax(ma, mb)));
    }

    template <typename T>
    __host__ __device__ inline bool is_finite(const R<T>& x) { return is_finite(x.a) && is_finite(x.b); }

} /* end of namespace detail */
} /* end of namespace cusvna */

#endif /* end of include guard: ROUNDING_CUH_Q3M8ZT1C */
