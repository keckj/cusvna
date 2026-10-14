#ifndef APPROX_CUH_W2K7ND4E
#define APPROX_CUH_W2K7ND4E

/* Rigorous affine approximations of elementary functions.
 *
 * For a function f and an input affine form x = x0 + rho, where x0 is the affine part (whose range
 * is X) and rho the independent rounding error term (|rho| <= err), an approximation is a tuple
 * (alpha, zeta, delta, err') such that:
 *     f(x0) - alpha*x0  belongs to  [zeta - delta, zeta + delta]   for all x0 in X  (residual)
 *     |f(x0 + rho) - f(x0)| <= err'                                                 (propagated error)
 * The residual is attached to a new noise symbol, the propagated error to the error term.
 *
 * The range of the residual g(x) = f(x) - alpha*x is computed with outward rounding by splitting X
 * into pieces where f is convex or concave: g is then bounded by its values at the endpoints on one
 * side and by a tangent of f on the other side. Pieces where f has an inflection point (known up to
 * rounding) are thin and are bounded with interval arithmetic assuming f is monotonic on them.
 *
 * Math library results are assumed to be within a given number of ulps (see the *_fn::ulps
 * constants which have been chosen as twice the maximum errors documented in the CUDA math API).
 *
 * Inputs out of the domain of a function either produce an 'entire' approximation (the result
 * can be anything) or, when the function is bounded on the part of the input inside its domain
 * (sqrt, pow, acosh), an approximation of the function restricted to its domain.
 */

#include <cusvna/rounding.cuh>

namespace cusvna {
namespace detail {

    template <typename T>
    struct Approx {
        T alpha, zeta, delta, err;
        bool entire;

        __host__ __device__ static Approx make_entire() {
            return Approx{T(0), T(0), T(0), infinity<T>(), true};
        }

        /* exact affine function alpha*x + zeta */
        __host__ __device__ static Approx make_affine(const T alpha, const T zeta, const T err) {
            return Approx{alpha, zeta, T(0), err, false};
        }
    };

    /* largest integer such that all smaller integers are exactly representable */
    template <typename T>
    __host__ __device__ inline T max_exact_integer() { return T(1) / fp<T>::u(); }

    /* bound of the propagated error for a function with lipschitz constant lip */
    template <typename T>
    __host__ __device__ inline T propagate(const T lip, const T err) {
        return (err == T(0) ? T(0) : mul_up(lip, err));
    }

    /* approximation from the slope alpha and an enclosure g of the residual */
    template <typename T>
    __host__ __device__ inline Approx<T> from_residual(const T alpha, const R<T>& g, const T err) {
        if (!is_finite(alpha) || !is_finite(g) || is_nan(err)) {
            return Approx<T>::make_entire();
        }
        const T zeta  = add_rn(mul_rn(g.a, T(0.5)), mul_rn(g.b, T(0.5)));
        const T delta = fmax(sub_up(g.b, zeta), sub_up(zeta, g.a));
        return Approx<T>{alpha, zeta, delta, err, false};
    }

    /* the affine part range enlarged by the error term */
    template <typename T>
    __host__ __device__ inline R<T> widen(const R<T>& X, const T err) {
        return R<T>(sub_down(X.a, err), add_up(X.b, err));
    }

    /* x^n for an integer n >= 0 */
    template <typename T>
    __host__ __device__ inline R<T> ipow(R<T> x, unsigned int n) {
        R<T> result = point(T(1));
        while (n) {
            if (n & 1u) { result = result * x; }
            n >>= 1;
            if (n) { x = sqr(x); }
        }
        return result;
    }

    /* slope of the secant of f over X, an approximation. For a degenerate interval the slope does not 
       matter because all symbol magnitudes are zero, zero yields the tightest enclosure. */
    template <typename Fn, typename T>
    __host__ __device__ inline T secant_slope(const Fn& fn, const R<T>& X) {
        if (X.a == X.b) {
            return T(0);
        }
        return (fn.f(X.b) - fn.f(X.a)) / (X.b - X.a);
    }

    /* approximation of the point u in [p,q] where f'(u) = alpha, the tangent at u bounds g */
    template <typename Fn, typename T>
    __host__ __device__ inline T tangent_point(const Fn& fn, const T alpha, const T p, const T q, const int curvature) {
        T u;
        if constexpr (Fn::has_dfinv) {
            u = fn.dfinv(alpha, p, q);
        }
        else {
            /* f' is increasing (convex) or decreasing (concave) on [p,q] */
            T lo = p;
            T hi = q;
            for (int i = 0; i < 16; ++i) {
                const T m = lo + (hi - lo) / T(2);
                if ((curvature > 0) == (fn.df(m) < alpha)) { lo = m; } else { hi = m; }
            }
            u = lo + (hi - lo) / T(2);
        }
        if (!(u >= p)) { u = p; } /* also handles nans */
        if (u > q) { u = q; }
        return u;
    }

    /* Enclosure of g(x) = f(x) - alpha*x over [p,q].
       curvature: +1 if f is convex on [p,q], -1 if concave, 0 if f is monotonic on [p,q]. */
    template <typename Fn, typename T>
    __host__ __device__ inline R<T> residual_piece(const Fn& fn, const T alpha, const T p, const T q, const int curvature) {
        const R<T> A = point(alpha);
        const R<T> P = point(p);
        const R<T> Q = point(q);
        const R<T> Fp = fn.F(p);
        const R<T> Fq = fn.F(q);

        if (curvature == 0) {
            return hull(Fp, Fq) - A * R<T>(p, q);
        }

        /* convex: g is maximal at an endpoint, f(x) >= f(u) + f'(u)(x-u) for all x
           concave: g is minimal at an endpoint, f(x) <= f(u) + f'(u)(x-u) for all x */
        const R<T> gp = Fp - A * P;
        const R<T> gq = Fq - A * Q;
        const T u = tangent_point(fn, alpha, p, q, curvature);
        const R<T> U = point(u);
        /* g(x) >= f(u) - alpha*u + (f'(u) - alpha)*(x - u) for convex f (<= for concave f) */
        const R<T> t = (fn.F(u) - A * U) + (fn.DF(u) - A) * (R<T>(p, q) - U);
        if (curvature > 0) {
            return R<T>(t.a, fmax(gp.b, gq.b));
        }
        else {
            return R<T>(fmin(gp.a, gq.a), t.b);
        }
    }

    /* enclosure of g over X when the curvature of f changes sign at 0 only */
    template <typename Fn, typename T>
    __host__ __device__ inline R<T> residual_split0(const Fn& fn, const T alpha, const R<T>& X,
            const int curvature_neg, const int curvature_pos) {
        if (X.b <= T(0)) {
            return residual_piece(fn, alpha, X.a, X.b, curvature_neg);
        }
        if (X.a >= T(0)) {
            return residual_piece(fn, alpha, X.a, X.b, curvature_pos);
        }
        return hull(residual_piece(fn, alpha, X.a, T(0), curvature_neg),
                    residual_piece(fn, alpha, T(0), X.b, curvature_pos));
    }

    /* Enclosure of g over X for trigonometric functions whose special points (inflection points
       and poles) are multiples of pi/2. Returns false if X contains a pole or is too large. */
    template <typename Fn, typename T>
    __host__ __device__ inline bool residual_periodic(const Fn& fn, const T alpha, const R<T>& X, R<T>& g) {
        const R<T> half_pi(mul_rn(fp<T>::pi_lo(), T(0.5)), mul_rn(fp<T>::pi_hi(), T(0.5)));
        const T mf = ::floor(X.a / half_pi.a) - T(1);
        if (!(::fabs(mf) + T(16) < max_exact_integer<T>())) {
            return false;
        }
        const long long m0 = (long long)(mf);

        bool first = true;
        T cursor = X.a;
        for (int i = 0; i < 8; ++i) {
            const long long m = m0 + i;
            const R<T> S = point(T(m)) * half_pi;
            if (i == 0) {
                if (!(S.b < X.a)) { return false; }
                continue;
            }
            if (S.b < cursor) {
                continue;
            }
            if (S.a > X.b) {
                const R<T> piece = residual_piece(fn, alpha, cursor, X.b, fn.curvature(m-1));
                g = (first ? piece : hull(g, piece));
                return true;
            }
            const int kind = fn.kind(m);
            if (kind == Fn::REGULAR) {
                continue;
            }
            if (kind == Fn::POLE || !(S.b - S.a < T(0.125))) {
                return false;
            }
            /* inflection point, f is monotonic around S */
            if (S.a > cursor) {
                const R<T> piece = residual_piece(fn, alpha, cursor, S.a, fn.curvature(m-1));
                g = (first ? piece : hull(g, piece));
                first = false;
            }
            const R<T> sliver = residual_piece(fn, alpha, fmax(S.a, cursor), fmin(S.b, X.b), 0);
            g = (first ? sliver : hull(g, sliver));
            first = false;
            cursor = S.b;
            if (cursor >= X.b) {
                return true;
            }
        }
        return false;
    }

    /* positive modulo */
    __host__ __device__ inline int pmod(const long long m, const int n) {
        const int r = int(m % n);
        return (r < 0 ? r + n : r);
    }


    /*** function descriptors: f and f' (approximations), F and DF (rigorous enclosures at a point) ***/

    struct exp_fn {
        static constexpr int ulps = 4;
        static constexpr bool has_dfinv = true;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::exp(x); }
        template <typename T> __host__ __device__ T df(const T x) const { return ::exp(x); }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return enclose(T(::exp(x)), ulps); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return F(x); }
        template <typename T> __host__ __device__ T dfinv(const T d, const T p, const T q) const { return ::log(d); }
    };

    struct log_fn {
        static constexpr int ulps = 4;
        static constexpr bool has_dfinv = true;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::log(x); }
        template <typename T> __host__ __device__ T df(const T x) const { return T(1) / x; }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return enclose(T(::log(x)), ulps); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return point(T(1)) / point(x); }
        template <typename T> __host__ __device__ T dfinv(const T d, const T p, const T q) const { return T(1) / d; }
    };

    struct sqrt_fn {
        static constexpr bool has_dfinv = true;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::sqrt(x); }
        template <typename T> __host__ __device__ T df(const T x) const { return T(0.5) / ::sqrt(x); }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return R<T>(sqrt_down(x), sqrt_up(x)); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return point(T(0.5)) / F(x); }
        template <typename T> __host__ __device__ T dfinv(const T d, const T p, const T q) const { return T(0.25) / (d*d); }
    };

    struct reciprocal_fn {
        static constexpr bool has_dfinv = true;
        template <typename T> __host__ __device__ T f(const T x)  const { return T(1) / x; }
        template <typename T> __host__ __device__ T df(const T x) const { return -T(1) / (x*x); }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return point(T(1)) / point(x); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return -sqr(F(x)); }
        template <typename T> __host__ __device__ T dfinv(const T d, const T p, const T q) const {
            return (p < T(0) ? -T(1) : T(1)) / ::sqrt(-d);
        }
    };

    /* x^n for an integer n */
    struct pown_fn {
        int n;
        static constexpr bool has_dfinv = false;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::pow(x, T(n)); }
        template <typename T> __host__ __device__ T df(const T x) const { return T(n) * ::pow(x, T(n-1)); }
        template <typename T> __host__ __device__ R<T> F(const T x) const { return pow(x, n); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return point(T(n)) * pow(x, n-1); }
        template <typename T> __host__ __device__ static R<T> pow(const T x, const int k) {
            return (k >= 0 ? ipow(point(x), (unsigned int)(k)) : point(T(1)) / ipow(point(x), (unsigned int)(-k)));
        }
    };

    /* x^y for a real y, x >= 0 */
    template <typename T>
    struct pow_fn {
        T y;
        static constexpr int ulps = 16;
        static constexpr bool has_dfinv = false;
        __host__ __device__ T f(const T x)  const { return ::pow(x, y); }
        __host__ __device__ T df(const T x) const { return y * ::pow(x, y - T(1)); }
        __host__ __device__ R<T> F(const T x)  const { return enclose(T(::pow(x, y)), ulps); }
        __host__ __device__ R<T> DF(const T x) const {
            if (x == T(0)) {
                return (y > T(1) ? point(T(0)) : point(infinity<T>()));
            }
            return point(y) * F(x) / point(x);
        }
    };

    /* kinds of the multiples of pi/2 for trigonometric functions */
    struct trigo_fn {
        static constexpr int REGULAR = 0;
        static constexpr int INFLECTION = 1;
        static constexpr int POLE = 2;
        static constexpr bool has_dfinv = false;
    };

    struct sin_fn : trigo_fn {
        static constexpr int ulps = 4;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::sin(x); }
        template <typename T> __host__ __device__ T df(const T x) const { return ::cos(x); }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return enclose(T(::sin(x)), ulps); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return enclose(T(::cos(x)), ulps); }
        /* sin(m*pi/2) = 0 for even m */
        __host__ __device__ int kind(const long long m) const { return (pmod(m, 2) == 0 ? INFLECTION : REGULAR); }
        /* curvature on (m*pi/2, (m+1)*pi/2): sin'' = -sin */
        __host__ __device__ int curvature(const long long m) const { return (pmod(m, 4) < 2 ? -1 : +1); }
    };

    struct cos_fn : trigo_fn {
        static constexpr int ulps = 4;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::cos(x); }
        template <typename T> __host__ __device__ T df(const T x) const { return -::sin(x); }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return enclose(T(::cos(x)), ulps); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return enclose(T(-::sin(x)), ulps); }
        __host__ __device__ int kind(const long long m) const { return (pmod(m, 2) == 1 ? INFLECTION : REGULAR); }
        /* cos > 0 on (-pi/2, pi/2), cos'' = -cos */
        __host__ __device__ int curvature(const long long m) const { return (pmod(m, 4) == 0 || pmod(m, 4) == 3 ? -1 : +1); }
    };

    struct tan_fn : trigo_fn {
        static constexpr int ulps = 8;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::tan(x); }
        template <typename T> __host__ __device__ T df(const T x) const { const T t = ::tan(x); return T(1) + t*t; }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return enclose(T(::tan(x)), ulps); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return point(T(1)) + sqr(F(x)); }
        __host__ __device__ int kind(const long long m) const { return (pmod(m, 2) == 0 ? INFLECTION : POLE); }
        /* tan > 0 on (0, pi/2), tan'' = 2 tan (1 + tan^2) */
        __host__ __device__ int curvature(const long long m) const { return (pmod(m, 2) == 0 ? +1 : -1); }
    };

    struct cot_fn : trigo_fn {
        static constexpr int ulps = 4;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::cos(x) / ::sin(x); }
        template <typename T> __host__ __device__ T df(const T x) const { const T c = f(x); return -T(1) - c*c; }
        template <typename T> __host__ __device__ R<T> F(const T x)  const {
            return enclose(T(::cos(x)), ulps) / enclose(T(::sin(x)), ulps);
        }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return -(point(T(1)) + sqr(F(x))); }
        __host__ __device__ int kind(const long long m) const { return (pmod(m, 2) == 1 ? INFLECTION : POLE); }
        /* cot > 0 on (0, pi/2), cot'' = 2 cot (1 + cot^2) */
        __host__ __device__ int curvature(const long long m) const { return (pmod(m, 2) == 0 ? +1 : -1); }
    };

    struct sinh_fn {
        static constexpr int ulps = 6;
        static constexpr bool has_dfinv = false;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::sinh(x); }
        template <typename T> __host__ __device__ T df(const T x) const { return ::cosh(x); }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return enclose(T(::sinh(x)), ulps); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return enclose(T(::cosh(x)), ulps); }
    };

    struct cosh_fn {
        static constexpr int ulps = 6;
        static constexpr bool has_dfinv = false;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::cosh(x); }
        template <typename T> __host__ __device__ T df(const T x) const { return ::sinh(x); }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return enclose(T(::cosh(x)), ulps); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return enclose(T(::sinh(x)), ulps); }
    };

    struct tanh_fn {
        static constexpr int ulps = 4;
        static constexpr bool has_dfinv = false;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::tanh(x); }
        template <typename T> __host__ __device__ T df(const T x) const { const T t = ::tanh(x); return T(1) - t*t; }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return enclose(T(::tanh(x)), ulps); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return point(T(1)) - sqr(F(x)); }
    };

    struct asinh_fn {
        static constexpr int ulps = 6;
        static constexpr bool has_dfinv = false;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::asinh(x); }
        template <typename T> __host__ __device__ T df(const T x) const { return T(1) / ::sqrt(T(1) + x*x); }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return enclose(T(::asinh(x)), ulps); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return point(T(1)) / sqrt(point(T(1)) + sqr(point(x))); }
    };

    struct acosh_fn {
        static constexpr int ulps = 8;
        static constexpr bool has_dfinv = false;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::acosh(x); }
        template <typename T> __host__ __device__ T df(const T x) const { return T(1) / ::sqrt(x*x - T(1)); }
        template <typename T> __host__ __device__ R<T> F(const T x)  const {
            const R<T> y = enclose(T(::acosh(x)), ulps);
            return R<T>(fmax(y.a, T(0)), y.b);
        }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return point(T(1)) / sqrt(sqr(point(x)) - point(T(1))); }
    };

    struct atanh_fn {
        static constexpr int ulps = 6;
        static constexpr bool has_dfinv = false;
        template <typename T> __host__ __device__ T f(const T x)  const { return ::atanh(x); }
        template <typename T> __host__ __device__ T df(const T x) const { return T(1) / (T(1) - x*x); }
        template <typename T> __host__ __device__ R<T> F(const T x)  const { return enclose(T(::atanh(x)), ulps); }
        template <typename T> __host__ __device__ R<T> DF(const T x) const { return point(T(1)) / (point(T(1)) - sqr(point(x))); }
    };


    /*** approximations, X is the range of the affine part and err the input error term ***/

    template <typename T>
    __host__ __device__ inline Approx<T> approx_exp(const R<T>& X, const T err) {
        if (!is_finite(X)) { return Approx<T>::make_entire(); }
        const exp_fn fn;
        const T alpha = secant_slope(fn, X);
        const R<T> g = residual_piece(fn, alpha, X.a, X.b, +1);
        const T lip = fn.F(add_up(X.b, err)).b;
        return from_residual(alpha, g, propagate(lip, err));
    }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_log(const R<T>& X, const T err) {
        if (!is_finite(X) || !(X.a > T(0))) { return Approx<T>::make_entire(); }
        const log_fn fn;
        const T alpha = secant_slope(fn, X);
        const R<T> g = residual_piece(fn, alpha, X.a, X.b, -1);
        const T xmin = sub_down(X.a, err);
        const T lip = (xmin > T(0) ? div_up(T(1), xmin) : infinity<T>());
        return from_residual(alpha, g, propagate(lip, err));
    }

    /* the input is restricted to x >= 0 */
    template <typename T>
    __host__ __device__ inline Approx<T> approx_sqrt(const R<T>& Xin, const T err) {
        if (!is_finite(Xin) || Xin.b < T(0)) { return Approx<T>::make_entire(); }
        const R<T> X(fmax(Xin.a, T(0)), Xin.b);
        /* |sqrt(x+r) - sqrt(x)| <= sqrt(|r|) */
        const T holder = (err == T(0) ? T(0) : sqrt_up(err));
        if (X.b == T(0)) {
            return Approx<T>::make_affine(T(0), T(0), holder);
        }
        const sqrt_fn fn;
        const T alpha = secant_slope(fn, X);
        const R<T> g = residual_piece(fn, alpha, X.a, X.b, -1);
        const T xmin = sub_down(X.a, err);
        const T lip = (xmin > T(0) ? div_up(T(1), mul_down(T(2), sqrt_down(xmin))) : infinity<T>());
        return from_residual(alpha, g, fmin(propagate(lip, err), holder));
    }

    /* min-range approximation */
    template <typename T>
    __host__ __device__ inline Approx<T> approx_reciprocal(const R<T>& X, const T err) {
        if (!is_finite(X) || (X.a <= T(0) && X.b >= T(0))) { return Approx<T>::make_entire(); }
        const reciprocal_fn fn;
        const bool positive = (X.a > T(0));
        const T alpha = (X.a == X.b ? T(0) : fn.df(positive ? X.b : X.a));
        const R<T> g = residual_piece(fn, alpha, X.a, X.b, positive ? +1 : -1);
        const R<T> Xf = widen(X, err);
        T lip = infinity<T>();
        if (Xf.a > T(0) || Xf.b < T(0)) {
            const T m = fmin(::fabs(Xf.a), ::fabs(Xf.b));
            lip = div_up(T(1), mul_down(m, m));
        }
        return from_residual(alpha, g, propagate(lip, err));
    }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_pown(const R<T>& X, const T err, const int n) {
        if (n == 0) { return Approx<T>::make_affine(T(0), T(1), T(0)); }
        if (n == 1) { return Approx<T>::make_affine(T(1), T(0), err); }
        if (!is_finite(X) || (n < 0 && X.a <= T(0) && X.b >= T(0))) { return Approx<T>::make_entire(); }

        const pown_fn fn{n};
        const bool even = (n % 2 == 0);
        const T alpha = secant_slope(fn, X);
        R<T> g;
        if (X.a >= T(0)) {
            g = residual_piece(fn, alpha, X.a, X.b, +1);
        }
        else if (X.b <= T(0)) {
            g = residual_piece(fn, alpha, X.a, X.b, even ? +1 : -1);
        }
        else {
            g = (even ? residual_piece(fn, alpha, X.a, X.b, +1) : residual_split0(fn, alpha, X, -1, +1));
        }

        const R<T> Xf = widen(X, err);
        T lip = infinity<T>();
        if (n > 0) {
            const T M = fmax(::fabs(Xf.a), ::fabs(Xf.b));
            lip = (point(T(n)) * pown_fn::pow(M, n-1)).b;
        }
        else if (Xf.a > T(0) || Xf.b < T(0)) {
            const T m = fmin(::fabs(Xf.a), ::fabs(Xf.b));
            lip = (point(T(-n)) * pown_fn::pow(m, n-1)).b;
        }
        return from_residual(alpha, g, propagate(lip, err));
    }

    /* the input is restricted to x >= 0 (x > 0 for negative exponents) */
    template <typename T>
    __host__ __device__ inline Approx<T> approx_pow(const R<T>& Xin, const T err, const T y) {
        if (!is_finite(y)) { return Approx<T>::make_entire(); }
        if (y == T(0)) { return Approx<T>::make_affine(T(0), T(1), T(0)); }
        if (y == T(1)) { return Approx<T>::make_affine(T(1), T(0), err); }
        if (!is_finite(Xin) || Xin.b < T(0) || (y < T(0) && Xin.a <= T(0))) { return Approx<T>::make_entire(); }

        const pow_fn<T> fn{y};
        const R<T> X(fmax(Xin.a, T(0)), Xin.b);
        const R<T> Xf = widen(X, err);
        T prop = infinity<T>();
        if (y > T(1)) {
            const T M = fmax(::fabs(Xf.a), ::fabs(Xf.b));
            prop = propagate(fn.DF(M).b, err);
        }
        else if (y > T(0)) {
            /* |x^y - z^y| <= |x - z|^y */
            const T holder = (err == T(0) ? T(0) : fn.F(err).b);
            const T lip = (Xf.a > T(0) ? fn.DF(Xf.a).b : infinity<T>());
            prop = fmin(propagate(lip, err), holder);
        }
        else if (Xf.a > T(0)) {
            prop = propagate(T(-1)*fn.DF(Xf.a).a, err);
        }

        if (X.b == T(0)) {
            return Approx<T>::make_affine(T(0), T(0), prop);
        }
        const T alpha = secant_slope(fn, X);
        const R<T> g = residual_piece(fn, alpha, X.a, X.b, (y > T(1) || y < T(0)) ? +1 : -1);
        return from_residual(alpha, g, prop);
    }

    template <typename Fn, typename T>
    __host__ __device__ inline Approx<T> approx_bounded_trigo(const R<T>& X, const T err) {
        /* the whole range [-1,1], the input error does not matter */
        const Approx<T> full{T(0), T(0), T(1), T(0), false};
        if (!is_finite(X) || !(sub_up(add_up(X.b, err), sub_down(X.a, err)) < mul_rn(T(2), fp<T>::pi_lo()))) {
            return full;
        }
        const Fn fn;
        const T alpha = secant_slope(fn, X);
        R<T> g;
        if (!residual_periodic(fn, alpha, X, g)) {
            return full;
        }
        return from_residual(alpha, g, propagate(T(1), err));
    }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_sin(const R<T>& X, const T err) { return approx_bounded_trigo<sin_fn>(X, err); }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_cos(const R<T>& X, const T err) { return approx_bounded_trigo<cos_fn>(X, err); }

    template <typename Fn, typename T>
    __host__ __device__ inline Approx<T> approx_unbounded_trigo(const R<T>& X, const T err) {
        if (!is_finite(X)) { return Approx<T>::make_entire(); }
        const Fn fn;
        const R<T> Xf = widen(X, err);
        R<T> g;
        /* the widened input must not contain any pole, the derivative is then maximal at an endpoint */
        if (!is_finite(Xf) || !residual_periodic(fn, T(0), Xf, g)) {
            return Approx<T>::make_entire();
        }
        const T lip = fmax(::fabs(fn.DF(Xf.a).a), ::fabs(fn.DF(Xf.b).b));
        const T lip2 = fmax(::fabs(fn.DF(Xf.a).b), ::fabs(fn.DF(Xf.b).a));
        const T alpha = secant_slope(fn, X);
        if (!residual_periodic(fn, alpha, X, g)) {
            return Approx<T>::make_entire();
        }
        return from_residual(alpha, g, propagate(fmax(lip, lip2), err));
    }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_tan(const R<T>& X, const T err) { return approx_unbounded_trigo<tan_fn>(X, err); }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_cot(const R<T>& X, const T err) { return approx_unbounded_trigo<cot_fn>(X, err); }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_sinh(const R<T>& X, const T err) {
        if (!is_finite(X)) { return Approx<T>::make_entire(); }
        const sinh_fn fn;
        const T alpha = secant_slope(fn, X);
        const R<T> g = residual_split0(fn, alpha, X, -1, +1);
        const R<T> Xf = widen(X, err);
        const T lip = cosh_fn().F(fmax(::fabs(Xf.a), ::fabs(Xf.b))).b;
        return from_residual(alpha, g, propagate(lip, err));
    }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_cosh(const R<T>& X, const T err) {
        if (!is_finite(X)) { return Approx<T>::make_entire(); }
        const cosh_fn fn;
        const T alpha = secant_slope(fn, X);
        const R<T> g = residual_piece(fn, alpha, X.a, X.b, +1);
        const R<T> Xf = widen(X, err);
        const T lip = sinh_fn().F(fmax(::fabs(Xf.a), ::fabs(Xf.b))).b;
        return from_residual(alpha, g, propagate(lip, err));
    }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_tanh(const R<T>& X, const T err) {
        if (!is_finite(X)) { return Approx<T>::make_entire(); }
        const tanh_fn fn;
        const T alpha = secant_slope(fn, X);
        const R<T> g = residual_split0(fn, alpha, X, +1, -1);
        return from_residual(alpha, g, propagate(T(1), err));
    }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_asinh(const R<T>& X, const T err) {
        if (!is_finite(X)) { return Approx<T>::make_entire(); }
        const asinh_fn fn;
        const T alpha = secant_slope(fn, X);
        const R<T> g = residual_split0(fn, alpha, X, +1, -1);
        return from_residual(alpha, g, propagate(T(1), err));
    }

    /* the input is restricted to x >= 1 */
    template <typename T>
    __host__ __device__ inline Approx<T> approx_acosh(const R<T>& Xin, const T err) {
        if (!is_finite(Xin) || Xin.b < T(1)) { return Approx<T>::make_entire(); }
        const acosh_fn fn;
        const R<T> X(fmax(Xin.a, T(1)), Xin.b);
        /* acosh is concave and increasing: |acosh(x+r) - acosh(x)| <= acosh(1+|r|) */
        const T holder = (err == T(0) ? T(0) : fn.F(add_up(T(1), err)).b);
        const T xmin = sub_down(X.a, err);
        const T lip = (xmin > T(1) ? fn.DF(xmin).b : infinity<T>());
        const T prop = fmin(propagate(lip, err), holder);
        if (X.b == T(1)) {
            return Approx<T>::make_affine(T(0), T(0), prop);
        }
        const T alpha = secant_slope(fn, X);
        const R<T> g = residual_piece(fn, alpha, X.a, X.b, -1);
        return from_residual(alpha, g, prop);
    }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_atanh(const R<T>& X, const T err) {
        if (!is_finite(X) || X.a <= -T(1) || X.b >= T(1)) { return Approx<T>::make_entire(); }
        const atanh_fn fn;
        const T alpha = secant_slope(fn, X);
        const R<T> g = residual_split0(fn, alpha, X, -1, +1);
        const R<T> Xf = widen(X, err);
        const T M = fmax(::fabs(Xf.a), ::fabs(Xf.b));
        const T lip = (M < T(1) ? fn.DF(M).b : infinity<T>());
        return from_residual(alpha, g, propagate(lip, err));
    }

    /* convex piecewise linear functions f with f(0) = 0, f(x) = sa*x for x < 0 and f(x) = sb*x for x > 0 */
    template <typename T>
    __host__ __device__ inline Approx<T> approx_kink(const R<T>& X, const T err, const T sa, const T sb) {
        if (!is_finite(X)) { return Approx<T>::make_entire(); }
        const T prop = propagate(fmax(::fabs(sa), ::fabs(sb)), err);
        if (X.a >= T(0)) { return Approx<T>::make_affine(sb, T(0), prop); }
        if (X.b <= T(0)) { return Approx<T>::make_affine(sa, T(0), prop); }
        /* g(x) = f(x) - alpha*x is minimal at 0 for sa <= alpha <= sb */
        T alpha = (sb*X.b - sa*X.a) / (X.b - X.a);
        alpha = fmin(fmax(alpha, sa), sb);
        const R<T> A = point(alpha);
        const R<T> ga = (point(sa) - A) * point(X.a);
        const R<T> gb = (point(sb) - A) * point(X.b);
        const R<T> g(T(0), fmax(ga.b, gb.b));
        return from_residual(alpha, g, prop);
    }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_abs(const R<T>& X, const T err) { return approx_kink(X, err, T(-1), T(1)); }

    template <typename T>
    __host__ __device__ inline Approx<T> approx_positive_part(const R<T>& X, const T err) { return approx_kink(X, err, T(0), T(1)); }

} /* end of namespace detail */
} /* end of namespace cusvna */

#endif /* end of include guard: APPROX_CUH_W2K7ND4E */
