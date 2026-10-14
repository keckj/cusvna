#ifndef AF1_CUH_Q8ZK3MTD
#define AF1_CUH_Q8ZK3MTD

/* AF1 affine forms (Messine 2002): a fixed number N of input noise symbols and a single error term.
 *
 * x = b + sum_k x[k] eps_k + err eps_err where all eps are unknown in [-1,1]. The N input symbols are
 * the only ones that correlate forms, every non-affine operation (product of forms, nonlinear functions)
 * and every rounding error goes into the error term err, which is never correlated with anything.
 *
 * Compared to AA, forms have a constant size of N+2 scalars and no symbol bookkeeping (no symbol types,
 * no tags), which is cheaper on GPUs, at the cost of losing the correlations between the generated
 * symbols (x*y - x*y does not cancel the quadratic term of x*y, for instance).
 *
 * The same polynomial approximations (approx.cuh) and rounding rules (rounding.cuh) as AA are used:
 * all operations are rigorous and the exact real result always belongs to the range of the form. */

#ifndef __CUDACC_RTC__
#include <ostream>
#endif

#include <cusvna/aa.cuh>

namespace cusvna {

namespace detail {
    /* integer index K of a noise symbol type id<K> */
    template <typename I>
    struct IdIndex { static constexpr int value = -1; };

    template <int K>
    struct IdIndex<id<K>> { static constexpr int value = K; };
}

template <typename T, int N>
class AF1 {
    static_assert(N >= 0, "The number of noise symbols must be nonnegative.");

    public:
        static constexpr int symbols = N;  /* number of input noise symbols */
        using type = T;

        __host__ __device__ AF1(const T& b=T(0));
        AF1(const AF1& other) = default;
        AF1& operator=(const AF1& other) = default;

        /* input variable enclosing interval through noise symbol k (0 <= k < N) */
        __host__ __device__ static AF1 variable(int k, const Interval<T>& interval);

        /* same as variable(K, interval) with a compile time check, this mirrors AA::from_interval<K> */
        template <int K> __host__ __device__ static AF1 from_interval(const Interval<T>& interval);

        /* uncorrelated form enclosing interval through its error term */
        __host__ __device__ static AF1 constant(const Interval<T>& interval);

        /* convert back an affine form to an interval, an entire form yields [-inf, inf] */
                 __host__ __device__ Interval<T> interval() const;
        explicit __host__ __device__ operator Interval<T>() const;

        /* range of the affine part (without the error term) */
        __host__ __device__ Interval<T> affine_interval() const;

        /* base coefficient of the affine form */
        __host__ __device__       T& b();
        __host__ __device__ const T& b() const;

        /* magnitude of the uncertainty corresponding to input noise symbol K (or k) */
        template <int K> __host__ __device__       T& x();
        template <int K> __host__ __device__ const T& x() const;
                         __host__ __device__       T& x(int k);
                         __host__ __device__ const T& x(int k) const;

        /* magnitude (>= 0) of the independent error term */
        __host__ __device__       T& err();
        __host__ __device__ const T& err() const;

        /* has_symbol<I> is true if I = detail::id<K> with 0 <= K < N */
        template <typename I> __host__ __device__ static constexpr bool has_symbol();

        /* symbol_magnitude<I,D> returns x<K>() for I = detail::id<K> if it exists else T(D), as AA::symbol_magnitude */
        template <typename I, int D=0> __host__ __device__ T symbol_magnitude() const;

        /* utility functions */
        __host__ __device__ T radius() const;             /* upper bound of the sum of absolute symbol magnitudes and error term */
        __host__ __device__ T affine_radius() const;      /* upper bound of the sum of absolute symbol magnitudes */
        __host__ __device__ Sign sign() const;            /* shortcut for Interval<T>(*this).sign() */
        __host__ __device__ bool is_entire() const;       /* true if the range of the form is unbounded or unknown */
        __host__ __device__ void set_entire();            /* the form may represent any value */
        __host__ __device__ void print() const;           /* alternative to std::cout to print on device */

        __host__ __device__ AF1 operator-() const;
        __host__ __device__ AF1 operator+() const;

        __host__ __device__ AF1& operator-=(const T& other);
        __host__ __device__ AF1& operator+=(const T& other);
        __host__ __device__ AF1& operator*=(const T& other);
        __host__ __device__ AF1& operator/=(const T& other);

        __host__ __device__ AF1& operator-=(const AF1& other);
        __host__ __device__ AF1& operator+=(const AF1& other);
        __host__ __device__ AF1& operator*=(const AF1& other);

    protected:
        T m_b;
        cuda::std::array<T,N> m_x;
        T m_err;
};


/* arithmetic operators */
template <typename T, int N> __host__ __device__ AF1<T,N> operator+(const AF1<T,N>& lhs, const AF1<T,N>& rhs);
template <typename T, int N> __host__ __device__ AF1<T,N> operator+(const AF1<T,N>& lhs, const detail::identity_t<T>& rhs);
template <typename T, int N> __host__ __device__ AF1<T,N> operator+(const detail::identity_t<T>& lhs, const AF1<T,N>& rhs);

template <typename T, int N> __host__ __device__ AF1<T,N> operator-(const AF1<T,N>& lhs, const AF1<T,N>& rhs);
template <typename T, int N> __host__ __device__ AF1<T,N> operator-(const AF1<T,N>& lhs, const detail::identity_t<T>& rhs);
template <typename T, int N> __host__ __device__ AF1<T,N> operator-(const detail::identity_t<T>& lhs, const AF1<T,N>& rhs);

template <typename T, int N> __host__ __device__ AF1<T,N> operator*(const AF1<T,N>& lhs, const AF1<T,N>& rhs);
template <typename T, int N> __host__ __device__ AF1<T,N> operator*(const AF1<T,N>& lhs, const detail::identity_t<T>& rhs);
template <typename T, int N> __host__ __device__ AF1<T,N> operator*(const detail::identity_t<T>& lhs, const AF1<T,N>& rhs);

template <typename T, int N> __host__ __device__ AF1<T,N> operator/(const AF1<T,N>& lhs, const AF1<T,N>& rhs);
template <typename T, int N> __host__ __device__ AF1<T,N> operator/(const AF1<T,N>& lhs, const detail::identity_t<T>& rhs);
template <typename T, int N> __host__ __device__ AF1<T,N> operator/(const detail::identity_t<T>& lhs, const AF1<T,N>& rhs);

/* The elementary functions of AA (reciprocal, sqrt, exp, log, pow, sqr, sin, ..., floor, fract, fmod, modf,
   remainder, remquo, tgamma, lgamma) are provided with the same semantics, see aa.cuh, the composite functions
   of composite.cuh as well. */

#ifndef __CUDACC_RTC__
    template <typename T, int N>
    __host__ std::ostream& operator<<(std::ostream& os, const AF1<T,N>& val);
#endif


/* implementation */
template <typename T, int N>
AF1<T,N>::AF1(const T& b): m_b(b), m_x{}, m_err(0) {}

template <typename T, int N>
AF1<T,N> AF1<T,N>::variable(const int k, const Interval<T>& interval) {
    AF1 out;
    if (!detail::is_finite(interval.a) || !detail::is_finite(interval.b)) {
        out.set_entire();
        return out;
    }
    /* enclose [a,b] */
    const T m = detail::add_rn(detail::mul_rn(interval.a, T(0.5)), detail::mul_rn(interval.b, T(0.5)));
    out.m_b = m;
    out.m_x[k] = detail::fmax(detail::sub_up(interval.b, m), detail::sub_up(m, interval.a));
    return out;
}

template <typename T, int N>
template <int K>
AF1<T,N> AF1<T,N>::from_interval(const Interval<T>& interval) {
    static_assert(K >= 0 && K < N, "Noise symbol index out of range.");
    return variable(K, interval);
}

template <typename T, int N>
AF1<T,N> AF1<T,N>::constant(const Interval<T>& interval) {
    AF1 out;
    if (!detail::is_finite(interval.a) || !detail::is_finite(interval.b)) {
        out.set_entire();
        return out;
    }
    const T m = detail::add_rn(detail::mul_rn(interval.a, T(0.5)), detail::mul_rn(interval.b, T(0.5)));
    out.m_b = m;
    out.m_err = detail::fmax(detail::sub_up(interval.b, m), detail::sub_up(m, interval.a));
    return out;
}

template <typename T, int N>
T& AF1<T,N>::b() { return m_b; }

template <typename T, int N>
const T& AF1<T,N>::b() const { return m_b; }

template <typename T, int N>
template <int K>
T& AF1<T,N>::x() {
    static_assert(K >= 0 && K < N, "Noise symbol index out of range.");
    return cuda::std::get<K>(m_x);
}

template <typename T, int N>
template <int K>
const T& AF1<T,N>::x() const {
    static_assert(K >= 0 && K < N, "Noise symbol index out of range.");
    return cuda::std::get<K>(m_x);
}

template <typename T, int N>
T& AF1<T,N>::x(const int k) { return m_x[k]; }

template <typename T, int N>
const T& AF1<T,N>::x(const int k) const { return m_x[k]; }

template <typename T, int N>
T& AF1<T,N>::err() { return m_err; }

template <typename T, int N>
const T& AF1<T,N>::err() const { return m_err; }

template <typename T, int N>
template <typename I>
__host__ __device__ constexpr bool AF1<T,N>::has_symbol() {
    return (detail::IdIndex<I>::value >= 0) && (detail::IdIndex<I>::value < N);
}

template <typename T, int N>
template <typename I, int D>
__host__ __device__ T AF1<T,N>::symbol_magnitude() const {
    if constexpr (has_symbol<I>()) {
        return this->template x<detail::IdIndex<I>::value>();
    }
    else {
        return T(D);
    }
}

template <typename T, int N>
T AF1<T,N>::affine_radius() const {
    T r(0);
    for (int k = 0; k < N; ++k) {
        r = detail::add_up(r, T(::fabs(m_x[k])));
    }
    return r;
}

template <typename T, int N>
T AF1<T,N>::radius() const {
    return detail::add_up(this->affine_radius(), m_err);
}

template <typename T, int N>
AF1<T,N>::operator Interval<T>() const {
    return impl::centered_interval(m_b, this->radius());
}

template <typename T, int N>
Interval<T> AF1<T,N>::interval() const {
    return Interval<T>(*this);
}

template <typename T, int N>
Interval<T> AF1<T,N>::affine_interval() const {
    return impl::centered_interval(m_b, this->affine_radius());
}

template <typename T, int N>
Sign AF1<T,N>::sign() const {
    return this->interval().sign();
}

template <typename T, int N>
bool AF1<T,N>::is_entire() const {
    const Interval<T> i = this->interval();
    return !(detail::is_finite(i.a) && detail::is_finite(i.b));
}

template <typename T, int N>
void AF1<T,N>::set_entire() {
    m_b = T(0);
    m_x = {};
    m_err = detail::infinity<T>();
}

template <typename T, int N>
AF1<T,N> AF1<T,N>::operator+() const {
    return AF1<T,N>(*this);
}

template <typename T, int N>
AF1<T,N> AF1<T,N>::operator-() const {
    AF1<T,N> out(*this);
    out.m_b = -out.m_b;
    for (int k = 0; k < N; ++k) {
        out.m_x[k] = -out.m_x[k];
    }
    return out;
}

template <typename T, int N>
AF1<T,N>& AF1<T,N>::operator+=(const T& other) {
    const T s = detail::add_rn(m_b, other);
    m_err = detail::add_up(m_err, impl::add_error(m_b, other, s));
    m_b = s;
    return *this;
}

template <typename T, int N>
AF1<T,N>& AF1<T,N>::operator-=(const T& other) {
    return (*this += (-other));
}

template <typename T, int N>
AF1<T,N>& AF1<T,N>::operator*=(const T& other) {
    T e = detail::mul_up(m_err, T(::fabs(other)));
    const T p = detail::mul_rn(m_b, other);
    e = detail::add_up(e, impl::mul_error(m_b, other, p));
    m_b = p;
    for (int k = 0; k < N; ++k) {
        const T q = detail::mul_rn(m_x[k], other);
        e = detail::add_up(e, impl::mul_error(m_x[k], other, q));
        m_x[k] = q;
    }
    m_err = e;
    return *this;
}

template <typename T, int N>
AF1<T,N>& AF1<T,N>::operator/=(const T& other) {
    T e = detail::div_up(m_err, T(::fabs(other)));
    const T p = detail::div_rn(m_b, other);
    e = detail::add_up(e, impl::div_error(m_b, other, p));
    m_b = p;
    for (int k = 0; k < N; ++k) {
        const T q = detail::div_rn(m_x[k], other);
        e = detail::add_up(e, impl::div_error(m_x[k], other, q));
        m_x[k] = q;
    }
    m_err = e;
    return *this;
}

template <typename T, int N>
AF1<T,N>& AF1<T,N>::operator+=(const AF1<T,N>& other) {
    const T s = detail::add_rn(m_b, other.m_b);
    T e = detail::add_up(detail::add_up(m_err, other.m_err), impl::add_error(m_b, other.m_b, s));
    m_b = s;
    for (int k = 0; k < N; ++k) {
        const T q = detail::add_rn(m_x[k], other.m_x[k]);
        e = detail::add_up(e, impl::add_error(m_x[k], other.m_x[k], q));
        m_x[k] = q;
    }
    m_err = e;
    return *this;
}

template <typename T, int N>
AF1<T,N>& AF1<T,N>::operator-=(const AF1<T,N>& other) {
    const T s = detail::sub_rn(m_b, other.m_b);
    T e = detail::add_up(detail::add_up(m_err, other.m_err), impl::add_error(m_b, -other.m_b, s));
    m_b = s;
    for (int k = 0; k < N; ++k) {
        const T q = detail::sub_rn(m_x[k], other.m_x[k]);
        e = detail::add_up(e, impl::add_error(m_x[k], -other.m_x[k], q));
        m_x[k] = q;
    }
    m_err = e;
    return *this;
}

template <typename T, int N>
AF1<T,N>& AF1<T,N>::operator*=(const AF1<T,N>& other) {
    /* (lb + L + el)(rb + R + er) = lb*rb + lb*R + rb*L + L*R + (lb+L)*er + (rb+R)*el + el*er
       where L*R (bounded by the product of the affine radii) and the last three terms go to the error term */
    const AF1<T,N>& lhs = *this;
    const AF1<T,N>& rhs = other;
    const T p = detail::mul_rn(lhs.m_b, rhs.m_b);
    const T lr = lhs.affine_radius();
    const T rr = rhs.affine_radius();
    T e = detail::add_up(impl::mul_error(lhs.m_b, rhs.m_b, p), detail::mul_up(lr, rr));
    if (lhs.m_err != T(0) || rhs.m_err != T(0)) {
        const T lm = detail::add_up(T(::fabs(lhs.m_b)), lr);
        const T rm = detail::add_up(T(::fabs(rhs.m_b)), rr);
        e = detail::add_up(e, detail::add_up(detail::mul_up(rm, lhs.m_err), detail::mul_up(lm, rhs.m_err)));
        e = detail::add_up(e, detail::mul_up(lhs.m_err, rhs.m_err));
    }
    cuda::std::array<T,N> x;
    for (int k = 0; k < N; ++k) {
        const T lx = lhs.m_x[k];
        const T rx = rhs.m_x[k];
        const T l = detail::mul_rn(lx, rhs.m_b);
        const T r = detail::mul_rn(rx, lhs.m_b);
        const T s = detail::add_rn(l, r);
        e = detail::add_up(e, detail::add_up(impl::add_error(l, r, s),
                    detail::add_up(impl::mul_error(lx, rhs.m_b, l), impl::mul_error(rx, lhs.m_b, r))));
        x[k] = s;
    }
    m_b = p;
    m_x = x;
    m_err = e;
    return *this;
}

template <typename T, int N>
__host__ __device__ AF1<T,N> operator+(const AF1<T,N>& lhs, const AF1<T,N>& rhs) { return +lhs += rhs; }

template <typename T, int N>
__host__ __device__ AF1<T,N> operator+(const AF1<T,N>& lhs, const detail::identity_t<T>& rhs) { return +lhs += T(rhs); }

template <typename T, int N>
__host__ __device__ AF1<T,N> operator+(const detail::identity_t<T>& lhs, const AF1<T,N>& rhs) { return +rhs += T(lhs); }

template <typename T, int N>
__host__ __device__ AF1<T,N> operator-(const AF1<T,N>& lhs, const AF1<T,N>& rhs) { return +lhs -= rhs; }

template <typename T, int N>
__host__ __device__ AF1<T,N> operator-(const AF1<T,N>& lhs, const detail::identity_t<T>& rhs) { return +lhs -= T(rhs); }

template <typename T, int N>
__host__ __device__ AF1<T,N> operator-(const detail::identity_t<T>& lhs, const AF1<T,N>& rhs) { return -rhs += T(lhs); }

template <typename T, int N>
__host__ __device__ AF1<T,N> operator*(const AF1<T,N>& lhs, const AF1<T,N>& rhs) { return +lhs *= rhs; }

template <typename T, int N>
__host__ __device__ AF1<T,N> operator*(const AF1<T,N>& lhs, const detail::identity_t<T>& rhs) { return +lhs *= T(rhs); }

template <typename T, int N>
__host__ __device__ AF1<T,N> operator*(const detail::identity_t<T>& lhs, const AF1<T,N>& rhs) { return +rhs *= T(lhs); }

template <typename T, int N>
__host__ __device__ AF1<T,N> operator/(const AF1<T,N>& lhs, const detail::identity_t<T>& rhs) { return +lhs /= T(rhs); }


namespace impl {
    /* Apply the approximation f(x) = alpha*x + zeta + delta*eps_new + err*eps_err, the new symbol is
       merged into the error term. */
    template <typename T, int N>
    __host__ __device__ AF1<T,N> apply_approx(const AF1<T,N>& x, const detail::Approx<T>& A) {
        AF1<T,N> out;
        if (A.entire) {
            out.set_entire();
            return out;
        }
        const T p = detail::mul_rn(A.alpha, x.b());
        const T s = detail::add_rn(p, A.zeta);
        out.b() = s;
        T e = detail::add_up(detail::add_up(A.err, A.delta), detail::add_up(mul_error(A.alpha, x.b(), p), add_error(p, A.zeta, s)));
        for (int k = 0; k < N; ++k) {
            const T q = detail::mul_rn(A.alpha, x.x(k));
            e = detail::add_up(e, mul_error(A.alpha, x.x(k), q));
            out.x(k) = q;
        }
        out.err() = e;
        return out;
    }
}

template <typename T, int N>
__host__ __device__ AF1<T,N> reciprocal(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_reciprocal(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> operator/(const AF1<T,N>& lhs, const AF1<T,N>& rhs) { return lhs * reciprocal(rhs); }

template <typename T, int N>
__host__ __device__ AF1<T,N> operator/(const detail::identity_t<T>& lhs, const AF1<T,N>& rhs) { return T(lhs) * reciprocal(rhs); }

template <typename T, int N>
__host__ __device__ AF1<T,N> sqrt(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_sqrt(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> exp(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_exp(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> log(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_log(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> pow(const AF1<T,N>& x, int n) { return impl::apply_approx(x, detail::approx_pown(x.affine_interval(), x.err(), n)); }

template <typename T, int N>
__host__ __device__ AF1<T,N> sqr(const AF1<T,N>& x) { return pow(x, 2); }

template <typename T, int N>
__host__ __device__ AF1<T,N> pow(const AF1<T,N>& x, const detail::identity_t<T>& y) { return impl::apply_approx(x, detail::approx_pow(x.affine_interval(), x.err(), T(y))); }

template <typename T, int N>
__host__ __device__ AF1<T,N> sin(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_sin(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> cos(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_cos(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> tan(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_tan(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> cot(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_cot(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> sinh(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_sinh(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> cosh(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_cosh(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> tanh(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_tanh(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> asinh(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_asinh(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> acosh(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_acosh(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> atanh(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_atanh(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> abs(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_abs(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> positive_part(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_positive_part(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> rsqrt(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_rsqrt(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> exp2(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_exp2(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> exp10(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_exp10(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> expm1(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_expm1(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> log2(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_log2(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> log10(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_log10(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> log1p(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_log1p(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> cbrt(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_cbrt(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> rcbrt(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_rcbrt(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> atan(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_atan(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> asin(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_asin(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> acos(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_acos(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> erf(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_erf(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> erfc(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_erfc(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> normcdf(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_normcdf(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> sinpi(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_sinpi(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> cospi(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_cospi(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> floor(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_step<detail::floor_fn>(x.interval())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> ceil(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_step<detail::ceil_fn>(x.interval())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> trunc(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_step<detail::trunc_fn>(x.interval())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> round(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_step<detail::round_fn>(x.interval())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> rint(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_step<detail::rint_fn>(x.interval())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> fract(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_fract(x.interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> fmod(const AF1<T,N>& x, const detail::identity_t<T>& c) { return impl::apply_approx(x, detail::approx_fmod(x.interval(), x.err(), T(c))); }

template <typename T, int N>
__host__ __device__ AF1<T,N> modf(const AF1<T,N>& x, AF1<T,N>* ip) {
    if (ip) { *ip = trunc(x); }
    return impl::apply_approx(x, detail::approx_modf(x.interval(), x.err()));
}

template <typename T, int N>
__host__ __device__ AF1<T,N> remquo(const AF1<T,N>& x, const detail::identity_t<T>& c, int* quo_lo, int* quo_hi) {
    Interval<T> quotient;
    const detail::Approx<T> A = detail::approx_reduction<detail::rint_fn>(x.interval(), x.err(), T(c), &quotient);
    detail::integer_range(quotient, quo_lo, quo_hi);
    return impl::apply_approx(x, A);
}

template <typename T, int N>
__host__ __device__ AF1<T,N> remainder(const AF1<T,N>& x, const detail::identity_t<T>& c) { return remquo(x, c, nullptr, nullptr); }

template <typename T, int N>
__host__ __device__ AF1<T,N> tgamma(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_tgamma(x.affine_interval(), x.err())); }

template <typename T, int N>
__host__ __device__ AF1<T,N> lgamma(const AF1<T,N>& x) { return impl::apply_approx(x, detail::approx_lgamma(x.affine_interval(), x.err())); }


/* composite functions, the generic ones of composite.cuh apply to AF1 through the traits below */
namespace detail {
    template <typename T, int N>
    struct IsAA<AF1<T,N>> : cuda::std::true_type {};

    template <typename T, int N>
    struct Scalar<AF1<T,N>> { using type = T; };
}

template <typename T, int N>
__host__ __device__ AF1<T,N> min(const AF1<T,N>& x, const AF1<T,N>& y) { return ((x + y) - abs(x - y)) * T(0.5); }

template <typename T, int N>
__host__ __device__ AF1<T,N> max(const AF1<T,N>& x, const AF1<T,N>& y) { return ((x + y) + abs(x - y)) * T(0.5); }

template <typename T, int N>
__host__ __device__ AF1<T,N> clamp(const AF1<T,N>& x, const AF1<T,N>& lo, const AF1<T,N>& hi) { return min(max(x, lo), hi); }

template <typename T, int N>
__host__ __device__ AF1<T,N> smoothstep(const detail::identity_t<T>& e0, const detail::identity_t<T>& e1, const AF1<T,N>& x) {
    const auto t = clamp((x - T(e0)) / (T(e1) - T(e0)), T(0), T(1));
    return sqr(t) * (T(3) - T(2)*t);
}

template <typename T, int N>
__host__ __device__ bool possibly_zero(const AF1<T,N>& x) {
    const Interval<T> i = x.interval();
    return (i.a <= T(0)) && (i.b >= T(0));
}

/* two-argument arctangent in [-pi, pi], see atan2 in composite.cuh */
template <typename T, int N>
__host__ __device__ AF1<T,N> atan2(const AF1<T,N>& y, const AF1<T,N>& x) {
    const Interval<T> X = x.interval();
    const Interval<T> Y = y.interval();
    if (X.a > T(0)) {
        return atan(y / x);
    }
    const AF1<T,N> half_pi = AF1<T,N>::constant(Interval<T>(detail::mul_rn(detail::fp<T>::pi_lo(), T(0.5)),
                                                            detail::mul_rn(detail::fp<T>::pi_hi(), T(0.5))));
    if (Y.a > T(0)) {
        return half_pi - atan(x / y);
    }
    if (Y.b < T(0)) {
        return -half_pi - atan(x / y);
    }
    AF1<T,N> out(T(0));
    out.err() = detail::fp<T>::pi_hi();
    return out;
}

template <typename T, int N>
__host__ __device__ AF1<T,N> atan2(const AF1<T,N>& y, const detail::identity_t<T>& x) { return atan2(y, AF1<T,N>(T(x))); }

template <typename T, int N>
__host__ __device__ AF1<T,N> atan2(const detail::identity_t<T>& y, const AF1<T,N>& x) { return atan2(AF1<T,N>(T(y)), x); }

namespace impl {
    /* x - n*y with n = Fn(x/y) and an affine divisor y, see impl::reduction in composite.cuh */
    template <typename Fn, typename T, int N>
    __host__ __device__ AF1<T,N> reduction(const AF1<T,N>& x, const AF1<T,N>& y, int* quo_lo, int* quo_hi) {
        const Interval<T> X = x.interval();
        const Interval<T> Y = y.interval();
        if (!detail::is_finite(X) || !detail::is_finite(Y) || (Y.a <= T(0) && Y.b >= T(0))) {
            detail::integer_range(Interval<T>(-detail::infinity<T>(), detail::infinity<T>()), quo_lo, quo_hi);
            AF1<T,N> out;
            out.set_entire();
            return out;
        }
        const Fn fn;
        const Interval<T> Q = detail::operator/(X, Y);
        const T n = fn(Q.a);
        detail::integer_range(Interval<T>(n, fn(Q.b)), quo_lo, quo_hi);
        if (n == fn(Q.b)) {
            return x - y * n;
        }
        const T M = detail::fmax(T(::fabs(Y.a)), T(::fabs(Y.b)));
        return AF1<T,N>::constant(detail::remainder_range(fn, X, M));
    }
}

template <typename T, int N>
__host__ __device__ AF1<T,N> fmod(const AF1<T,N>& x, const AF1<T,N>& y) { return impl::reduction<detail::trunc_fn>(x, y, nullptr, nullptr); }

template <typename T, int N>
__host__ __device__ AF1<T,N> fmod(const detail::identity_t<T>& x, const AF1<T,N>& y) { return fmod(AF1<T,N>(T(x)), y); }

template <typename T, int N>
__host__ __device__ AF1<T,N> remquo(const AF1<T,N>& x, const AF1<T,N>& y, int* quo_lo, int* quo_hi) {
    return impl::reduction<detail::rint_fn>(x, y, quo_lo, quo_hi);
}

template <typename T, int N>
__host__ __device__ AF1<T,N> remainder(const AF1<T,N>& x, const AF1<T,N>& y) { return remquo(x, y, nullptr, nullptr); }

template <typename T, int N>
__host__ __device__ AF1<T,N> remainder(const detail::identity_t<T>& x, const AF1<T,N>& y) { return remainder(AF1<T,N>(T(x)), y); }

template <typename T, int N>
__host__ __device__ AF1<T,N> remquo(const detail::identity_t<T>& x, const AF1<T,N>& y, int* quo_lo, int* quo_hi) {
    return remquo(AF1<T,N>(T(x)), y, quo_lo, quo_hi);
}

/* x^y = exp(y*log(x)) for x > 0 (entire otherwise) */
template <typename T, int N>
__host__ __device__ AF1<T,N> pow(const AF1<T,N>& x, const AF1<T,N>& y) { return exp(y * log(x)); }

/* c^y = exp(y*log(c)) for a scalar c > 0 (entire otherwise) */
template <typename T, int N>
__host__ __device__ AF1<T,N> pow(const detail::identity_t<T>& c, const AF1<T,N>& y) {
    const T base(c);
    const Interval<T> L = (base > T(0) ? detail::log_fn().F(base)
                                       : Interval<T>(-detail::infinity<T>(), detail::infinity<T>()));
    return exp(y * AF1<T,N>::constant(L));
}


template <typename T, int N>
__host__ __device__ void AF1<T,N>::print() const {
    printf("AF1(b=%+2.1e, x=[", double(m_b));
    for (int k = 0; k < N; ++k) {
        printf("%+2.1e%s", double(m_x[k]), (k == N-1 ? "" : ", "));
    }
    printf("], err=%+2.1e)\n", double(m_err));
}

#ifndef __CUDACC_RTC__
    template <typename T, int N>
    __host__ std::ostream& operator<<(std::ostream& os, const AF1<T,N>& val) {
        os << "AF1(b=" << val.b() << ", x=[";
        for (int k = 0; k < N; ++k) {
            os << val.x(k) << (k == N-1 ? "" : ", ");
        }
        os << "], err=" << val.err() << ")";
        return os;
    }
#endif

} /* end of namespace cusvna */

#endif /* end of include guard: AF1_CUH_Q8ZK3MTD */
