/* This test checks the soundness of all operations and elementary functions with rounding, in single
   and double precision. Enclosures are checked against pointwise evaluations of the same expressions
   in extended precision (long double). Bounds must never be nan, and must be finite for inputs inside
   the domain of the functions. Point inputs must give tight enclosures.
   The same tests are run on AA forms, and on AF1 forms when CUSVNA_TEST_AF1 is defined. Forms are
   evaluated in host code, or in device code when CUSVNA_TEST_DEVICE is defined (nvcc --extended-lambda). */
#include <cmath>
#include <cstdlib>
#include <math.h>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

#ifdef CUSVNA_TEST_AF1
#include "cusvna/af1.cuh"
template <typename T> using Form = cusvna::AF1<T,6>;
#else
#include "cusvna/aa.cuh"
template <typename T> using Form = cusvna::AA<T>;
#endif

using LD = long double;

#ifdef CUSVNA_TEST_DEVICE
#define HD __host__ __device__

template <typename T, typename F, typename X, typename Y>
__global__ void evaluate_kernel(F f, const X x, const Y y, cusvna::Interval<T>* out) {
    *out = f(x, y, T(0));
}

/* evaluates f(x, y) in device code (a single thread) */
template <typename T, typename F, typename X, typename Y>
cusvna::Interval<T> evaluate(F f, const X& x, const Y& y) {
    cusvna::Interval<T>* d_out;
    cusvna::Interval<T> out;
    if (cudaMalloc(&d_out, sizeof(out)) != cudaSuccess) { std::cerr << "cudaMalloc failed" << std::endl; std::exit(2); }
    evaluate_kernel<T><<<1, 1>>>(f, x, y, d_out);
    const cudaError_t err = cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "kernel failed: " << cudaGetErrorString(err) << std::endl; std::exit(2); }
    cudaFree(d_out);
    return out;
}
#else
#define HD

template <typename T, typename F, typename X, typename Y>
cusvna::Interval<T> evaluate(F f, const X& x, const Y& y) { return f(x, y, T(0)); }
#endif

/* types of the forms of the first two variables */
template <typename T, int K>
using Var = decltype(Form<T>::template from_interval<K>(std::declval<const cusvna::Interval<T>&>()));

template <typename F, typename R>
struct Expr {
    F form;         /* (Var<T,0>, Var<T,1>, T) -> Interval<T> */
    R reference;    /* (LD, LD, T) -> LD */
};

template <typename F, typename R>
Expr<F,R> make_expr(F form, R reference) { return Expr<F,R>{form, reference}; }

/* the same expression of x, y and t (whose type gives the floating point type T of the forms) evaluated
   on forms, in host or device code, and in extended precision */
#define EXPR(...) make_expr( \
    [=] HD (Var<T,0> x, Var<T,1> y, T t) -> cusvna::Interval<T> { return [&]() { __VA_ARGS__ }().interval(); }, \
    [=] (LD x, LD y, T t) -> LD { __VA_ARGS__ })

/* reference implementations of the functions that are not in the standard library */
LD reciprocal(const LD x)    { return LD(1) / x; }
LD sqr(const LD x)           { return x * x; }
LD cot(const LD x)           { return std::cos(x) / std::sin(x); }
LD positive_part(const LD x) { return (x > 0 ? x : LD(0)); }
LD rsqrt(const LD x)         { return LD(1) / std::sqrt(x); }
LD rcbrt(const LD x)         { return LD(1) / std::cbrt(x); }
LD exp10(const LD x)         { return std::pow(LD(10), x); }
LD normcdf(const LD x)       { return std::erfc(-x / std::sqrt(LD(2))) / 2; }
LD reduce2(const LD x)       { return x - 2 * std::nearbyint(x / 2); }   /* exact */
LD sinpi(const LD x)         { return std::sin(3.14159265358979323846264338327950288L * reduce2(x)); }
LD cospi(const LD x)         { return std::cos(3.14159265358979323846264338327950288L * reduce2(x)); }
LD min(const LD x, const LD y) { return std::fmin(x, y); }
LD max(const LD x, const LD y) { return std::fmax(x, y); }
template <typename U, typename std::enable_if<std::is_floating_point<U>::value && !std::is_same<U,LD>::value, int>::type = 0>
LD min(const LD x, const U y) { return std::fmin(x, LD(y)); }
template <typename U, typename std::enable_if<std::is_floating_point<U>::value && !std::is_same<U,LD>::value, int>::type = 0>
LD min(const U x, const LD y) { return std::fmin(LD(x), y); }
template <typename U, typename std::enable_if<std::is_floating_point<U>::value && !std::is_same<U,LD>::value, int>::type = 0>
LD max(const LD x, const U y) { return std::fmax(x, LD(y)); }
template <typename U, typename std::enable_if<std::is_floating_point<U>::value && !std::is_same<U,LD>::value, int>::type = 0>
LD max(const U x, const LD y) { return std::fmax(LD(x), y); }
LD clamp(const LD x, const LD lo, const LD hi) { return min(max(x, lo), hi); }
LD norm3d(const LD x, const LD y, const LD z) { return std::hypot(x, y, z); }
LD rhypot(const LD x, const LD y) { return LD(1) / std::hypot(x, y); }
LD rnorm3d(const LD x, const LD y, const LD z) { return LD(1) / std::hypot(x, y, z); }
LD lerp(const LD a, const LD b, const LD t) { return a + t*(b - a); }
LD smoothstep(const LD e0, const LD e1, const LD x) { const LD t = clamp((x - e0)/(e1 - e0), LD(0), LD(1)); return t*t*(3 - 2*t); }
LD smooth_min(const LD a, const LD b, const LD k) { const LD h = max(k - std::fabs(a - b), LD(0)) / k; return min(a, b) - h*h*k/4; }
LD smooth_max(const LD a, const LD b, const LD k) { return -smooth_min(-a, -b, k); }
LD fract(const LD x)         { return x - std::floor(x); }

static int failures = 0;
static int checks = 0;

void fail(const std::string& msg) {
    std::cerr << "FAILED: " << msg << std::endl;
    ++failures;
}

template <typename T>
const char* type_name() { return std::is_same<T,float>::value ? "float" : "double"; }

template <typename T>
bool is_finite(const T x) { return std::isfinite(x); }

/* check that f(u, v) belongs to f(x, y).interval() for (u, v) sampled in X x Y, f is an EXPR */
template <typename T, typename F>
void check_enclosure(const std::string& name, F&& f, const cusvna::Interval<T>& X, const cusvna::Interval<T>& Y,
        const bool expect_finite, const int n=32) {
    ++checks;
    const std::string what = std::string(type_name<T>()) + " " + name + " on X=[" + std::to_string(X.a) + "," +
        std::to_string(X.b) + "], Y=[" + std::to_string(Y.a) + "," + std::to_string(Y.b) + "]";

    const auto x = Form<T>::template from_interval<0>(X);
    const auto y = Form<T>::template from_interval<1>(Y);
    const cusvna::Interval<T> r = evaluate<T>(f.form, x, y);

    if (r.a != r.a || r.b != r.b) {
        fail(what + ": nan bounds");
        return;
    }
    if (expect_finite && !(is_finite(r.a) && is_finite(r.b))) {
        fail(what + ": unexpected unbounded result [" + std::to_string(r.a) + ", " + std::to_string(r.b) + "]");
        return;
    }
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            /* sample points are representable in T so that reference evaluations are accurate */
            const LD u = LD(T(LD(X.a) + (LD(X.b) - LD(X.a)) * i / n));
            const LD v = LD(T(LD(Y.a) + (LD(Y.b) - LD(Y.a)) * j / n));
            const LD ref = f.reference(u, v, T(0));
            if (ref != ref) {
                continue; /* outside of the domain */
            }
            const LD tol = std::ldexp(std::fmax(std::fabs(ref), LD(1)), -60);
            if (ref < LD(r.a) - tol || ref > LD(r.b) + tol) {
                std::cerr.precision(20);
                std::cerr << "  f(" << u << ", " << v << ") = " << ref << " not in [" << r.a << ", " << r.b << "]" << std::endl;
                fail(what + ": not enclosed");
                return;
            }
        }
    }
}

/* check that the enclosure of a point evaluation is at most max_ulps wide */
template <typename T, typename F>
void check_tight(const std::string& name, F&& f, const T x0, const int max_ulps) {
    ++checks;
    const auto x = Form<T>::template from_interval<0>(cusvna::Interval<T>(x0, x0));
    const auto y = Form<T>::template from_interval<1>(cusvna::Interval<T>(T(0), T(0)));
    const cusvna::Interval<T> r = evaluate<T>(f.form, x, y);
    const LD ref = f.reference(LD(x0), LD(0), T(0));
    /* exact zeros are enclosed with a few smallest normal numbers */
    const LD ulp = std::fmax(std::numeric_limits<T>::epsilon() * std::fabs(ref), LD(std::numeric_limits<T>::min()));
    const LD width = LD(r.b) - LD(r.a);
    const LD tol = std::ldexp(std::fmax(std::fabs(ref), LD(1)), -60);
    if (!(LD(r.a) <= ref + tol && ref - tol <= LD(r.b))) {
        std::cerr.precision(20);
        std::cerr << "  " << type_name<T>() << " " << name << "(" << x0 << ") = " << ref << " not in [" << r.a << ", " << r.b << "]" << std::endl;
        fail(std::string(type_name<T>()) + " " + name + ": point not enclosed");
        return;
    }
    if (!(width <= max_ulps * ulp)) {
        std::cerr.precision(20);
        std::cerr << "  " << type_name<T>() << " " << name << "(" << x0 << ") = [" << r.a << ", " << r.b << "] is "
            << width / ulp << " ulps wide" << std::endl;
        fail(std::string(type_name<T>()) + " " + name + ": point enclosure is not tight");
    }
}

/* unary functions, evaluated on a set of intervals */
template <typename T, typename F, typename D>
void check_unary(const std::string& name, F&& f, D&& in_domain, const int max_ulps) {
    using I = cusvna::Interval<T>;
    const T e = std::numeric_limits<T>::epsilon();
    const I intervals[] = {
        I(0.5, 0.75), I(1, 4), I(-3, -2), I(-1, 1), I(-0.25, 0.5), I(0, 1), I(-1, 0), I(2, 2), I(T(0.1), T(0.1)),
        I(T(-1.7), T(-1.7)), I(0, 0), I(1, 1), I(-1, -1), I(1, 1+64*e), I(T(0.3), T(0.3)+4*e), I(-20, -19), I(10, 12),
        I(T(-1e-3), T(2e-3)), I(3, 30), I(-100, 100), I(T(0.99), T(0.999)), I(T(-0.999), T(-0.5)), I(T(1.5), T(1.6)),
        I(T(3.1), T(3.2)), I(T(4.7), T(4.75)), I(T(6.2), T(6.3)), I(T(-7.9), T(-7.8)), I(T(1.56), T(1.58))
    };
    for (const I& X : intervals) {
        check_enclosure<T>(name, f, X, I(0, 0), in_domain(LD(X.a), LD(X.b)));
    }
    const T points[] = { T(0.1), T(0.5), T(0.9), T(1.3), T(2), T(-0.3), T(-1.1), T(-2.5), T(7.25) };
    for (const T x0 : points) {
        if (in_domain(LD(x0), LD(x0)) && x0 != T(0)) {
            check_tight<T>(name, f, x0, max_ulps);
        }
    }
}

bool contains_multiple(const LD a, const LD b, const LD period, const LD offset) {
    /* does [a,b] contain a point offset + k*period */
    return std::floor((a - offset) / period) != std::floor((b - offset) / period) || std::fmod(a - offset, period) == 0;
}

template <typename T>
void run() {
    using I = cusvna::Interval<T>;
    const LD pi = 3.14159265358979323846264338327950288L;
    const LD max_exp = std::log(LD(std::numeric_limits<T>::max())) - 1;

    auto always = [](LD a, LD b) { return true; };
    auto exp_ok = [&](LD a, LD b) { return b < max_exp; };
    auto positive = [](LD a, LD b) { return a > 0; };
    auto nonnegative_part = [](LD a, LD b) { return b >= 0; };
    auto nonzero = [](LD a, LD b) { return a > 0 || b < 0; };
    auto no_tan_pole = [&](LD a, LD b) { return !contains_multiple(a, b, pi, pi/2) && b - a < pi; };
    auto no_cot_pole = [&](LD a, LD b) { return !contains_multiple(a, b, pi, 0) && b - a < pi; };
    auto acosh_ok = [](LD a, LD b) { return b >= 1; };
    auto atanh_ok = [](LD a, LD b) { return a > -1 && b < 1; };

    check_unary<T>("-x",         EXPR(return -x;), always, 0);
    check_unary<T>("reciprocal", EXPR(return reciprocal(x);), nonzero, 16);
    check_unary<T>("sqrt",       EXPR(return sqrt(x);), nonnegative_part, 16);
    check_unary<T>("exp",        EXPR(return exp(x);), exp_ok, 24);
    check_unary<T>("log",        EXPR(return log(x);), positive, 24);
    check_unary<T>("sqr",        EXPR(return sqr(x);), always, 16);
    check_unary<T>("x^3",        EXPR(return pow(x, 3);), always, 16);
    check_unary<T>("x^4",        EXPR(return pow(x, 4);), always, 16);
    check_unary<T>("x^-2",       EXPR(return pow(x, -2);), nonzero, 16);
    check_unary<T>("x^-3",       EXPR(return pow(x, -3);), nonzero, 16);
    check_unary<T>("x^0.5",      EXPR(using S = decltype(t); return pow(x, S(0.5));), nonnegative_part, 72);
    check_unary<T>("x^2.5",      EXPR(using S = decltype(t); return pow(x, S(2.5));), nonnegative_part, 72);
    check_unary<T>("x^-1.5",     EXPR(using S = decltype(t); return pow(x, S(-1.5));), positive, 72);
    check_unary<T>("sin",        EXPR(return sin(x);), always, 24);
    check_unary<T>("cos",        EXPR(return cos(x);), always, 24);
    check_unary<T>("tan",        EXPR(return tan(x);), no_tan_pole, 40);
    check_unary<T>("cot",        EXPR(return cot(x);), no_cot_pole, 48);
    check_unary<T>("sinh",       EXPR(return sinh(x);), exp_ok, 32);
    check_unary<T>("cosh",       EXPR(return cosh(x);), exp_ok, 32);
    check_unary<T>("tanh",       EXPR(return tanh(x);), always, 24);
    check_unary<T>("asinh",      EXPR(return asinh(x);), always, 32);
    check_unary<T>("acosh",      EXPR(return acosh(x);), acosh_ok, 40);
    check_unary<T>("atanh",      EXPR(return atanh(x);), atanh_ok, 32);
    check_unary<T>("abs",        EXPR(return abs(x);), always, 0);
    check_unary<T>("positive_part", EXPR(return positive_part(x);), always, 0);
    
    auto arc_ok = [](LD a, LD b) { return b >= -1 && a <= 1; };
    auto log1p_ok = [](LD a, LD b) { return a > -1; };
    auto exp2_ok = [&](LD a, LD b) { return b < max_exp / std::log(LD(2)); };
    auto exp10_ok = [&](LD a, LD b) { return b < max_exp / std::log(LD(10)); };
    check_unary<T>("rsqrt",      EXPR(return rsqrt(x);), positive, 16);
    check_unary<T>("exp2",       EXPR(return exp2(x);), exp2_ok, 24);
    check_unary<T>("exp10",      EXPR(return exp10(x);), exp10_ok, 72);
    check_unary<T>("expm1",      EXPR(return expm1(x);), exp_ok, 24);
    check_unary<T>("log2",       EXPR(return log2(x);), positive, 24);
    check_unary<T>("log10",      EXPR(return log10(x);), positive, 24);
    check_unary<T>("log1p",      EXPR(return log1p(x);), log1p_ok, 24);
    check_unary<T>("cbrt",       EXPR(return cbrt(x);), always, 24);
    check_unary<T>("rcbrt",      EXPR(return rcbrt(x);), nonzero, 24);
    check_unary<T>("atan",       EXPR(return atan(x);), always, 32);
    check_unary<T>("asin",       EXPR(return asin(x);), arc_ok, 32);
    check_unary<T>("acos",       EXPR(return acos(x);), arc_ok, 32);
    check_unary<T>("erf",        EXPR(return erf(x);), always, 32);
    check_unary<T>("erfc",       EXPR(return erfc(x);), always, 48);
    check_unary<T>("normcdf",    EXPR(return normcdf(x);), always, 64);
    check_unary<T>("sinpi",      EXPR(return sinpi(x);), always, 24);
    check_unary<T>("cospi",      EXPR(return cospi(x);), always, 24);
    auto gamma_ok = [](LD a, LD b) { return a > 0 && b < (std::is_same<T,float>::value ? 34 : 170); };
    check_unary<T>("tgamma",     EXPR(return tgamma(x);), gamma_ok, 64);
    check_unary<T>("lgamma",     EXPR(return lgamma(x);), positive, 48);
    check_unary<T>("floor",      EXPR(return floor(x);), always, 0);
    check_unary<T>("ceil",       EXPR(return ceil(x);), always, 0);
    check_unary<T>("trunc",      EXPR(return trunc(x);), always, 0);
    check_unary<T>("round",      EXPR(return round(x);), always, 0);
    check_unary<T>("rint",       EXPR(return rint(x);), always, 0);
    check_unary<T>("fract",      EXPR(return fract(x);), always, 4);
    check_unary<T>("fmod(x, 0.75)",  EXPR(using S = decltype(t); return fmod(x, S(0.75));), always, 4);
    check_unary<T>("fmod(x, -1.5)",  EXPR(using S = decltype(t); return fmod(x, S(-1.5));), always, 4);
    check_unary<T>("modf",       EXPR(decltype(trunc(x)) ip; return modf(x, &ip);), always, 4);
    check_unary<T>("modf ip",    EXPR(decltype(trunc(x)) ip; modf(x, &ip); return ip;), always, 0);
    check_unary<T>("modf(3x) + ip", EXPR(using S = decltype(t); decltype(trunc(x*S(3))) ip; auto f = modf(x*S(3), &ip); return f + ip;), always, 8);
    check_unary<T>("remainder(x, 0.75)", EXPR(using S = decltype(t); return remainder(x, S(0.75));), always, 4);
    check_unary<T>("remainder(x, -1.5)", EXPR(using S = decltype(t); return remainder(x, S(-1.5));), always, 4);
    check_unary<T>("fract(3x) + floor(3x)", EXPR(using S = decltype(t); return fract(x*S(3)) + floor(x*S(3));), always, 8);

    /* composed expressions */
    const I U(-1, 1), V(0, 1), W(1, 2);
    auto check = [&](const std::string& name, auto&& f, const I& X, const I& Y) { check_enclosure<T>(name, f, X, Y, true); };

    check("(x+y)^2 - (x-y)^2", EXPR(return (x+y)*(x+y) - (x-y)*(x-y);), U, U);
    check("sqrt(x*x + y*y)",   EXPR(return sqrt(x*x + y*y);), W, W);
    check("hypot sqr",         EXPR(return sqrt(sqr(x) + sqr(y));), U, W);
    check("x / (y + 2)",       EXPR(using S = decltype(t); return x / (y + S(2));), U, V);
    check("exp(sin(x))*cos(y)",EXPR(return exp(sin(x)) * cos(y);), U, U);
    check("tanh(x*y) + atanh(x/2)", EXPR(using S = decltype(t); return tanh(x*y) + atanh(x*S(0.5));), U, U);
    check("|x - y| + max(x-y,0)", EXPR(return abs(x - y) + positive_part(x - y);), U, V);
    check("x^3 - x*x*x",       EXPR(return pow(x, 3) - x*x*x;), U, V);
    check("log(1 + x^2) - log(y)", EXPR(using S = decltype(t); return log(S(1) + sqr(x)) - log(y);), U, W);
    check("sinh(x) - cosh(y) + asinh(x*y)", EXPR(return sinh(x) - cosh(y) + asinh(x*y);), U, U);
    check("test2 f", EXPR(
            using S = decltype(t);
            auto g = sqrt(x*x - x + S(0.5)) / sqrt(x*x + S(0.5));
            return sqrt(g*g - g + S(0.5)) / sqrt(g*g + S(0.5));), I(T(0.25), T(0.375)), V);
    check("test3 f", EXPR(
            using S = decltype(t);
            auto xy = x*y;
            return x*x + y*y + xy - xy*xy/S(2) - S(1)/S(4);), I(-2, -1), I(1, 2));

    /* composite functions */
    check("min(x, y)",          EXPR(return min(x, y);), U, V);
    check("max(x, y)",          EXPR(return max(x, y);), U, V);
    check("min(x, y) disjoint", EXPR(return min(x, y);), I(-2, -1), V);
    check("max(x, 0.25)",       EXPR(using S = decltype(t); return max(x, S(0.25));), U, V);
    check("min(0.25, x)",       EXPR(using S = decltype(t); return min(S(0.25), x);), U, V);
    check("fmin/fmax",          EXPR(return fmin(x, y) * fmax(x, y);), U, V);
    check("clamp(x, -0.5, 0.5)",EXPR(using S = decltype(t); return clamp(x, S(-0.5), S(0.5));), U, V);
    check("clamp(x, y-1, y)",   EXPR(using S = decltype(t); return clamp(x, y - S(1), y);), U, V);
    check("hypot(x, y)",        EXPR(return hypot(x, y);), U, W);
    check("hypot(x, y) at 0",   EXPR(return hypot(x, y);), U, U);
    check("norm3d(x, y, x*y)",  EXPR(return norm3d(x, y, x*y);), U, W);
    check("hypot(x, y, x-y)",   EXPR(return hypot(x, y, x - y);), U, W);
    check("rhypot(x, y)",       EXPR(return rhypot(x, y);), U, W);
    check("rnorm3d(x, y, y)",   EXPR(return rnorm3d(x, y, y);), U, W);
    check("fma(x, y, x)",       EXPR(return fma(x, y, x);), U, W);
    check("lerp(x, y, 0.3)",    EXPR(using S = decltype(t); return lerp(x, y, S(0.3));), U, W);
    check("lerp(0, 1, x*y)",    EXPR(using S = decltype(t); return lerp(S(0), S(1), x*y);), U, V);
    check("smoothstep",         EXPR(using S = decltype(t); return smoothstep(S(-0.5), S(0.5), x);), U, V);
    check("smoothstep inside",  EXPR(using S = decltype(t); return smoothstep(S(-2), S(2), x);), U, V);
    check("smooth_min",         EXPR(using S = decltype(t); return smooth_min(x, y, S(0.5));), U, V);
    check("smooth_max",         EXPR(using S = decltype(t); return smooth_max(x, y, S(0.25));), U, V);
    check("sdf union of disks", EXPR( 
            using S = decltype(t);
            const auto d0 = hypot(x - S(0.5), y) - S(0.5);
            const auto d1 = hypot(x + S(0.5), y - S(0.25)) - S(0.4);
            return min(d0, d1);), U, U);
    check("asin(x/2) + acos(y)",EXPR(using S = decltype(t); return asin(x*S(0.5)) + acos(y);), U, V);
    check("atan(x*y) * erf(x)", EXPR(return atan(x*y) * erf(x);), U, W);
    check("sinpi(x)*cospi(y)",  EXPR(return sinpi(x) * cospi(y);), U, V);
    check("log1p(expm1(x))",    EXPR(return log1p(expm1(x));), V, V);
    check("cbrt(x*y) - rcbrt(y)",EXPR(return cbrt(x*y) - rcbrt(y);), U, W);
    
    /* atan2 on boxes in all quadrants, across the axes, the branch cut and the origin */
    const T centers[] = { T(-2), T(-0.75), T(-0.1), T(0), T(0.1), T(0.75), T(2) };
    const T radii[] = { T(0), T(0.05), T(0.5) };
    for (const T cx : centers) {
        for (const T cy : centers) {
            for (const T r : radii) {
                const I X(cx - r, cx + r);
                const I Y(cy - r, cy + r);
                check("atan2(y, x)", EXPR(return atan2(y, x);), X, Y);
            }
        }
    }
    check("atan2(y, 0.5)",      EXPR(using S = decltype(t); return atan2(y, S(0.5));), U, U);
    check("atan2(-1, x)",       EXPR(using S = decltype(t); return atan2(S(-1), x);), U, U);
    check("atan2(x*y, x+2)",    EXPR(using S = decltype(t); return atan2(x*y, x + S(2));), U, U);
    check("fmod(x, y) crossing",EXPR(return fmod(x, y);), U, W);
    check("fmod(x, y) exact",   EXPR(return fmod(x, y);), V, I(3, 4));
    check("fmod(x+5, y)",       EXPR(using S = decltype(t); return fmod(x + S(5), y);), I(T(0.1), T(0.2)), I(2, T(2.05)));
    check("fmod(x+5, -y)",      EXPR(using S = decltype(t); return fmod(x + S(5), -y);), I(T(0.1), T(0.2)), I(2, T(2.05)));
    check("fmod(-x-5, y) range",EXPR(using S = decltype(t); return fmod(-x - S(5), y);), V, I(T(1.2), T(1.3)));
    check("fmod(7, y)",         EXPR(using S = decltype(t); return fmod(S(7), y);), V, I(2, T(2.2)));
    check("remainder(x, y) crossing", EXPR(return remainder(x, y);), U, W);
    check("remainder(x, y) exact", EXPR(return remainder(x, y);), V, I(3, 4));
    check("remainder(x+5, y)",  EXPR(using S = decltype(t); return remainder(x + S(5), y);), I(T(0.1), T(0.2)), I(2, T(2.05)));
    check("remainder(x+5, -y)", EXPR(using S = decltype(t); return remainder(x + S(5), -y);), I(T(0.1), T(0.2)), I(2, T(2.05)));
    check("remainder(7, y)",    EXPR(using S = decltype(t); return remainder(S(7), y);), V, I(2, T(2.2)));
    check("tgamma(x*y + 2)",    EXPR(using S = decltype(t); return tgamma(x*y + S(2));), U, V);
    check("lgamma(x) - lgamma(y)", EXPR(return lgamma(x) - lgamma(y);), W, I(T(0.1), T(0.5)));
    check("pow(x, y)",          EXPR(return pow(x, y);), W, U);
    check("pow(x, x*y)",        EXPR(return pow(x, x*y);), W, U);
    check("pow(2, y)",          EXPR(using S = decltype(t); return pow(S(2), y);), U, U);
    check("pow(0.3, x+y)",      EXPR(using S = decltype(t); return pow(S(0.3), x + y);), U, U);

    {
        /* atan2 of the same arguments cancels */
        const auto x = Form<T>::template from_interval<0>(W);
        const auto y = Form<T>::template from_interval<1>(U);
        ++checks;
        if (!((atan2(y, x) - atan2(y, x)).affine_radius() == T(0))) { fail(std::string(type_name<T>()) + " atan2 does not cancel"); }
        
        /* fmod(x+5, y) = x + 5 - 2y exactly when trunc((x+5)/y) = 2 */
        const auto z = Form<T>::template from_interval<2>(I(2, T(2.05)));
        const auto w = Form<T>::template from_interval<3>(I(T(0.1), T(0.2)));
        ++checks;
        if (!((fmod(w + T(5), z) - (w + T(5) - z*T(2))).affine_radius() == T(0))) { fail(std::string(type_name<T>()) + " fmod is not affine"); }
        
        /* derivative enclosures of the gamma functions (secant slopes) contain the derivative */
        auto derivative_check = [&](const std::string& name, auto fn, auto ref) {
            const T points[] = { T(0.3), T(1), T(1.4616), T(2.5), T(7) };
            for (const T u : points) {
                ++checks;
                const cusvna::Interval<T> D = fn.DF(u);
                const LD h = 1e-5L;
                const LD d = (ref(LD(u) + h) - ref(LD(u) - h)) / (2*h);   /* error of order h^2 */
                const LD tol = std::fmax(std::fabs(d), LD(1)) * 1e-9L;
                if (!(LD(D.a) <= d + tol && d - tol <= LD(D.b))) {
                    std::cerr << "  " << name << "'(" << u << ") = " << d << " not in [" << D.a << ", " << D.b << "]" << std::endl;
                    fail(std::string(type_name<T>()) + " " + name + " derivative enclosure");
                }
            }
        };
        derivative_check("tgamma", cusvna::detail::tgamma_fn(), [](LD v) { return std::tgamma(v); });
        derivative_check("lgamma", cusvna::detail::lgamma_fn(), [](LD v) { return std::lgamma(v); });

        /* remquo: the quotient range contains rint(x/c) and the remainder is exact when it is known */
        auto quotient_check = [&](const std::string& name, const I& X, const I& Y, const bool known) {
            const auto a = Form<T>::template from_interval<4>(X);
            const auto b = Form<T>::template from_interval<5>(Y);
            int lo, hi, slo, shi;
            const auto r = remquo(a, b, &lo, &hi);
            const auto rs = remquo(a, Y.a, &slo, &shi);
            ++checks;
            if (known != (lo == hi) || (Y.a == Y.b && (slo != lo || shi != hi))) {
                fail(std::string(type_name<T>()) + " remquo " + name + ": unexpected quotient range [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
            }
            if (lo == hi && !((r - (a - b*T(lo))).affine_radius() == T(0))) {
                fail(std::string(type_name<T>()) + " remquo " + name + ": remainder is not x - quo*y");
            }
            for (int i = 0; i <= 16; ++i) {
                for (int j = 0; j <= 16; ++j) {
                    const LD u = LD(T(LD(X.a) + (LD(X.b) - LD(X.a)) * i / 16));
                    const LD v = LD(T(LD(Y.a) + (LD(Y.b) - LD(Y.a)) * j / 16));
                    const LD n = std::nearbyint(u / v);
                    const LD ns = std::nearbyint(u / LD(Y.a));
                    if (n < lo || n > hi || ns < slo || ns > shi) {
                        fail(std::string(type_name<T>()) + " remquo " + name + ": quotient not in range");
                        return;
                    }
                }
            }
        };
        quotient_check("known", I(T(2.1), T(2.4)), I(T(0.75), T(0.75)), true);
        quotient_check("known affine", I(T(4.1), T(4.2)), I(2, T(2.05)), true);
        quotient_check("negative", I(T(-4.2), T(-4.1)), I(2, T(2.05)), true);
        quotient_check("half crossing", I(T(5.1), T(5.2)), I(2, T(2.05)), false);
        quotient_check("ambiguous", I(T(2.1), T(2.4)), I(T(0.9), T(1)), false);
        quotient_check("ambiguous scalar", I(T(2.1), T(2.4)), I(T(0.5), T(0.5)), false);
        quotient_check("wide", I(T(21), T(24)), I(T(0.9), T(1)), false);
        {
            int lo, hi;
            const auto a = Form<T>::template from_interval<4>(I(1, 2));
            const auto b = Form<T>::template from_interval<5>(I(-1, 1));
            const auto r = remquo(a, b, &lo, &hi);
            ++checks;
            if (!r.is_entire() || lo != std::numeric_limits<int>::min() || hi != std::numeric_limits<int>::max()) {
                fail(std::string(type_name<T>()) + " remquo by zero");
            }
        }
        
        /* interval utilities */
        checks += 6;
        if (!(cusvna::hull(I(0, 1), I(3, 4)).a == T(0) && cusvna::hull(I(0, 1), I(3, 4)).b == T(4))) { fail("hull"); }
        if (!cusvna::is_empty(cusvna::intersect(I(0, 1), I(3, 4)))) { fail("intersect empty"); }
        if (!(cusvna::intersect(I(0, 2), I(1, 4)).a == T(1) && cusvna::intersect(I(0, 2), I(1, 4)).b == T(2))) { fail("intersect"); }
        if (!cusvna::contains(I(0, 1), T(0.5)) || cusvna::contains(I(0, 1), T(2))) { fail("contains"); }
        if (!cusvna::subset(I(1, 2), I(0, 3)) || cusvna::subset(I(0, 3), I(1, 2))) { fail("subset"); }
        if (!(cusvna::hull(cusvna::intersect(I(0, 1), I(3, 4)), I(5, 6)).a == T(5))) { fail("hull with empty"); }
    }

    {
        /* min(x, y) + max(x, y) cancels to x + y */
        const auto x = Form<T>::template from_interval<0>(U);
        const auto y = Form<T>::template from_interval<1>(V);
        const auto d = (min(x, y) + max(x, y)) - (x + y);
        ++checks;
        if (!(d.affine_radius() == T(0))) { fail(std::string(type_name<T>()) + " min(x,y) + max(x,y) - (x+y) does not cancel"); }
        
        /* predicates */
        const auto z = Form<T>::template from_interval<2>(I(2, 3));
        checks += 8;
        if (!certainly_lt(x, z) || !certainly_le(x, z) || certainly_lt(x, y) || certainly_ge(x, y)) { 
            fail(std::string(type_name<T>()) + " certainly_lt/le/ge"); 
        }
        if (!certainly_gt(z, T(1)) || !certainly_ge(T(4), z) || certainly_gt(z, T(2.5))) {
            fail(std::string(type_name<T>()) + " certainly_gt/ge with scalars");
        }
        if (!possibly_zero(x) || possibly_zero(z)) {
            fail(std::string(type_name<T>()) + " possibly_zero");
        }
    }

    /* rounding errors */
    /* x + big is inexact in T but exact in long double */
    const T big = (std::is_same<T,float>::value ? T(1e8) : T(128));
    check("(x + big) - big", EXPR(using S = decltype(t); return (x + S(big)) - S(big);), I(T(0.1), T(0.2)), V);
    check("(x + tiny) - 1",  EXPR(using S = decltype(t); return (x + std::numeric_limits<S>::epsilon()/S(8)) - S(1);), I(1, 1), V);
    check("sum 0.1 ten times", EXPR(
            using S = decltype(t);
            auto s = x;
            for (int k = 0; k < 9; ++k) { s = s + S(0.1); }
            return s;), I(T(0.1), T(0.1)), V);
    check("x * 3 / 3",       EXPR(using S = decltype(t); return x * S(3) / S(3);), I(T(1)/T(3), T(1)/T(3)), V);
    check("x * y with rounding", EXPR(return x * y;), I(T(0.1), T(0.1)), I(T(0.3), T(0.3)));
    check("x / y",           EXPR(return x / y;), I(T(1), T(1)), I(T(3), T(3)));
    check("interval ctor",   EXPR(return x + y;), I(T(0.1), T(0.7)), I(T(1e-3), T(1e-3)));
    check("polynomial horner", EXPR(
            using S = decltype(t);
            return ((((x*S(0.3) - S(1.7))*x + S(2.1))*x - S(0.9))*x + S(0.1));), I(T(0.4), T(0.6)), V);
}

int main() {
    run<float>();
    run<double>();

    std::cout << checks << " checks, " << failures << " failure(s)." << std::endl;
    return (failures ? 1 : 0);
}
