/* This test checks the soundness of all operations and elementary functions with rounding, in single
   and double precision. Enclosures are checked against pointwise evaluations of the same expressions
   in extended precision (long double). Bounds must never be nan, and must be finite for inputs inside
   the domain of the functions. Point inputs must give tight enclosures. */
#include <cmath>
#include <math.h>
#include <iostream>
#include <string>
#include <type_traits>

#include "cusvna/aa.cuh"

using LD = long double;

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
LD clamp(const LD x, const LD lo, const LD hi) { return min(max(x, lo), hi); }
LD norm3d(const LD x, const LD y, const LD z) { return std::hypot(x, y, z); }
LD rhypot(const LD x, const LD y) { return LD(1) / std::hypot(x, y); }
LD rnorm3d(const LD x, const LD y, const LD z) { return LD(1) / std::hypot(x, y, z); }
LD lerp(const LD a, const LD b, const LD t) { return a + t*(b - a); }
LD smoothstep(const LD e0, const LD e1, const LD x) { const LD t = clamp((x - e0)/(e1 - e0), 0, 1); return t*t*(3 - 2*t); }
LD smooth_min(const LD a, const LD b, const LD k) { const LD h = max(k - std::fabs(a - b), 0) / k; return min(a, b) - h*h*k/4; }
LD smooth_max(const LD a, const LD b, const LD k) { return -smooth_min(-a, -b, k); }

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

/* check that f(u, v) belongs to f(x, y).interval() for (u, v) sampled in X x Y
   f is called as f(x, y, T()) where T gives the floating point type of the AA computation */
template <typename T, typename F>
void check_enclosure(const std::string& name, F&& f, const cusvna::Interval<T>& X, const cusvna::Interval<T>& Y,
        const bool expect_finite, const int n=32) {
    ++checks;
    const std::string what = std::string(type_name<T>()) + " " + name + " on X=[" + std::to_string(X.a) + "," +
        std::to_string(X.b) + "], Y=[" + std::to_string(Y.a) + "," + std::to_string(Y.b) + "]";

    const auto x = cusvna::AA<T>::template from_interval<0>(X);
    const auto y = cusvna::AA<T>::template from_interval<1>(Y);
    const cusvna::Interval<T> r = f(x, y, T(0)).interval();

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
            const LD ref = f(u, v, T(0));
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
    const auto x = cusvna::AA<T>::template from_interval<0>(cusvna::Interval<T>(x0, x0));
    const auto y = cusvna::AA<T>::template from_interval<1>(cusvna::Interval<T>(T(0), T(0)));
    const cusvna::Interval<T> r = f(x, y, T(0)).interval();
    const LD ref = f(LD(x0), LD(0), T(0));
    const LD ulp = std::numeric_limits<T>::epsilon() * std::fmax(std::fabs(ref), LD(std::numeric_limits<T>::min()));
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
        auto g = [&](auto x, auto y, auto t) { return f(x, t); };
        check_enclosure<T>(name, g, X, I(0, 0), in_domain(LD(X.a), LD(X.b)));
    }
    const T points[] = { T(0.1), T(0.5), T(0.9), T(1.3), T(2), T(-0.3), T(-1.1), T(-2.5), T(7.25) };
    for (const T x0 : points) {
        if (in_domain(LD(x0), LD(x0)) && x0 != T(0)) {
            auto g = [&](auto x, auto y, auto t) { return f(x, t); };
            check_tight<T>(name, g, x0, max_ulps);
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

    check_unary<T>("-x",         [](auto x, auto t) { return -x; }, always, 0);
    check_unary<T>("reciprocal", [](auto x, auto t) { return reciprocal(x); }, nonzero, 16);
    check_unary<T>("sqrt",       [](auto x, auto t) { return sqrt(x); }, nonnegative_part, 16);
    check_unary<T>("exp",        [](auto x, auto t) { return exp(x); }, exp_ok, 24);
    check_unary<T>("log",        [](auto x, auto t) { return log(x); }, positive, 24);
    check_unary<T>("sqr",        [](auto x, auto t) { return sqr(x); }, always, 16);
    check_unary<T>("x^3",        [](auto x, auto t) { return pow(x, 3); }, always, 16);
    check_unary<T>("x^4",        [](auto x, auto t) { return pow(x, 4); }, always, 16);
    check_unary<T>("x^-2",       [](auto x, auto t) { return pow(x, -2); }, nonzero, 16);
    check_unary<T>("x^-3",       [](auto x, auto t) { return pow(x, -3); }, nonzero, 16);
    check_unary<T>("x^0.5",      [](auto x, auto t) { using S = decltype(t); return pow(x, S(0.5)); }, nonnegative_part, 72);
    check_unary<T>("x^2.5",      [](auto x, auto t) { using S = decltype(t); return pow(x, S(2.5)); }, nonnegative_part, 72);
    check_unary<T>("x^-1.5",     [](auto x, auto t) { using S = decltype(t); return pow(x, S(-1.5)); }, positive, 72);
    check_unary<T>("sin",        [](auto x, auto t) { return sin(x); }, always, 24);
    check_unary<T>("cos",        [](auto x, auto t) { return cos(x); }, always, 24);
    check_unary<T>("tan",        [](auto x, auto t) { return tan(x); }, no_tan_pole, 40);
    check_unary<T>("cot",        [](auto x, auto t) { return cot(x); }, no_cot_pole, 48);
    check_unary<T>("sinh",       [](auto x, auto t) { return sinh(x); }, exp_ok, 32);
    check_unary<T>("cosh",       [](auto x, auto t) { return cosh(x); }, exp_ok, 32);
    check_unary<T>("tanh",       [](auto x, auto t) { return tanh(x); }, always, 24);
    check_unary<T>("asinh",      [](auto x, auto t) { return asinh(x); }, always, 32);
    check_unary<T>("acosh",      [](auto x, auto t) { return acosh(x); }, acosh_ok, 40);
    check_unary<T>("atanh",      [](auto x, auto t) { return atanh(x); }, atanh_ok, 32);
    check_unary<T>("abs",        [](auto x, auto t) { return abs(x); }, always, 0);
    check_unary<T>("positive_part", [](auto x, auto t) { return positive_part(x); }, always, 0);
    
    auto arc_ok = [](LD a, LD b) { return b >= -1 && a <= 1; };
    auto log1p_ok = [](LD a, LD b) { return a > -1; };
    auto exp2_ok = [&](LD a, LD b) { return b < max_exp / std::log(LD(2)); };
    auto exp10_ok = [&](LD a, LD b) { return b < max_exp / std::log(LD(10)); };
    check_unary<T>("rsqrt",      [](auto x, auto t) { return rsqrt(x); }, positive, 16);
    check_unary<T>("exp2",       [](auto x, auto t) { return exp2(x); }, exp2_ok, 24);
    check_unary<T>("exp10",      [](auto x, auto t) { return exp10(x); }, exp10_ok, 72);
    check_unary<T>("expm1",      [](auto x, auto t) { return expm1(x); }, exp_ok, 24);
    check_unary<T>("log2",       [](auto x, auto t) { return log2(x); }, positive, 24);
    check_unary<T>("log10",      [](auto x, auto t) { return log10(x); }, positive, 24);
    check_unary<T>("log1p",      [](auto x, auto t) { return log1p(x); }, log1p_ok, 24);
    check_unary<T>("cbrt",       [](auto x, auto t) { return cbrt(x); }, always, 24);
    check_unary<T>("rcbrt",      [](auto x, auto t) { return rcbrt(x); }, nonzero, 24);
    check_unary<T>("atan",       [](auto x, auto t) { return atan(x); }, always, 32);
    check_unary<T>("asin",       [](auto x, auto t) { return asin(x); }, arc_ok, 32);
    check_unary<T>("acos",       [](auto x, auto t) { return acos(x); }, arc_ok, 32);
    check_unary<T>("erf",        [](auto x, auto t) { return erf(x); }, always, 32);
    check_unary<T>("erfc",       [](auto x, auto t) { return erfc(x); }, always, 48);
    check_unary<T>("normcdf",    [](auto x, auto t) { return normcdf(x); }, always, 64);
    check_unary<T>("sinpi",      [](auto x, auto t) { return sinpi(x); }, always, 24);
    check_unary<T>("cospi",      [](auto x, auto t) { return cospi(x); }, always, 24);

    /* composed expressions */
    const I U(-1, 1), V(0, 1), W(1, 2);
    auto check = [&](const std::string& name, auto&& f, const I& X, const I& Y) { check_enclosure<T>(name, f, X, Y, true); };

    check("(x+y)^2 - (x-y)^2", [](auto x, auto y, auto t) { return (x+y)*(x+y) - (x-y)*(x-y); }, U, U);
    check("sqrt(x*x + y*y)",   [](auto x, auto y, auto t) { return sqrt(x*x + y*y); }, W, W);
    check("hypot sqr",         [](auto x, auto y, auto t) { return sqrt(sqr(x) + sqr(y)); }, U, W);
    check("x / (y + 2)",       [](auto x, auto y, auto t) { using S = decltype(t); return x / (y + S(2)); }, U, V);
    check("exp(sin(x))*cos(y)",[](auto x, auto y, auto t) { return exp(sin(x)) * cos(y); }, U, U);
    check("tanh(x*y) + atanh(x/2)", [](auto x, auto y, auto t) { using S = decltype(t); return tanh(x*y) + atanh(x*S(0.5)); }, U, U);
    check("|x - y| + max(x-y,0)", [](auto x, auto y, auto t) { return abs(x - y) + positive_part(x - y); }, U, V);
    check("x^3 - x*x*x",       [](auto x, auto y, auto t) { return pow(x, 3) - x*x*x; }, U, V);
    check("log(1 + x^2) - log(y)", [](auto x, auto y, auto t) { using S = decltype(t); return log(S(1) + sqr(x)) - log(y); }, U, W);
    check("sinh(x) - cosh(y) + asinh(x*y)", [](auto x, auto y, auto t) { return sinh(x) - cosh(y) + asinh(x*y); }, U, U);
    check("test2 f", [](auto x, auto y, auto t) {
            using S = decltype(t);
            auto g = sqrt(x*x - x + S(0.5)) / sqrt(x*x + S(0.5));
            return sqrt(g*g - g + S(0.5)) / sqrt(g*g + S(0.5)); }, I(T(0.25), T(0.375)), V);
    check("test3 f", [](auto x, auto y, auto t) {
            using S = decltype(t);
            auto xy = x*y;
            return x*x + y*y + xy - xy*xy/S(2) - S(1)/S(4); }, I(-2, -1), I(1, 2));

    /* composite functions */
    check("min(x, y)",          [](auto x, auto y, auto t) { return min(x, y); }, U, V);
    check("max(x, y)",          [](auto x, auto y, auto t) { return max(x, y); }, U, V);
    check("min(x, y) disjoint", [](auto x, auto y, auto t) { return min(x, y); }, I(-2, -1), V);
    check("max(x, 0.25)",       [](auto x, auto y, auto t) { using S = decltype(t); return max(x, S(0.25)); }, U, V);
    check("min(0.25, x)",       [](auto x, auto y, auto t) { using S = decltype(t); return min(S(0.25), x); }, U, V);
    check("fmin/fmax",          [](auto x, auto y, auto t) { return fmin(x, y) * fmax(x, y); }, U, V);
    check("clamp(x, -0.5, 0.5)",[](auto x, auto y, auto t) { using S = decltype(t); return clamp(x, S(-0.5), S(0.5)); }, U, V);
    check("clamp(x, y-1, y)",   [](auto x, auto y, auto t) { using S = decltype(t); return clamp(x, y - S(1), y); }, U, V);
    check("hypot(x, y)",        [](auto x, auto y, auto t) { return hypot(x, y); }, U, W);
    check("hypot(x, y) at 0",   [](auto x, auto y, auto t) { return hypot(x, y); }, U, U);
    check("norm3d(x, y, x*y)",  [](auto x, auto y, auto t) { return norm3d(x, y, x*y); }, U, W);
    check("hypot(x, y, x-y)",   [](auto x, auto y, auto t) { return hypot(x, y, x - y); }, U, W);
    check("rhypot(x, y)",       [](auto x, auto y, auto t) { return rhypot(x, y); }, U, W);
    check("rnorm3d(x, y, y)",   [](auto x, auto y, auto t) { return rnorm3d(x, y, y); }, U, W);
    check("fma(x, y, x)",       [](auto x, auto y, auto t) { return fma(x, y, x); }, U, W);
    check("lerp(x, y, 0.3)",    [](auto x, auto y, auto t) { using S = decltype(t); return lerp(x, y, S(0.3)); }, U, W);
    check("lerp(0, 1, x*y)",    [](auto x, auto y, auto t) { using S = decltype(t); return lerp(S(0), S(1), x*y); }, U, V);
    check("smoothstep",         [](auto x, auto y, auto t) { using S = decltype(t); return smoothstep(S(-0.5), S(0.5), x); }, U, V);
    check("smoothstep inside",  [](auto x, auto y, auto t) { using S = decltype(t); return smoothstep(S(-2), S(2), x); }, U, V);
    check("smooth_min",         [](auto x, auto y, auto t) { using S = decltype(t); return smooth_min(x, y, S(0.5)); }, U, V);
    check("smooth_max",         [](auto x, auto y, auto t) { using S = decltype(t); return smooth_max(x, y, S(0.25)); }, U, V);
    check("sdf union of disks", [](auto x, auto y, auto t) { 
            using S = decltype(t);
            const auto d0 = hypot(x - S(0.5), y) - S(0.5);
            const auto d1 = hypot(x + S(0.5), y - S(0.25)) - S(0.4);
            return min(d0, d1); }, U, U);
    check("asin(x/2) + acos(y)",[](auto x, auto y, auto t) { using S = decltype(t); return asin(x*S(0.5)) + acos(y); }, U, V);
    check("atan(x*y) * erf(x)", [](auto x, auto y, auto t) { return atan(x*y) * erf(x); }, U, W);
    check("sinpi(x)*cospi(y)",  [](auto x, auto y, auto t) { return sinpi(x) * cospi(y); }, U, V);
    check("log1p(expm1(x))",    [](auto x, auto y, auto t) { return log1p(expm1(x)); }, V, V);
    check("cbrt(x*y) - rcbrt(y)",[](auto x, auto y, auto t) { return cbrt(x*y) - rcbrt(y); }, U, W);
    
    {
        /* min(x, y) + max(x, y) cancels to x + y */
        const auto x = cusvna::AA<T>::template from_interval<0>(U);
        const auto y = cusvna::AA<T>::template from_interval<1>(V);
        const auto d = (min(x, y) + max(x, y)) - (x + y);
        ++checks;
        if (!(d.affine_radius() == T(0))) { fail(std::string(type_name<T>()) + " min(x,y) + max(x,y) - (x+y) does not cancel"); }
        
        /* predicates */
        const auto z = cusvna::AA<T>::template from_interval<2>(I(2, 3));
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
    check("(x + big) - big", [=](auto x, auto y, auto t) { using S = decltype(t); return (x + S(big)) - S(big); }, I(T(0.1), T(0.2)), V);
    check("(x + tiny) - 1",  [](auto x, auto y, auto t) { using S = decltype(t); return (x + std::numeric_limits<S>::epsilon()/S(8)) - S(1); }, I(1, 1), V);
    check("sum 0.1 ten times", [](auto x, auto y, auto t) {
            using S = decltype(t);
            auto s = x;
            for (int k = 0; k < 9; ++k) { s = s + S(0.1); }
            return s; }, I(T(0.1), T(0.1)), V);
    check("x * 3 / 3",       [](auto x, auto y, auto t) { using S = decltype(t); return x * S(3) / S(3); }, I(T(1)/T(3), T(1)/T(3)), V);
    check("x * y with rounding", [](auto x, auto y, auto t) { return x * y; }, I(T(0.1), T(0.1)), I(T(0.3), T(0.3)));
    check("x / y",           [](auto x, auto y, auto t) { return x / y; }, I(T(1), T(1)), I(T(3), T(3)));
    check("interval ctor",   [](auto x, auto y, auto t) { return x + y; }, I(T(0.1), T(0.7)), I(T(1e-3), T(1e-3)));
    check("polynomial horner", [](auto x, auto y, auto t) {
            using S = decltype(t);
            return ((((x*S(0.3) - S(1.7))*x + S(2.1))*x - S(0.9))*x + S(0.1)); }, I(T(0.4), T(0.6)), V);
}

int main() {
    run<float>();
    run<double>();

    std::cout << checks << " checks, " << failures << " failure(s)." << std::endl;
    return (failures ? 1 : 0);
}
