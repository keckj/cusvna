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
