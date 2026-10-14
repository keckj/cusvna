/* This test checks the soundness of nonlinear operations and of generated noise symbols: two different quantities whose noise symbols share the
   same type must not be correlated, while reusing the same quantity must keep its correlations.
   Range enclosures are checked against pointwise evaluations of the same expressions. */
#include <cmath>
#include <iostream>

#include "cusvna/aa.cuh"

using T = double;
using Interval = cusvna::Interval<T>;
using AA = cusvna::AA<T>;

static int failures = 0;

T reciprocal(const T& x) { return T(1)/x; }

void check(bool ok, const std::string& msg) {
    if (!ok) {
        std::cerr << "FAILED: " << msg << std::endl;
        ++failures;
    }
}

/* check that f(u,v) belongs to f(x,y).interval() for (u,v) sampled in the input intervals */
template <typename F>
void check_enclosure(const std::string& name, F&& f, const Interval& X, const Interval& Y, int n=64) {
    const auto x = AA::from_interval<0>(X);
    const auto y = AA::from_interval<1>(Y);
    const Interval r = f(x, y).interval();

    T vmin = +INFINITY;
    T vmax = -INFINITY;
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            const T u = X.a + i*X.length()/n;
            const T v = Y.a + j*Y.length()/n;
            const T fuv = f(u, v);
            vmin = fmin(vmin, fuv);
            vmax = fmax(vmax, fuv);
        }
    }

    const T eps = T(1e-12) * fmax(T(1), fmax(fabs(vmin), fabs(vmax)));
    const bool ok = (r.a <= vmin + eps) && (r.b >= vmax - eps);
    if (!ok) {
        std::cerr << "  " << name << ": cusvna " << r << " does not enclose sampled range "
            << Interval(vmin, vmax) << std::endl;
    }
    check(ok, name + " enclosure");
}

int main() {
    const Interval U(-1, 1);
    const Interval V( 0, 1);

    /* different quantities sharing the same generated symbol types */
    check_enclosure("(x+y)*(x+y) - (x-y)*(x-y)", [](auto x, auto y) { return (x+y)*(x+y) - (x-y)*(x-y); }, U, U);
    check_enclosure("(x+y)*(x+y) + (x-y)*(x-y)", [](auto x, auto y) { return (x+y)*(x+y) + (x-y)*(x-y); }, U, U);
    check_enclosure("(x+y)*(x-y) - (x-y)*(x+y)", [](auto x, auto y) { return (x+y)*(x-y) - (x-y)*(x+y); }, U, U);
    for (int k = 0; k < 8; ++k) {
        /* subintervals of V, so that nonlinear approximation errors are not hidden by the linear part */
        const Interval W(V.a + k*V.length()/8, V.a + (k+1)*V.length()/8);
        check_enclosure("sqrt(x) - sqrt(1-x)",       [](auto x, auto y) { return sqrt(x) - sqrt(1.0-x); }, W, W);
        check_enclosure("sqrt(x+1) + sqrt(2-x)",     [](auto x, auto y) { return sqrt(x+1.0) + sqrt(2.0-x); }, W, W);
        check_enclosure("sqrt(x+1) - sqrt(2*x+1)",   [](auto x, auto y) { return sqrt(x+1.0) - sqrt(2.0*x+1.0); }, W, W);
        check_enclosure("1/(x+1) - 1/(2-x)",         [](auto x, auto y) { return reciprocal(x+1.0) - reciprocal(2.0-x); }, W, W);
        check_enclosure("log(x+1) - log(2-x)",       [](auto x, auto y) { return log(x+1.0) - log(2.0-x); }, W, W);
        check_enclosure("exp(x)",                    [](auto x, auto y) { return exp(x); }, W, W);
        check_enclosure("exp(4*x-2)",                [](auto x, auto y) { return exp(4.0*x-2.0); }, W, W);
        check_enclosure("exp(x) - exp(x+1)",         [](auto x, auto y) { return exp(x) - exp(x+1.0); }, W, W);
    }
    check_enclosure("exp(x)",                    [](auto x, auto y) { return exp(x); }, V, V);
    check_enclosure("exp(x+y)",                  [](auto x, auto y) { return exp(x+y); }, U, U);
    check_enclosure("x*sqrt(x+1)*sqrt(2-x)",     [](auto x, auto y) { return x*sqrt(x+1.0)*sqrt(2.0-x); }, V, V);
    check_enclosure("(x*y)*(x*y) - (x*x)*(y*y)", [](auto x, auto y) { return (x*y)*(x*y) - (x*x)*(y*y); }, U, U);

    /* same quantities, correlations must be preserved: noise symbols cancel exactly, only the
       independent rounding error terms remain */
    {
        const auto x = AA::from_interval<0>(U);
        const auto y = AA::from_interval<1>(U);
        
        auto cancels = [](const auto& d, const auto& v) { 
            return d.affine_radius() == T(0) && d.radius() <= T(2)*v.err();
        };
        
        const auto z = x*y;
        check(cancels(z - z, z), "z - z == 0 with z = x*y");
        check(cancels(x*y - x*y, z), "x*y - x*y == 0");
        check(cancels(x*y + 1.0 - x*y, z), "x*y + 1 - x*y == 1");
        
        const auto s = sqrt(y + 2.0);
        check(cancels(s - s, s), "s - s == 0 with s = sqrt(y+2)");
        check(cancels(sqrt(y + 2.0) - sqrt(y + 2.0), s), "sqrt(y+2) - sqrt(y+2) == 0");
        check(cancels(z*s - z*s, z*s), "z*s - z*s == 0");
        check(cancels(exp(z) - exp(z), exp(z)), "exp(z) - exp(z) == 0");
        check(cancels(sin(s*z) - sin(s*z), sin(s*z)), "sin(s*z) - sin(s*z) == 0");
    }

    if (failures) {
        std::cerr << failures << " check(s) failed." << std::endl;
        return 1;
    }
    return 0;
}
