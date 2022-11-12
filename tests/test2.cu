/* This test compares outputs between cusvna and libaffa, corresponds to example 2 */
#include "tests/utils.h"
#include "aa.h" /* libaffa */
#include "cusvna/aa.cuh"

using I = int;
using T = double;
using Interval = cusvna::Interval<T>;

template <typename R>
auto f(const R& x) {
    auto y = sqrt(x*x - x + 0.5) / sqrt(x*x + 0.5); // y(x)
    auto z = sqrt(y*y - y + 0.5) / sqrt(y*y + 0.5); // y(y(x))
    return z;
}

int main() {
    T xmin(0), xmax(1);
    I nintervals(24);
    T dx((xmax-xmin)/nintervals);
    
    for (I i = 0; i < nintervals; ++i) {
        Interval u(xmin + i*dx, xmin + (i+1)*dx);
        auto x = new_aa_from_interval(u);
        auto y = f(x);
        
        interval u_(xmin + i*dx, xmin + (i+1)*dx);
        AAF x_(u_);
        AAF y_ = f(x_);

        assert_tighter(y, y_);
    }
}
