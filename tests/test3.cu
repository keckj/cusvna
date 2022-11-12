/* This test compares outputs between cusvna and libaffa, corresponds to example 2 */
#include "tests/utils.h"
#include "aa.h" /* libaffa */
#include "cusvna/aa.cuh"

using T = double;
using I = unsigned;
using Interval = cusvna::Interval<T>;

template <typename X, typename Y>
auto f(const X& x, const Y& y) {
    auto xy = x*y;
    return x*x + y*y + xy - xy*xy/T(2) - T(1)/T(4);
}

int main() {
    T xmin(-2), xmax(2);
    T ymin(-2), ymax(2);
    I nintervals(4);
    T dx((xmax-xmin)/nintervals);
    T dy((ymax-ymin)/nintervals);
    
    for (I i = 0; i < nintervals; ++i) {
        Interval u(ymin + i*dy, ymin + (i+1)*dy);
        interval u_(ymin + i*dy, ymin + (i+1)*dy);
        auto y = aa_from_interval(u);
        AAF y_(u_);

        for (I j = 0; j < nintervals; ++j) {
            Interval v(xmin + j*dx, xmin + (j+1)*dx);
            interval v_(xmin + j*dx, xmin + (j+1)*dx);
            auto x = aa_from_interval(v);
            AAF x_(v_);

            auto z = f(x,y);
            AAF z_ = f(x_,y_);

            assert_tighter(z, z_);
        }
    }
}
