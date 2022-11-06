/* This test compares outputs between cusvna and libaffa, corresponds to example 2 */
#include "tests/utils.h"
#include "aa.h" /* libaffa */
#include "cusvna/aa.cuh"

using T = double;
using I = unsigned;
using G = cusvna::LocalThreadIndex<I>;
constexpr I N = 16;
using AA = cusvna::AA<T,I,G,N>;
using Interval = cusvna::Interval<T>;

template <typename R>
R f(const R& x, const R& y) {
    R xy = x*y;
    return x*x + y*y + xy - xy*xy/T(2) - T(1)/T(4);
}

int main() {
    T xmin(-2), xmax(2);
    T ymin(-2), ymax(2);
    I nintervals(4);
    T dx((xmax-xmin)/nintervals);
    T dy((ymax-ymin)/nintervals);
    
    G g;
    for (I i = 0; i < nintervals; ++i) {
        Interval u(ymin + i*dy, ymin + (i+1)*dy);
        interval u_(ymin + i*dy, ymin + (i+1)*dy);
        AA y(u,g);
        AAF y_(u_);

        for (I j = 0; j < nintervals; ++j) {
            Interval v(xmin + j*dx, xmin + (j+1)*dx);
            interval v_(xmin + j*dx, xmin + (j+1)*dx);
            AA x(v,g);
            AAF x_(v_);

            AA z = f(x,y);
            AAF z_ = f(x_,y_);

            assert_equal(z, z_);
        }
    }
}
