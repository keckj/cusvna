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
R f(const R& x) {
    R y(x);
    y = sqrt(y*y - y + 0.5) / sqrt(y*y + 0.5); // y(x)
    y = sqrt(y*y - y + 0.5) / sqrt(y*y + 0.5); // y(y(x))
    return y;
}

int main() {
    T xmin(0), xmax(1);
    I nintervals(24);
    T dx((xmax-xmin)/nintervals);
    
    G g; g(); /* libaffa index 0 is reserved */
    for (I i = 0; i < nintervals; ++i) {
        Interval u(xmin + i*dx, xmin + (i+1)*dx);
        AA x(u,g);
        AA y = f(x);
        
        interval u_(xmin + i*dx, xmin + (i+1)*dx);
        AAF x_(u_);
        AAF y_ = f(x_);

        assert_equal(y, y_);
    }
}
