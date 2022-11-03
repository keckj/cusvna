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
    y = sqrt(y*y - y + 0.5) / sqrt(y*y + 0.5); // y(y(x))
    y = sqrt(y*y - y + 0.5) / sqrt(y*y + 0.5); // y(y(x))
    return y;
}

int main() {
    G g; 
    g(); /* libaffa index 0 is reserved */

    Interval u(1.1, 1.2);
    AA x(u,g);
    AA y = f(x);
    
    interval u_(1.1, 1.2);
    AAF x_(u_);
    AAF y_ = f(x_);

    assert_equal(y, y_);
}
