/* This test compares outputs between cusvna and libaffa, corresponds to example 1 */

#include <iostream>

#include "aa.h" /* libaffa */

#include "tests/utils.h"
#include "cusvna/aa.cuh"
   
using T = double;
using I = unsigned;
using G = cusvna::LocalThreadIndex<I>;
constexpr I N = 4;
using AA = cusvna::AA<T,I,G,N>;
using Interval = cusvna::Interval<T>;

int main() {
    G g; 
    g(); /* libaffa index 0 is reserved */

    Interval u(-2,2); interval u_(-2,2);
    Interval v(-1,1); interval v_(-1,1);

    AA x(u,g); AAF x_(u_);
    AA r(v,g); AAF r_(v_);
    AA s(v,g); AAF s_(v_);

    assert_equal(x, x_);
    assert_equal(r, r_);
    assert_equal(s, s_);
    
    AA temp1(10.0+x+r); AAF temp1_(10.0+x_+r_);
    AA temp2(10.0-x-s); AAF temp2_(10.0-x_-s_);
    
    assert_equal(temp1, temp1_);
    assert_equal(temp2, temp2_);

    AA  z  = (10.0+x+r)*(10.0-x+s);
    AAF z_ = (10.0+x_+r_)*(10.0-x_+s_);
    assert_equal(z, z_);
    assert_equal(Interval(z), z_.convert());
}
