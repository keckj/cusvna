/* This test compares outputs between cusvna and libaffa, corresponds to example 1 */
#include "tests/utils.h"

using Interval = cusvna::Interval<double>;
   
int main() {
    Interval u(-2,2); interval u_(-2,2);
    Interval v(-1,1); interval v_(-1,1);

    auto x = new_aa_from_interval(u); AAF x_(u_);
    auto r = new_aa_from_interval(v); AAF r_(v_);
    auto s = new_aa_from_interval(v); AAF s_(v_);

    assert_equal(x, x_);
    assert_equal(r, r_);
    assert_equal(s, s_);
    
    auto temp1(10.0+x+r); AAF temp1_(10.0+x_+r_);
    auto temp2(10.0-x-s); AAF temp2_(10.0-x_-s_);
    
    assert_equal(temp1, temp1_);
    assert_equal(temp2, temp2_);

    auto  z  = (10.0+x+r)*(10.0-x+s);
    AAF z_ = (10.0+x_+r_)*(10.0-x_+s_);
    assert_equal(z, z_);
    assert_equal(Interval(z), z_.convert());
}
