/* This test compares outputs between primitives */
#include "tests/utils.h"
#include "aa.h" /* libaffa */
#include "cusvna/aa_variable.cuh"

using T = double;
using Interval = cusvna::Interval<T>;
   
int main() {
    Interval u(-2,2); interval u_(-2,2);
    Interval v(0.5,1); interval v_(0.5,1);

    auto x = aa_from_interval(u); AAF x_(u_);
    auto y = aa_from_interval(v); AAF y_(v_);
    auto z = aa_from_interval(v); AAF z_(v_);
    auto w = aa_from_interval(v); AAF w_(v_);

    assert_equal(x, x_, "x");
    assert_equal(y, y_, "y");
    assert_equal(z, z_, "z");
    assert_equal(w, w_, "w");
    
    assert_equal(+x, x_, "+x");
    assert_equal(+y, y_, "+y");
    assert_equal(+z, z_, "+z");
    assert_equal(+w, w_, "+w");
    
    assert_equal(-x, -x_, "-x");
    assert_equal(-y, -y_, "-y");
    assert_equal(-z, -z_, "-z");
    assert_equal(-w, -w_, "-w");
    
    assert_equal(x+1.0, x_+1.0, "x+1");
    assert_equal(y-2.0, y_-2.0, "y-2");
    assert_equal(z*3.0, z_*3.0, "z*3");
    assert_equal(w/4.0, w_/4.0, "w/4");
    
    assert_equal(1.0+x, 1.0+x_, "1+x");
    assert_equal(2.0-y, 2.0-y_, "2-y");
    assert_equal(3.0*z, 3.0*z_, "3*z");

    x += 1.0; x_=x_+1.0; assert_equal(x, x_, "x+=1");
    y -= 2.0; y_=y_-2.0; assert_equal(y, y_, "y-=2");
    z *= 3.0; z_=z_*3.0; assert_equal(z, z_, "z*=3");
    w /= 4.0; w_=w_/4.0; assert_equal(w, w_, "w/=4");
    
    assert_equal(x*x, x_*x_, "x*x");
    assert_equal(y*y, y_*y_, "y*y");
    assert_equal(x*y, x_*y_, "x*y");
    assert_equal(w*w, w_*w_, "w*w");
    
    /*assert_equal(w.reciprocal(), inv(w_), "1/w");*/
    /*assert_equal(2.0/w, 2.0*inv(w_), "2/w");*/
    /*assert_equal(w/w, w_*inv(w_), "w/w");*/
    /*assert_equal(z/w, z_/w_, "z/w");*/
    
    /*assert_equal(x*y/y, x_*y_/y_, "x*y/y");*/
    /*assert_equal(y*y/y, y_*y_/y_, "y*y/y");*/
    /*assert_equal(w*w/w, w_*w_/w_, "w*w/w");*/
    
    /*assert_equal(sqrt(w), sqrt(w_), "sqrt(w)");*/
    /*assert_equal(exp(w), exp(w_), "exp(w)");*/
    /*assert_equal(log(w), log(w_), "log(w)");*/

    /*assert_equal(sinh(w), sinh(w_), "sinh(w)");*/
    /*assert_equal(cosh(w), cosh(w_), "cosh(w)");*/
    /*assert_equal(tanh(w), tanh(w_), "tanh(w)");*/
    /*assert_equal(cotanh(w), inv(tanh(w_)), "cotanh(w)");*/
}
