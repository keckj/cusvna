/* This test compares outputs between cusvna and libaffa, corresponds to example 1 */

#include <iostream>

#include "cusvna/aa.cuh"
    
using T = float;
using I = unsigned short;
using G = cusvna::LocalThreadIndex<I>;
constexpr I N = 2;

using AA = cusvna::AA<T,I,G,N>;
using Interval = cusvna::Interval<T>;

int main() {
    G g;

    Interval u(-2,2);
    Interval v(-1,1);

    std::cout << u << std::endl;
    std::cout << v << std::endl;

    AA x(u,g);
    AA r(v,g);
    AA s(v,g);

    AA temp1 = (10.0f+x+r);
    AA temp2 = (10.0f-x+s);

    /*AAF z = (10.0f+x+r)*(10.0f-x+s);*/
}
