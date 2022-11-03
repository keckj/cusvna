
#include "cusvna/aa.cuh"

using T = float;
using I = unsigned char;
using G = cusvna::LocalThreadIndex<I>;
constexpr I N = 2;
    
using AAF = cusvna::AA<T,I,G,N>;
using interval = cusvna::Interval<T>;

/* Reproduce example 1 of libaffa */
__global__ void kernel() {
  G g;

  interval u(-2,2);
  interval v(-1,1);

  AAF x(u,g);
  AAF r(v,g);
  AAF s(v,g);

  AAF temp1 = (10.0f+x+r);
  AAF temp2 = (10.0f-x+s);

  /*AAF z = (10.0f+x+r)*(10.0f-x+s);*/
}

int main()
{
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0; 
}
