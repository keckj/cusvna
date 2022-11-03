#ifndef INTERVAL_H_V9EGLT0S
#define INTERVAL_H_V9EGLT0S

#include <cuda/std/cassert>

namespace cusvna {

    template <typename T>
    struct Interval {
        T a, b;
        Interval() = default;
        Interval(const Interval& other) = default;
        Interval& operator=(const Interval& other) = default;
        __host__ __device__ Interval(const T& _a, const T& _b): a(_a), b(_b) { assert(a <= b); }
        __host__ __device__ T length() const { return b-a; };
        __host__ __device__ T radius() const { return (b-a)/T(2); };
        __host__ __device__ T midpoint() const { return (a+b)/T(2); };
    };

    template <typename T>
    __host__ std::ostream& operator<<(std::ostream& os, const Interval<T>& interval) {
        os << '[' << interval.a << ", " << interval.b << ']';
        return os;
    }

} /* end of namespace cusvna */

#endif /* end of include guard: INTERVAL_H_V9EGLT0S */
