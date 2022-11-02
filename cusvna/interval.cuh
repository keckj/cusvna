#ifndef INTERVAL_H_V9EGLT0S
#define INTERVAL_H_V9EGLT0S

namespace cusvna {

    template <typename T>
    struct Interval {
        T a, b;
        Interval() = default;
        Interval(const Interval& other) = default;
        Interval& operator=(const Interval& other) = default;
        __device__ Interval(const T& _a, const T& _b): a(_a), b(_b) { assert(a <= b); }
        __device__ T length() const { return b-a; };
        __device__ T radius() const { return (b-a)/T(2); };
        __device__ T midpoint() const { return (a+b)/T(2); };
    };

} /* end of namespace cusvna */

#endif /* end of include guard: INTERVAL_H_V9EGLT0S */
