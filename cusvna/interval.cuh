#ifndef INTERVAL_H_V9EGLT0S
#define INTERVAL_H_V9EGLT0S

#ifndef __CUDACC_RTC__
#include <ostream>
#endif

#include <cuda/std/cassert>

namespace cusvna {

    enum Sign: unsigned char {
        UNKNOWN  = 0,
        POSITIVE = 1,
        NEGATIVE = 2,
    };

    template <typename T>
    struct Interval {
        using type = T;
        T a, b;
        Interval() = default;
        Interval(const Interval& other) = default;
        Interval& operator=(const Interval& other) = default;
        __host__ __device__ Interval(const T& _a, const T& _b): a(_a), b(_b) { }
        __host__ __device__ T length() const { return (b-a); };
        __host__ __device__ T radius() const { return (b-a)/T(2); };
        __host__ __device__ T midpoint() const { return (a+b)/T(2); };
        __host__ __device__ Sign sign() const {
            return (a>b || a!=a || b!=b ? Sign::UNKNOWN : (b<=T(0) ? Sign::NEGATIVE : (a>=T(0) ? Sign::POSITIVE : Sign::UNKNOWN)));
        }
    };

#ifndef __CUDACC_RTC__
    template <typename T>
    __host__ std::ostream& operator<<(std::ostream& os, const Interval<T>& interval) {
        os << '[' << interval.a << ", " << interval.b << ']';
        return os;
    }
    
    __host__ std::ostream& operator<<(std::ostream& os, const Sign& sign) {
        switch (sign) {
            case Sign::UNKNOWN:
                os << "UNKNOWN";
                break;
            case Sign::NEGATIVE:
                os << "NEGATIVE";
                break;
            case Sign::POSITIVE:
                os << "POSITIVE";
                break;
        }
        return os;
    }
#endif

} /* end of namespace cusvna */

#endif /* end of include guard: INTERVAL_H_V9EGLT0S */
