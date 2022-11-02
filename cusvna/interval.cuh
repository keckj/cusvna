#ifndef INTERVAL_H_V9EGLT0S
#define INTERVAL_H_V9EGLT0S

namespace cusvna {

    template <typename T>
    class Interval {
        T a, b;
        Interval() = default;
        Interval(const Interval& other) = default;
        Interval(const T& _a, const T& _b): a(_a), b(_b) {}
        T length() const { return abs(b-a); };
    };

} /* end of namespace cusvna */

#endif /* end of include guard: INTERVAL_H_V9EGLT0S */
