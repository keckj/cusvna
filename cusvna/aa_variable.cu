
#ifndef __CUDACC_RTC__
#include <iostream>
#endif

#include <cuda/std/array>

#include <cusvna/detail.cuh>
#include <cusvna/interval.cuh>

namespace cusvna {

template <typename T, int... Is>
class AA {
    /* Is... represent the indices of noise symbols. It is assumed to be sorted. */
    static_assert(detail::is_ascending_seq<Is...>(), "Indices are not sorted in strictly ascending order.");

    public:
        __host__ __device__ AA(const T& b=T(0));
        __host__ __device__ AA(const AA& other);
        __host__ __device__ AA(const Interval<T>& interval);
        __host__ __device__ AA(AA&& other) = delete;

        __host__ __device__ AA& operator=(const AA& other);
        __host__ __device__ AA& operator=(AA&& other) = delete;

        explicit __host__ __device__ operator Interval<T>() const;  /* cast operator to convert back an affine form to an interval */

                         static constexpr int N = sizeof...(Is);  /* number of active noise symbols 0 <= n <= N. */
        template <int I> static constexpr int idx();              /* returns Is[I] which represents the index of noise symbol number I. */

                         T& b();  /* base coefficient of the affine form */
        template <int I> T& x();  /* x<I> is the magnitude vector of the uncertainty coerresponding to noise symbol number I. */

                         const T& b() const;
        template <int I> const T& x() const;

    protected:
        T m_b;
        cuda::std::array<T,N> m_x;
};

    
/* implementation */   
template <typename T, int... Is>
AA<T,Is...>::AA(const T& b): m_b(b), m_x{} {}

template <typename T, int... Is>
AA<T,Is...>::AA(const AA& other): m_b(other.m_b), m_x(other.m_x) {}

template <typename T, int... Is>
AA<T,Is...>::AA(const Interval<T>& interval): m_b(interval.midpoint()), m_x{interval.radius()} {
    static_assert(N==1, "Can only construct from interval when there is only one noise symbol.");
}

template <typename T, int... Is>
template <int I> 
constexpr int AA<T,Is...>::idx() { 
    static_assert(I<N); 
    return cuda::std::get<I>(cuda::std::make_tuple(Is...));
}

template <typename T, int... Is>
T& AA<T,Is...>::b() {
    return m_b;
}
        
template <typename T, int... Is>
const T& AA<T,Is...>::b() const {
    return m_b;
}
    
template <typename T, int... Is>
template <int I> 
T& AA<T,Is...>::x() {
    static_assert(I<N); 
    return cuda::std::get<I>(m_x);
}
        
template <typename T, int... Is>
template <int I> 
const T& AA<T,Is...>::x() const {
    static_assert(I<N); 
    return cuda::std::get<I>(m_x);
}
        
template <typename T, int... Is, int... Js, typename ResultType=detail::merged_t<AA<T,Is...>, AA<T,Js...>>>
ResultType operator+(const AA<T,Is...>& lhs, const AA<T,Js...>& rhs) {
    return ResultType();
}

#ifndef __CUDACC_RTC__
    template <typename T, int... Is>
    __host__ std::ostream& operator<<(std::ostream& os, const AA<T,Is...>& val) {
        using V = AA<T,Is...>;
        os << "AA(b=" << val.b() << ", x=[";
        for (int i = 0; i < V::N; ++i) {
            os << V::template idx<0>() << ": " << val.template x<0>();
            if (i != V::N-1) { os << ", "; }
        }
        os << "])";
        return os;
    }
#endif

} /* end of namespace cusvna */

int main() {
    using T = float;
    using cusvna::AA, cusvna::Interval;
    const Interval<T> ii(-1,2.0);
    AA<T,__COUNTER__> x(ii);
    AA<T,__COUNTER__> y(ii);
    auto z = x + y;
    std::cout << x << std::endl;
    std::cout << y << std::endl;
    std::cout << z << std::endl;
}
