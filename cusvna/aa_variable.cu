
#ifndef __CUDACC_RTC__
#include <iostream>
#endif

#include <cuda/std/array>

#include <cusvna/detail.cuh>
#include <cusvna/interval.cuh>

namespace cusvna {

using detail::type_id, detail::type_id_t;

template <typename T, type_id_t... Is>
class AA {
    /* Is... represent the unique indices of noise symbols. 
       Is cannot be sorted because type_id_t does not support comparisson (just equality). */

    public:
        __host__ __device__ AA(const T& b=T(0));
        __host__ __device__ AA(const AA& other);
        __host__ __device__ AA(const Interval<T>& interval);

        __host__ __device__ AA& operator=(const AA& other);

        explicit __host__ __device__ operator Interval<T>() const;  /* cast operator to convert back an affine form to an interval */

                                             static constexpr int N = sizeof...(Is);   /* number of active noise symbols 0 <= n <= N. */
        template <int I> __host__ __device__ static constexpr type_id_t idx();               /* returns Is[I] which represents the index of noise symbol number I. */
        
        /* base coefficient of the affine form */
        __host__ __device__       T& b();        
        __host__ __device__ const T& b() const;
        
        /* x<I> is the magnitude vector of the uncertainty coerresponding to noise symbol number I. */
        template <int I> __host__ __device__       T& x();
        template <int I> __host__ __device__ const T& x() const;
            
        __host__ __device__ T radius() const;             /* sum of absolute symbol magnitudes */
        __host__ __device__ Interval<T> interval() const; /* cast to an interval */
        __host__ __device__ Sign sign() const;            /* shortcut for Interval<T>(*this).sign() */
        __host__ __device__ void print() const;           /* alternative to std::cout for nvrtc */
            
        __host__ __device__ AA operator-() const;
        __host__ __device__ AA operator+() const;

        __host__ __device__ AA& operator-=(const T& other);
        __host__ __device__ AA& operator+=(const T& other);
        __host__ __device__ AA& operator*=(const T& other);
        __host__ __device__ AA& operator/=(const T& other);
            
        __host__ __device__ AA& operator-=(const AA& other);
        __host__ __device__ AA& operator+=(const AA& other);
    
        /* apply N times a functor with arguments (k, Is[k], x[k], args...) where k=range(N) */
        template <typename Functor, typename... Args> __host__ __device__       AA& apply(Functor&& f, Args&&... args);
        template <typename Functor, typename... Args> __host__ __device__ const AA& apply(Functor&& f, Args&&... args) const;

        /* apply an integer templated functor on range(N) with arguments (*this, args...) */
        template <template <int> class Functor, typename... Args> __host__ __device__       AA& constexpr_apply(Args&&... args);
        template <template <int> class Functor, typename... Args> __host__ __device__ const AA& constexpr_apply(Args&&... args) const;
        

    private:
        template <typename Functor, typename... Args, int... Ks> __host__ __device__ void apply_impl(Functor&& f, Args&&... args, detail::integer_sequence<Ks...>) const;
        template <typename Functor, typename... Args, int... Ks> __host__ __device__ void apply_impl(Functor&& f, Args&&... args, detail::integer_sequence<Ks...>);
        template <template <int> class Functor, int K, typename... Args> __host__ __device__ void constexpr_apply_impl(Args&&... args) const;
        template <template <int> class Functor, int K, typename... Args> __host__ __device__ void constexpr_apply_impl(Args&&... args);

    protected:
        T m_b;
        cuda::std::array<T,N> m_x;
};

    
/* implementation */   
template <typename T, type_id_t... Is>
AA<T,Is...>::AA(const T& b): m_b(b), m_x{} {}

template <typename T, type_id_t... Is>
AA<T,Is...>::AA(const AA& other): m_b(other.m_b), m_x(other.m_x) {}

template <typename T, type_id_t... Is>
AA<T,Is...>::AA(const Interval<T>& interval): m_b(interval.midpoint()), m_x{interval.radius()} {
    static_assert(N==1, "Can only construct from interval when there is only one noise symbol.");
}

template <typename T, type_id_t... Is>
template <int I> 
constexpr type_id_t AA<T,Is...>::idx() { 
    static_assert(I<N); 
    return cuda::std::get<I>(cuda::std::make_tuple(Is...));
}

template <typename T, type_id_t... Is>
T& AA<T,Is...>::b() {
    return m_b;
}
        
template <typename T, type_id_t... Is>
const T& AA<T,Is...>::b() const {
    return m_b;
}
    
template <typename T, type_id_t... Is>
template <int I> 
T& AA<T,Is...>::x() {
    static_assert(I<N); 
    return cuda::std::get<I>(m_x);
}
        
template <typename T, type_id_t... Is>
template <int I> 
const T& AA<T,Is...>::x() const {
    static_assert(I<N); 
    return cuda::std::get<I>(m_x);
}
        
template <typename T, type_id_t... Is>
template <typename Functor, typename... Args> 
AA<T,Is...>& AA<T,Is...>::apply(Functor&& f, Args&&... args) {
    this->apply_impl(std::forward<Functor>(f), std::forward<Args>(args)..., detail::make_integer_sequence<N>{});
    return *this;
}

template <typename T, type_id_t... Is>
template <typename Functor, typename... Args> 
const AA<T,Is...>& AA<T,Is...>::apply(Functor&& f, Args&&... args) const {
    this->apply_impl(std::forward<Functor>(f), std::forward<Args>(args)..., detail::make_integer_sequence<N>{});
    return *this;
}

template <typename T, type_id_t... Is>
template <typename Functor, typename... Args, int... Ks> 
void AA<T,Is...>::apply_impl(Functor&& f, Args&&... args, detail::integer_sequence<Ks...>) {
    (f(Ks, this->idx<Ks>(), this->x<Ks>(), args...), ...);
}

template <typename T, type_id_t... Is>
template <typename Functor, typename... Args, int... Ks> 
void AA<T,Is...>::apply_impl(Functor&& f, Args&&... args, detail::integer_sequence<Ks...>) const {
    (f(Ks, this->idx<Ks>(), this->x<Ks>(), args...), ...);
}
        
template <typename T, type_id_t... Is>
template <template <int> class Functor, typename... Args>
const AA<T,Is...>& AA<T,Is...>::constexpr_apply(Args&&... args) const {
    if constexpr (N > 0) {
        this->constexpr_apply_impl<Functor,0,Args...>(std::forward<Args>(args)...);
    }
    return *this;
}
    
template <typename T, type_id_t... Is>
template <template <int> class Functor, int K, typename... Args>
void AA<T,Is...>::constexpr_apply_impl(Args&&... args) const {
    Functor<K>()(*this, args...);
    if constexpr (K+1<N) {
        this->constexpr_apply_impl<Functor,K+1,Args...>(std::forward<Args>(args)...);
    }
}

template <typename T, type_id_t... Is>
template <template <int> class Functor, typename... Args>
AA<T,Is...>& AA<T,Is...>::constexpr_apply(Args&&... args) {
    if constexpr (N > 0) {
        this->constexpr_apply_impl<Functor,0,Args...>(std::forward<Args>(args)...);
    }
    return *this;
}
    
template <typename T, type_id_t... Is>
template <template <int> class Functor, int K, typename... Args>
void AA<T,Is...>::constexpr_apply_impl(Args&&... args) {
    Functor<K>()(*this, args...);
    if constexpr (K+1<N) {
        this->constexpr_apply_impl<Functor,K+1,Args...>(std::forward<Args>(args)...);
    }
}

#if 0
template <typename T, type_id_t... Is, type_id_t... Js, typename ResultType=detail::merged_t<AA<T,Is...>, AA<T,Js...>>>
ResultType operator+(const AA<T,Is...>& lhs, const AA<T,Js...>& rhs) {
    ResultType out;
    return out;
}
#endif
    
template <typename T, type_id_t... Is>
T AA<T,Is...>::radius() const {
    T r(0);
    this->apply([&](int k, type_id_t i, const T& xi) { r += ::abs(xi); });
    return r;
};
    
template <typename T, type_id_t... Is>
AA<T,Is...>::operator Interval<T>() const {
    const T c = this->b();
    const T r = this->radius();
    return Interval<T>(c-r, c+r);
}
    
template <typename T, type_id_t... Is>
Interval<T> AA<T,Is...>::interval() const {
    return Interval<T>(*this);
};

template <typename T, type_id_t... Is>
Sign AA<T,Is...>::sign() const {
    return this->interval().sign();
};
        
template <typename T, type_id_t... Is>
AA<T,Is...> AA<T,Is...>::operator+() const {
    AA<T,Is...> out(*this);
    out.b() = +out.b();
    return out.apply([](int k, type_id_t i, T& xi) { xi = +xi; });
}

template <typename T, type_id_t... Is>
AA<T,Is...> AA<T,Is...>::operator-() const {
    AA<T,Is...> out(*this);
    out.b() = -out.b();
    return out.apply([](int k, type_id_t i, T& xi) { xi = -xi; });
}

template <typename T, type_id_t... Is>
AA<T,Is...>& AA<T,Is...>::operator+=(const T& other)  {
    this->b() += other;
    return *this;
}

template <typename T, type_id_t... Is>
AA<T,Is...>& AA<T,Is...>::operator-=(const T& other)  {
    this->b() -= other;
    return *this;
}

template <typename T, type_id_t... Is>
AA<T,Is...>& AA<T,Is...>::operator*=(const T& other)  {
    this->b() *= other;
    return this->apply([&](int k, type_id_t i, T& xi) { xi *= other; });
}

template <typename T, type_id_t... Is>
AA<T,Is...>& AA<T,Is...>::operator/=(const T& other)  {
    this->b() /= other;
    return this->apply([&](int k, type_id_t i, T& xi) { xi /= other; });
}

namespace impl {
    template <int K>
    struct inplace_add_functor {
        template <typename T, type_id_t... Is>
        __host__ __device__ void operator()(AA<T,Is...>& lhs, const AA<T,Is...>& rhs) const {
            lhs.template x<K>() += rhs.template x<K>(); 
        }
    };
    
    template <int K>
    struct inplace_subtract_functor {
        template <typename T, type_id_t... Is>
        __host__ __device__ void operator()(AA<T,Is...>& lhs, const AA<T,Is...>& rhs) const {
            lhs.template x<K>() -= rhs.template x<K>(); 
        }
    };
}

template <typename T, type_id_t... Is>
AA<T,Is...>& AA<T,Is...>::operator+=(const AA<T,Is...>& other)  {
    this->b() += other.b();
    return this->constexpr_apply<impl::inplace_add_functor>(other);
}

template <typename T, type_id_t... Is>
AA<T,Is...>& AA<T,Is...>::operator-=(const AA<T,Is...>& other)  {
    this->b() -= other.b();
    return this->constexpr_apply<impl::inplace_subtract_functor>(other);
}
    

#ifndef __CUDACC_RTC__
    /* example for apply_constexpr, there is no templated lambdas until c++ 20 so we use a struct */
    namespace impl {
        template <int K>
        struct print_functor {
            template <typename T, type_id_t... Is>
            __host__ __device__ void operator()(const AA<T,Is...>& val, std::ostream& os) const {
#ifndef __CUDA_ARCH__
                using V = AA<T,Is...>;
                os << intptr_t(V::template idx<K>()) - intptr_t(cusvna::detail::type_id<cusvna::detail::integer_sequence<0>>) << ": " << val.template x<K>();
                if constexpr (K != V::N-1) {
                    os << ", ";
                }
#endif /* endif __CUDA_ARCH__ */
            }
        };
    }

    template <typename T, type_id_t... Is>
    __host__ std::ostream& operator<<(std::ostream& os, const AA<T,Is...>& val) {
        using V = AA<T,Is...>;

        os << "AA(b=" << val.b() << ", x=[";
        val.template constexpr_apply<impl::print_functor>(os);
        os << "])";

        return os;
    }
#endif /* endif __CUDACC_RTC__ */

} /* end of namespace cusvna */


#define aa_from_interval(interval) cusvna::AA<decltype(interval)::type, cusvna::detail::type_id<cusvna::detail::integer_sequence<__COUNTER__>>>(interval)


int main() {
    using T = float;
    using cusvna::Interval;
    const Interval<T> ii(-2,2);
    auto x = aa_from_interval(ii);
    auto y = aa_from_interval(ii);
    std::cout << x << std::endl;
    std::cout << y << std::endl;
    x += 1.0f;
    x *= 2.0f;
    y -= 1.0f;
    y /= 2.0f;
    x += x;
    y -= y;
    std::cout << ii << std::endl;
    std::cout << x << std::endl;
    std::cout << y << std::endl;
    std::cout << (+x).radius() << std::endl;
    std::cout << (-y).radius() << std::endl;
    std::cout << x.interval() << std::endl;
    std::cout << x.sign() << std::endl;
}
