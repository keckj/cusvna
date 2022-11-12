#ifndef AA_CUH_NLBP0A1W
#define AA_CUH_NLBP0A1W

#ifndef __CUDACC_RTC__
#include <ostream>
#include <typeinfo>
#include <typeindex>
#endif

#include <cusvna/detail.cuh>
#include <cusvna/interval.cuh>

namespace cusvna {

template <typename T, typename... Is>
class AA {
    /* Is... represent the unique indices of noise symbols (actually represented by a type).
       Is cannot be sorted because there is no constexpr comparison of types in C++17 */
    static_assert(detail::IsSet<detail::type_id_sequence<Is...>>::value, "Multiple indices repeating.");

    public:
        static constexpr int N = sizeof...(Is);  /* number of active noise symbols N >= 0. */

        __host__ __device__ AA(const T& b=T(0));
        __host__ __device__ AA(const AA& other);
        __host__ __device__ AA(const Interval<T>& interval);
        __host__ __device__ AA& operator=(const AA& other);
        
        /* create a new variable form an interval */
        template <int K> __host__ __device__ static AA<T,detail::id<K>> from_interval(const Interval<T>& interval);
        
        /* convert back an affine form to an interval */
                 __host__ __device__ Interval<T> interval() const;
        explicit __host__ __device__ operator Interval<T>() const;

        /* Is[K] represents the index of noise symbol number K. */
        template <int K> using idx = typename detail::ExtractFromSequence<K, detail::type_id_sequence<Is...>>::type;  
        
        /* returns typeid(Is[K]) which is the pretty integer index of noise symbol number K 
           by using type introspection when this is possible (host without nvrtc), else just return a K. */
        template <int K> __host__ __device__ static size_t pretty_idx();       
        
        /* base coefficient of the affine form */
        __host__ __device__       T& b();        
        __host__ __device__ const T& b() const;
        
        /* x<K> is the magnitude of the uncertainty corresponding to Kth noise symbol. */
        template <int K> __host__ __device__       T& x();
        template <int K> __host__ __device__ const T& x() const;
        
        /* x<I> is the magnitude of the uncertainty corresponding to noise symbol index I. */
        template <typename I> __host__ __device__       T& x();
        template <typename I> __host__ __device__ const T& x() const;
        
        /* has_symbol<I> is true if I represents an existing noise symbol in this affine arithmetic form */
        template <typename I> __host__ __device__ static constexpr bool has_symbol();
        
        /* symbol_offset<I> is the offset K of I into Is for an existing symbol */
        template <typename I> __host__ __device__ static constexpr int symbol_offset();
        
        /* get<I,D> returns x<I> for an existing noise symbol else return default value T(D) */
        template <typename I, int D=0> __host__ __device__ T symbol_magnitude() const;
       
        /* utility functions */
        __host__ __device__ T radius() const;             /* sum of absolute symbol magnitudes */
        __host__ __device__ Sign sign() const;            /* shortcut for Interval<T>(*this).sign() */
        __host__ __device__ void print() const;           /* alternative to std::cout to print on device */
            
        /* type preserving operators */
        __host__ __device__ AA operator-() const;
        __host__ __device__ AA operator+() const;
        
        __host__ __device__ AA& operator-=(const T& other);
        __host__ __device__ AA& operator+=(const T& other);
        __host__ __device__ AA& operator*=(const T& other);
        __host__ __device__ AA& operator/=(const T& other);
            
        __host__ __device__ AA& operator-=(const AA& other);
        __host__ __device__ AA& operator+=(const AA& other);
        
        /* apply N times a function with arguments (K, x[K], args...) where K=range(N) is not constexpr  */
        template <typename Functor, typename... Args> __host__ __device__       AA& apply(Functor&& f, Args&&... args);
        template <typename Functor, typename... Args> __host__ __device__ const AA& apply(Functor&& f, Args&&... args) const;

        /* apply an integer templated functor on constexpr K=range(N) with arguments (*this, args...) */
        template <template <int> class Functor, typename... Args> __host__ __device__       AA& constexpr_apply(Args&&... args);
        template <template <int> class Functor, typename... Args> __host__ __device__ const AA& constexpr_apply(Args&&... args) const;
        
    protected:
        T m_b;
        cuda::std::array<T,N> m_x;
       
    private:
        template <typename Functor, typename... Args, int... Ks> __host__ __device__ void apply_impl(Functor&& f, Args&&... args, detail::integer_sequence<Ks...>) const;
        template <typename Functor, typename... Args, int... Ks> __host__ __device__ void apply_impl(Functor&& f, Args&&... args, detail::integer_sequence<Ks...>);
        template <template <int> class Functor, int K, typename... Args> __host__ __device__ void constexpr_apply_impl(Args&&... args) const;
        template <template <int> class Functor, int K, typename... Args> __host__ __device__ void constexpr_apply_impl(Args&&... args);
};


namespace op {
    static constexpr int MUL = 1;
    static constexpr int RECIPROCAL = 2;
    static constexpr int SQRT = 3;
    static constexpr int EXP = 4;
    static constexpr int LOG = 5;
}


/* type preserving operators */
template <typename T, typename... Is> __host__ __device__ AA<T,Is...> operator+(const AA<T,Is...>& lhs, const AA<T,Is...>& rhs);
template <typename T, typename... Is> __host__ __device__ AA<T,Is...> operator+(const AA<T,Is...>& lhs, const T& rhs);
template <typename T, typename... Is> __host__ __device__ AA<T,Is...> operator+(const T& lhs, const AA<T,Is...>& rhs);

template <typename T, typename... Is> __host__ __device__ AA<T,Is...> operator-(const AA<T,Is...>& lhs, const AA<T,Is...>& rhs);
template <typename T, typename... Is> __host__ __device__ AA<T,Is...> operator-(const AA<T,Is...>& lhs, const T& rhs);
template <typename T, typename... Is> __host__ __device__ AA<T,Is...> operator-(const T& lhs, const AA<T,Is...>& rhs);

template <typename T, typename... Is> __host__ __device__ AA<T,Is...> operator*(const AA<T,Is...>& lhs, const T& rhs);
template <typename T, typename... Is> __host__ __device__ AA<T,Is...> operator*(const T& lhs, const AA<T,Is...>& rhs);

template <typename T, typename... Is> __host__ __device__ AA<T,Is...> operator/(const AA<T,Is...>& lhs, const T& rhs);


/* type altering operators */
template <typename T, typename... Is, typename... Js, typename ResultType=detail::merged_t<AA<T,Is...>, AA<T,Js...>>>
__host__ __device__ ResultType operator+(const AA<T,Is...>& lhs, const AA<T,Js...>& rhs);

template <typename T, typename... Is, typename... Js, typename ResultType=detail::merged_t<AA<T,Is...>, AA<T,Js...>>>
__host__ __device__ ResultType operator-(const AA<T,Is...>& lhs, const AA<T,Js...>& rhs);

template <typename T, typename... Is, typename... Js,
          typename NewSymbolId = detail::binop_type_id<AA<T,Is...>, AA<T,Js...>, op::MUL>,
          typename ResultType = detail::insert_new_symbol_t<detail::merged_t<AA<T,Is...>, AA<T,Js...>>, NewSymbolId>>
__host__ __device__ ResultType operator*(const AA<T,Is...>& lhs, const AA<T,Js...>& rhs);

template <typename T, typename... Is>
__host__ __device__ auto operator/(const T& lhs, const AA<T,Is...>& rhs);

template <typename T, typename... Is, typename... Js>
__host__ __device__ auto operator/(const AA<T,Is...>& lhs, const AA<T,Js...>& rhs);


/* functions that add a introduce a new error symbol */
template <typename T, typename... Is, 
          typename NewSymbolId = detail::unop_type_id<AA<T,Is...>, op::RECIPROCAL>,
          typename ResultType = detail::insert_new_symbol_t<AA<T,Is...>, NewSymbolId>>
__host__ __device__ ResultType reciprocal(const AA<T,Is...>& val);

template <typename T, typename... Is, 
          typename NewSymbolId = detail::unop_type_id<AA<T,Is...>, op::SQRT>,
          typename ResultType = detail::insert_new_symbol_t<AA<T,Is...>, NewSymbolId>>
__host__ __device__ ResultType sqrt(const AA<T, Is...>& val);

template <typename T, typename... Is, 
          typename NewSymbolId = detail::unop_type_id<AA<T,Is...>, op::EXP>,
          typename ResultType = detail::insert_new_symbol_t<AA<T,Is...>, NewSymbolId>>
__host__ __device__ ResultType exp(const AA<T, Is...>& val);

template <typename T, typename... Is, 
          typename NewSymbolId = detail::unop_type_id<AA<T,Is...>, op::LOG>,
          typename ResultType = detail::insert_new_symbol_t<AA<T,Is...>, NewSymbolId>>
__host__ __device__ ResultType log(const AA<T, Is...>& val);


/* other operators */
#ifndef __CUDACC_RTC__
    template <typename T, typename... Is> 
    __host__ std::ostream& operator<<(std::ostream& os, const AA<T,Is...>& val);
#endif

    
/* implementation */   
template <typename T, typename... Is>
AA<T,Is...>::AA(const T& b): m_b(b), m_x{} {}

template <typename T, typename... Is>
AA<T,Is...>::AA(const AA& other): m_b(other.m_b), m_x(other.m_x) {}

template <typename T, typename... Is>
AA<T,Is...>::AA(const Interval<T>& interval): m_b(interval.midpoint()), m_x{interval.radius()} {
    static_assert(N==1, "Can only construct from interval when there is only one noise symbol.");
}
        
template <typename T, typename... Is>
template <int K>
__host__ __device__ AA<T,detail::id<K>> AA<T,Is...>::from_interval(const Interval<T>& interval) {
    return AA<T,detail::id<K>>(interval);
}

template <typename T, typename... Is>
template <int K> 
size_t AA<T,Is...>::pretty_idx() {
    static_assert(K<N); 
#ifdef __CUDACC_RTC__
    return K;
#else
#ifdef __CUDA_ARCH__
    return K;
#else
    return std::type_index(typeid(idx<K>)).hash_code();
#endif  /* __CUDA_ARCH__ */
#endif  /* __CUDACC_RTC__ */
}

template <typename T, typename... Is>
T& AA<T,Is...>::b() {
    return m_b;
}
        
template <typename T, typename... Is>
const T& AA<T,Is...>::b() const {
    return m_b;
}
    
template <typename T, typename... Is>
template <int K> 
T& AA<T,Is...>::x() {
    static_assert(K<N); 
    return cuda::std::get<K>(m_x);
}
        
template <typename T, typename... Is>
template <int K> 
const T& AA<T,Is...>::x() const {
    static_assert(K<N); 
    return cuda::std::get<K>(m_x);
}

template <typename T, typename... Is>
template <typename I> 
T& AA<T,Is...>::x() {
    static_assert(has_symbol<I>()); 
    return x<symbol_offset<I>()>();
}
        
template <typename T, typename... Is>
template <typename I> 
const T& AA<T,Is...>::x() const {
    static_assert(has_symbol<I>()); 
    return x<symbol_offset<I>()>();
}
        
template <typename T, typename... Is>
template <typename I> 
__host__ __device__ constexpr bool AA<T,Is...>::has_symbol() {
    return detail::IdInSequence<I, Is...>::value;
}

template <typename T, typename... Is>
template <typename I>
__host__ __device__ constexpr int AA<T,Is...>::symbol_offset() {
    return detail::OffsetInSequence<I, detail::type_id_sequence<Is...>>::value;
}

template <typename T, typename... Is>
template <typename I, int D>
__host__ __device__ T AA<T,Is...>::symbol_magnitude() const {
    if constexpr (has_symbol<I>()) {
        return x<I>();
    }
    else {
        return T(D);
    }
}
        
template <typename T, typename... Is>
template <typename Functor, typename... Args> 
AA<T,Is...>& AA<T,Is...>::apply(Functor&& f, Args&&... args) {
    this->apply_impl(cuda::std::forward<Functor>(f), cuda::std::forward<Args>(args)..., detail::make_integer_sequence<N>{});
    return *this;
}

template <typename T, typename... Is>
template <typename Functor, typename... Args> 
const AA<T,Is...>& AA<T,Is...>::apply(Functor&& f, Args&&... args) const {
    this->apply_impl(cuda::std::forward<Functor>(f), cuda::std::forward<Args>(args)..., detail::make_integer_sequence<N>{});
    return *this;
}

template <typename T, typename... Is>
template <typename Functor, typename... Args, int... Ks> 
void AA<T,Is...>::apply_impl(Functor&& f, Args&&... args, detail::integer_sequence<Ks...>) {
    (f(Ks, this->x<Ks>(), args...), ...);
}

template <typename T, typename... Is>
template <typename Functor, typename... Args, int... Ks> 
void AA<T,Is...>::apply_impl(Functor&& f, Args&&... args, detail::integer_sequence<Ks...>) const {
    (f(Ks, this->x<Ks>(), args...), ...);
}
        
template <typename T, typename... Is>
template <template <int> class Functor, typename... Args>
const AA<T,Is...>& AA<T,Is...>::constexpr_apply(Args&&... args) const {
    if constexpr (N > 0) {
        this->constexpr_apply_impl<Functor,0,Args...>(cuda::std::forward<Args>(args)...);
    }
    return *this;
}
    
template <typename T, typename... Is>
template <template <int> class Functor, int K, typename... Args>
void AA<T,Is...>::constexpr_apply_impl(Args&&... args) const {
    Functor<K>()(*this, args...);
    if constexpr (K+1<N) {
        this->constexpr_apply_impl<Functor,K+1,Args...>(cuda::std::forward<Args>(args)...);
    }
}

template <typename T, typename... Is>
template <template <int> class Functor, typename... Args>
AA<T,Is...>& AA<T,Is...>::constexpr_apply(Args&&... args) {
    if constexpr (N > 0) {
        this->constexpr_apply_impl<Functor,0,Args...>(cuda::std::forward<Args>(args)...);
    }
    return *this;
}
    
template <typename T, typename... Is>
template <template <int> class Functor, int K, typename... Args>
void AA<T,Is...>::constexpr_apply_impl(Args&&... args) {
    Functor<K>()(*this, args...);
    if constexpr (K+1<N) {
        this->constexpr_apply_impl<Functor,K+1,Args...>(cuda::std::forward<Args>(args)...);
    }
}

template <typename T, typename... Is>
T AA<T,Is...>::radius() const {
    T r(0);
    this->apply([&](int k, const T& xi) { r += ::abs(xi); });
    return r;
};
    
template <typename T, typename... Is>
AA<T,Is...>::operator Interval<T>() const {
    const T c = this->b();
    const T r = this->radius();
    return Interval<T>(c-r, c+r);
}
    
template <typename T, typename... Is>
Interval<T> AA<T,Is...>::interval() const {
    return Interval<T>(*this);
};

template <typename T, typename... Is>
Sign AA<T,Is...>::sign() const {
    return this->interval().sign();
};
        
template <typename T, typename... Is>
AA<T,Is...> AA<T,Is...>::operator+() const {
    AA<T,Is...> out(*this);
    out.b() = +out.b();
    return out.apply([](int k, T& xi) { xi = +xi; });
}

template <typename T, typename... Is>
AA<T,Is...> AA<T,Is...>::operator-() const {
    AA<T,Is...> out(*this);
    out.b() = -out.b();
    return out.apply([](int k, T& xi) { xi = -xi; });
}

template <typename T, typename... Is>
AA<T,Is...>& AA<T,Is...>::operator+=(const T& other)  {
    this->b() += other;
    return *this;
}

template <typename T, typename... Is>
AA<T,Is...>& AA<T,Is...>::operator-=(const T& other)  {
    this->b() -= other;
    return *this;
}

template <typename T, typename... Is>
AA<T,Is...>& AA<T,Is...>::operator*=(const T& other)  {
    this->b() *= other;
    return this->apply([&](int k, T& xi) { xi *= other; });
}

template <typename T, typename... Is>
AA<T,Is...>& AA<T,Is...>::operator/=(const T& other)  {
    this->b() /= other;
    return this->apply([&](int k, T& xi) { xi /= other; });
}

namespace impl {
    template <int K>
    struct inplace_add_functor {
        template <typename T, typename... Is>
        __host__ __device__ void operator()(AA<T,Is...>& lhs, const AA<T,Is...>& rhs) const {
            lhs.template x<K>() += rhs.template x<K>(); 
        }
    };
    
    template <int K>
    struct inplace_subtract_functor {
        template <typename T, typename... Is>
        __host__ __device__ void operator()(AA<T,Is...>& lhs, const AA<T,Is...>& rhs) const {
            lhs.template x<K>() -= rhs.template x<K>(); 
        }
    };
    
    template <int K>
    struct outplace_add_functor {
        template <typename T, typename... Is, typename... Js, typename... Ks>
        __host__ __device__ void operator()(AA<T,Is...>& out, const AA<T,Js...>& lhs, const AA<T,Ks...>& rhs) const {
            using I = typename AA<T,Is...>::template idx<K>;
            out.template x<K>() = lhs.template symbol_magnitude<I,0>() + rhs.template symbol_magnitude<I,0>();
        }
    };
    
    template <int K>
    struct outplace_subtract_functor {
        template <typename T, typename... Is, typename... Js, typename... Ks>
        __host__ __device__ void operator()(AA<T,Is...>& out, const AA<T,Js...>& lhs, const AA<T,Ks...>& rhs) const {
            using I = typename AA<T,Is...>::template idx<K>;
            out.template x<K>() = lhs.template symbol_magnitude<I,0>() - rhs.template symbol_magnitude<I,0>();
        }
    };
    
    template <int K>
    struct outplace_multiply_functor {
        template <typename T, typename... Is, typename... Js, typename... Ks>
        __host__ __device__ void operator()(AA<T,Is...>& out, const AA<T,Js...>& lhs, const AA<T,Ks...>& rhs) const {
            using I = typename AA<T,Is...>::template idx<K>;
            if constexpr (AA<T,Js...>::template has_symbol<I>() || AA<T,Ks...>::template has_symbol<I>()) {
                out.template x<I>() = lhs.template symbol_magnitude<I,0>() * rhs.b() + rhs.template symbol_magnitude<I,0>() * lhs.b();
            }
            else {
                out.template x<I>() = lhs.radius() * rhs.radius();
            }
        }
    };
    
    template <int K>
    struct affine_functor {
        template <typename T, typename... Is, typename... Js>
        __host__ __device__ void operator()(AA<T,Is...>& out, const AA<T,Js...>& base, 
                const T& alpha, const T& zeta, const T& delta) const {
            using I = typename AA<T,Is...>::template idx<K>;
            if constexpr (AA<T,Js...>::template has_symbol<I>()) {
                out.template x<I>() = alpha * base.template x<I>();
            }
            else {
                out.b() = alpha * base.b() + zeta;
                out.template x<I>() = delta;
            }
        }
    };

    template <int K>
    struct print_functor {
        template <typename T, typename... Is>
        __host__ __device__ void operator()(const AA<T,Is...>& val) const {
            using V = AA<T,Is...>;
            printf("%3i: %+2.1e", int(V::template pretty_idx<K>() & 0xff), float(val.template x<K>()));
            if (K != V::N-1) { printf(", "); }
        }
    };

#ifndef __CUDACC_RTC__
    template <int K>
    struct ostream_functor {
        template <typename T, typename... Is>
        __host__ __device__ void operator()(const AA<T,Is...>& val, std::ostream& os) const {
#ifndef __CUDA_ARCH__
            using V = AA<T,Is...>;
            os << (V::template pretty_idx<K>() & 0xff) << ": " << val.template x<K>();
            if constexpr (K != V::N-1) {
                os << ", ";
            }
#endif /* endif __CUDA_ARCH__ */
        }
    };
#endif /* endif __CUDACC_RTC__ */

} /* end of namespace impl */

template <typename T, typename... Is>
AA<T,Is...>& AA<T,Is...>::operator+=(const AA<T,Is...>& other)  {
    this->b() += other.b();
    return this->constexpr_apply<impl::inplace_add_functor>(other);
}

template <typename T, typename... Is>
AA<T,Is...>& AA<T,Is...>::operator-=(const AA<T,Is...>& other)  {
    this->b() -= other.b();
    return this->constexpr_apply<impl::inplace_subtract_functor>(other);
}
    
template <typename T, typename... Is>
__host__ __device__ AA<T,Is...> operator+(const AA<T,Is...>& lhs, const AA<T,Is...>& rhs) {
    return +lhs += rhs;
}

template <typename T, typename... Is>
__host__ __device__ AA<T,Is...> operator+(const AA<T,Is...>& lhs, const T& rhs) {
    return +lhs += rhs;
}

template <typename T, typename... Is>
__host__ __device__ AA<T,Is...> operator+(const T& lhs, const AA<T,Is...>& rhs) {
    return +rhs += lhs;
}

template <typename T, typename... Is>
__host__ __device__ AA<T,Is...> operator-(const AA<T,Is...>& lhs, const AA<T,Is...>& rhs) {
    return +lhs -= rhs;
}

template <typename T, typename... Is>
__host__ __device__ AA<T,Is...> operator-(const AA<T,Is...>& lhs, const T& rhs) {
    return +lhs -= rhs;
}

template <typename T, typename... Is>
__host__ __device__ AA<T,Is...> operator-(const T& lhs, const AA<T,Is...>& rhs) {
    return -rhs += lhs;
}

template <typename T, typename... Is>
__host__ __device__ AA<T,Is...> operator*(const AA<T,Is...>& lhs, const T& rhs) {
    return +lhs *= rhs;
}

template <typename T, typename... Is>
__host__ __device__ AA<T,Is...> operator*(const T& lhs, const AA<T,Is...>& rhs) {
    return +rhs *= lhs;
}

template <typename T, typename... Is>
__host__ __device__ AA<T,Is...> operator/(const AA<T,Is...>& lhs, const T& rhs) {
    return +lhs /= rhs;
}

template <typename T, typename... Is, typename... Js, typename ResultType>
__host__ __device__ ResultType operator+(const AA<T,Is...>& lhs, const AA<T,Js...>& rhs) {
    ResultType out(lhs.b() + rhs.b());
    return out.template constexpr_apply<impl::outplace_add_functor>(lhs, rhs);
}

template <typename T, typename... Is, typename... Js, typename ResultType>
__host__ __device__ ResultType operator-(const AA<T,Is...>& lhs, const AA<T,Js...>& rhs) {
    ResultType out(lhs.b() - rhs.b());
    return out.template constexpr_apply<impl::outplace_subtract_functor>(lhs, rhs);
}

template <typename T, typename... Is, typename... Js, typename NewSymbolId, typename ResultType>
__host__ __device__ ResultType operator*(const AA<T,Is...>& lhs, const AA<T,Js...>& rhs) {
    ResultType out(lhs.b() * rhs.b());
    return out.template constexpr_apply<impl::outplace_multiply_functor>(lhs, rhs);
}

template <typename T, typename... Is>
__host__ __device__ auto operator/(const T& lhs, const AA<T,Is...>& rhs) {
    return lhs * reciprocal(rhs);
}

template <typename T, typename... Is, typename... Js>
__host__ __device__ auto operator/(const AA<T,Is...>& lhs, const AA<T,Js...>& rhs) {
    return lhs * reciprocal(rhs);
}

template <typename T, typename... Is, typename NewSymbolId, typename ResultType>
__host__ __device__ ResultType reciprocal(const AA<T,Is...>& val) {
    /* Self-Validated Numerical Methods and Applications - section 3.12 */
    ResultType out;
    
    Interval<T> interval(val);
    if (interval.a <= T(0) && interval.b >= T(0)) {
        out.template x<NewSymbolId>() = cuda::std::numeric_limits<T>::infinity();
        return out;
    }

    const T fa = ::fabs(interval.a);
    const T fb = ::fabs(interval.b);
    const T a = min(fa, fb);
    const T b = max(fa, fb);
    
    const T alpha = -T(1)/(b*b);
    const T dmin = T(1)/a - alpha*a;
    const T dmax = T(2)/b; // - alpha*b;
    
    const T zeta = (interval.a < T(0) ? T(-1) : T(1)) * (dmin + dmax)/T(2);
    const T delta = (dmax - dmin)/T(2);

    return out.template constexpr_apply<impl::affine_functor>(val, alpha, zeta, delta);
}

template <typename T, typename... Is, typename NewSymbolId, typename ResultType>
__host__ __device__ ResultType sqrt(const AA<T, Is...>& val) {
    /* Self-Validated Numerical Methods and Applications - section 3.9.1 */

    Interval<T> interval(val);
    const T a = max(interval.a, T(0));  /* we assume a is always positive */
    const T b = max(interval.b, T(0));  /* we assume b is always positive */
    
    const T t = ::sqrt(a) + ::sqrt(b);
    const T alpha = T(1)/t;
    const T zeta = t/T(8) + ::sqrt(a*b)/(T(2)*t);
    const T delta = (a+b-T(2)*::sqrt(a*b)) / (T(8)*t);

    return ResultType().template constexpr_apply<impl::affine_functor>(val, alpha, zeta, delta);
}

template <typename T, typename... Is, typename NewSymbolId, typename ResultType>
__host__ __device__ ResultType exp(const AA<T, Is...>& val) {
    /* Self-Validated Numerical Methods and Applications - section 3.12 */

    Interval<T> interval(val);
    const T a = interval.a;
    const T b = interval.b;

    const T ea = ::exp(a);
    const T eb = ::exp(b);
    const T alpha = (eb - ea)/(b - a);
    
    const T zeta  = alpha * (T(1) - ::log(alpha));
    const T delta = (alpha * (::log(alpha) - T(1) - a) + ea)/T(2);

    return ResultType().template constexpr_apply<impl::affine_functor>(val, alpha, zeta, delta);
}

template <typename T, typename... Is, typename NewSymbolId, typename ResultType>
__host__ __device__ ResultType log(const AA<T, Is...>& val) {
    Interval<T> interval(val);
    const T a = interval.a;
    const T b = interval.b;

    const T la = ::log(a);
    const T lb = ::log(b);
    const T alpha = (lb - la)/(b - a);

    const T xs = T(1)/alpha;
    const T ys = alpha * (xs - a) + la;
    
    const T zeta =  -alpha*xs + (::log(xs)+ys)/T(2);
    const T delta = (::log(xs) - ys)/T(2);

    return ResultType().template constexpr_apply<impl::affine_functor>(val, alpha, zeta, delta);
}


    template <typename T, typename... Is>
    __host__ __device__ void AA<T,Is...>::print() const {
        printf("AA(b=%+2.1e, x=[", float(this->b()));
        this->template constexpr_apply<impl::print_functor>();
        printf("])\n");
    }

#ifndef __CUDACC_RTC__
    template <typename T, typename... Is>
    __host__ std::ostream& operator<<(std::ostream& os, const AA<T,Is...>& val) {
        using V = AA<T,Is...>;

        os << "AA(b=" << val.b() << ", x=[";
        val.template constexpr_apply<impl::ostream_functor>(os);
        os << "])";

        return os;
    }
#endif

} /* end of namespace cusvna */


/* NVRTC does not support __COUNTER__ */
#ifndef __CUDACC_RTC__
#define new_aa_from_interval(interval) cusvna::AA<decltype(interval)::type>::from_interval<__COUNTER__>(interval)
#endif


#endif /* end of include guard: AA_CUH_NLBP0A1W */
