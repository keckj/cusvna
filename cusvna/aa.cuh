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
        using tag_t = detail::tag_t;

        __host__ __device__ AA(const T& b=T(0));
        __host__ __device__ AA(const Interval<T>& interval);
        AA(const AA& other) = default;
        AA& operator=(const AA& other) = default;
        
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
        
        /* tag<K> is the runtime tag of the Kth noise symbol. Generated noise symbols (introduced by
           non-affine operations) are typed after the types of their operands only, the tag tells apart
           two symbols of the same type that originate from operands with different contents. 
           Input symbols always have a zero tag. */
        template <int K> __host__ __device__       tag_t& tag();
        template <int K> __host__ __device__ const tag_t& tag() const;
        
        /* tag<I> is the runtime tag of the noise symbol index I. */
        template <typename I> __host__ __device__       tag_t& tag();
        template <typename I> __host__ __device__ const tag_t& tag() const;
        
        /* has_symbol<I> is true if I represents an existing noise symbol in this affine arithmetic form */
        template <typename I> __host__ __device__ static constexpr bool has_symbol();
        
        /* symbol_offset<I> is the offset K of I into Is for an existing symbol */
        template <typename I> __host__ __device__ static constexpr int symbol_offset();
        
        /* get<I,D> returns x<I> for an existing noise symbol else return default value T(D) */
        template <typename I, int D=0> __host__ __device__ T symbol_magnitude() const;
        
        /* symbol_tag<I> returns tag<I> for an existing noise symbol else return 0 */
        template <typename I> __host__ __device__ tag_t symbol_tag() const;
        
        /* hash of the content of the affine form (base coefficient, symbol magnitudes and tags) */
        __host__ __device__ tag_t content_hash(tag_t seed) const;
       
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
        cuda::std::array<tag_t,N> m_tag;
       
    private:
        template <int... Ks> __host__ __device__ tag_t content_hash_impl(tag_t h, detail::integer_sequence<Ks...>) const;
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
    static constexpr int MERGE = 6;  /* runtime only, merge of two generated symbols of the same type */
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
AA<T,Is...>::AA(const T& b): m_b(b), m_x{}, m_tag{} {}

template <typename T, typename... Is>
AA<T,Is...>::AA(const Interval<T>& interval): m_b(interval.midpoint()), m_x{interval.radius()}, m_tag{} {
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
template <int K> 
detail::tag_t& AA<T,Is...>::tag() {
    static_assert(K<N); 
    return cuda::std::get<K>(m_tag);
}
        
template <typename T, typename... Is>
template <int K> 
const detail::tag_t& AA<T,Is...>::tag() const {
    static_assert(K<N); 
    return cuda::std::get<K>(m_tag);
}

template <typename T, typename... Is>
template <typename I> 
detail::tag_t& AA<T,Is...>::tag() {
    static_assert(has_symbol<I>()); 
    return tag<symbol_offset<I>()>();
}
        
template <typename T, typename... Is>
template <typename I> 
const detail::tag_t& AA<T,Is...>::tag() const {
    static_assert(has_symbol<I>()); 
    return tag<symbol_offset<I>()>();
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
template <typename I>
__host__ __device__ detail::tag_t AA<T,Is...>::symbol_tag() const {
    if constexpr (has_symbol<I>()) {
        return tag<I>();
    }
    else {
        return detail::tag_t(0);
    }
}

template <typename T, typename... Is>
__host__ __device__ detail::tag_t AA<T,Is...>::content_hash(detail::tag_t seed) const {
    return this->content_hash_impl(detail::hash_combine(seed, detail::bits(m_b)), detail::make_integer_sequence<N>{});
}

template <typename T, typename... Is>
template <int... Ks>
__host__ __device__ detail::tag_t AA<T,Is...>::content_hash_impl(detail::tag_t h, detail::integer_sequence<Ks...>) const {
    /* input symbols have constant zero tags, only hash magnitudes for those */
    ((h = detail::hash_combine(h, detail::bits(this->x<Ks>())),
      h = (detail::is_generated<idx<Ks>> ? detail::hash_combine(h, this->tag<Ks>()) : h)), ...);
    return h;
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
    /* runtime tag of a new symbol introduced by operation opcode */
    __host__ __device__ inline detail::tag_t seed(int opcode) {
        return detail::hash_combine(detail::tag_t(0), detail::tag_t(opcode));
    }
    
    /* Linear combination l*eps_l + r*eps_r of two occurences of noise symbol I with tags ltag and rtag.
       Input symbols, or generated symbols with the same tag, represent the same quantity (eps_l == eps_r).
       Otherwise eps_l and eps_r are two different generated quantities that happen to share the same 
       symbol type, they are merged into a new symbol with magnitude |l|+|r| (and a new tag). */
    template <typename I, typename T>
    __host__ __device__ void combine_symbol(T& out, detail::tag_t& out_tag,
            const T l, const detail::tag_t ltag, const T r, const detail::tag_t rtag) {
        if constexpr (!detail::is_generated<I>) {
            out = l + r;
            out_tag = detail::tag_t(0);
        }
        else {
            if (ltag == rtag || r == T(0)) {
                out = l + r;
                out_tag = ltag;
            }
            else if (l == T(0)) {
                out = r;
                out_tag = rtag;
            }
            else {
                out = ::fabs(l) + ::fabs(r);
                detail::tag_t h = seed(op::MERGE);
                h = detail::hash_combine(h, ltag);
                h = detail::hash_combine(h, detail::bits(l));
                h = detail::hash_combine(h, rtag);
                h = detail::hash_combine(h, detail::bits(r));
                out_tag = h;
            }
        }
    }

    template <int K>
    struct inplace_add_functor {
        template <typename T, typename... Is>
        __host__ __device__ void operator()(AA<T,Is...>& lhs, const AA<T,Is...>& rhs) const {
            using I = typename AA<T,Is...>::template idx<K>;
            combine_symbol<I>(lhs.template x<K>(), lhs.template tag<K>(),
                    lhs.template x<K>(), lhs.template tag<K>(),
                    rhs.template x<K>(), rhs.template tag<K>());
        }
    };
    
    template <int K>
    struct inplace_subtract_functor {
        template <typename T, typename... Is>
        __host__ __device__ void operator()(AA<T,Is...>& lhs, const AA<T,Is...>& rhs) const {
            using I = typename AA<T,Is...>::template idx<K>;
            combine_symbol<I>(lhs.template x<K>(), lhs.template tag<K>(),
                    lhs.template x<K>(), lhs.template tag<K>(),
                    -rhs.template x<K>(), rhs.template tag<K>());
        }
    };
    
    template <int K>
    struct outplace_add_functor {
        template <typename T, typename... Is, typename... Js, typename... Ks>
        __host__ __device__ void operator()(AA<T,Is...>& out, const AA<T,Js...>& lhs, const AA<T,Ks...>& rhs) const {
            using I = typename AA<T,Is...>::template idx<K>;
            combine_symbol<I>(out.template x<K>(), out.template tag<K>(),
                    lhs.template symbol_magnitude<I,0>(), lhs.template symbol_tag<I>(),
                    rhs.template symbol_magnitude<I,0>(), rhs.template symbol_tag<I>());
        }
    };
    
    template <int K>
    struct outplace_subtract_functor {
        template <typename T, typename... Is, typename... Js, typename... Ks>
        __host__ __device__ void operator()(AA<T,Is...>& out, const AA<T,Js...>& lhs, const AA<T,Ks...>& rhs) const {
            using I = typename AA<T,Is...>::template idx<K>;
            combine_symbol<I>(out.template x<K>(), out.template tag<K>(),
                     lhs.template symbol_magnitude<I,0>(), lhs.template symbol_tag<I>(),
                    -rhs.template symbol_magnitude<I,0>(), rhs.template symbol_tag<I>());
        }
    };
    
    template <int K>
    struct outplace_multiply_functor {
        template <typename T, typename... Is, typename... Js, typename... Ks>
        __host__ __device__ void operator()(AA<T,Is...>& out, const AA<T,Js...>& lhs, const AA<T,Ks...>& rhs,
                const detail::tag_t new_tag) const {
            using I = typename AA<T,Is...>::template idx<K>;
            if constexpr (AA<T,Js...>::template has_symbol<I>() || AA<T,Ks...>::template has_symbol<I>()) {
                combine_symbol<I>(out.template x<I>(), out.template tag<I>(),
                        lhs.template symbol_magnitude<I,0>() * rhs.b(), lhs.template symbol_tag<I>(),
                        rhs.template symbol_magnitude<I,0>() * lhs.b(), rhs.template symbol_tag<I>());
            }
            else {
                out.template x<I>() = lhs.radius() * rhs.radius();
                out.template tag<I>() = new_tag;
            }
        }
    };
    
    template <int K>
    struct affine_functor {
        template <typename T, typename... Is, typename... Js>
        __host__ __device__ void operator()(AA<T,Is...>& out, const AA<T,Js...>& base, 
                const T& alpha, const T& zeta, const T& delta, const detail::tag_t new_tag) const {
            using I = typename AA<T,Is...>::template idx<K>;
            if constexpr (AA<T,Js...>::template has_symbol<I>()) {
                out.template x<I>() = alpha * base.template x<I>();
                out.template tag<I>() = base.template tag<I>();
            }
            else {
                out.b() = alpha * base.b() + zeta;
                out.template x<I>() = delta;
                out.template tag<I>() = new_tag;
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
    const detail::tag_t new_tag = rhs.content_hash(lhs.content_hash(impl::seed(op::MUL)));
    return out.template constexpr_apply<impl::outplace_multiply_functor>(lhs, rhs, new_tag);
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
        out.template tag<NewSymbolId>() = val.content_hash(impl::seed(op::RECIPROCAL));
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

    const detail::tag_t new_tag = val.content_hash(impl::seed(op::RECIPROCAL));
    return out.template constexpr_apply<impl::affine_functor>(val, alpha, zeta, delta, new_tag);
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

    const detail::tag_t new_tag = val.content_hash(impl::seed(op::SQRT));
    return ResultType().template constexpr_apply<impl::affine_functor>(val, alpha, zeta, delta, new_tag);
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

    const detail::tag_t new_tag = val.content_hash(impl::seed(op::EXP));
    return ResultType().template constexpr_apply<impl::affine_functor>(val, alpha, zeta, delta, new_tag);
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

    const detail::tag_t new_tag = val.content_hash(impl::seed(op::LOG));
    return ResultType().template constexpr_apply<impl::affine_functor>(val, alpha, zeta, delta, new_tag);
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
