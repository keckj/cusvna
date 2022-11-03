#ifndef AA_H_TLKSOS6P
#define AA_H_TLKSOS6P

#include <cuda/std/functional>

#include "cusvna/interval.cuh"
#include "cusvna/index.cuh"

namespace cusvna {
    
    template <typename T, typename I, typename G, I N>
    class AA {
        /* Affine Arithmetic with fixed number of active noise symbols:
         *    Coefficients type T
         *    Integer type I
         *    Symbol index generator type G
         *    Sparse representation with at most N different noise symbols 
         */
        public:
                     __host__ __device__ AA();                                                                        /* default constructor, only to be used for array allocation */
                     __host__ __device__ AA(const AA<T,I,G,N>& other);                                                /* copy constructor */
            explicit __host__ __device__ AA(G& index_generator);                                                      /* minimal constructor */
            explicit __host__ __device__ AA(G& index_generator, const T& base_coefficient);                           /* base coeff constructor */
            explicit __host__ __device__ AA(const Interval<T>& interval, G& index_generator);                         /* construct from an interval */
            explicit __host__ __device__ AA(const AA<T,I,G,N>& base, const T& alpha, const T& beta, const T& gamma);  /* affine constructor */

            explicit __host__ __device__ operator Interval<T>() const;  /* cast operator to convert back an affine form to an interval */

            __host__ __device__ const T& x0() const;             /* base coefficient x0 */
            __host__ __device__ const I& n() const;              /* number of active noise symbols 0 <= n <= N. */
            __host__ __device__ const T* x() const;              /* x is the magnitude vector of the uncertainty coerresponding to noise symbols. */
            __host__ __device__ const I* idx() const;            /* idx represent the index vector of noise symbols, idx is assumed to be sorted. */
            __host__ __device__ const T& x(const I& i) const;    /* x(i) is the magnitude of the uncertainty coerresponding to noise symbol idx[i]. */
            __host__ __device__ const I& idx(const I& i) const;  /* idx(i) represent the index of noise symbol number i, idx is assumed to be sorted. */

            __host__ __device__ T& x0();
            __host__ __device__ I& n();
            __host__ __device__ T* x();
            __host__ __device__ I* idx();
            __host__ __device__ T& x(const I& i);
            __host__ __device__ I& idx(const I& i);
            __host__ __device__ G& index_generator() const;

            __host__ __device__ AA<T,I,G,N>& append_symbol(const I& symbol_index, const T& magnitude);
            __host__ __device__ AA<T,I,G,N>& append_new_symbol(const T& magnitude);

            __host__ __device__ T radius() const; /* sum of absolute symbol magnitudes */
            __host__ __device__ Interval<T> interval() const; /* cast to an interval */
            
            __host__ __device__ AA<T,I,G,N>& operator=(const AA<T,I,G,N>& other);
            
            __host__ __device__ const I& operator()(const I& i) const;  /* alias of idx(i) */
            __host__ __device__ const T& operator[](const I& i) const;  /* alias of x(i)   */

            __host__ __device__ I& operator()(const I& i);
            __host__ __device__ T& operator[](const I& i);

            __host__ __device__ AA<T,I,G,N> operator-() const;
            __host__ __device__ AA<T,I,G,N> operator+() const;
            __host__ __device__ AA<T,I,G,N> reciprocal() const;  /* multiplicative inverse */

            __host__ __device__ AA<T,I,G,N>& operator-=(const T& other);
            __host__ __device__ AA<T,I,G,N>& operator+=(const T& other);
            __host__ __device__ AA<T,I,G,N>& operator*=(const T& other);
            __host__ __device__ AA<T,I,G,N>& operator/=(const T& other);

            __host__ __device__ AA<T,I,G,N>& operator-=(const AA<T,I,G,N>& other);
            __host__ __device__ AA<T,I,G,N>& operator+=(const AA<T,I,G,N>& other);
            __host__ __device__ AA<T,I,G,N>& operator*=(const AA<T,I,G,N>& other);
            __host__ __device__ AA<T,I,G,N>& operator/=(const AA<T,I,G,N>& other);
        private:
            T m_coeffs[N+1]; 
            I m_indices[N+1];
            G* m_index_generator; /* symbol index generator, no static members in device code */
    };
        
    /* operators */
    template <typename T, typename I, typename G, I N, typename F>
    __host__ __device__ AA<T,I,G,N>& inplace_unary_op(AA<T,I,G,N>& input, F&& op);
    
    template <typename T, typename I, typename G, I N, typename F>
    __host__ __device__ AA<T,I,G,N> binary_op(const AA<T,I,G,N>& lhs, const AA<T,I,G,N>& rhs, F&& op, const T& identity);

    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator+(const AA<T,I,G,N>& lhs, const AA<T,I,G,N>& rhs);
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator+(const AA<T,I,G,N>& lhs, const T& rhs);
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator+(const T& lhs, const AA<T,I,G,N>& rhs);
    
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator-(const AA<T,I,G,N>& lhs, const AA<T,I,G,N>& rhs);
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator-(const AA<T,I,G,N>& lhs, const T& rhs);
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator-(const T& lhs, const AA<T,I,G,N>& rhs);
    
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator*(const AA<T,I,G,N>& lhs, const AA<T,I,G,N>& rhs);
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator*(const AA<T,I,G,N>& lhs, const T& rhs);
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator*(const T& lhs, const AA<T,I,G,N>& rhs);
    
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator/(const AA<T,I,G,N>& lhs, const AA<T,I,G,N>& rhs);
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator/(const AA<T,I,G,N>& lhs, const T& rhs);
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ operator*(const T& lhs, const AA<T,I,G,N>& rhs);
    
    template <typename T, typename I, typename G, I N> AA<T,I,G,N> __host__ __device__ sqrt(const AA<T,I,G,N>& val);
    

    /* implementation */   
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>::AA() : 
        m_coeffs{0}, m_indices{0}, m_index_generator(nullptr)  {
        static_assert(N>=0, "N<0");
    }

    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>::AA(G& index_generator) : 
        m_coeffs{0}, m_indices{0}, m_index_generator(&index_generator)  {
        static_assert(N>=0, "N<0");
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>::AA(G& index_generator, const T& base_coefficient) : 
        m_coeffs{0}, m_indices{0}, m_index_generator(&index_generator)  {
        static_assert(N>=0, "N<0");
        this->x0() = base_coefficient;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>::AA(const Interval<T>& interval, G& index_generator):
        m_coeffs{0}, m_indices{0}, m_index_generator(&index_generator) {
        static_assert(N>=0, "N<0");
        this->x0() = interval.midpoint();
        this->append_new_symbol(interval.radius());
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>::AA(const AA<T,I,G,N>& other) : 
        m_coeffs{0}, m_indices{0}, m_index_generator(&other.index_generator()) {
        static_assert(N>=0, "N<0");
        *this = other;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>::AA(const AA<T,I,G,N>& base, const T& alpha, const T& beta, const T& delta) : 
        m_coeffs{0}, m_indices{0}, m_index_generator(&base.index_generator()) {
        static_assert(N>=0, "N<0");
        this->x0() = alpha * base.x0() + beta;
        this->n() = base.n();
        for (I i = 0; i < base.n(); ++i) {
            this->idx(i) = base.idx(i);
            this->x(i) = alpha * base.x(i);
        }
        this->append_new_symbol(delta);
    }

    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>& AA<T,I,G,N>::operator=(const AA<T,I,G,N>& other) {
        if (this != &other) {
            this->x0() = other.x0();
            this->n() = other.n();
            for (I i = 0; i < other.n(); ++i) {
                this->idx(i) = other.idx(i);
                this->x(i) = other.x(i);
            }
        }
        return *this;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>::operator Interval<T>() const {
        const T c = this->x0();
        const T r = this->radius();
        return Interval<T>(c-r, c+r);
    }

    template <typename T, typename I, typename G, I N>
    __host__ __device__ G& AA<T,I,G,N>::index_generator() const {
        /* m_index_generator is nullptr if and only if index_generator is used with 
           the default constructor. Default constructor should only be used for allocation */
        assert(m_index_generator != nullptr);
        return *this->m_index_generator;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ T& AA<T,I,G,N>::x0() {
        return m_coeffs[0];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ I& AA<T,I,G,N>::n() {
        return m_indices[0];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ T* AA<T,I,G,N>::x() {
        return &m_coeffs[1];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ I* AA<T,I,G,N>::idx() {
        return &m_indices[1];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ T& AA<T,I,G,N>::x(const I& i) {
        assert(i<this->n());
        return m_coeffs[i+1];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ I& AA<T,I,G,N>::idx(const I& i) {
        assert(i<this->n());
        return m_indices[i+1];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ I& AA<T,I,G,N>::operator()(const I& i) {
        return this->idx(i);
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ T& AA<T,I,G,N>::operator[](const I& i) {
        return this->x(i);
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ const T& AA<T,I,G,N>::x0() const {
        return m_coeffs[0];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ const I& AA<T,I,G,N>::n() const {
        return m_indices[0];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ const T* AA<T,I,G,N>::x() const {
        return &m_coeffs[1];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ const I* AA<T,I,G,N>::idx() const {
        return &m_indices[1];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ const T& AA<T,I,G,N>::x(const I& i) const {
        assert(i<this->n());
        return m_coeffs[i+1];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ const I& AA<T,I,G,N>::idx(const I& i) const {
        assert(i<this->n());
        return m_indices[i+1];
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ const I& AA<T,I,G,N>::operator()(const I& i) const {
        return this->idx(i);
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ const T& AA<T,I,G,N>::operator[](const I& i) const {
        return this->x(i);
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> AA<T,I,G,N>::operator+() const {
        return AA<T,I,G,N>(*this);
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> AA<T,I,G,N>::operator-() const {
        AA<T,I,G,N> out(*this);
        inplace_unary_op(out, [](T& a){ a = -a; });
        return out;
    }
            
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>& AA<T,I,G,N>::append_symbol(const I& symbol_index, const T& magnitude) {
        I i = this->n();
        assert(i < N);
        assert(i==0 || this->idx(i-1) < symbol_index);
        ++this->n();
        this->idx(i) = symbol_index;
        this->x(i) = magnitude;
        return *this;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>& AA<T,I,G,N>::append_new_symbol(const T& magnitude) {
        I symbol_index = this->index_generator()();
        I i = this->n();
        assert(i < N);
        assert(i==0 || this->idx(i-1) < symbol_index);
        ++this->n();
        this->idx(i) = symbol_index;
        this->x(i) = magnitude;
        return *this;
    }
            
    template <typename T, typename I, typename G, I N>
    __host__ __device__ T AA<T,I,G,N>::radius() const {
        T r(0);
        for (I i = 0; i < this->n(); ++i) {
            r += abs(this->x(i));
        }
        return r;
    };
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ Interval<T> AA<T,I,G,N>::interval() const {
        return Interval<T>(*this);
    };
    
    /* operators */
    template <typename T, typename I, typename G, I N, typename F>
    __host__ __device__ AA<T,I,G,N>& inplace_unary_op(AA<T,I,G,N>& input, F&& op) {
        op(input.x0());
        for (I i = 0; i < input.n(); ++i) {
            op(input.x(i));
        }
        return input;
    }

    template <typename T, typename I, typename G, I N, typename F>
    __host__ __device__ AA<T,I,G,N> binary_op(const AA<T,I,G,N>& lhs, const AA<T,I,G,N>& rhs, F&& op, const T& identity) {
        AA<T,I,G,N> out(lhs.index_generator());
        out.x0() = op(lhs.x0(), rhs.x0());

        const I lhs_n(lhs.n()), rhs_n(rhs.n());
        I lhs_i(0), rhs_i(0);
        while(lhs_i < lhs_n || rhs_i < rhs_n) {
            const I lhs_idx(lhs_i >= lhs_n ? G::last_symbol_index : lhs(lhs_i));
            const I rhs_idx(rhs_i >= rhs_n ? G::last_symbol_index : rhs(rhs_i));
            const bool le = lhs_idx <= rhs_idx;
            const bool ge = lhs_idx >= rhs_idx;
            out.append_symbol( (le ? lhs_idx : rhs_idx), op((le ? lhs[lhs_i] : identity), (ge ? rhs[rhs_i] : identity)));
            lhs_i += le;
            rhs_i += ge;
        }
        return out;
    }
        

    /*  OPERATOR +  */
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator+(const AA<T,I,G,N>& lhs, const AA<T,I,G,N>& rhs) {
        return binary_op(lhs, rhs, [](const T& a, const T& b){ return a + b; }, T(0));
    }

    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>& AA<T,I,G,N>::operator+=(const T& other) {
        this->x0() += other;
        return *this;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>& AA<T,I,G,N>::operator+=(const AA<T,I,G,N>& other) {
        return *this = *this + other;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator+(const AA<T,I,G,N>& lhs, const T& rhs) {
        return AA<T,I,G,N>(lhs) += rhs;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator+(const T& lhs, const AA<T,I,G,N>& rhs) {
        return AA<T,I,G,N>(rhs) += lhs;
    }
    

    /*  OPERATOR -  */
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator-(const AA<T,I,G,N>& lhs, const AA<T,I,G,N>& rhs) {
        return binary_op(lhs, rhs, [](const T& a, const T& b){ return a - b; }, T(0));
    }

    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>& AA<T,I,G,N>::operator-=(const T& other) {
        this->x0() -= other;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>& AA<T,I,G,N>::operator-=(const AA<T,I,G,N>& other) {
        return *this = *this - other;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator-(const AA<T,I,G,N>& lhs, const T& rhs) {
        return +lhs -= rhs;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator-(const T& lhs, const AA<T,I,G,N>& rhs) {
        return -rhs += lhs;
    }

    
    /*  OPERATOR *  */
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator*(const AA<T,I,G,N>& lhs, const AA<T,I,G,N>& rhs) {
        AA<T,I,G,N> out(lhs.index_generator());

        const T lhs_x0(lhs.x0()), rhs_x0(rhs.x0());
        const I lhs_n(lhs.n()), rhs_n(rhs.n());
        out.x0() = lhs_x0 * rhs_x0;
        I lhs_i(0), rhs_i(0);
        while(lhs_i < lhs_n || rhs_i < rhs_n) {
            const I lhs_idx(lhs_i >= lhs_n ? G::last_symbol_index : lhs(lhs_i));
            const I rhs_idx(rhs_i >= rhs_n ? G::last_symbol_index : rhs(rhs_i));
            const bool le = lhs_idx <= rhs_idx;
            const bool ge = lhs_idx >= rhs_idx;
            out.append_symbol( (le ? lhs_idx : rhs_idx), 
                    (le ? lhs[lhs_i] : T(0)) * rhs_x0
                  + (ge ? rhs[rhs_i] : T(0)) * lhs_x0);
            lhs_i += le;
            rhs_i += ge;
        }
        out.append_new_symbol(lhs.radius() * rhs.radius());
        return out;
    }

    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>& AA<T,I,G,N>::operator*=(const T& other) {
        return inplace_unary_op(*this, [&](T& x){ x *= other; });
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>& AA<T,I,G,N>::operator*=(const AA<T,I,G,N>& other) {
        return *this = *this * other;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator*(const AA<T,I,G,N>& lhs, const T& rhs) {
        return +lhs *= rhs;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator*(const T& lhs, const AA<T,I,G,N>& rhs) {
        return +rhs *= lhs;
    }
    

    /*  OPERATOR /  */
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator/(const AA<T,I,G,N>& lhs, const AA<T,I,G,N>& rhs) {
        return lhs * rhs.reciprocal();
    }

    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>& AA<T,I,G,N>::operator/=(const T& other) {
        return inplace_unary_op(*this, [&](T& x){ x /= other; });
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>& AA<T,I,G,N>::operator/=(const AA<T,I,G,N>& other) {
        return *this = *this / other;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator/(const AA<T,I,G,N>& lhs, const T& rhs) {
        return +lhs *= T(1)/rhs;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> operator/(const T& lhs, const AA<T,I,G,N>& rhs) {
        return lhs * rhs.reciprocal();
    }
    
    template <typename T, typename I, typename G, I N>
    AA<T,I,G,N> __host__ __device__ AA<T,I,G,N>::reciprocal() const {
        Interval<T> interval(*this);
        T a = interval.a;
        T b = interval.b;

        if (a <= T(0) && b >= T(0)) {
            return AA<T,I,G,N>(this->index_generator(), cuda::std::numeric_limits<T>::infinity());
        }

        const T aa = ::fabs(a);
        const T bb = ::fabs(b);

        a = min(aa, bb);
        b = max(aa, bb);
        
        const T alpha = -T(1)/(b*b);
        const Interval<T> ii(T(1)/a - alpha*a, T(2)/b);
        const T beta = (interval.a < T(0) ? T(-1) : T(1)) * ii.midpoint();
        const T delta = ii.radius();
        
        return AA<T,I,G,N>(*this, alpha, beta, delta);
    }

    
    /* SQRT */
    template <typename T, typename I, typename G, I N>
    AA<T,I,G,N> __host__ __device__ sqrt(const AA<T,I,G,N>& val) {
        Interval<T> interval(val);
        const T a = max(interval.a, T(0));  /* we assume val is always positive */
        const T b = max(interval.b, T(0));  /* we assume val is always positive */

        const T t = ::sqrt(a) + ::sqrt(b);
        const T alpha = T(1)/t;
        const T beta = t/T(8) + ::sqrt(a*b)/(T(2)*t);
        const T rdelta = ::sqrt(b) - ::sqrt(a);
        const T delta = rdelta * rdelta / (T(8)*t);

        return AA<T,I,G,N>(val, alpha, beta, delta);
    }

} /* end of namespace cusvna */
    
template <typename T, typename I, typename G, I N>
__host__ std::ostream& operator<<(std::ostream& os, const cusvna::AA<T,I,G,N>& a)
{
    os << "AA(b=" << a.x0() << ", x=[";
    for (I i = 0; i < a.n(); ++i) {
        os << a.idx(i) << ": " << a.x(i);
        if (i != a.n()-1) { os << ", "; }
    }
    os << "])";
    return os;
}

#endif /* end of include guard: AA_H_TLKSOS6P */
