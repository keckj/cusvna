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
         *    Sparse representation with at most N noise symbols 
         */
        public:
            __host__ __device__ AA(G& index_generator);
            __host__ __device__ AA(const Interval<T>& interval, G& index_generator);
            __host__ __device__ AA(const AA<T,I,G,N>& other);
            __host__ __device__ AA<T,I,G,N>& operator=(const AA<T,I,G,N>& other);
            
            __host__ __device__ const T& x0() const;             /* base coefficient x0 */
            __host__ __device__ const I& n() const;              /* number of active noise symbols 0 <= n <= N. */
            __host__ __device__ const T* x() const;              /* x is the magnitude vector of the uncertainty coerresponding to noise symbols. */
            __host__ __device__ const I* idx() const;            /* idx represent the index vector of noise symbols, idx is assumed to be sorted. */
            __host__ __device__ const T& x(const I& i) const;    /* x[i] is the magnitude of the uncertainty coerresponding to noise symbol idx[i]. */
            __host__ __device__ const I& idx(const I& i) const;  /* idx[i] represent the index of noise symbol number i, idx is assumed to be sorted. */
            __host__ __device__ const I& operator()(const I& i) const;  /* alias of idx(i) */
            __host__ __device__ const T& operator[](const I& i) const;  /* alias of x(i)   */

            __host__ __device__ T& x0();
            __host__ __device__ I& n();
            __host__ __device__ T* x();
            __host__ __device__ I* idx();
            __host__ __device__ T& x(const I& i);
            __host__ __device__ I& idx(const I& i);
            __host__ __device__ I& operator()(const I& i);
            __host__ __device__ T& operator[](const I& i);
            __host__ __device__ G& index_generator() const;

            __host__ __device__ AA<T,I,G,N>& append_symbol(const I& symbol_index, const T& magnitude);
            __host__ __device__ AA<T,I,G,N>& append_new_symbol(const T& magnitude);

            __host__ __device__ AA<T,I,G,N> operator-() const;
            __host__ __device__ AA<T,I,G,N> operator+() const;

            __host__ __device__ AA<T,I,G,N>& operator-=(const T& other);
            __host__ __device__ AA<T,I,G,N>& operator+=(const T& other);
            __host__ __device__ AA<T,I,G,N>& operator*=(const T& other);

            __host__ __device__ AA<T,I,G,N>& operator-=(const AA<T,I,G,N>& other);
            __host__ __device__ AA<T,I,G,N>& operator+=(const AA<T,I,G,N>& other);
            __host__ __device__ AA<T,I,G,N>& operator*=(const AA<T,I,G,N>& other);
        private:
            T m_coeffs[N+1]; 
            I m_indices[N+1];
            G& m_index_generator; /* symbol index generator, no static variables here */
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
    
    /* implementation */   
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>::AA(G& index_generator) : 
        m_coeffs{0}, m_indices{0}, m_index_generator(index_generator)  {
        static_assert(N>=0, "N<0");
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>::AA(const Interval<T>& interval, G& index_generator):
        m_coeffs{0}, m_indices{0}, m_index_generator(index_generator) {
        static_assert(N>=0, "N<0");
        this->x0() = interval.midpoint();
        this->append_new_symbol(interval.radius());
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N>::AA(const AA<T,I,G,N>& other) : 
        m_coeffs{0}, m_indices{0}, m_index_generator(other.index_generator()) {
        static_assert(N>=0, "N<0");
        *this = other;
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
    __host__ __device__ G& AA<T,I,G,N>::index_generator() const {
        return this->m_index_generator;
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
        AA<T,I,G,N> out(*this);
        inplace_unary_op(out, [](T& a){ return +a; });
        return out;
    }
    
    template <typename T, typename I, typename G, I N>
    __host__ __device__ AA<T,I,G,N> AA<T,I,G,N>::operator-() const {
        AA<T,I,G,N> out(*this);
        inplace_unary_op(out, [](T& a){ return -a; });
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
        return inplace_unary_op(*this, [&](T& x){ x += other; });
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
        return inplace_unary_op(*this, [&](T& x){ x -= other; });
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
        assert(false); /* not implemented */
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

} /* end of namespace cusvna */
    
template <typename T, typename I, typename G, I N>
__host__ std::ostream& operator<<(std::ostream& os, const cusvna::AA<T,I,G,N>& a)
{
    os << a.n();
    return os;
}

#endif /* end of include guard: AA_H_TLKSOS6P */
