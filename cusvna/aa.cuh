#ifndef AA_H_TLKSOS6P
#define AA_H_TLKSOS6P

#include "cusvna/interval.cuh"

namespace cusvna {
    
    template <typename T, typename I, I N>
    class AA {
        /* Affine Arithmetic with fixed number of active noise symbols (sparse representation with N indices) */
        public:
            __device__ AA();
            __device__ AA(const T& base);
            __device__ AA(const AA<T,I,N>& other);
            __device__ AA<T,I,N>& operator=(const AA<T,I,N>& other);
            
            __device__ const T& x0() const;             /* base coefficient x0 */
            __device__ const I& n() const;              /* number of active noise symbols 0 <= n <= N. */
            __device__ const T* x() const;              /* x is the magnitude vector of the uncertainty coerresponding to noise symbols. */
            __device__ const I* idx() const;            /* idx represent the index vector of noise symbols, idx is assumed to be sorted. */
            __device__ const T& x(const I& i) const;    /* x[i] is the magnitude of the uncertainty coerresponding to noise symbol idx[i]. */
            __device__ const I& idx(const I& i) const;  /* idx[i] represent the index of noise symbol number i, idx is assumed to be sorted. */
            __device__ const I& operator()(const I& i) const;  /* alias of idx(i) */
            __device__ const T& operator[](const I& i) const;  /* alias of x(i)   */

            __device__ T& x0();
            __device__ I& n();
            __device__ T* x();
            __device__ I* idx();
            __device__ T& x(const I& i);
            __device__ I& idx(const I& i);
            __device__ I& operator()(const I& i);
            __device__ T& operator[](const I& i);

            __device__ AA<T,I,N>& append_symbol(const I& symbol_index, const T& magnitude);

            __device__ AA<T,I,N>& operator+=(const T& other);
            __device__ AA<T,I,N>& operator+=(const AA<T,I,N>& other);
        protected:
            T m_coeffs[N+1]; 
            I m_indices[N+1];
    };
        
    /* operators */
    template <typename T, typename I, I N> AA<T,I,N> __device__ operator+(const AA<T,I,N>& lhs, const AA<T,I,N>& rhs);
    template <typename T, typename I, I N> AA<T,I,N> __device__ operator+(const AA<T,I,N>& lhs, const T& rhs);
    template <typename T, typename I, I N> AA<T,I,N> __device__ operator+(const T& lhs, const AA<T,I,N>& rhs);
    
    /* implementation */   
    template <typename T, typename I, I N>
    __device__ AA<T,I,N>::AA() : m_coeffs{0}, m_indices{0} {
        static_assert(N>=0, "N<0");
    }
    
    template <typename T, typename I, I N>
    __device__ AA<T,I,N>::AA(const T& base) : m_coeffs{0}, m_indices{0} {
        static_assert(N>=0, "N<0");
        this->x0() = base;
    }
    
    template <typename T, typename I, I N>
    __device__ AA<T,I,N>::AA(const AA<T,I,N>& other) : m_coeffs{0}, m_indices{0} {
        static_assert(N>=0, "N<0");
        *this = other;
    }

    template <typename T, typename I, I N>
    __device__ AA<T,I,N>& AA<T,I,N>::operator=(const AA<T,I,N>& other) {
        if (this != &other) {
            this->x0() = other.x0();
            I& i = this->n();
            for (i = 0; i < other.n(); ++i) {
                this->idx(i) = other.idx(i);
                this->x(i) = other.x(i);
            }
        }
        return *this;
    }
    
    template <typename T, typename I, I N>
    __device__ T& AA<T,I,N>::x0() {
        return m_coeffs[0];
    }
    
    template <typename T, typename I, I N>
    __device__ I& AA<T,I,N>::n() {
        return m_indices[0];
    }
    
    template <typename T, typename I, I N>
    __device__ T* AA<T,I,N>::x() {
        return &m_coeffs[1];
    }
    
    template <typename T, typename I, I N>
    __device__ I* AA<T,I,N>::idx() {
        return &m_indices[1];
    }
    
    template <typename T, typename I, I N>
    __device__ T& AA<T,I,N>::x(const I& i) {
        assert(i<this->n());
        return m_coeffs[i+1];
    }
    
    template <typename T, typename I, I N>
    __device__ I& AA<T,I,N>::idx(const I& i) {
        assert(i<this->n());
        return m_indices[i+1];
    }
    
    template <typename T, typename I, I N>
    __device__ I& AA<T,I,N>::operator()(const I& i) {
        return this->idx(i);
    }
    
    template <typename T, typename I, I N>
    __device__ T& AA<T,I,N>::operator[](const I& i) {
        return this->x(i);
    }
    
    template <typename T, typename I, I N>
    __device__ const T& AA<T,I,N>::x0() const {
        return m_coeffs[0];
    }
    
    template <typename T, typename I, I N>
    __device__ const I& AA<T,I,N>::n() const {
        return m_indices[0];
    }
    
    template <typename T, typename I, I N>
    __device__ const T* AA<T,I,N>::x() const {
        return &m_coeffs[1];
    }
    
    template <typename T, typename I, I N>
    __device__ const I* AA<T,I,N>::idx() const {
        return &m_indices[1];
    }
    
    template <typename T, typename I, I N>
    __device__ const T& AA<T,I,N>::x(const I& i) const {
        assert(i<this->n());
        return m_coeffs[i+1];
    }
    
    template <typename T, typename I, I N>
    __device__ const I& AA<T,I,N>::idx(const I& i) const {
        assert(i<this->n());
        return m_indices[i+1];
    }
    
    template <typename T, typename I, I N>
    __device__ const I& AA<T,I,N>::operator()(const I& i) const {
        return this->idx(i);
    }
    
    template <typename T, typename I, I N>
    __device__ const T& AA<T,I,N>::operator[](const I& i) const {
        return this->x(i);
    }
            
    template <typename T, typename I, I N>
    __device__ AA<T,I,N>& AA<T,I,N>::append_symbol(const I& symbol_index, const T& magnitude) {
        I& i = this->n();
        assert(i < N);
        assert(i==0 || this->idx(i-1) < symbol_index);
        this->idx(i) = symbol_index;
        this->x(i) = magnitude;
        ++i;
        return *this;
    }
    
    
    /* operators */
    template <typename T, typename I, I N>
    __device__ AA<T,I,N> operator+(const AA<T,I,N>& lhs, const AA<T,I,N>& rhs) {
        AA<T,I,N> out(lhs.x0() + rhs.x0());
        I lhs_i(0), rhs_i(0);
        while(lhs_i < lhs.n() || rhs_i < rhs.n()) {
            const I lhs_idx(lhs(lhs_i)), rhs_idx(rhs(rhs_i));
            const bool le = lhs_idx <= rhs_idx;
            const bool ge = lhs_idx >= rhs_idx;
            out.append_symbol( (le ? lhs_idx : rhs_idx), le*lhs[lhs_i] + ge*rhs[rhs_i]);
            lhs_i += le;
            rhs_i += ge;
        }
        return out;
    };

    template <typename T, typename I, I N>
    __device__ AA<T,I,N>& AA<T,I,N>::operator+=(const T& other) {
        this->x0() += other;
        for (I i = 0; i < this->n(); ++i) {
            this->x(i) += other;
        }
        return *this;
    }
    
    template <typename T, typename I, I N>
    __device__ AA<T,I,N>& AA<T,I,N>::operator+=(const AA<T,I,N>& other) {
        return *this = *this + other;
    }
    
    template <typename T, typename I, I N>
    __device__ AA<T,I,N> operator+(const AA<T,I,N>& lhs, const T& rhs) {
        AA<T,I,N> out(lhs);
        return out += rhs;
    };
    
    template <typename T, typename I, I N>
    __device__ AA<T,I,N> operator+(const T& lhs, const AA<T,I,N>& rhs) {
        AA<T,I,N> out(rhs);
        return out += lhs;
    };


} /* end of namespace cusvna */

#endif /* end of include guard: AA_H_TLKSOS6P */
