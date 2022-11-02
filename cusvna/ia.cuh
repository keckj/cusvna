
#ifndef IA_H_59FBPTE3
#define IA_H_59FBPTE3

namespace cusvna {
    
    template <typename T, typename I, I max_noise_symbols>
    class IA {
        public:
            IA(): x0(0), noise_symbols_count(0), noise_symbols(), noise_indices();
        protected:
            T base;
            I noise_symbols_count;

            T noise_symbols[max_noise_symbols]; 
            I noise_indices[max_noise_symbols];
    };

} /* end of namespace cusvna */

#endif /* end of include guard: IA_H_59FBPTE3 */
