# cusvna

Cusvna (CUDA Self-Validated Numerical Analysis) is a header-only library providing computation models 
in which approximate results are automatically computed with guaranteed error bounds. 
The library currently focus on one such model, the affine arithmetic model (AA).

## Affine arithmetic 

Affine arithmetic is a model designed to provide tighter bounds when compared to interval arithmetic.
Unlike Interval Arithmetic, it keeps track of correlations between computed and input quantities, and is therefore resistant to the explosion error observed in long interval computations. 
