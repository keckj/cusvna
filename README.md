# cusvna

Cusvna (CUDA Self-Validated Numerical Analysis) is a header-only library providing computation models 
in which approximate results are automatically computed with guaranteed error bounds. 
The library currently focus on one such model, the affine arithmetic model (AA).

## Affine arithmetic 

Affine arithmetic is a model designed to provide tighter bounds when compared to interval arithmetic.
Unlike Interval Arithmetic, it keeps track of correlations between computed and input quantities, and is therefore resistant to the explosion error observed in long interval computations. 

An affine form represents an unknown real quantity as

    x = b + x_1 e_1 + ... + x_N e_N + err e_err,      e_i, e_err unknown in [-1, 1]

The noise symbols `e_i` are shared between forms and carry correlations. The error term `err` accumulates 
rounding errors and is independent from everything else.

In cusvna, noise symbols are identified at compile time by types: `cusvna::AA<T, Is...>` stores exactly
`sizeof...(Is)` coefficients in registers, there is no dynamic allocation and forms can be used in device
code, including code compiled at runtime with NVRTC.

```c++
#include <cusvna/aa.cuh>

using Interval = cusvna::Interval<float>;
using AA = cusvna::AA<float>;

__device__ bool may_contain_root(const Interval& X, const Interval& Y) {
    const auto x = AA::from_interval<0>(X);   /* input noise symbols are numbered by the user */
    const auto y = AA::from_interval<1>(Y);
    const auto f = sqr(x) + sqr(y) - 1.0f;    /* f(x, y) is guaranteed to lie in f.interval() */
    return f.sign() == cusvna::Sign::UNKNOWN;
}
```

## Supported operations

| Operations | Notes |
|---|---|
| `+`, `-`, `*`, `/` between forms and with scalars | |
| `reciprocal(x)` | min-range approximation, entire if `0` belongs to `x` |
| `sqrt(x)`, `exp(x)`, `log(x)` | Chebyshev approximations |
| `sqr(x)`, `pow(x, int n)` | `sqr(x)` is tighter than `x*x` |
| `pow(x, T y)` | for `x >= 0` (`x > 0` if `y < 0`) |
| `sin(x)`, `cos(x)`, `tan(x)`, `cot(x)` | entire if a pole belongs to `x` |
| `sinh(x)`, `cosh(x)`, `tanh(x)`, `asinh(x)`, `acosh(x)`, `atanh(x)` | |
| `abs(x)`, `positive_part(x)` | `positive_part(x) = max(x, 0)` |
| `rsqrt(x)`, `cbrt(x)`, `rcbrt(x)` | |
| `exp2(x)`, `exp10(x)`, `expm1(x)`, `log2(x)`, `log10(x)`, `log1p(x)` | |
| `atan(x)`, `asin(x)`, `acos(x)` | `asin`, `acos` for `-1 <= x <= 1` |
| `erf(x)`, `erfc(x)`, `normcdf(x)` | |
| `sinpi(x)`, `cospi(x)` | `sin(pi*x)` and `cos(pi*x)` without rounding `pi*x`, exact at integers and half-integers |
| `min(x, y)`, `max(x, y)`, `fmin`, `fmax`, `clamp(x, lo, hi)` | `min(x, y) + max(x, y)` cancels to `x + y` |
| `hypot(x, y)`, `hypot(x, y, z)`, `norm3d`, `rhypot`, `rnorm3d` | |
| `fma(x, y, z)`, `lerp(a, b, t)`, `smoothstep(e0, e1, x)` | |
| `smooth_min(a, b, k)`, `smooth_max(a, b, k)` | polynomial smooth minimum of distances |
| `atan2(y, x)` | whole range `[-pi, pi]` when the input box meets the branch cut `y = 0, x <= 0` |
| `pow(x, y)`, `pow(c, y)` with an affine exponent | `exp(y*log(x))`, for `x > 0` |
| `floor`, `ceil`, `trunc`, `round`, `rint`, `fract`, `fmod(x, c)` | exact (`fract` and `fmod` stay correlated with `x`) when no discontinuity is crossed, else the range of the result |
| `certainly_lt`, `certainly_le`, `certainly_gt`, `certainly_ge`, `possibly_zero` | rigorous predicates |
| `hull`, `intersect`, `contains`, `subset`, `is_empty` | on `cusvna::Interval` |
| `x.interval()`, `x.sign()`, `x.radius()`, `x.is_entire()` | |

These cover all the operations of [libaffa](https://github.com/ogay/libaffa), which is used as a reference
in the tests (`inv` is `reciprocal`, `cotan` is `cot` and `half_plane` is `positive_part`).

## Guarantees

For any operation, the exact real result of the computation belongs to the range of the computed form,
including rounding errors. In particular `x.interval()` and `x.sign()` are rigorous. Bounds are never NaN.

* **Rounding**: all operations run in the default round-to-nearest mode. Rounding errors of affine
  operations are computed exactly with error-free transformations (TwoSum, TwoProd) and accumulated, 
  outward rounding is emulated by moving inexact bounds by at least one ulp.
* **Nonlinear functions**: the residual `f(x) - alpha*x` is bounded with outward rounding by splitting 
  the input range into pieces where `f` is convex or concave. The results of math library functions are
  assumed to be within twice the maximum error documented in the CUDA math API (see `cusvna/approx.cuh`).
* **Domains**: if an input range is not entirely inside the domain of a function, the result is entire
  (its interval is `[-inf, inf]`), except for `sqrt`, `pow(x, T y)`, `acosh`, `asin` and `acos` which are evaluated on
  the part of their input inside their domain when it is not empty. Overflows produce entire forms. 
  Entire forms propagate through all operations, except through bounded functions such as `sin` and `cos`.

These guarantees hold under the following assumptions:

* IEEE-754 binary32 (`float`) or binary64 (`double`) arithmetic in round-to-nearest mode,
* **no fast math**: do not compile with `--use_fast_math` (nvcc, nvrtc) or `-ffast-math` (host), 
  `-ftz=true`, `-prec-div=false` or `-prec-sqrt=false`,
* host code must not contract floating point operations into fused multiply-adds (`-ffp-contract=off`,
  which is the default of gcc with `-std=c++17`). Device code uses non-contractible intrinsics.

### Noise symbols and hash collisions

The type of a noise symbol created by a nonlinear operation (`*`, `exp`, ...) only depends on the types of
the operands, so two values of the same type share the same symbol types. To tell them apart, each 
generated symbol carries a 64-bit runtime tag, a hash of the content of the operands (base coefficient, 
symbol magnitudes and tags). Symbols with equal tags are treated as the same quantity (`z - z == 0`), 
symbols with different tags are merged conservatively. A hash collision between two different values
would wrongly correlate them; for non-adversarial inputs the probability is about `2^-64` per comparison
of two generated symbols of the same type.

Input symbols are under the responsibility of the user: two forms created with the same index
(`from_interval<K>` or `new_aa_from_interval` from the same line) represent the same quantity.

## Defects found in libaffa

The following libaffa functions do not return enclosures, they have been fixed in cusvna:

| Function | Input | libaffa | Exact range |
|---|---|---|---|
| `exp` | `[0, 1]` | `[0.682, 2.612]` | `[1, 2.718]` |
| `sin` | `[-4.41, 1.21]` | `[-0.951, 0.996]` | `[-1, 0.955]` |
| `abs` | `[-1, 1]` | `[-0.5, 0.5]` | `[0, 1]` |

In addition, `exp` only uses the tangent intercept instead of the mean of the secant and tangent 
intercepts, `sin` estimates its approximation error from 8 samples, `atanh` is not implemented (no return
value) and no operation accounts for rounding errors.

## Tests

Tests are compared against libaffa, which must be cloned in `ext/libaffa` (`git submodule update --init ext/libaffa`).

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build
```
