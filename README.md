# cusvna

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**cusvna** (CUDA Self-Validated Numerical Analysis) is a header-only C++17 library for CUDA providing
computations with **guaranteed error bounds**. It implements [affine arithmetic](https://en.wikipedia.org/wiki/Affine_arithmetic)
(AA), a model that keeps track of the correlations between quantities and therefore yields much tighter
bounds than interval arithmetic on long computations.

* **Rigorous**: the exact real result of a computation always lies within the computed bounds,
  including floating point rounding errors.
* **GPU friendly**: no dynamic allocation, affine forms live in registers. Works in `__device__` code
  compiled with nvcc or at runtime with NVRTC, and in host code compiled with nvcc.
* **Complete**: arithmetic, all usual elementary functions of the CUDA math API, signed distance
  helpers (`min`, `max`, `hypot`, `smooth_min`, ...) and rigorous predicates, in `float` and `double`.

Typical applications are robust root isolation and subdivision schemes (quadtrees, octrees), ray tracing
of implicit surfaces and level sets, and any numerical method that must not miss a solution.

## Example

```c++
#include <cusvna/aa.cuh>

using Interval = cusvna::Interval<float>;
using AA = cusvna::AA<float>;

/* returns false only if the unit circle certainly does not cross the box X x Y */
__device__ bool may_contain_root(const Interval& X, const Interval& Y) {
    const auto x = AA::from_interval<0>(X);   /* input noise symbols are numbered by the user */
    const auto y = AA::from_interval<1>(Y);
    const auto f = sqr(x) + sqr(y) - 1.0f;    /* f(x, y) is guaranteed to lie in f.interval() */
    return possibly_zero(f);
}
```

Functions are found by argument dependent lookup, so the same templated code can be evaluated on floating
point numbers and on affine forms.

## Installation

cusvna is header-only: add the root of this repository to your include paths and include `<cusvna/aa.cuh>`.

Requirements:

* CUDA 11 or newer (the library uses `cuda::std` from libcu++, shipped with the CUDA toolkit),
* C++17 (`nvcc -std=c++17`, `nvrtc --std=c++17`),
* **no fast math**, see [Guarantees](#guarantees).

With NVRTC (for instance through CuPy), pass the CUDA include directory and the root of this repository
as include paths, see `examples/utils.py`.

## Affine arithmetic in cusvna

An affine form represents an unknown real quantity as

    x = b + x_1 e_1 + ... + x_N e_N + err e_err,      e_i, e_err unknown in [-1, 1]

The noise symbols `e_i` are shared between forms and carry correlations: `x - x` is exactly zero, while
interval arithmetic would return `[-2r, 2r]`. Nonlinear operations approximate their result by an affine
function of their input and attach the approximation error to a new noise symbol. The error term
`err` accumulates rounding errors and is independent from everything else.

In cusvna, noise symbols are identified at compile time by types: `cusvna::AA<T, Is...>` stores exactly
`sizeof...(Is)` coefficients and the type of the result of an operation is computed at compile time from
the types of its operands.

* Input symbols are created by `AA::from_interval<K>(interval)`, where `K` identifies the input.
  Two forms created with the same `K` represent the same quantity. The macro `new_aa_from_interval(interval)`
  uses a unique `K` for each line of code (not available with NVRTC).
* Since the types of the symbols created by nonlinear operations only depend on the types of the operands,
  each of them also carries a 64-bit runtime tag: a hash of the content of the operands. Symbols with equal
  tags represent the same quantity, symbols with different tags are merged conservatively.

## Supported operations

| Operations | Notes |
|---|---|
| `+`, `-`, `*`, `/` between forms and with scalars | |
| `reciprocal(x)` | min-range approximation, entire if `0` belongs to `x` |
| `sqrt(x)`, `exp(x)`, `log(x)` | Chebyshev approximations |
| `sqr(x)`, `pow(x, int n)` | `sqr(x)` is tighter than `x*x` |
| `pow(x, T y)` | for `x >= 0` (`x > 0` if `y < 0`) |
| `pow(x, y)`, `pow(c, y)` with an affine exponent | `exp(y*log(x))`, for `x > 0` |
| `rsqrt(x)`, `cbrt(x)`, `rcbrt(x)` | |
| `exp2(x)`, `exp10(x)`, `expm1(x)`, `log2(x)`, `log10(x)`, `log1p(x)` | |
| `sin(x)`, `cos(x)`, `tan(x)`, `cot(x)` | entire if a pole belongs to `x` |
| `sinpi(x)`, `cospi(x)` | `sin(pi*x)` and `cos(pi*x)` without rounding `pi*x`, exact at integers and half-integers |
| `atan(x)`, `asin(x)`, `acos(x)`, `atan2(y, x)` | `atan2` returns `[-pi, pi]` when the input box meets the branch cut `y = 0, x <= 0` |
| `sinh(x)`, `cosh(x)`, `tanh(x)`, `asinh(x)`, `acosh(x)`, `atanh(x)` | |
| `erf(x)`, `erfc(x)`, `normcdf(x)` | |
| `tgamma(x)`, `lgamma(x)` | for `x > 0` only (entire otherwise) |
| `abs(x)`, `positive_part(x)` | `positive_part(x) = max(x, 0)` |
| `min(x, y)`, `max(x, y)`, `fmin`, `fmax`, `clamp(x, lo, hi)` | `min(x, y) + max(x, y)` cancels to `x + y` |
| `hypot(x, y)`, `hypot(x, y, z)`, `norm3d`, `rhypot`, `rnorm3d` | |
| `fma(x, y, z)`, `lerp(a, b, t)`, `smoothstep(e0, e1, x)` | |
| `smooth_min(a, b, k)`, `smooth_max(a, b, k)` | polynomial smooth minimum of distances |
| `floor`, `ceil`, `trunc`, `round`, `rint`, `fract`, `modf(x, &ip)` | exact when no discontinuity is crossed, else the range of the result |
| `fmod(x, y)`, `remainder(x, y)` | `y` may be a scalar or a form, exact and correlated with `x` and `y` when the integer quotient is known |
| `remquo(x, y, &quo_lo, &quo_hi)` | returns the range of the whole integer quotient `rint(x/y)` (not only its low bits as in C), `quo_lo == quo_hi` when it is known |
| `certainly_lt`, `certainly_le`, `certainly_gt`, `certainly_ge`, `possibly_zero` | rigorous predicates |
| `x.interval()`, `x.sign()`, `x.radius()`, `x.is_entire()` | conversions back to intervals |
| `hull`, `intersect`, `contains`, `subset`, `is_empty` | on `cusvna::Interval` |

These cover all the operations of [libaffa](https://github.com/ogay/libaffa), which is used as a reference
in the tests (`inv` is `reciprocal`, `cotan` is `cot` and `half_plane` is `positive_part`).

## Guarantees

For any operation, the exact real result of the computation belongs to the range of the computed form,
including rounding errors. In particular `x.interval()`, `x.sign()` and the predicates are rigorous. 
Bounds are never NaN.

* **Rounding**: all operations run in the default round-to-nearest mode. Rounding errors of affine
  operations are computed exactly with error-free transformations (TwoSum, TwoProd) and accumulated,
  outward rounding is emulated by moving inexact bounds by at least one ulp.
* **Nonlinear functions**: the residual `f(x) - alpha*x` is bounded with outward rounding by splitting
  the input range into pieces where `f` is convex or concave. The results of math library functions are
  assumed to be within twice the maximum error documented in the CUDA math API (see `cusvna/approx.cuh`).
  `tgamma` and `lgamma` are restricted to positive inputs where this assumption holds; CUDA does not bound
  the relative error of `lgamma` for negative inputs close to its zeros.
* **Domains**: if an input range is not entirely inside the domain of a function, the result is entire
  (its interval is `[-inf, inf]`), except for `sqrt`, `pow(x, T y)`, `acosh`, `asin` and `acos` which are
  evaluated on the part of their input inside their domain when it is not empty. Overflows produce entire
  forms. Entire forms propagate through all operations, except through bounded functions such as `sin`.

These guarantees hold under the following assumptions:

* IEEE-754 binary32 (`float`) or binary64 (`double`) arithmetic in round-to-nearest mode,
* **no fast math**: do not compile with `--use_fast_math` (nvcc, nvrtc) or `-ffast-math` (host),
  `-ftz=true`, `-prec-div=false` or `-prec-sqrt=false`,
* host code must not contract floating point operations into fused multiply-adds (`-ffp-contract=off`,
  which is the default of gcc with `-std=c++17`). Device code uses non-contractible intrinsics,
* **hash collisions**: a collision between the runtime tags of two different values would wrongly correlate
  them. For non-adversarial inputs the probability is about `2^-64` per comparison of two generated symbols
  of the same type.

## Examples

The `examples` directory contains Python scripts that compile CUDA kernels at runtime with CuPy (NVRTC):

| Script | Description |
|---|---|
| `00_1d.py` | enclosures of a 1D function on refined subdivisions |
| `01_2d.py` | sign classification of a 2D function on refined grids |
| `02_quadtree.py` | adaptive quadtree refinement around the zero level set of a 2D function |
| `03_octtree.py` | adaptive octree refinement around an implicit surface, meshed with marching cubes |
| `04_raytracing.py` | ray tracing of implicit surfaces with OptiX |
| `05_levelset.py` | interactive OptiX viewer of level sets |

```sh
git clone --recursive https://github.com/keckj/cusvna.git
pip install -r cusvna/examples/requirements.txt
python cusvna/examples/02_quadtree.py
```

The OptiX examples require the NVIDIA OptiX SDK and the `python-optix` submodule.

## Tests

The tests compare cusvna against libaffa, which must be cloned in `ext/libaffa`
(`git submodule update --init ext/libaffa`), and check the soundness of all operations against extended
precision evaluations.

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

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

## License

cusvna is released under the [MIT License](LICENSE).
