<p align="center">
  <img src="logo/cusvna-logo.svg" alt="cusvna" width="600">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/CUDA-11%2B-76B900?logo=nvidia&logoColor=white" alt="CUDA 11+"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white" alt="C++17"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/NVRTC-supported-76B900?logo=nvidia&logoColor=white" alt="NVRTC supported"></a>
  <img src="https://img.shields.io/badge/header--only-yes-brightgreen" alt="Header-only">
  <img src="https://img.shields.io/badge/precision-float%20%7C%20double-orange" alt="float and double">
</p>

**cusvna** (CUDA Self-Validated Numerical Analysis) is a header-only C++17 library for CUDA providing
computations with **guaranteed error bounds**. It implements [affine arithmetic](https://en.wikipedia.org/wiki/Affine_arithmetic)
(AA), a model that keeps track of the correlations between quantities and therefore yields much tighter
bounds than interval arithmetic on long computations. Two models are available: full affine arithmetic
(`cusvna::AA`) and the cheaper fixed-size AF1 forms (`cusvna::AF1`), see [AF1 forms](#af1-forms).

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
point numbers and on affine forms. Replacing `cusvna::AA<float>` by `cusvna::AF1<float, 2>` (from
`<cusvna/af1.cuh>`) evaluates the same code with AF1 forms.

## Installation

cusvna is header-only: add the root of this repository to your include paths and include `<cusvna/aa.cuh>`
and/or `<cusvna/af1.cuh>`.

Requirements:

* CUDA 11 or newer (the library uses `cuda::std` from libcu++, shipped with the CUDA toolkit),
* C++17 (`nvcc -std=c++17`, `nvrtc --std=c++17`),
* **no fast math**, even with `CUSVNA_IGNORE_ROUNDING`, see [Guarantees](#guarantees).

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

## AF1 forms

`cusvna::AF1<T, N>` implements the AF1 model of Messine: a form has exactly `N` input noise symbols and a
single error term,

    x = b + x_1 e_1 + ... + x_N e_N + err e_err

Affine operations are exact up to rounding as in AA, but nonlinear operations (products of forms, elementary
functions) add their approximation error to `err` instead of creating a new noise symbol. Forms have a
constant size of `N + 2` scalars and need no symbol types, tags or merges, which makes them cheaper. The
price is that correlations between results of nonlinear operations are lost: on `x = [0.1, 0.3]`,
`sin(x) - sin(x)` has a radius of `1e-3` (twice the approximation error of `sin`) with AF1, and `2e-8`
(rounding errors only) with AA.

* Input symbols are created by `AF1::from_interval<K>(interval)` with `0 <= K < N` (checked at compile
  time), or `AF1::variable(k, interval)` with a runtime index. `AF1::constant(interval)` encloses an
  interval without any symbol.
* All the operations listed below are available with the same names and the same guarantees, forms of
  different models cannot be mixed.

AF1 is a good default when nonlinear subexpressions are not reused, which is the case of most signed
distance functions. AA is tighter when the results of nonlinear operations appear several times in an
expression. See [Performance](#performance).

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

* **Rounding**: device code uses the directed rounding instructions of the hardware (`__fadd_ru`,
  `__dmul_rd`, ...), which are exact and as fast as round-to-nearest ones. Rounding errors of affine
  operations are computed exactly (TwoSum, fused multiply-add residuals) and accumulated. In host code, or
  in device code when `CUSVNA_EMULATED_ROUNDING` is defined, everything runs in the default round-to-nearest
  mode and outward rounding is emulated with error-free transformations, moving inexact bounds by at least
  one ulp. Emulated rounding is 2 to 6 times slower on GPUs, see [Performance](#performance).
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
* **hash collisions** (AA only): a collision between the runtime tags of two different values would wrongly
  correlate them. For non-adversarial inputs the probability is about `2^-64` per comparison of two generated
  symbols of the same type.

### Ignoring rounding errors

When `CUSVNA_IGNORE_ROUNDING` is defined, cusvna ignores all floating point rounding errors: directed
operations become round-to-nearest operations, the error terms of affine operations are not computed, the
results of math library functions are taken as exact, and operations may be contracted into fused
multiply-adds. Approximation errors of nonlinear functions are still bounded. The results are **no longer
rigorous**: enclosures may miss the exact result by a few ulps, and deep subdivisions may discard cells that
contain zeros. This mode is meant for visualization, where it is up to 2 times faster, see
[Performance](#performance).

**Fast math is strictly discouraged, even with `CUSVNA_IGNORE_ROUNDING`.** Its approximate math functions
(`__sinf`, `__expf`, ...) are far less accurate than the CUDA math library and break the enclosures: with
`--use_fast_math`, `05_levelset.py` misses most surfaces and runs 5 times slower because interval marching
cannot conclude.

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
python cusvna/examples/02_quadtree.py            # AF1 forms
python cusvna/examples/02_quadtree.py aa         # AA forms
python cusvna/examples/02_quadtree.py --fast     # AF1 forms, rounding errors ignored
```

`02_quadtree.py` to `05_levelset.py` take an optional `af1` (default) or `aa` argument selecting the affine
forms, and a `--fast` option that defines `CUSVNA_IGNORE_ROUNDING` (see
[Ignoring rounding errors](#ignoring-rounding-errors)). The examples never use fast math. The OptiX examples
require the NVIDIA OptiX SDK and the `python-optix` submodule.

## Performance

Time of the evaluations on an RTX 5090 (float), with hardware directed rounding (default) and with
`CUSVNA_EMULATED_ROUNDING`:

| Benchmark | AA | AF1 | AA, emulated rounding | AF1, emulated rounding |
|---|---|---|---|---|
| `02_quadtree.py`, level 25 (126M active cells) | 3.0 ms | 2.4 ms | 6.7 ms | 5.9 ms |
| `03_octtree.py`, level 11 (20M active cells) | 4.1 ms | 1.7 ms | 11.9 ms | 9.1 ms |
| `03_octtree.py`, level 12 (81M active cells) | 16.4 ms | 6.8 ms | 48 ms | 36 ms |
| `05_levelset.py`, 1024x768, 1 sample per pixel | 8.5-10 ms | 7-9 ms | 29-31 ms | 25-26 ms |

Both models produce the same numbers of active cells up to a few cells and the same images.

With `CUSVNA_IGNORE_ROUNDING` (`--fast` option of the examples):

| Benchmark | AF1 | AF1, rounding ignored | AA | AA, rounding ignored |
|---|---|---|---|---|
| `02_quadtree.py`, level 22 (9.5M / 8.4M active cells) | 0.29 ms | 0.14 ms | 0.35 ms | 0.16 ms |
| `03_octtree.py`, level 11 (20M active cells) | 1.9 ms | 0.8 ms | 4.5 ms | 3.4 ms |
| `03_octtree.py`, level 12 (81M active cells) | 7.3 ms | 3.0 ms | 18 ms | 13.3 ms |
| `05_levelset.py`, 1024x768, 1 sample per pixel | 8-9 ms | 7-8 ms | 9-11 ms | 7.3-8 ms |

Ignoring rounding errors keeps fewer cells in the quadtree (8.4M instead of 9.5M at level 22, 57M instead
of 126M at level 25): in single precision, the rounding errors of deep levels are larger than the cells.
The octree keeps 0.02% fewer cells and the images of `05_levelset.py` are the same.

The examples compile their device code for the architecture of the GPU (`-arch=compute_XY`, see
`nvrtc_arch_flags` in `examples/utils.py`): NVRTC targets `compute_52` by default, which made the
level set example about 30% slower on this GPU.

## Tests

The tests compare cusvna against libaffa, which must be cloned in `ext/libaffa`
(`git submodule update --init ext/libaffa`), and check the soundness of all operations against extended
precision evaluations, for AA and AF1 forms, in float and double, in host code (emulated rounding) and in
device code (hardware rounding and CUDA math library).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build
```

Add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` if `nvcc` is not in your `PATH`.

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
