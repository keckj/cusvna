/* this file is part of example 05_levelset.py */
#include "examples/cuda/levelset_renderer.cuh"

/* must match the Shape enum of 05_levelset.py, FLOOR = 0 is defined by levelset_renderer.cuh */
enum Shape : int {
    GYROID      = 1,
    METABALLS   = 2,
    TWISTED_BOX = 3,
    KUMMER      = 4,
    BARTH       = 5,
    HEART       = 6,
    TETRAHEDRAL = 7,
};

/*----------------------------------------------------------------------------*/
/* Level sets                                                                 */
/*----------------------------------------------------------------------------*/

/* All the level sets are written once as templates, evaluated on affine forms to march the rays, on
   floats to refine the roots and on first order autodiff numbers to compute the normals. Functions of
   affine forms and autodiff numbers are found by argument dependent lookup in namespaces cusvna and
   autodiff, their float counterparts and the ones missing in autodiff are declared here. Shapes are given
   in local coordinates, they are negative inside and must fit in the cylinder x^2+z^2 <= 1, |y| <= 1. */
namespace shapes {
    using real = autodiff::Real<1, float>;
    __device__ inline float sqr(float x) { return x*x; }
    __device__ inline float abs(float x) { return fabsf(x); }
    __device__ inline float min(float x, float y) { return fminf(x, y); }
    __device__ inline float max(float x, float y) { return fmaxf(x, y); }
    __device__ inline float sin(float x) { return sinf(x); }
    __device__ inline float cos(float x) { return cosf(x); }
    __device__ inline float sqrt(float x) { return sqrtf(x); }
    __device__ inline float cbrt(float x) { return cbrtf(x); }
    __device__ inline float positive_part(float x) { return fmaxf(x, 0.0f); }
    __device__ inline real sqr(const real& x) { return x*x; }
    __device__ inline real positive_part(const real& x) { return max(x, 0.0f); }

    template <typename X, typename Y, typename Z>
    __device__ auto length(const X& x, const Y& y, const Z& z) {
        return sqrt(sqr(x) + sqr(y) + sqr(z));
    }

    /* a ball carved by a flowing gyroid lattice */
    template <typename V>
    __device__ auto gyroid(const V& x, const V& y, const V& z, const float t) {
        const float k = 6.0f;
        const auto X = k*x;
        const auto Y = k*y - 0.8f*t;
        const auto Z = k*z;
        const auto g = sin(X)*cos(Y) + sin(Y)*cos(Z) + sin(Z)*cos(X);
        const float thickness = 0.35f + 0.25f*sinf(0.7f*t);
        const auto lattice = (abs(g) - thickness) * (1.0f/k);
        const auto ball = length(x, y, z) - 0.95f;
        return max(lattice, ball);
    }

    /* five soft objects with compact polynomial falloff, w(d) = (1 - d^2/R^2)^2 for d < R */
    template <typename V>
    __device__ auto metaballs(const V& x, const V& y, const V& z, const float t) {
        auto blob = [&](const float a, const float b, const float c, const float phase, const float R) {
            const float3 center = 0.3f * make_float3(sinf(a*t + phase), cosf(b*t + 2.0f*phase), sinf(c*t + 3.0f*phase));
            const auto d2 = sqr(x - center.x) + sqr(y - center.y) + sqr(z - center.z);
            return sqr(positive_part(1.0f - d2*(1.0f/(R*R))));
        };
        return 0.25f - (blob(1.1f, 0.7f, 0.9f, 0.0f, 0.44f) + blob(0.6f, 1.3f, 0.5f, 1.3f, 0.40f)
                     + blob(0.9f, 0.4f, 1.4f, 2.6f, 0.36f) + blob(1.7f, 0.9f, 0.3f, 3.9f, 0.32f)
                     + blob(0.3f, 1.6f, 1.1f, 5.2f, 0.34f));
    }

    /* a rounded box twisted around the y axis */
    template <typename V>
    __device__ auto twisted_box(const V& x, const V& y, const V& z, const float t) {
        const float twist = 2.2f*sinf(0.6f*t);
        const auto a = twist*y;
        const auto c = cos(a);
        const auto s = sin(a);
        const auto u = c*x - s*z;
        const auto w = s*x + c*z;
        const float r = 0.12f;
        const auto qx = abs(u) - 0.3f;
        const auto qy = abs(y) - 0.78f;
        const auto qz = abs(w) - 0.3f;
        const auto outside = length(positive_part(qx), positive_part(qy), positive_part(qz));
        const auto inside = min(max(qx, max(qy, qz)), 0.0f);
        return outside + inside - r;
    }

    /* Kummer quartic surface clipped by a ball, mu morphs it around its 16 nodes configuration (mu = 1.3) */
    template <typename V>
    __device__ auto kummer(const V& x0, const V& y0, const V& z0, const float t) {
        const float k = 2.4f;
        const auto x = k*x0;
        const auto y = k*z0;
        const auto z = k*y0;
        const float mu2 = sqr(1.3f + 0.2f*sinf(0.5f*t));
        const float lambda = (3.0f*mu2 - 1.0f) / (3.0f - mu2);
        const float s2 = 1.41421356f;
        const auto p = 1.0f - z - s2*x;
        const auto q = 1.0f - z + s2*x;
        const auto r = 1.0f + z + s2*y;
        const auto s = 1.0f + z - s2*y;
        const auto f = sqr(sqr(x) + sqr(y) + sqr(z) - mu2) - lambda*((p*q)*(r*s));
        const auto ball = sqr(x) + sqr(y) + sqr(z) - sqr(0.95f*k);
        return max(f, ball);
    }

    /* Barth sextic (65 nodes for w = 1) clipped by a ball, w breathes around 1 */
    template <typename V>
    __device__ auto barth(const V& x0, const V& y0, const V& z0, const float t) {
        const float k = 1.9f;
        const auto x = k*x0;
        const auto y = k*y0;
        const auto z = k*z0;
        const float phi = 1.61803399f;
        const float phi2 = phi*phi;
        const float w = 1.0f + 0.12f*sinf(0.8f*t);
        const auto xx = sqr(x);
        const auto yy = sqr(y);
        const auto zz = sqr(z);
        const auto f = 4.0f*((phi2*xx - yy)*(phi2*yy - zz))*(phi2*zz - xx)
                     - (1.0f + 2.0f*phi)*sqr(xx + yy + zz - w*w);
        const auto ball = xx + yy + zz - sqr(0.95f*k);
        return max(-f, ball);
    }

    /* Taubin heart surface (x^2 + 9/4 y^2 + z^2 - 1)^3 - z^3 (x^2 + 9/80 y^2) = 0, beating. The cube root
       of the equation has the same sign and a gradient that does not vanish on the surface where z = 0. */
    template <typename V>
    __device__ auto heart(const V& x0, const V& y0, const V& z0, const float t) {
        const float beat = 1.0f + 0.12f*powf(fabsf(sinf(2.5f*t)), 6.0f);
        const float k = 1.35f/beat;
        const auto x = k*x0;
        const auto y = k*z0;
        const auto z = k*y0 + 0.1f;
        return sqr(x) + 2.25f*sqr(y) + sqr(z) - 1.0f - z*cbrt(sqr(x) + 0.1125f*sqr(y));
    }

    /* the tetrahedral quartic r^4 + 8xyz - 10r^2 + c = 0, its handles open and close with c */
    template <typename V>
    __device__ auto tetrahedral(const V& x0, const V& y0, const V& z0, const float t) {
        const float k = 4.0f;
        const auto x = k*x0;
        const auto y = k*y0;
        const auto z = k*z0;
        const float c = 20.0f + 4.0f*sinf(0.6f*t);
        const auto r2 = sqr(x) + sqr(y) + sqr(z);
        return sqr(r2) + 8.0f*((x*y)*z) - 10.0f*r2 + c;
    }
} /* end of namespace shapes */


/* calls visitor(f) where f(x, y, z) is the level set function of the given shape */
template <typename Visitor>
__device__ void visit_shape(const int shape, const float t, Visitor&& visitor) {
    switch (shape) {
        case GYROID:
            visitor([t](const auto& x, const auto& y, const auto& z) { return shapes::gyroid(x, y, z, t); });
            break;
        case METABALLS:
            visitor([t](const auto& x, const auto& y, const auto& z) { return shapes::metaballs(x, y, z, t); });
            break;
        case TWISTED_BOX:
            visitor([t](const auto& x, const auto& y, const auto& z) { return shapes::twisted_box(x, y, z, t); });
            break;
        case KUMMER:
            visitor([t](const auto& x, const auto& y, const auto& z) { return shapes::kummer(x, y, z, t); });
            break;
        case BARTH:
            visitor([t](const auto& x, const auto& y, const auto& z) { return shapes::barth(x, y, z, t); });
            break;
        case HEART:
            visitor([t](const auto& x, const auto& y, const auto& z) { return shapes::heart(x, y, z, t); });
            break;
        case TETRAHEDRAL:
            visitor([t](const auto& x, const auto& y, const auto& z) { return shapes::tetrahedral(x, y, z, t); });
            break;
        default:
            break;
    }
}


__device__ float3 albedo(const int shape) {
    switch (shape) {
        case GYROID:      return make_float3(1.00f, 0.55f, 0.10f);
        case METABALLS:   return make_float3(0.90f, 0.15f, 0.45f);
        case TWISTED_BOX: return make_float3(0.05f, 0.65f, 0.60f);
        case KUMMER:      return make_float3(0.20f, 0.35f, 0.95f);
        case BARTH:       return make_float3(0.45f, 0.80f, 0.15f);
        case HEART:       return make_float3(0.85f, 0.05f, 0.08f);
        case TETRAHEDRAL: return make_float3(0.55f, 0.25f, 0.85f);
        default:          return make_float3(0.8f);
    }
}
