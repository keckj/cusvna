/* this file is part of example 05_levelset.py */
#include <optix.h>

#include "ext/python-optix/examples/cuda/vec_math.h"
#include "ext/python-optix/examples/cuda/helpers.h"

#include <cusvna/aa.cuh>
#include <cusvna/af1.cuh>

/* must match the Shape enum of 05_levelset.py */
enum Shape : int {
    FLOOR       = 0,
    GYROID      = 1,
    METABALLS   = 2,
    TWISTED_BOX = 3,
    KUMMER      = 4,
    BARTH       = 5,
    HEART       = 6,
    TETRAHEDRAL = 7,
};

/* must match the object dtype of 05_levelset.py */
struct Object {
    float3 center;
    float  scale;   /* the shape lives in [-scale, scale]^3 around center, in any rotation around y */
    float  spin;    /* angular velocity around the y axis */
    int    shape;
    float  phase;   /* animation time offset */
    float  pad;
};

struct HitGroupData {
    const Object* objects;
};

struct Params {
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    unsigned int           ray_marching_steps;
    float                  ray_marching_threshold;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
    unsigned int           rendering_mode;
    int                    subframe_index;
    float                  time;
    unsigned int           marching_method;
    unsigned int           samples_per_pixel;
    unsigned int           pad;
    float4*                accum_buffer;
};

enum RenderingMode : unsigned int {
    SHADED  = 0,
    NORMALS = 1,
    STEPS   = 2,
};

enum MarchingMethod : unsigned int {
    SIGN_TEST   = 0,
    ROOT_WINDOW = 1,
};

enum RayType : unsigned int {
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_SHADOW    = 1,
};

extern "C" {
__constant__ Params params;
}


/*----------------------------------------------------------------------------*/
/* Level sets                                                                 */
/*----------------------------------------------------------------------------*/

/* All the level sets are written once as templates, evaluated on affine forms to march the rays and on
   floats to refine the roots and compute the normals. Functions of affine forms are found by argument
   dependent lookup in namespace cusvna, their float counterparts are declared here. Shapes are given in
   local coordinates, they are negative inside and must fit in the cylinder x^2+z^2 <= 1, |y| <= 1. */
namespace shapes {
    __device__ inline float sqr(float x) { return x*x; }
    __device__ inline float abs(float x) { return fabsf(x); }
    __device__ inline float min(float x, float y) { return fminf(x, y); }
    __device__ inline float max(float x, float y) { return fmaxf(x, y); }
    __device__ inline float sin(float x) { return sinf(x); }
    __device__ inline float cos(float x) { return cosf(x); }
    __device__ inline float sqrt(float x) { return sqrtf(x); }
    __device__ inline float cbrt(float x) { return cbrtf(x); }
    __device__ inline float positive_part(float x) { return fmaxf(x, 0.0f); }

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


/* Rotation of an object around its vertical axis, maps world coordinates to local coordinates */
struct Frame {
    float3 center;
    float  inv_scale;
    float  c, s;

    __device__ Frame(const Object& object, const float t)
        : center(object.center), inv_scale(1.0f/object.scale) {
        sincosf(object.spin * t, &s, &c);
    }

    __device__ float3 to_local_vector(const float3& v) const {
        return inv_scale * make_float3(c*v.x - s*v.z, v.y, s*v.x + c*v.z);
    }

    __device__ float3 to_local_point(const float3& p) const {
        return to_local_vector(p - center);
    }

    __device__ float3 to_world_normal(const float3& n) const {
        return normalize(make_float3(c*n.x + s*n.z, n.y, -s*n.x + c*n.z));
    }
};


/* Finds the first root of f(o + t*d) for t in [t0, t1] up to the threshold delta. Each step evaluates f on a
   segment [t, t+s] of the ray with affine arithmetic, using a single noise symbol e for the ray parameter
   t = t + s/2 + e*s/2, and the step size adapts as in "Spelunking the Deep: Guaranteed Queries on General
   Neural Implicit Surfaces via Range Analysis" (Sharp & Jacobson 2022, procedure 2).

   SIGN_TEST only uses the range of f on the segment: the segment is skipped if f has a certain sign.

   ROOT_WINDOW also uses the affine form f = b + a*e + (other terms bounded by R) of f along the ray: a root
   is only possible where |b + a*e| <= R, which is a window [lo, hi] of e. Everything before the window is
   skipped and, since R shrinks quadratically with the length of the segment around a simple root, the
   window converges quickly to the root. */
template <typename F>
__device__ bool march(const F& f, const float3& o, const float3& d, const float t0, const float t1,
                      const float delta, float& t_hit, unsigned int& steps) {
#ifdef CUSVNA_AF1
    using AA = cusvna::AF1<float,1>;
#else
    using AA = cusvna::AA<float>;
#endif
    using Interval = cusvna::Interval<float>;
    using RaySymbol = cusvna::detail::id<0>;

    auto eval = [&](const float t) { return f(o.x + t*d.x, o.y + t*d.y, o.z + t*d.z); };
    const bool inside = (eval(t0) < 0.0f);

    float t = t0;
    float sigma = 0.125f*(t1 - t0);
    for (steps = 0; steps < params.ray_marching_steps && t < t1; ++steps) {
        const float s = fminf(sigma, t1 - t);
        const auto T = AA::from_interval<0>(Interval(t, t + s));
        const auto w = f(o.x + d.x*T, o.y + d.y*T, o.z + d.z*T);

        float lo = -1.0f, hi = 1.0f;
        if (params.marching_method == ROOT_WINDOW) {
            const float b = w.b();
            const float a = w.template symbol_magnitude<RaySymbol>();
            const float R = (w.radius() - fabsf(a)) * (1.0f + 1e-5f);
            if (a != 0.0f) {
                const float e0 = (-R - b) / a;
                const float e1 = (+R - b) / a;
                lo = fmaxf(lo, fminf(e0, e1));
                hi = fminf(hi, fmaxf(e0, e1));
            }
        }

        if (w.sign() != cusvna::Sign::UNKNOWN || !(lo <= hi)) {
            /* no root in [t, t+s] */
            t += s;
            sigma *= 1.5f;
            continue;
        }

        const float ta = t + 0.5f*s*(lo + 1.0f);
        const float tb = t + 0.5f*s*(hi + 1.0f);
        if (tb - ta > delta) {
            /* no root in [t, ta], look closer */
            t = ta;
            sigma = 0.5f*s;
        }
        else {
            /* root isolated in [ta, tb], refine it if the sign changes */
            if ((eval(tb) < 0.0f) != inside) {
                float a = ta, b = tb;
                for (int i = 0; i < 12; ++i) {
                    const float m = 0.5f*(a + b);
                    ((eval(m) < 0.0f) != inside ? b : a) = m;
                }
                t_hit = b;
                return true;
            }
            /* tangency or two close roots, move on by at least delta */
            t = fmaxf(tb, t + delta);
        }
    }
    return false;
}

/* entry and exit parameters of the ray in a box, returns false if the ray misses the box */
__device__ bool clip_to_box(const float3& o, const float3& d, const float3& lo, const float3& hi,
                            float& t0, float& t1) {
    const float3 inv_d = 1.0f / d;
    const float3 ta = (lo - o) * inv_d;
    const float3 tb = (hi - o) * inv_d;
    const float3 tmin = fminf(ta, tb);
    const float3 tmax = fmaxf(ta, tb);
    t0 = fmaxf(t0, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z)));
    t1 = fminf(t1, fminf(tmax.x, fminf(tmax.y, tmax.z)));
    return t0 < t1;
}

extern "C" __global__ void __intersection__levelset() {
    const HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const Object object = data->objects[optixGetPrimitiveIndex()];

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    if (object.shape == FLOOR) {
        const float t = (object.center.y - ray_orig.y) / ray_dir.y;
        if (t > optixGetRayTmin() && t < optixGetRayTmax())
            optixReportIntersection(t, 0, 0u);
        return;
    }

    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();
    const float3 extent = make_float3(object.scale);
    if (!clip_to_box(ray_orig, ray_dir, object.center - extent, object.center + extent, t0, t1))
        return;

    const float t = params.time + object.phase;
    const Frame frame(object, t);
    const float3 o = frame.to_local_point(ray_orig);
    const float3 d = frame.to_local_vector(ray_dir);

    /* shadow rays only need to know if they are blocked, use a coarser threshold */
    const bool shadow_ray = optixGetRayFlags() & OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    const float delta = params.ray_marching_threshold * (shadow_ray ? 4.0f : 1.0f);

    visit_shape(object.shape, t, [&](const auto& f) {
        float t_hit;
        unsigned int steps;
        if (march(f, o, d, t0, t1, delta, t_hit, steps))
            optixReportIntersection(t_hit, 0, steps);
    });
}


/*----------------------------------------------------------------------------*/
/* Shading                                                                    */
/*----------------------------------------------------------------------------*/

__device__ const float3 sun_direction = {-0.45f, 0.75f, -0.48f};

__device__ float3 sky(const float3& dir) {
    const float3 sun = normalize(sun_direction);
    const float h = fmaxf(dir.y, 0.0f);
    const float3 horizon = make_float3(0.85f, 0.72f, 0.62f);
    const float3 zenith  = make_float3(0.12f, 0.25f, 0.55f);
    float3 color = lerp(horizon, zenith, powf(h, 0.5f));
    const float sd = fmaxf(dot(dir, sun), 0.0f);
    color += make_float3(1.0f, 0.75f, 0.45f) * (0.25f*powf(sd, 8.0f) + 2.0f*powf(sd, 512.0f));
    if (dir.y < 0.0f)
        color = lerp(color, make_float3(0.35f, 0.3f, 0.28f), fminf(-4.0f*dir.y, 1.0f));
    return color;
}

/* 0 to 1 colormap (blue - cyan - green - yellow - red) */
__device__ float3 heatmap(const float x) {
    const float t = clamp(x, 0.0f, 1.0f);
    return clamp(make_float3(1.5f - fabsf(4.0f*t - 3.0f),
                             1.5f - fabsf(4.0f*t - 2.0f),
                             1.5f - fabsf(4.0f*t - 1.0f)), 0.0f, 1.0f);
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

template <typename F>
__device__ float3 gradient(const F& f, const float3& p) {
    const float h = 2e-3f;
    return make_float3(f(p.x + h, p.y, p.z) - f(p.x - h, p.y, p.z),
                       f(p.x, p.y + h, p.z) - f(p.x, p.y - h, p.z),
                       f(p.x, p.y, p.z + h) - f(p.x, p.y, p.z - h));
}


static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float2 jitter, float3& origin, float3& direction)
{
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + jitter.x) / static_cast<float>(dim.x),
            (static_cast<float>(idx.y) + jitter.y) / static_cast<float>(dim.y)
    ) - 1.0f;

    origin    = params.cam_eye;
    direction = normalize(d.x * U + d.y * V + W);
}

static __forceinline__ __device__ float3 trace_radiance(const float3& origin, const float3& direction) {
    unsigned int p0 = 0, p1 = 0, p2 = 0;
    optixTrace(params.handle, origin, direction,
               0.0f, 1e16f, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
               RAY_TYPE_RADIANCE, 1, RAY_TYPE_RADIANCE,
               p0, p1, p2);
    return make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
}

static __forceinline__ __device__ bool trace_occlusion(const float3& origin, const float3& direction) {
    unsigned int occluded = 1u;
    optixTrace(params.handle, origin, direction,
               0.0f, 1e16f, 0.0f, OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               RAY_TYPE_RADIANCE, 1, RAY_TYPE_SHADOW,
               occluded);
    return occluded != 0u;
}

static __forceinline__ __device__ void setPayload(const float3& p) {
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    /* supersampling: the subpixel offsets follow the R2 low discrepancy sequence, the same for every pixel,
       so that accumulating subframes converges without noise while the scene is still */
    const unsigned int spp = max(params.samples_per_pixel, 1u);
    float3 color = make_float3(0.0f);
    for (unsigned int s = 0; s < spp; s++) {
        const float k = static_cast<float>(params.subframe_index * spp + s);
        const float jx = 0.5f + 0.75487766624669276f * k;
        const float jy = 0.5f + 0.56984029099805326f * k;
        float3 ray_origin, ray_direction;
        computeRay(make_uint3(idx.x, idx.y, 0), dim, make_float2(jx - floorf(jx), jy - floorf(jy)), ray_origin, ray_direction);
        color += trace_radiance(ray_origin, ray_direction);
    }

    /* progressive accumulation, restarted by the host whenever the image changes */
    float4 accum = make_float4(color.x, color.y, color.z, static_cast<float>(spp));
    if (params.accum_buffer) {
        const unsigned int pixel = idx.y * params.image_width + idx.x;
        if (params.subframe_index > 0)
            accum += params.accum_buffer[pixel];
        params.accum_buffer[pixel] = accum;
    }
    color = make_float3(accum.x, accum.y, accum.z) / accum.w;

    /* vignetting */
    const float2 uv = make_float2(idx.x / float(dim.x) - 0.5f, idx.y / float(dim.y) - 0.5f);
    color *= 1.0f - 0.6f*dot(uv, uv);

    params.image[idx.y * params.image_width + idx.x] = make_color(color);
}


extern "C" __global__ void __miss__radiance()
{
    const float3 dir = optixGetWorldRayDirection();
    setPayload(params.rendering_mode == SHADED ? sky(dir) : make_float3(0.05f));
}

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(0u);
}


extern "C" __global__ void __closesthit__ch()
{
    const HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const Object object = data->objects[optixGetPrimitiveIndex()];
    const unsigned int steps = optixGetAttribute_0();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  t_hit    = optixGetRayTmax();
    const float3 p = ray_orig + t_hit*ray_dir;

    /* normal and material */
    float3 normal = make_float3(0.0f, 1.0f, 0.0f);
    float3 color;
    float  shininess = 64.0f;
    if (object.shape == FLOOR) {
        const float checker = (int(floorf(p.x)) + int(floorf(p.z))) & 1 ? 1.0f : 0.0f;
        color = lerp(make_float3(0.55f, 0.53f, 0.50f), make_float3(0.75f, 0.73f, 0.70f), checker);
        shininess = 16.0f;
    }
    else {
        const float t = params.time + object.phase;
        const Frame frame(object, t);
        const float3 q = frame.to_local_point(p);
        visit_shape(object.shape, t, [&](const auto& f) { normal = frame.to_world_normal(gradient(f, q)); });
        color = albedo(object.shape);
    }

    /* two sided surfaces: clipped algebraic surfaces show their back faces, tint them */
    if (dot(normal, ray_dir) > 0.0f) {
        normal = -normal;
        color = lerp(color, make_float3(1.0f, 0.95f, 0.85f), 0.6f);
    }

    if (params.rendering_mode == NORMALS) {
        setPayload(normal * 0.5f + 0.5f);
        return;
    }
    if (params.rendering_mode == STEPS) {
        setPayload(object.shape == FLOOR ? make_float3(0.1f) : heatmap(logf(1.0f + steps) / logf(1.0f + params.ray_marching_steps)));
        return;
    }

    /* sun light with hard shadows, hemispheric ambient, Blinn-Phong highlights and Fresnel rim */
    const float3 L = normalize(sun_direction);
    const float3 V = -ray_dir;
    const float3 H = normalize(L + V);
    const float ndotl = dot(normal, L);

    float shadow = 0.0f;
    if (ndotl > 0.0f)
        shadow = trace_occlusion(p + 1e-3f*normal, L) ? 0.0f : 1.0f;

    const float3 sun_color = make_float3(1.0f, 0.92f, 0.8f) * 2.2f;
    const float3 ambient = lerp(make_float3(0.25f, 0.2f, 0.18f), make_float3(0.3f, 0.4f, 0.6f), 0.5f + 0.5f*normal.y);
    const float fresnel = 0.04f + 0.96f*powf(1.0f - fmaxf(dot(normal, V), 0.0f), 5.0f);
    const float specular = shadow * powf(fmaxf(dot(normal, H), 0.0f), shininess) * (shininess + 8.0f) / 25.0f;

    float3 radiance = color * (ambient + sun_color * fmaxf(ndotl, 0.0f) * shadow);
    radiance += sun_color * specular * 0.25f;

    /* the objects reflect the sky, the floor is matte */
    if (object.shape != FLOOR)
        radiance += fresnel * 0.5f * sky(reflect(ray_dir, normal));

    /* distance fog */
    const float fog = 1.0f - expf(-0.0015f * t_hit * t_hit);
    setPayload(lerp(radiance, sky(ray_dir), fog));
}
