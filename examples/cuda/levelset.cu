/* this file is part of example 04_raytracing.py */
#include <optix.h>

#include "examples/cuda/vec_math.cuh"
#include "examples/cuda/helpers.cuh"

#include <cusvna/aa.cuh>

#define float3_as_ints( u ) __float_as_int( u.x ), __float_as_int( u.y ), __float_as_int( u.z )

namespace levelset {
    const unsigned int NUM_ATTRIBUTE_VALUES = 4u;

    struct LevelsetHitGroupData {
        float3* centers;
        float* radii;
    };
}

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
};


struct MissData {
    float3 bg_color;
};

extern "C" {
__constant__ Params params;
}

__device__ float3 df(const float3& p, const float3& origin, const float radius) {
    const float3 dx = p - origin;
    return dx / norm3df(dx.x, dx.y, dx.z);
}

#ifdef SDF
__device__ float f(const float3& p, const float3& origin, const float radius) {
    const float3 dx = p - origin;
    return norm3df(dx.x, dx.y, dx.z) - radius;
}
#else
template <typename T>
__device__ T f(const T& x, const T& y, const T& z,
                   const float3& origin, const float radius) {
    const T dx(x-origin.x);
    const T dy(y-origin.y);
    const T dz(z-origin.z);
    return dx*dx + dy*dy + dz*dz - radius*radius;
}

__device__ float f(const float3& p, const float3& origin, const float radius) {
    return f<float>(p.x, p.y, p.z, origin, radius);
}
#endif

extern "C" __global__ void __intersection__levelset() {
    const levelset::LevelsetHitGroupData* hit_group_data = \
            reinterpret_cast<levelset::LevelsetHitGroupData*>( optixGetSbtDataPointer() );

    const unsigned int primitive_index = optixGetPrimitiveIndex();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const float3 origin = hit_group_data->centers[primitive_index];
    const float  radius = hit_group_data->radii[primitive_index];
   
#ifdef SDF
    /* Ray marching using signed distance field with SDF sphere tracing */
    float t = ray_tmin;
    for(int i = 0; i < params.ray_marching_steps && t <= ray_tmax; ++i) {
        const float3 p = ray_orig + t*ray_dir;
        const float d = f(p, origin, radius);
        if (abs(d) < params.ray_marching_threshold) {
            const float3 normal = df(p, origin, radius);
            optixReportIntersection(t, 0, float3_as_ints(normal), i);
            break;
        }
        t += d;
    }
#else
    /* Ray marching using general interval tracing */
    using T = float;
    using I = unsigned int;
    using G = cusvna::LocalThreadIndex<I>;
    constexpr I N = 8;
    using AA = cusvna::AA<T,I,G,N>;

    /* See Spelunking the Deep: Guaranteed Queries on General Neural Implicit Surfaces via Range Analysis, procedure 2 */
    const float np = 1.5f;
    const float nm = 0.5f;

    const float f0 = f(ray_orig, origin, radius);
    const float delta = params.ray_marching_threshold;

    float t = 0;
    float sigma = ray_tmax / 10.0;
    for(int i = 0; i < params.ray_marching_steps && t <= ray_tmax; ++i) {
        const float3 pc = ray_orig + (t+delta)*ray_dir;
        const float  fc = f(pc, origin, radius);
        if (((f0 < 0) ^ (fc < 0)) && t >= ray_tmin) {
            const float3 normal = df(pc, origin, radius);
            optixReportIntersection(t, 0, float3_as_ints(normal), i);
            break;
        }
        const float3 c = ray_orig + (t+sigma/2)*ray_dir;
        const float3 v = (sigma/2)*ray_dir;
        
        G g;
        const AA xv = AA(g, c.x).append_new_symbol(v.x);
        const AA yv = AA(g, c.y).append_new_symbol(v.y);
        const AA zv = AA(g, c.z).append_new_symbol(v.z);

        const AA w = f(xv, yv, zv, origin, radius);
        if (w.sign() == cusvna::Sign::UNKNOWN) {
            t += delta;
            sigma *= nm;
        }
        else {
            t += max(sigma, delta);
            sigma *= np;
        }
    }
#endif
}

static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
{
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
    ) - 1.0f;

    origin    = params.cam_eye;
    direction = normalize( d.x * U + d.y * V + W );
}


static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                prd
)
{
    unsigned int p0, p1, p2;
    p0 = __float_as_int( prd->x );
    p1 = __float_as_int( prd->y );
    p2 = __float_as_int( prd->z );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2 );
    prd->x = __int_as_float( p0 );
    prd->y = __int_as_float( p1 );
    prd->z = __int_as_float( p2 );
}


static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_int( p.x ) );
    optixSetPayload_1( __float_as_int( p.y ) );
    optixSetPayload_2( __float_as_int( p.z ) );
}


static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
            __int_as_float( optixGetPayload_0() ),
            __int_as_float( optixGetPayload_1() ),
            __int_as_float( optixGetPayload_2() )
    );
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    computeRay( make_uint3( idx.x, idx.y, 0 ), dim, ray_origin, ray_direction );


    float3       payload_rgb = make_float3( 0.5f, 0.5f, 0.5f );
    trace( params.handle,
           ray_origin,
           ray_direction,
           0.00f,  // tmin
           10.0f,  // tmax
           &payload_rgb );

    params.image[idx.y * params.image_width + idx.x] = make_color( payload_rgb );
}


extern "C" __global__ void __miss__ms()
{
    MissData* miss_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    setPayload(  miss_data->bg_color );
}


extern "C" __global__ void __closesthit__ch()
{
    const float3 normal =
            make_float3(
                    __int_as_float( optixGetAttribute_0() ),
                    __int_as_float( optixGetAttribute_1() ),
                    __int_as_float( optixGetAttribute_2() )
            );
    const int steps = optixGetAttribute_3();
    
    float3 color;
    if (params.rendering_mode == 0) {
        color = normalize( optixTransformNormalFromObjectToWorldSpace( normal ) ) * 0.5f + 0.5f;
    }
    else {
        color = normalize(make_float3(steps, 0, params.ray_marching_steps-1-steps));
    }
    setPayload(color);
}
