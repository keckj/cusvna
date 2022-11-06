/* this file is part of example 04_raytracing.py */
#include <optix.h>

#include "examples/cuda/vec_math.cuh"
#include "examples/cuda/helpers.cuh"

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
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};


struct MissData {
    float3 bg_color;
};

extern "C" __global__ void __intersection__levelset() {
    const levelset::LevelsetHitGroupData* hit_group_data = reinterpret_cast<levelset::LevelsetHitGroupData*>( optixGetSbtDataPointer() );

    const unsigned int primitive_index = optixGetPrimitiveIndex();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const float3 O      = ray_orig - hit_group_data->centers[primitive_index];

    const float  l      = 1.0f / length( ray_dir );
    const float3 D      = ray_dir * l;
    const float  radius = hit_group_data->radii[primitive_index];
    float b    = dot( O, D );
    float c    = dot( O, O ) - radius * radius;
    float disc = b * b - c;
    if( disc > 0.0f )
    {
        float sdisc        = sqrtf( disc );
        float root1        = ( -b - sdisc );
        float root11       = 0.0f;
        bool  check_second = true;

        const bool do_refine = fabsf( root1 ) > ( 10.0f * radius );

        if( do_refine )
        {
            // refine root1
            float3 O1 = O + root1 * D;
            b         = dot( O1, D );
            c         = dot( O1, O1 ) - radius * radius;
            disc      = b * b - c;

            if( disc > 0.0f )
            {
                sdisc  = sqrtf( disc );
                root11 = ( -b - sdisc );
            }
        }

        float  t;
        float3 normal;
        t = ( root1 + root11 ) * l;
        if( t > ray_tmin && t < ray_tmax )
        {
            normal = ( O + ( root1 + root11 ) * D ) / radius;
            if( optixReportIntersection( t, 0, float3_as_ints( normal ), __float_as_int( radius ) ) )
                check_second = false;
        }

        if( check_second )
        {
            float root2 = ( -b + sdisc ) + ( do_refine ? root1 : 0 );
            t           = root2 * l;
            normal      = ( O + root2 * D ) / radius;
            if( t > ray_tmin && t < ray_tmax )
                optixReportIntersection( t, 0, float3_as_ints( normal ), __float_as_int( radius ) );
        }
    }
}

extern "C" {
__constant__ Params params;
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
           1e16f,  // tmax
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
    const float3 shading_normal =
            make_float3(
                    __int_as_float( optixGetAttribute_0() ),
                    __int_as_float( optixGetAttribute_1() ),
                    __int_as_float( optixGetAttribute_2() )
            );
    setPayload( normalize( optixTransformNormalFromObjectToWorldSpace( shading_normal ) ) * 0.5f + 0.5f );
}
