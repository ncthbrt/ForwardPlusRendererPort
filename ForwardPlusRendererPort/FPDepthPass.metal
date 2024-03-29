/*
Abstract:
Metal shaders for performing the depth pre-pass.
*/
#include <metal_stdlib>

using namespace metal;

// Include header shared between C code and .metal files.
#include "FPShaderTypes.h"

// Include header shared between all .metal files.
#include "FPShaderCommon.h"

struct VertexOutput
{
    float4 position [[position]];
};


// A depth pre-pass is necessary in forward plus rendering to produce
// minimum and maximum depth bounds for light culling.
vertex VertexOutput depth_pre_pass_vertex(Vertex in                         [[ stage_in ]],
                                          constant FPFrameData & frameData [[ buffer(FPBufferIndexFrameData) ]])
{
    // Make the position a float4 to perform 4x4 matrix math on it.
    VertexOutput out;
    float4 position = float4(in.position, 1.0);
    
    // Calculate the position in clip space.
    out.position = frameData.projectionMatrix * frameData.modelViewMatrix * position;
    return out;
}

fragment ColorData depth_pre_pass_fragment(VertexOutput in [[ stage_in ]])
{
    // Populate on-tile geometry buffer data.
    ColorData f;
    
    f.lighting = half4(in.position.z/20, 0, 0, 1);
    // Set the depth in clip space, which you use in `FPCulling` to perform per-tile light culling.
    f.depth = in.position.z;
    
    
    return f;
}
