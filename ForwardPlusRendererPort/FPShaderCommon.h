/*
Abstract:
Header containing types and enumeration constants shared between Metal shaders (but not Swift source).
*/
#ifndef FPShaderCommon_h
#define FPShaderCommon_h

// Per-tile data computed by the culling kernel.
struct TileData
{
    atomic_int numLights;
    float minDepth;
    float maxDepth;
};

// Per-vertex inputs populated by the vertex buffer laid out with the `MTLVertexDescriptor` Metal API.
struct Vertex
{
    float3 position [[attribute(FPVertexAttributePosition)]];
    float2 texCoord [[attribute(FPVertexAttributeTexcoord)]];
    half3 normal    [[attribute(FPVertexAttributeNormal)]];
    half3 tangent   [[attribute(FPVertexAttributeTangent)]];
    half3 bitangent [[attribute(FPVertexAttributeBitangent)]];
};

// Outputs for the color attachments.
struct ColorData
{
    half4 lighting [[color(FPRenderTargetLighting)]];
    float depth    [[color(FPRenderTargetDepth)]];
};

#endif /* FPShaderCommon_h */
