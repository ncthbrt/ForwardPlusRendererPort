/*
Abstract:
Header containing types and enumeration constants shared between Metal shaders and Swift source.
*/

#ifndef FPShaderTypes_h
#define FPShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
typedef metal::int32_t EnumBackingType;
#else
#import <Foundation/Foundation.h>
typedef NSInteger EnumBackingType;
#endif

#include <simd/simd.h>

// Buffer index values shared between shader and C code to ensure that Metal shader buffer inputs
//   match Metal API buffer set calls.
typedef NS_ENUM(EnumBackingType, FPBufferIndices)
{
    FPBufferIndexMeshPositions     = 0,
    FPBufferIndexMeshGenerics      = 1,
    FPBufferIndexFrameData         = 2,
    FPBufferIndexLightsData        = 3,
    FPBufferIndexLightsPosition    = 4
};

// Attribute index values shared between shader and C code to ensure that Metal shader vertex
//   attribute indices match Metal API vertex descriptor attribute indices.
typedef NS_ENUM(EnumBackingType, FPVertexAttributes)
{
    FPVertexAttributePosition  = 0,
    FPVertexAttributeTexcoord  = 1,
    FPVertexAttributeNormal    = 2,
    FPVertexAttributeTangent   = 3,
    FPVertexAttributeBitangent = 4
};

// Texture index values shared between shader and C code to ensure that Metal shader texture
//   indices match Metal API texture set calls.
typedef NS_ENUM(EnumBackingType, FPTextureIndices)
{
    FPTextureIndexBaseColor = 0,
    FPTextureIndexSpecular  = 1,
    FPTextureIndexNormal    = 2,
    
    FPNumTextureIndices
};

// Threadgroup space buffer indices.
typedef NS_ENUM(EnumBackingType, FPThreadgroupIndices)
{
    FPThreadgroupBufferIndexLightList  = 0,
    FPThreadgroupBufferIndexTileData  = 1,
};

typedef NS_ENUM(EnumBackingType, FPRenderTargetIndices)
{
    FPRenderTargetLighting  = 0,  //Required for the procedural blending.
    FPRenderTargetDepth = 1
};

// Structures shared between shader and C code to ensure the layout of uniform data accessed in
//    Metal shaders matches the layout of frame data set in C code.

// Per-light characteristics.
typedef struct
{
    vector_float3 lightColor;
    float lightRadius;
    float lightSpeed;
} FPPointLight;

// Data constant across all threads, vertices, and fragments.
typedef struct
{
    // Per-frame constants.
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 projectionMatrixInv;
    matrix_float4x4 viewMatrix;
    matrix_float4x4 viewMatrixInv;
    vector_float2 depthUnproject;
    vector_float3 screenToViewSpace;
    
    // Per-mesh constants.
    matrix_float4x4 modelViewMatrix;
    matrix_float3x3 normalMatrix;
    matrix_float4x4 modelMatrix;
    
    // Per-light properties.
    vector_float3 ambientLightColor;
    vector_float3 directionalLightDirection;
    vector_float3 directionalLightColor;
    uint framebufferWidth;
    uint framebufferHeight;
} FPFrameData;

// Simple vertex used to render the fairies.
typedef struct
{
    vector_float2 position;
} FPSimpleVertex;

#define FPNumSamples 1
#define FPNumLights 1024
#define FPMaxLightsPerTile 64
#define FPTileWidth 16
#define FPTileHeight 16

// Size of an on-tile structure containing information such as maximum tile depth, minimum tile
//   depth, and a list of lights in the tile.
#define FPTileDataSize 256

// Temporary buffer used for depth reduction.
// Buffer size needs to be at least tile width * tile height * 4.
#define FPThreadgroupBufferSize MAX(FPMaxLightsPerTile*sizeof(uint32_t), FPTileWidth*FPTileHeight*sizeof(uint32_t))


#endif /* FPShaderTypes_h */

