#include <metal_stdlib>
using namespace metal;

// 1. Fragment Shader for YCbCr to RGB conversion
// ARKit delivers high-resolution visual data in Bi-Planar YCbCr format.
// This executes in milliseconds natively on GPU, bypassing CPU freeze.

typedef struct {
    float4 position [[position]];
    float2 texCoord;
} ColorInOut;

// YCbCr to RGB Conversion Matrix (ITU-R BT.601)
constant float3x3 ycbcrToRGBTransform = float3x3(
    float3(1.0000, 1.0000, 1.0000),
    float3(0.0000, -0.3441, 1.7720),
    float3(1.4020, -0.7141, 0.0000)
);

fragment float4 ycbcrToRGBFragment(ColorInOut in [[stage_in]],
                                   texture2d<float, access::sample> capturedImageTextureY [[ texture(0) ]],
                                   texture2d<float, access::sample> capturedImageTextureCbCr [[ texture(1) ]]) {
    
    constexpr sampler colorSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);
    
    float y = capturedImageTextureY.sample(colorSampler, in.texCoord).r;
    float2 cbcr = capturedImageTextureCbCr.sample(colorSampler, in.texCoord).rg - float2(0.5, 0.5);
    
    float3 ycbcr = float3(y, cbcr.x, cbcr.y);
    float3 rgb = ycbcrToRGBTransform * ycbcr;
    
    return float4(rgb, 1.0);
}

// 2. Compute Shader for Depth Normalization
// Normalizes the unscaled absolute depth (meters) to [0, 1] range for CoreML inference inputs

kernel void normalizeDepth(texture2d<float, access::read> inTexture [[texture(0)]],
                           texture2d<float, access::write> outTexture [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }
    
    // Read raw depth in meters
    float depth = inTexture.read(gid).r;
    
    // Supposing typical skin-test distance is within 0.1 to 0.5 meters
    // We normalize this range to feed clearly into the Neural Engine
    float minDepth = 0.1;
    float maxDepth = 0.5;
    float normalized = clamp((depth - minDepth) / (maxDepth - minDepth), 0.0, 1.0);
    
    outTexture.write(float4(normalized, 0.0, 0.0, 1.0), gid);
}
