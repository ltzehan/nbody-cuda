//
//	Extends CUDA's built in float4 type
//	Using float4 over float3 for coalesced access
//

#pragma once

#include <cstdio>
#include <cuda_runtime.h>

// overload for built-in make_float4
__device__ __host__
inline float4 make_float4(float x, float y, float z) {
	float4 f = { x, y, z, 0 };
	return f;
}