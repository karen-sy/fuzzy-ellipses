#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <tuple>
#include <string>
#include "bindings.h"


//-------------------------------------------------------------------------
// Descriptors.

struct RasterizeDescriptor {
    int image_height;
    int image_width;
    int P;
    float tan_fovx;
    float tan_fovy;
};



//-------------------------------------------------------------------------
// Prototypes.

void RasterizeGaussiansCUDAJAX(
	cudaStream_t stream,
	void **buffers,
	const char *opaque, std::size_t opaque_len
);
