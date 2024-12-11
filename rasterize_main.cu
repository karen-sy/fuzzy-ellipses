/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer_impl.h" // #include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include "rasterize_main.h"


void RasterizeGaussiansCUDAJAX(
	cudaStream_t stream,
	void **buffers,
	const char *opaque, std::size_t opaque_len
)
{
	// init
	cudaStreamSynchronize(stream);
    const RasterizeDescriptor &descriptor =
        *UnpackDescriptor<RasterizeDescriptor>(opaque, opaque_len);
	const int P = descriptor.P;
	const float image_height = descriptor.image_height;
	const float image_width = descriptor.image_width;
	const float tan_fovx = descriptor.tan_fovx;
	const float tan_fovy = descriptor.tan_fovy;

	/*  inputs (excluding inputs from descriptor):
		[means3D, scales, rotations, weights,
		camera_rays, camera_rot, camera_trans,
		camera_view_matrix, camera_proj_matrix]

		outputs:
		[zs_final, est_alpha]
	*/
    // gaussian parameterization
	float* means = reinterpret_cast<float *> (buffers[0]);
	float* weights = reinterpret_cast<float *> (buffers[1]);
	float* scales = reinterpret_cast<float *> (buffers[2]);
	float* quats = reinterpret_cast<float *> (buffers[3]);

    // camera parameterization
	float* camera_rays = reinterpret_cast<float *> (buffers[4]);
	float* camera_rot = reinterpret_cast<float *> (buffers[5]);
	float* camera_trans = reinterpret_cast<float *> (buffers[6]);
	float* camera_view_matrix = reinterpret_cast<float *> (buffers[7]);
	float* camera_proj_matrix = reinterpret_cast<float *> (buffers[8]);

	// outputs.
	float *zs_final = reinterpret_cast<float *> (buffers[9]);
	float *est_alpha = reinterpret_cast<float *> (buffers[10]);

    /////////////////////////////
    // Launch function
    /////////////////////////////

    auto memory_pool = GpuMemoryPool();

    // reset function for allocated memory
    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(zs_final, 0, image_height * image_width * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            est_alpha, 0, image_height * image_width * sizeof(float)));
        memory_pool.reset();
    };

    // kernel launch function
    auto f = [&]() {
        CudaRasterizer::Rasterizer::forwardJAX(
			stream,
            image_height, image_width, P,
            means,
            scales,
            quats,
            weights,
            camera_rays,
            camera_rot,
            camera_trans,
            camera_view_matrix,
            camera_proj_matrix,
            tan_fovx,
            tan_fovy,

            zs_final,
            est_alpha,
            memory_pool
            );
    };
	// run
	if (P != 0) {
    	reset();
		cudaStreamSynchronize(stream);
    	f();   // try this in many loop, then unpack lambda.
	}
	cudaStreamSynchronize(stream);
}
