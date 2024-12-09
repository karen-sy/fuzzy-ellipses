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

// void _RasterizeGaussiansCUDAJAX(
// 	cudaStream_t stream,
// 	void **buffers,
// 	const char *opaque, std::size_t opaque_len
// )
// {
// 	/* Old repo version */

// 	// init
// 	cudaStreamSynchronize(stream);
//     const RasterizeDescriptor &descriptor =
//         *UnpackDescriptor<RasterizeDescriptor>(opaque, opaque_len);

// 	// inputs.[bg, means3D, colors_precomp, opacities, scales, rotations,
//                     //   viewmatrix, projmatrix, campos,cov3D_precomp,
//                     // shs]
//     const float *background = reinterpret_cast<const float *> (buffers[0]);
//     const float *means3D = reinterpret_cast<const float *> (buffers[1]);
//     const float *colors = reinterpret_cast<const float *> (buffers[2]);
//     const float *opacity = reinterpret_cast<const float *> (buffers[3]);
//     const float *scales = reinterpret_cast<const float *> (buffers[4]);
//     const float *rotations = reinterpret_cast<const float *> (buffers[5]);
// 	float scale_modifier = 1.0;
//     const float *viewmatrix = reinterpret_cast<const float *> (buffers[6]);
//     const float *projmatrix = reinterpret_cast<const float *> (buffers[7]);
// 	const float tan_fovx = descriptor.tan_fovx;
// 	const float tan_fovy = descriptor.tan_fovy;

//     const float *campos = reinterpret_cast<const float *> (buffers[8]);
// 	const int degree = descriptor.degree;
// 	const bool prefiltered = false;
// 	const bool debug = false;

// 	const int P = descriptor.P;
// 	const int H = descriptor.image_height;
// 	const int W = descriptor.image_width;

// 	// outputs.
//     int *out_num_rendered = reinterpret_cast<int *> (buffers[9]);
//     float *out_color = reinterpret_cast<float *> (buffers[10]);
//     int *radii = reinterpret_cast<int *> (buffers[11]);
//     char *geomBuffer = reinterpret_cast<char *> (buffers[12]);
//     char *binningBuffer = reinterpret_cast<char *> (buffers[13]);
//     char *imgBuffer = reinterpret_cast<char *> (buffers[14]);

//     cudaMemset(geomBuffer, '\0', descriptor.geombuffer_sz*sizeof(char));
//     cudaMemset(binningBuffer, '\0', descriptor.binningbuffer_sz*sizeof(char));
//     cudaMemset(imgBuffer, '\0',  descriptor.imgbuffer_sz*sizeof(char));

// 	std::function<char*(size_t)> geomFunc = resizeFunctionalDummy(geomBuffer);
// 	std::function<char*(size_t)> binningFunc = resizeFunctionalDummy(binningBuffer);
// 	std::function<char*(size_t)> imgFunc = resizeFunctionalDummy(imgBuffer);
// 	cudaStreamSynchronize(stream);

// 	auto cov3D_precomp = nullptr;
// 	auto sh = nullptr;
// 	int rendered = 0;
// 	if(P != 0)
// 	{
// 		int M = 1;
// 		rendered = CudaRasterizer::Rasterizer::forwardJAX(
// 		stream,  // NEW
// 		geomFunc,
// 		binningFunc,
// 		imgFunc,
// 		P, degree, M,
// 		background,
// 		W, H,
// 		means3D,
// 		sh,
// 		colors,
// 		opacity, // wtf why was there a difference | .contiguous().data<float>() vs. .contiguous().data_ptr<float>()
// 		scales,
// 		scale_modifier,
// 		rotations,
// 		cov3D_precomp,
// 		viewmatrix,
// 		projmatrix,
// 		campos,
// 		tan_fovx,
// 		tan_fovy,
// 		prefiltered,
// 		out_color,
// 		radii,
// 		debug);
// 	}
// 	cudaStreamSynchronize(stream);
// 	cudaMemcpy(out_num_rendered, &rendered, sizeof(int), cudaMemcpyDefault);
// 	cudaStreamSynchronize(stream);

// }
