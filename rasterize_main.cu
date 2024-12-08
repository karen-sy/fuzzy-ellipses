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
#include "rasterize_points.h"


void RasterizeGaussiansCUDAJAX(
	cudaStream_t stream,
	void **buffers,
	const char *opaque, std::size_t opaque_len
)
{
	cudaStreamSynchronize(stream);
    const RasterizeDescriptor &descriptor =
        *UnpackDescriptor<RasterizeDescriptor>(opaque, opaque_len);

	// inputs.[bg, means3D, colors_precomp, opacities, scales, rotations,
                    //   viewmatrix, projmatrix, campos,cov3D_precomp,
                    // shs]
    const float *background = reinterpret_cast<const float *> (buffers[0]);
    const float *means3D = reinterpret_cast<const float *> (buffers[1]);
    const float *colors = reinterpret_cast<const float *> (buffers[2]);
    const float *opacity = reinterpret_cast<const float *> (buffers[3]);
    const float *scales = reinterpret_cast<const float *> (buffers[4]);
    const float *rotations = reinterpret_cast<const float *> (buffers[5]);
	float scale_modifier = 1.0;
    const float *viewmatrix = reinterpret_cast<const float *> (buffers[6]);
    const float *projmatrix = reinterpret_cast<const float *> (buffers[7]);
	const float tan_fovx = descriptor.tan_fovx;
	const float tan_fovy = descriptor.tan_fovy;

    const float *campos = reinterpret_cast<const float *> (buffers[8]);
	const int degree = descriptor.degree;
	const bool prefiltered = false;
	const bool debug = false;

	const int P = descriptor.P;
	const int H = descriptor.image_height;
	const int W = descriptor.image_width;

	// outputs.
    int *out_num_rendered = reinterpret_cast<int *> (buffers[9]);
    float *out_color = reinterpret_cast<float *> (buffers[10]);
    int *radii = reinterpret_cast<int *> (buffers[11]);
    char *geomBuffer = reinterpret_cast<char *> (buffers[12]);
    char *binningBuffer = reinterpret_cast<char *> (buffers[13]);
    char *imgBuffer = reinterpret_cast<char *> (buffers[14]);

    cudaMemset(geomBuffer, '\0', descriptor.geombuffer_sz*sizeof(char));
    cudaMemset(binningBuffer, '\0', descriptor.binningbuffer_sz*sizeof(char));
    cudaMemset(imgBuffer, '\0',  descriptor.imgbuffer_sz*sizeof(char));

	std::function<char*(size_t)> geomFunc = resizeFunctionalDummy(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctionalDummy(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctionalDummy(imgBuffer);
	cudaStreamSynchronize(stream);

	auto cov3D_precomp = nullptr;
	auto sh = nullptr;
	int rendered = 0;
	if(P != 0)
	{
		int M = 1;
		// rendered = CudaRasterizer::Rasterizer::forwardJAX(
		// stream,  // NEW
		// geomFunc,
		// binningFunc,
		// imgFunc,
		// P, degree, M,
		// background,
		// W, H,
		// means3D,
		// sh,
		// colors,
		// opacity, // wtf why was there a difference | .contiguous().data<float>() vs. .contiguous().data_ptr<float>()
		// scales,
		// scale_modifier,
		// rotations,
		// cov3D_precomp,
		// viewmatrix,
		// projmatrix,
		// campos,
		// tan_fovx,
		// tan_fovy,
		// prefiltered,
		// out_color,
		// radii,
		// debug);
        rendered = CudaRasterizer::Rasterizer::forwardJAX(
            stream,
            scene.height, scene.width, scene.n_gaussians(),
            means_gpu.data,
            scales_gpu.data,
            quats_gpu.data,
            weights_gpu.data,
            camera_rays_gpu.data,
            camera_rot_gpu.data,
            camera_trans_gpu.data,
            camera_rays_xfm_gpu.data,
            camera_view_matrix_gpu.data,
            camera_proj_matrix_gpu.data,
            scene.tan_fovx,
            scene.tan_fovy,
            // cov3d_gpu.data,
            radii_gpu.data,
            tiles_touched_gpu.data,

            zs_final_gpu.data,
            est_alpha_gpu.data,
            memory_pool
        );



	}
	cudaStreamSynchronize(stream);
	cudaMemcpy(out_num_rendered, &rendered, sizeof(int), cudaMemcpyDefault);
	cudaStreamSynchronize(stream);

}
