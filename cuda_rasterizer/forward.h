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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "thrust_primitives.h"  // namespace thrust_primitives
#include "blas_primitives.h"    // namespace blas

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
    __global__ void preprocessCUDA(int P,
        float n_stds,
        const float* orig_points,
        const glm::vec3* scales,
        const glm::vec4* rotations,
        float* prcs,
        const float* viewmatrix,
        const float* projmatrix,
        const int W, int H,
        const float tan_fovx, float tan_fovy,
        const float focal_x, float focal_y,
        int* radii,
        float2* points_xy,  // means2D
        float* cov3Ds,
        const dim3 grid,
        uint32_t* tiles_touched
    );

	// Main rasterization method.
    __global__ void gaussianRayRasterizeCUDA(
        int img_height,
        int img_width,
        int num_gaussians,
        uint64_t *tile_gaussian_keys,
        uint2* tile_work_ranges,
        uint32_t max_gaussians_in_tile,
        float* prc_arr, // TODO add triu to generalize
        float* w_arr,
        float* meansI_arr,
        float* camera_rays, // gpu array
        float* camera_trans,    // gpu array
        float const beta_2,
        float const beta_3,
        float* est_alpha_final,    // output
        float* zs_final    // output
    );
}


#endif



// /*
//  * Copyright (C) 2023, Inria
//  * GRAPHDECO research group, https://team.inria.fr/graphdeco
//  * All rights reserved.
//  *
//  * This software is free for non-commercial, research and evaluation use
//  * under the terms of the LICENSE.md file.
//  *
//  * For inquiries contact  george.drettakis@inria.fr
//  */

// #ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
// #define CUDA_RASTERIZER_FORWARD_H_INCLUDED

// #include <cuda.h>
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
// #define GLM_FORCE_CUDA
// #include <glm/glm.hpp>

// namespace FORWARD
// {

// 	// Perform initial steps for each Gaussian prior to rasterization.
// 	void preprocessJAX(
// 		cudaStream_t stream, // NEW

// 		int P, int D, int M,
// 		const float* orig_points,
// 		const glm::vec3* scales,
// 		const float scale_modifier,
// 		const glm::vec4* rotations,
// 		const float* opacities,
// 		const float* shs,
// 		bool* clamped,
// 		const float* cov3D_precomp,
// 		const float* colors_precomp,
// 		const float* viewmatrix,
// 		const float* projmatrix,
// 		const glm::vec3* cam_pos,
// 		const int W, int H,
// 		const float focal_x, float focal_y,
// 		const float tan_fovx, float tan_fovy,
// 		int* radii,
// 		float2* points_xy_image,
// 		float* depths,
// 		float* cov3Ds,
// 		float* colors,
// 		float4* conic_opacity,
// 		const dim3 grid,
// 		uint32_t* tiles_touched,
// 		bool prefiltered);

// 	// Main rasterization method.
// 	void renderJAX(
// 		cudaStream_t stream, // new

// 		const dim3 grid, dim3 block,
// 		const uint2* ranges,
// 		const uint32_t* point_list,
// 		int W, int H,
// 		const float2* points_xy_image,
// 		const float* features,
// 		const float4* conic_opacity,
// 		float* final_T,
// 		uint32_t* n_contrib,
// 		const float* bg_color,
// 		float* out_color);
// }


// #endif
