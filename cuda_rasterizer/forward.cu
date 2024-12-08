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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;



// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D, float* prc_out)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::mat3 invS = glm::mat3(1.0f);
	invS[0][0] = 1 / S[0][0];
	invS[1][1] = 1 / S[1][1];
	invS[2][2] = 1 / S[2][2];

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

    // Compute precision (root inverse Sigma = R.T @ S^-1)
    // Symmetric, so only need upper right
    glm::mat3 prc = glm::transpose(R) * invS;
    prc_out[0] = prc[0][0];
    prc_out[1] = prc[0][1];
    prc_out[2] = prc[0][2];
    prc_out[3] = prc[1][0];
    prc_out[4] = prc[1][1];
    prc_out[5] = prc[1][2];
    prc_out[6] = prc[2][0];
    prc_out[7] = prc[2][1];
    prc_out[8] = prc[2][2];

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}


// Perform initial steps for each Gaussian prior to rasterization.
__global__ void FORWARD::preprocessCUDA(int P,
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
	)
{
    const float scale_modifier = 1.0f;
    bool prefiltered = false;

	auto idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	const float* cov3D;
    computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6, prcs + idx * 9);
    cov3D = cov3Ds + idx * 6;

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// // Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	// float det_inv = 1.f / det;
	// float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.
	float mid = 0.5f * (cov.x + cov.z);  // avg of diagonal elements of covariance matrix (2x2 symmetric)
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(n_stds * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Store some useful helper data for the next steps.
	radii[idx] = my_radius;
	points_xy[idx] = point_image;
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


__global__ void FORWARD::gaussianRayRasterizeCUDA(
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
){
    // TODO rename variables eventually
    extern __shared__ float shmem[];

    // shared input pixel (used interchangeably as "ray") buffers
    float* r_shmem = shmem;
    float* t_shmem = r_shmem + N_PIXELS_PER_BLOCK*3;
    if ((threadIdx.x == 0) && (threadIdx.y == 0)){
        for (uint8_t vec_offset = 0; vec_offset < 3; vec_offset++){     // t \in R^3
            t_shmem[vec_offset] = camera_trans[vec_offset];
        }
    }

    // shared input gaussian buffers
    float* means_shmem = t_shmem + 3;
    float* prc_shmem = means_shmem + max_gaussians_in_tile*3;  // linearized (9,) for each gaussian
    float* w_shmem = prc_shmem + max_gaussians_in_tile*9;  // (1,) for each gaussian

    // shared output pixel buffers
    float est_alpha_exp_factors[N_PIXELS_PER_BLOCK];
    float wgts[N_PIXELS_PER_BLOCK];  // atomicAdd over gaussians, then atomicAdd the wget_shmem into global mem wget
    float zs_finals[N_PIXELS_PER_BLOCK];  // divide by jnp.where(wgt == 0, 1, wgt) after kernel

    // Identify current pixel
    int linear_tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int tile_offset_x = blockIdx.x * TILE_WIDTH;
    int tile_offset_y = blockIdx.y * TILE_HEIGHT;
    int block_pixel_id = threadIdx.y * TILE_WIDTH + threadIdx.x; // 0 to N_PIXELS_PER_BLOCK
    int global_pixel_id = (tile_offset_y + threadIdx.y) * img_width + (tile_offset_x + threadIdx.x);

    if ((block_pixel_id >= N_PIXELS_PER_BLOCK) || (global_pixel_id >= img_height * img_width)){
        return;
    }

    //// Load global -> shared for input buffers
    // pixels
    for (uint8_t vec_offset = 0; vec_offset < 3; vec_offset++){     // r, t \in R^3, passed in as [*r0, *t0, *r1, *t1, ...]
        r_shmem[block_pixel_id * 3 + vec_offset] = camera_rays[global_pixel_id * 3 + vec_offset];
    }

    //// Initialize output buffers (size (height*width, )) to 0
    est_alpha_exp_factors[block_pixel_id] = 0.0f;
    wgts[block_pixel_id] = 0.0f;
    zs_finals[block_pixel_id] = 0.0f;

    __syncthreads();

    // load value from shmem
    float* r = &r_shmem[block_pixel_id * 3];
    float* t = &t_shmem[0];
    float prev_batch_normalizer = -INFINITY;

    // identify current workload of circles to render in tile
    uint32_t start_gidx = tile_work_ranges[linear_tile_idx].x;
    uint32_t end_gidx = tile_work_ranges[linear_tile_idx].y;    // exclusive range
    uint32_t num_gaussians_in_tile = end_gidx - start_gidx;

    // #ifdef DEBUG_MODE
    // if ((threadIdx.x == 0) && (threadIdx.y == 0) && (num_gaussians_in_tile != 0)){
    //     printf("tile (%d, %d): start_gidx %d, end_gidx %d, num_gaussians_in_tile %d\n", tile_offset_y, tile_offset_x, start_gidx, end_gidx, num_gaussians_in_tile);
    // }
    // #endif

    // each batch processes N_GAUSSIANS_PER_BATCH gaussians; sequentially iterate over batches
    float exp_batch[N_GAUSSIANS_PER_BATCH] = {0.0f};  // (1,) for each gaussian
    float z_batch[N_GAUSSIANS_PER_BATCH] = {0.0f};  // (1,) for each gaussian
    float exp_normalizer_batch = prev_batch_normalizer;

    // Load gaussians across threads (divide work among threads)
    int num_gaussians_per_thread = (num_gaussians_in_tile + blockDim.x - 1) / blockDim.x;

    for (int gaussian_work_id = num_gaussians_per_thread * block_pixel_id; gaussian_work_id < num_gaussians_per_thread * (block_pixel_id+1); gaussian_work_id++){
        // uint32_t gaussian_work_id = block_pixel_id;
        if (gaussian_work_id >= num_gaussians_in_tile){
            break;
        }
        // get key for gaussian
        uint32_t gidx = start_gidx + gaussian_work_id;  // index into tile_gaussian_keys
        uint64_t key = tile_gaussian_keys[gidx];

        int gaussian_id = key & 0xFFFFFFFF;

        for (int vec_offset = 0; vec_offset < 3; vec_offset++){
            means_shmem[gaussian_work_id * 3 + vec_offset] = meansI_arr[gaussian_id * 3 + vec_offset];
        }

        for (int row_offset = 0; row_offset < 3; row_offset++){
            for (int col_offset = row_offset; col_offset < 3; col_offset++){
                int vec_offset = row_offset * 3 + col_offset;
                int symmetric_vec_offset = col_offset * 3 + row_offset;
                float prc_val = prc_arr[gaussian_id * 9 + vec_offset];
                prc_shmem[gaussian_work_id * 9 + vec_offset] = prc_val; // symmetric
                prc_shmem[gaussian_work_id * 9 + symmetric_vec_offset] = prc_val; // symmetric
            }
        }

        w_shmem[gaussian_work_id] = w_arr[gaussian_id];
    }
    __syncthreads();


    for (uint32_t gaussian_id_offset = 0; gaussian_id_offset < num_gaussians_in_tile; gaussian_id_offset+=N_GAUSSIANS_PER_BATCH){
        uint32_t N_GAUSSIAN_CURR_BATCH = min(N_GAUSSIANS_PER_BATCH, num_gaussians_in_tile - gaussian_id_offset);

        // Do computations over gaussians in batch
        for (uint32_t g_id_in_batch = 0; g_id_in_batch < N_GAUSSIAN_CURR_BATCH; g_id_in_batch++){
            uint32_t gaussian_work_id = gaussian_id_offset + g_id_in_batch;

            if (gaussian_work_id >= num_gaussians_in_tile){
                break;
            }

            float* meansI = &means_shmem[gaussian_work_id * 3];
            float* prc = &prc_shmem[gaussian_work_id * 9];
            float w = w_shmem[gaussian_work_id];

            // shift the mean to be relative to ray start
            float p[3];
            thrust_primitives::subtract_vec(meansI, t, p, 3);

            // compute \sigma^{-0.5} p, which is reused
            float projp[3];
            blas::matmul_331(prc, p, projp);

            // compute v^T \sigma^{-1} v
            float vsv_sqrt[3];  // prc @ r  (reuse)
            blas::matmul_331(prc, r, vsv_sqrt);

            float vsv;
            float _vsv[3];
            thrust_primitives::pow2_vec(vsv_sqrt, _vsv, 3);
            thrust_primitives::sum_vec(_vsv, &vsv, 3);

            // compute p^T \sigma^{-1} v
            float _psv[3];
            float psv;
            thrust_primitives::multiply_vec(projp, vsv_sqrt, _psv, 3);
            thrust_primitives::sum_vec(_psv, &psv, 3);

            // distance to get maximum likelihood point for this gaussian
            // scale here is based on r!
            // if r = [x, y, 1], then depth. if ||r|| = 1, then distance
            float z = psv / vsv;

            // get the intersection point
            float v[3];
            thrust_primitives::mul_scalar(r, z, v, 3);
            thrust_primitives::subtract_vec(v, p, v, 3);

            // compute intersection's unnormalized Gaussian log likelihood
            float _std[3];
            blas::matmul_331(prc, v, _std);
            thrust_primitives::pow2_vec(_std, _std, 3);

            // multiply by weight
            float std;
            thrust_primitives::sum_vec(_std, &std, 3);
            std = -0.5 * std + w;

            // alpha is based on distance from all gaussians. (Eq. 8)
            float est_alpha_exp = exp(std);
            est_alpha_exp_factors[block_pixel_id] -= est_alpha_exp ;

            // compute the algebraic weights in the paper (Eq. 7)
            exp_batch[g_id_in_batch] = -z * beta_2 + beta_3 * std;
            z_batch[g_id_in_batch] = thrust_primitives::nan_to_num(z);
            exp_normalizer_batch = fmax(exp_normalizer_batch, exp_batch[g_id_in_batch]);
        }

        // correct exponentiation for numerical stability
        thrust_primitives::sub_scalar(exp_batch, exp_normalizer_batch, exp_batch, N_GAUSSIAN_CURR_BATCH); // in-place subtraction

        // get mask for z > 0
        float stencil[N_GAUSSIANS_PER_BATCH];
        thrust_primitives::positive_mask_vec(z_batch, stencil, N_GAUSSIAN_CURR_BATCH);

        // calculate w_intersection (into exp_batch)
        thrust_primitives::clipped_exp(exp_batch, exp_batch, N_GAUSSIAN_CURR_BATCH);  // nan_to_num(exp(w_intersection)) in-place
        thrust_primitives::multiply_vec(exp_batch, stencil, exp_batch, N_GAUSSIAN_CURR_BATCH);  // w_intersection

        // update normalization factor for weights for the pixel (sum over gaussian intersections)
        float correction_factor = exp(prev_batch_normalizer - exp_normalizer_batch); // correct the previous exponentiation in current zs_final with new normalizer
        float exp_batch_sum;
        thrust_primitives::sum_vec(exp_batch, &exp_batch_sum, N_GAUSSIAN_CURR_BATCH);
        wgts[block_pixel_id] *= correction_factor;
        wgts[block_pixel_id] += exp_batch_sum;  // weighted sum of w_intersection

        // compute weighted (but unnormalized) z
        // zs_finals[block_pixel_id] /= correction_factor;
        float z_batch_sum;
        thrust_primitives::multiply_vec(z_batch, exp_batch, z_batch, N_GAUSSIAN_CURR_BATCH);  // z * w_intersection in-place
        thrust_primitives::sum_vec(z_batch, &z_batch_sum, N_GAUSSIAN_CURR_BATCH);
        zs_finals[block_pixel_id] *= correction_factor;
        zs_finals[block_pixel_id] += z_batch_sum;  // weighted sum of z

        prev_batch_normalizer = exp_normalizer_batch;
        __syncthreads();
    }

    zs_final[global_pixel_id] = zs_finals[block_pixel_id] / (wgts[block_pixel_id] + 1e-10f);
    est_alpha_final[global_pixel_id] = 1 - exp(est_alpha_exp_factors[block_pixel_id]);
}



// void FORWARD::renderJAX(
// 	cudaStream_t stream,  // new
// 	const dim3 grid, dim3 block,
// 	const uint2* ranges,
// 	const uint32_t* point_list,
// 	int W, int H,
// 	const float2* means2D,
// 	const float* colors,
// 	const float4* conic_opacity,
// 	float* final_T,
// 	uint32_t* n_contrib,
// 	const float* bg_color,
// 	float* out_color)
// {
// 	renderCUDA<NUM_CHANNELS> << <grid, block, 0, stream >> > (
// 		ranges,
// 		point_list,
// 		W, H,
// 		means2D,
// 		colors,
// 		conic_opacity,
// 		final_T,
// 		n_contrib,
// 		bg_color,
// 		out_color);
// }


// void FORWARD::preprocessJAX(
// 	cudaStream_t stream, // NEW

// 	int P, int D, int M,
// 	const float* means3D,
// 	const glm::vec3* scales,
// 	const float scale_modifier,
// 	const glm::vec4* rotations,
// 	const float* opacities,
// 	const float* shs,
// 	bool* clamped,
// 	const float* cov3D_precomp,
// 	const float* colors_precomp,
// 	const float* viewmatrix,
// 	const float* projmatrix,
// 	const glm::vec3* cam_pos,
// 	const int W, int H,
// 	const float focal_x, float focal_y,
// 	const float tan_fovx, float tan_fovy,
// 	int* radii,
// 	float2* means2D,
// 	float* depths,
// 	float* cov3Ds,
// 	float* rgb,
// 	float4* conic_opacity,
// 	const dim3 grid,
// 	uint32_t* tiles_touched,
// 	bool prefiltered)
// {
// 	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256, 0, stream >> > (
// 		P, D, M,
// 		means3D,
// 		scales,
// 		scale_modifier,
// 		rotations,
// 		opacities,
// 		shs,
// 		clamped,
// 		cov3D_precomp,
// 		colors_precomp,
// 		viewmatrix,
// 		projmatrix,
// 		cam_pos,
// 		W, H,
// 		tan_fovx, tan_fovy,
// 		focal_x, focal_y,
// 		radii,
// 		means2D,
// 		depths,
// 		cov3Ds,
// 		rgb,
// 		conic_opacity,
// 		grid,
// 		tiles_touched,
// 		prefiltered
// 		);
// }
