#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <format>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <glm/glm.hpp>

#include "../cuda_rasterizer/auxiliary.h"
#include "../cuda_rasterizer/thrust_primitives.h"  // namespace thrust_primitives
#include "../cuda_rasterizer/blas_primitives.h"    // namespace blas

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

class GpuMemoryPool {
  public:
    GpuMemoryPool() = default;

    ~GpuMemoryPool();

    GpuMemoryPool(GpuMemoryPool const &) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool const &) = delete;
    GpuMemoryPool(GpuMemoryPool &&) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool &&) = delete;

    void *alloc(size_t size);
    void reset();

  private:
    std::vector<void *> allocations_;
    std::vector<size_t> capacities_;
    size_t next_idx_ = 0;
};


////////////////////////////////////////////////////////////////////////////////
// Projection helpers
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


////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation
// image partitioning

// gaussians partitionning

namespace fmb {

// forward.cu
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

__host__ float getNumStds(int num_gaussians){
    // get the number of standard deviations to use for the ellipse
    // controls the "blur" effect (includes more gaussians and slows computation) (
    // (default 3.0f in gsplat; around >5.0f seems to replicate FMB results exactly)
    // TODO can calculate a better heuristic based on total mem available / avg per pixel amortized

    if (num_gaussians <= 1000){
        return 1.0f;
    }
    else if (num_gaussians <= 5000){
        return 0.75f;
    }
    else if (num_gaussians <= 100'000){
        return 0.50f;
    }
    else{
        return 0.10f;  // cannot support more than 100k gaussians
    }
}

// forward.cu
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

// rasterizer_impl.cu
__global__ void getGaussianTileKeys(
	int P,
	const float2* points_xy,
	// const float* depths,
	const uint32_t* psum_touched_tiles,
	uint64_t* tile_gaussian_keys,
	uint32_t* tile_gaussian_counts,
	int* radii,
	dim3 grid)
{
	uint32_t gaussian_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (gaussian_idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[gaussian_idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (gaussian_idx == 0) ? 0 : psum_touched_tiles[gaussian_idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[gaussian_idx], radii[gaussian_idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a
		// key/value pair. The key is |  tile ID  |      gaussian_id   |,
		// and the value is the ID of the Gaussian. Sorting the values
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile, and then by the ID.
        // note that we do not need to sort by depth as in the gsplat case
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
                atomicAdd(&tile_gaussian_counts[key], 1);
				key <<= 32;
				key |= gaussian_idx;
				tile_gaussian_keys[off] = key;
				off++;
			}
		}
	}
}

// rasterizer_impl.cu
__global__ void getPerTileRanges(int n_tile_gaussian,
                                    uint64_t *tile_gaussian_keys,
                                    uint2 *ranges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tile_gaussian) return;

    uint64_t key = tile_gaussian_keys[idx];
    uint32_t tile_idx = key >> 32;

    if (idx == 0) {
        ranges[tile_idx].x = 0;
    }
    else {
        uint32_t prev_tile_idx = tile_gaussian_keys[idx-1] >> 32;
        // printf("idx %d | tile_idx: %d, prev_tile_idx: %d\n", idx, tile_idx, prev_tile_idx);

        if (tile_idx != prev_tile_idx) {
            // printf("idx %d | tile_idx: %d, prev_tile_idx: %d\n", idx, tile_idx, prev_tile_idx);
            ranges[prev_tile_idx].y = idx;
            ranges[tile_idx].x = idx;
        }
    }
    if (idx == n_tile_gaussian - 1) {
        ranges[tile_idx].y = n_tile_gaussian;
    }
}

// rasterizer_impl.cu (as forwardJAX)
void renderLaunchFunction(
                    int img_height, int img_width, int num_gaussians,
                    float* means3d,  // (N, 3)
                    float const* g_scales, // (N, 3) -- gsplat covariance parameterization
                    float const* g_rots, // (N, 4)  -- gsplat covariance parameterization
                    float* weights_log, // (N,)
                    float* _camera_rays,  // (H*W, 3) before cam pose transformation
                    float* rot, // (3, 3) // supply in rotation form not quaternion
                    float* trans, // (3, )
                    float* viewmatrix, float* projmatrix,
                    float tan_fovx, float tan_fovy,
                    float* zs_final,    // final gpu output
                    float* est_alpha,   // final gpu output
                    GpuMemoryPool &memory_pool
                    )
{
    int total_num_pixels = img_height * img_width;
    uint32_t total_num_tiles = ((img_width + TILE_WIDTH - 1) / TILE_WIDTH) * ((img_height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    dim3 num_threads_render(TILE_WIDTH, TILE_HEIGHT); // Do not change this without thought
    dim3 num_blocks_render((img_width + TILE_WIDTH - 1)/TILE_WIDTH, (img_height + TILE_HEIGHT - 1)/TILE_HEIGHT);

    float beta_2 = 21.4;
    float beta_3 = 2.66;
    float* camera_rays = reinterpret_cast<float *>(memory_pool.alloc(total_num_pixels * 3 * sizeof(float)));
    blas::matmul(total_num_pixels, 3, 3, _camera_rays, rot, camera_rays);  // _camera_rays @ rot

    ////////////////////////////////////////////////////
    /// Cull workloads
    ////////////////////////////////////////////////////
	const float focal_y = img_height / (2.0f * tan_fovy);
	const float focal_x = img_width / (2.0f * tan_fovx);

    dim3 num_threads_gaussian(256);  // align x, y order
    dim3 num_blocks_gaussian((num_gaussians + num_threads_gaussian.x - 1) / num_threads_gaussian.x);

    #ifdef DEBUG_MODE
    printf("Launching preprocessCUDA with %u blocks per grid\n", num_blocks_gaussian);
    #endif

    // float* prc_full_debug = reinterpret_cast<float *>(memory_pool.alloc(num_gaussians * 9 * sizeof(float)));

    int* radii = reinterpret_cast<int *>(memory_pool.alloc(num_gaussians * sizeof(int)));
    float2* means2d = reinterpret_cast<float2 *>(memory_pool.alloc(num_gaussians * sizeof(float2)));
    float* cov3ds = reinterpret_cast<float *>(memory_pool.alloc(num_gaussians * 6 * sizeof(float)));
    float* prec_full = reinterpret_cast<float *>(memory_pool.alloc(num_gaussians * 9 * sizeof(float)));
    uint32_t* tiles_touched = reinterpret_cast<uint32_t *>(memory_pool.alloc(num_gaussians * sizeof(uint32_t)));

    float n_std = getNumStds(num_gaussians);
    preprocessCUDA<<<num_blocks_gaussian, num_threads_gaussian>>>(
        num_gaussians,
        n_std,
        means3d,
        (glm::vec3*)g_scales,
        (glm::vec4*)g_rots,
        prec_full, // prc_full_debug,
        viewmatrix,
        projmatrix,
        img_width, img_height,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        radii,
        means2d,
        cov3ds,
        num_blocks_render,  // must be a grid3 with x and y
        tiles_touched
    );

    uint32_t *psum_touched_tiles = tiles_touched; //reinterpret_cast<uint32_t *>(memory_pool.alloc(num_gaussians * sizeof(int)));
    thrust::inclusive_scan(thrust::device, tiles_touched, tiles_touched + num_gaussians, psum_touched_tiles);  // [2,5,6,8,9]

    // Retrieve total number of gaussian instances to launch and resize aux buffers
    uint32_t n_tile_gaussians;
    cudaMemcpy(&n_tile_gaussians, &psum_touched_tiles[num_gaussians - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);

    #ifdef VERBOSE
        printf("total number of tile-gaussian pairs (input %d): %d\n", num_gaussians, n_tile_gaussians);
    #endif
    #ifdef DEBUG_MODE
    for (int i = 0; i < num_gaussians; i+=20){
        printf("gaussian %d-%d: ", i, i+19);
        for (int j = 0; j < 20; j++){
            if (i+j >= num_gaussians){
                break;
            }
            printf("%d, ", tiles_touched_host[i+j]);
        }
        printf("\n");
    }
    #endif

	// For each tile-gaussian instance, encode into [ tile_id | gaussian_id ] key
    // ex) [2, 5, 6, 8, 9] -> [t0_0, t1_0, t2_1, t3_1, t4_1, t5_2, t6_3, t7_3, t7_4]
    // Where t[0-7] are specific, not necessarily unique tile indices for the tile-gaussian pair.
    uint64_t *tile_gaussian_keys = reinterpret_cast<uint64_t *>(memory_pool.alloc(n_tile_gaussians * sizeof(uint64_t)));
    uint32_t *tile_gaussian_counts = reinterpret_cast<uint32_t *>(memory_pool.alloc(total_num_tiles * sizeof(uint32_t)));
    getGaussianTileKeys<<<num_blocks_gaussian, num_threads_gaussian>>>(num_gaussians,
                                                    means2d,
                                                    // depths,
                                                    psum_touched_tiles,
                                                    tile_gaussian_keys,
                                                    tile_gaussian_counts,
                                                    radii,
                                                    num_blocks_render
                                                    );

    // Get maximum number of gaussians in tile for shared memory allocation in rasterize
    uint32_t max_gaussians_in_tile = thrust::reduce(thrust::device, tile_gaussian_counts,
                                                    tile_gaussian_counts + total_num_tiles,
                                                    0, thrust::maximum<uint32_t>());

    // Sort tile-gaussian keys; result will be ordered by tile id, then by gaussian id
    thrust::sort(thrust::device, tile_gaussian_keys, tile_gaussian_keys + n_tile_gaussians); // in-place sort

    // Identify start and end indices of per-tile workloads in sorted tile-gaussian key list
    int num_threads_total = 256;  // TODO calibrate
    int num_blocks_total = (n_tile_gaussians + num_threads_total - 1) / num_threads_total;
    #ifdef VERBOSE
    printf("total_num_pixels: %d, total_num_tiles: %d\n", total_num_pixels, total_num_tiles);
    printf("total number of tile-gaussian pairs: %d\n\n\n", n_tile_gaussians);
    printf("Launching getPerTileRanges with %u blocks per grid\n", num_blocks_total);
    printf("Launching getPerTileRanges with %u threads per block\n", num_threads_total);
    #endif
    uint2 *tile_work_ranges = reinterpret_cast<uint2 *>(memory_pool.alloc(total_num_tiles * sizeof(uint2)));
    getPerTileRanges<<<num_blocks_total, num_threads_total>>>(n_tile_gaussians,
                                                                tile_gaussian_keys,
                                                                tile_work_ranges
                                                                );

    // ////////////////////////////////////////////////////
    // /// Render workloads
    // ////////////////////////////////////////////////////
    #ifdef DEBUG_MODE
    printf("Launching gaussianRayRasterizeCUDA with (%u,%u) blocks per grid\n", num_blocks_render.x, num_blocks_render.y);
    printf("Launching gaussianRayRasterizeCUDA with (%u,%u)=%u threads per block\n", num_threads_render.x, num_threads_render.y, num_threads_render.x*num_threads_render.y);
    #endif


    // launch gaussian_ray kernel to fill in needed values for final calculations
    cudaEvent_t start_record, stop_record;
    cudaEventCreate(&start_record); cudaEventCreate(&stop_record);
    cudaEventRecord(start_record);

    uint32_t shmem_size = sizeof(float) * (3 * N_PIXELS_PER_BLOCK + 3 + (3 + 9 + 1) * max_gaussians_in_tile);
    cudaFuncSetAttribute(gaussianRayRasterizeCUDA, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    gaussianRayRasterizeCUDA<<<num_blocks_render, num_threads_render, shmem_size>>>(
        img_height, img_width, num_gaussians,
        tile_gaussian_keys,
        tile_work_ranges,
        max_gaussians_in_tile,
        prec_full,
        weights_log,
        means3d,
        camera_rays,
        trans,
        beta_2,
        beta_3,
        est_alpha,
        zs_final
    );
    cudaEventRecord(stop_record); cudaEventSynchronize(stop_record);
    float time_record = 0;
    cudaEventElapsedTime(&time_record, start_record, stop_record);

    #ifdef VERBOSE
        printf("Time elapsed for gaussianRayRasterizeCUDA: %f ms\n", time_record);
    #endif
}
} // namespace fmb

////////////////////////////////////////////////////////////////////////////////
///          TESTING HARNESS                                                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);  // 32-bit float
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

std::vector<int> read_int_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<int32_t> data(size);    // 32-bit int
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(int32_t));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

GpuMemoryPool::~GpuMemoryPool() {
    for (auto ptr : allocations_) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void *GpuMemoryPool::alloc(size_t size) {
    if (next_idx_ < allocations_.size()) {
        auto idx = next_idx_++;
        if (size > capacities_.at(idx)) {
            CUDA_CHECK(cudaFree(allocations_.at(idx)));
            CUDA_CHECK(cudaMalloc(&allocations_.at(idx), size));
            CUDA_CHECK(cudaMemset(allocations_.at(idx), 0, size));
            capacities_.at(idx) = size;
        }
        return allocations_.at(idx);
    } else {
        void *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        CUDA_CHECK(cudaMemset(ptr, 0, size));
        allocations_.push_back(ptr);
        capacities_.push_back(size);
        next_idx_++;
        return ptr;
    }
}

void GpuMemoryPool::reset() {
    next_idx_ = 0;
    for (int32_t i = 0; i < allocations_.size(); i++) {
        CUDA_CHECK(cudaMemset(allocations_.at(i), 0, capacities_.at(i)));
    }
}

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Scene {
    int32_t width;
    int32_t height;
    std::vector<float> means;  // (3N, ) contiguous
    std::vector<float> quats;    // (4N, ) contiguous
    std::vector<float> scales;  // (3N, ) contiguous
    std::vector<float> weights;  // (N, )
    std::vector<float> camera_rays;  // (H*W*3, ) contiguous
    std::vector<float> camera_rot; // (3*3, )
    std::vector<float> camera_trans; // (3, )

    std::vector<float> _camera_rays_xfm;
    float tan_fovx;
    float tan_fovy;
    // float fx;
    // float fy;
    // float cx;
    // float cy;
    std::vector<float> camera_view_matrix;
    std::vector<float> camera_proj_matrix;

    int32_t n_pixels() const { return width * height; }
    int32_t n_gaussians() const { return weights.size(); }
};

struct Image {
    int32_t width;
    int32_t height;
    std::vector<float> zs;  // (width, height)
    std::vector<float> alphas; // (width, height)
};

float max_abs_diff(Image const &a, Image const &b) {
    if (a.width != b.width || a.height != b.height ||
        a.zs.size() != b.zs.size() || a.alphas.size() != b.alphas.size()) {
        return std::numeric_limits<float>::infinity();
    }

    float max_diff = 0.0f;
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        float diff_z = std::abs(a.zs.at(idx) - b.zs.at(idx));
        float diff_alpha = std::abs(a.alphas.at(idx) - b.alphas.at(idx));
        max_diff = std::max(max_diff, diff_z);
        max_diff = std::max(max_diff, diff_alpha);
    }
    return max_diff;
}

struct Results {
    bool correct;
    float max_abs_diff;
    Image image_expected;
    Image image_actual;
    double time_ms;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename T> struct GpuBuf {
    T *data;

    explicit GpuBuf(size_t n) { CUDA_CHECK(cudaMalloc(&data, n * sizeof(T))); }

    explicit GpuBuf(std::vector<T> const &host_data) {
        CUDA_CHECK(cudaMalloc(&data, host_data.size() * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(
            data,
            host_data.data(),
            host_data.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    ~GpuBuf() { CUDA_CHECK(cudaFree(data)); }
};

// rasterize_main.cu  (as RasterizeGaussiansCUDAJAX)
Results run_config(Mode mode, Scene const &scene) {
    auto img_expected = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f)
        };

    /////////////////////////////
    // Parse inputs
    /////////////////////////////

    std::string data_dir = "../data/" + std::to_string(scene.width) + "_" + std::to_string(scene.height) + "_" + std::to_string(scene.n_gaussians());
    img_expected.zs = read_data(data_dir + "/zs.bin", scene.n_pixels());
    img_expected.alphas = read_data(data_dir + "/alphas.bin", scene.n_pixels());

    // gaussian parameterization
    auto means_gpu = GpuBuf<float>(scene.means);
    auto weights_gpu = GpuBuf<float>(scene.weights);
    auto scales_gpu = GpuBuf<float>(scene.scales);
    auto quats_gpu = GpuBuf<float>(scene.quats);

    // camera parameterization
    auto camera_rays_gpu = GpuBuf<float>(scene.camera_rays);
    auto camera_rot_gpu = GpuBuf<float>(scene.camera_rot);
    auto camera_trans_gpu = GpuBuf<float>(scene.camera_trans);
    auto camera_view_matrix_gpu = GpuBuf<float>(scene.camera_view_matrix);
    auto camera_proj_matrix_gpu = GpuBuf<float>(scene.camera_proj_matrix);

    // need to do cudamemset for these below
    auto zs_final_gpu = GpuBuf<float>(scene.height * scene.width);
    auto est_alpha_gpu = GpuBuf<float>(scene.height * scene.width);

    /////////////////////////////
    // Launch function
    /////////////////////////////

    auto memory_pool = GpuMemoryPool();

    // reset function for allocated memory
    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(zs_final_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            est_alpha_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        memory_pool.reset();
    };

    #ifdef DEBUG_MODE
    printf("scene.height = %d\n", scene.height);
    printf("scene.width = %d\n", scene.width);
    printf("scene.n_gaussians = %d\n", scene.n_gaussians());
    printf("scene.means[0:3] = %f %f %f\n", scene.means[0], scene.means[1], scene.means[2]);
    printf("scene.scales[0:3] = %f %f %f\n", scene.scales[0], scene.scales[1], scene.scales[2]);
    printf("scene.quats[0:3] = %f %f %f\n", scene.quats[0], scene.quats[1], scene.quats[2]);
    printf("scene.weights[0:3] = %f %f %f\n", scene.weights[0], scene.weights[1], scene.weights[2]);
    printf("scene.camera_rays[0:3] = %f %f %f\n", scene.camera_rays[0], scene.camera_rays[1], scene.camera_rays[2]);
    printf("scene.camera_rot[0:3] = %f %f %f\n", scene.camera_rot[0], scene.camera_rot[1], scene.camera_rot[2]);
    printf("scene.camera_trans[0:3] = %f %f %f\n", scene.camera_trans[0], scene.camera_trans[1], scene.camera_trans[2]);
    printf("scene.tan_fovx = %f\n", scene.tan_fovx);
    printf("scene.tan_fovy = %f\n", scene.tan_fovy);
    printf("scene.camera_view_matrix[0:9] = %f %f %f %f %f %f %f %f %f\n", scene.camera_view_matrix[0], scene.camera_view_matrix[1], scene.camera_view_matrix[2], scene.camera_view_matrix[3], scene.camera_view_matrix[4], scene.camera_view_matrix[5], scene.camera_view_matrix[6], scene.camera_view_matrix[7], scene.camera_view_matrix[8]);
    printf("scene.camera_proj_matrix[0:9] = %f %f %f %f %f %f %f %f %f\n", scene.camera_proj_matrix[0], scene.camera_proj_matrix[1], scene.camera_proj_matrix[2], scene.camera_proj_matrix[3], scene.camera_proj_matrix[4], scene.camera_proj_matrix[5], scene.camera_proj_matrix[6], scene.camera_proj_matrix[7], scene.camera_proj_matrix[8]);
    #endif

    // kernel launch function
    auto f = [&]() {
        fmb::renderLaunchFunction(
            scene.height, scene.width, scene.n_gaussians(),
            means_gpu.data,
            scales_gpu.data,
            quats_gpu.data,
            weights_gpu.data,
            camera_rays_gpu.data,
            camera_rot_gpu.data,
            camera_trans_gpu.data,
            camera_view_matrix_gpu.data,
            camera_proj_matrix_gpu.data,
            scene.tan_fovx,
            scene.tan_fovy,

            zs_final_gpu.data,
            est_alpha_gpu.data,
            memory_pool
            );
    };
    reset();
    f();


    /////////////////////////////
    // Parse outputs and return
    /////////////////////////////

    // copy back kernel results to host img_actual
    auto img_actual = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f), // z
        std::vector<float>(scene.height * scene.width, 0.0f) // alpha
    };

    CUDA_CHECK(cudaMemcpy(
        img_actual.zs.data(),
        zs_final_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.alphas.data(),
        est_alpha_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));

    float max_diff = max_abs_diff(img_expected, img_actual);

    float thresh = 10.0f;
    // set it very high, since after culling, results are not expected to be
    // numerically identical to all-ray-intersection evaluations
    if (max_diff > thresh) {
        return Results{
            false,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    if (mode == Mode::TEST) {
        return Results{
            true,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    double time_ms = benchmark_ms(50.0, reset, f);  // shorten from 1000.0

    return Results{
        true,
        max_diff,
        std::move(img_expected),
        std::move(img_actual),
        time_ms,
    };
}

Scene gen_ycb_box(int _width, int _height, int _n_gaussians) {
    /*
        YCB box scene with an input # of gaussians
    */
    auto scene = Scene{};
    std::string data_dir = "../data/" + std::to_string(_width) + "_" + std::to_string(_height) + "_" + std::to_string(_n_gaussians);


    // gaussians
    auto w_h_g = read_int_data(data_dir + "/width_height_gaussians.bin", 3);
    int32_t width = w_h_g[0];
    int32_t height = w_h_g[1];
    int32_t n_gaussians = w_h_g[2];  // = n_gaussians_dir
    scene.width = width;
    scene.height = height;
    scene.means = read_data(data_dir + "/means.bin", 3 * n_gaussians);
    scene.quats = read_data(data_dir + "/quats.bin", 4 * n_gaussians);
    scene.scales = read_data(data_dir + "/scales.bin", 3 * n_gaussians);
    scene.weights = read_data(data_dir + "/weights.bin", n_gaussians);

    // rays
    scene.camera_rays = read_data(data_dir + "/camera_rays.bin", 3 * width * height);  // TODO
    scene.camera_rot = read_data(data_dir + "/camera_rot.bin", 9);
    scene.camera_trans = read_data(data_dir + "/camera_trans.bin", 3);

    // intrinsics
    auto tan_fov = read_data(data_dir + "/tan_fovs.bin", 2);
    // auto intrinsics = read_data(data_dir + "/intrinsics.bin", 4);
    scene.tan_fovx = tan_fov[0];
    scene.tan_fovy = tan_fov[1];
    // scene.fx = intrinsics[0];
    // scene.fy = intrinsics[1];
    // scene.cx = intrinsics[2];
    // scene.cy = intrinsics[3];

    scene.camera_view_matrix = read_data(data_dir + "/camera_view_matrix.bin", 16);
    scene.camera_proj_matrix = read_data(data_dir + "/camera_proj_matrix.bin", 16);

    return scene;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void write_bmp(
    std::string const &fname,
    uint32_t width,
    uint32_t height,
    const std::vector<uint8_t> &pixels) {
    BMPHeader header;
    header.width = width;
    header.height = height;

    uint32_t rowSize = (width * 3 + 3) & (~3); // Align to 4 bytes
    header.imageSize = rowSize * height;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));

    // Write pixel data with padding
    std::vector<uint8_t> padding(rowSize - width * 3, 0);
    for (int32_t idx_y = height - 1; idx_y >= 0;
         --idx_y) { // BMP stores pixels from bottom to top
        const uint8_t *row = &pixels[idx_y * width * 3];
        file.write(reinterpret_cast<const char *>(row), width * 3);
        if (!padding.empty()) {
            file.write(reinterpret_cast<const char *>(padding.data()), padding.size());
        }
    }
}

uint8_t float_to_byte(float _x, float norm_factor) {
    float x = _x / norm_factor;
    if (x < 0) {
        return 0;
    } else if (x >= 1) {
        return 255;
    } else {
        return x * 255.0f;
    }
}

void write_zs_image(std::string const &fname, Image const &img, float norm_factor) {
    auto pixels = std::vector<uint8_t>(img.width * img.height * 3);
    for (int32_t idx = 0; idx < img.width * img.height; idx++) {
        // for grayscale, make all channels have same value
        // BMP stores pixels in BGR order
        float val = img.zs.at(idx);
        pixels.at(idx * 3) = float_to_byte(val, norm_factor);
        pixels.at(idx * 3 + 1) = float_to_byte(val, norm_factor);
        pixels.at(idx * 3 + 2) = float_to_byte(val, norm_factor);
    }
    write_bmp(fname, img.width, img.height, pixels);
}

void write_alphas_image(std::string const &fname, Image const &img, float norm_factor) {
    auto pixels = std::vector<uint8_t>(img.width * img.height * 3);
    for (int32_t idx = 0; idx < img.width * img.height; idx++) {
        // for grayscale, make all channels have same value
        // BMP stores pixels in BGR order
        float val = img.alphas.at(idx);
        pixels.at(idx * 3) = float_to_byte(val, norm_factor);
        pixels.at(idx * 3 + 1) = float_to_byte(val, norm_factor);
        pixels.at(idx * 3 + 2) = float_to_byte(val, norm_factor);
    }
    write_bmp(fname, img.width, img.height, pixels);
}



struct SceneTest {
    std::string name;
    Mode mode;
    Scene scene;
};

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    // create image output dir if not exists
    if (!std::filesystem::exists("fmb_out/")) {
        if (std::filesystem::create_directory("fmb_out/")) {
            std::cout << "Directory created: " << "fmb_out/" << '\n';
        } else {
            std::cerr << "Failed to create directory: " << "fmb_out/" << '\n';
        }
    } else {
        std::cout << "Directory already exists: " << "fmb_out/" << '\n';
    }

    auto scenes = std::vector<SceneTest>();
    scenes.push_back({"ycb_box_150", Mode::TEST, gen_ycb_box(640, 480, 150)});
    scenes.push_back({"ycb_box_500", Mode::TEST, gen_ycb_box(640, 480, 500)});
    scenes.push_back({"ycb_box_2500", Mode::TEST, gen_ycb_box(640, 480, 2500)});
    scenes.push_back({"ycb_box_5000", Mode::TEST, gen_ycb_box(640, 480, 5000)});
    scenes.push_back({"ycb_box_10000", Mode::TEST, gen_ycb_box(640, 480, 10'000)});
    scenes.push_back({"ycb_box_50000", Mode::TEST, gen_ycb_box(640, 480, 50'000)});
    scenes.push_back({"ycb_box_100000", Mode::TEST, gen_ycb_box(640, 480, 100'000)});


    scenes.push_back({"ycb_box_150", Mode::BENCHMARK, gen_ycb_box(640, 480, 150)});
    scenes.push_back({"ycb_box_500", Mode::BENCHMARK, gen_ycb_box(640, 480, 500)});
    scenes.push_back({"ycb_box_2500", Mode::BENCHMARK, gen_ycb_box(640, 480, 2500)});
    scenes.push_back({"ycb_box_5000", Mode::BENCHMARK, gen_ycb_box(640, 480, 5000)});
    scenes.push_back({"ycb_box_10000", Mode::BENCHMARK, gen_ycb_box(640, 480, 10'000)});
    scenes.push_back({"ycb_box_50000", Mode::BENCHMARK, gen_ycb_box(640, 480, 50'000)});
    scenes.push_back({"ycb_box_100000", Mode::BENCHMARK, gen_ycb_box(640, 480, 100'000)});
    scenes.push_back({"ycb_box_5000", Mode::BENCHMARK, gen_ycb_box(640, 480, 5000)});
    scenes.push_back({"ycb_box_10000", Mode::BENCHMARK, gen_ycb_box(640, 480, 10'000)});
    scenes.push_back({"ycb_box_50000", Mode::BENCHMARK, gen_ycb_box(640, 480, 50'000)});
    scenes.push_back({"ycb_box_100000", Mode::BENCHMARK, gen_ycb_box(640, 480, 100'000)});

    int32_t fail_count = 0;

    int32_t count = 0;
    for (auto const &scene_test : scenes) {
        auto i = count++;
        printf("\nTesting scene '%s'\n", scene_test.name.c_str());
        auto results = run_config(scene_test.mode, scene_test.scene);

        // normalize results by GT results max values
        float z_normalizer = *std::max_element(results.image_expected.zs.begin(), results.image_expected.zs.end());
        float alpha_normalizer = *std::max_element(results.image_expected.alphas.begin(), results.image_expected.alphas.end());

        float z_normalizer_actual = *std::max_element(results.image_actual.zs.begin(), results.image_actual.zs.end());
        float alpha_normalizer_actual = *std::max_element(results.image_actual.alphas.begin(), results.image_actual.alphas.end());

        if (scene_test.mode == Mode::TEST) {
            printf("max(expected zs) vs max(actual zs): %f %f\n", z_normalizer, z_normalizer_actual);
            printf("max(expected alpha) vs max(actual alpha): %f %f\n", alpha_normalizer, alpha_normalizer_actual);
        }

        write_zs_image(
            std::string("fmb_out/img") + std::to_string(i) + "_" + scene_test.name +
                "_z_jax.bmp",
            results.image_expected,
            z_normalizer);
        write_alphas_image(
            std::string("fmb_out/img") + std::to_string(i) + "_" + scene_test.name +
                "_alphas_jax.bmp",
            results.image_expected,
            alpha_normalizer);
        write_zs_image(
            std::string("fmb_out/img") + std::to_string(i) + "_" + scene_test.name +
                "_z_cuda.bmp",
            results.image_actual,
            z_normalizer);
        write_alphas_image(
            std::string("fmb_out/img") + std::to_string(i) + "_" + scene_test.name +
                "_alphas_cuda.bmp",
            results.image_actual,
            alpha_normalizer);
        if (!results.correct) {
            printf("Result did not match expected image\n");
            printf("Max absolute difference: %.2e\n", results.max_abs_diff);
            fail_count++;
            // continue;
        } else {
            printf("Max absolute difference: %.2e\n", results.max_abs_diff);
            printf("OK\n");
        }

        if (scene_test.mode == Mode::BENCHMARK) {
            printf("Time: %f ms (%f FPS)\n", results.time_ms, 1000.0f/results.time_ms);
        }
    }

    if (fail_count) {
        printf("\nCorrectness: %d tests failed\n", fail_count);
    } else {
        printf("\nCorrectness: All tests passed\n");
    }

    return 0;
}
