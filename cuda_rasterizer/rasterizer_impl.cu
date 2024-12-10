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

#include "rasterizer_impl.h"
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

#include "auxiliary.h"
#include "forward.h"			// namespace FORWARD


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

__host__ float getNumStds(int num_gaussians){
    // get the number of standard deviations to use for the ellipse
    // controls the "blur" effect (includes more gaussians and slows computation) (
    // (default 3.0f in gsplat; around >5.0f seems to replicate FMB results exactly)
    // TODO can calculate a better heuristic based on total mem available / avg per pixel amortized

    if (num_gaussians <= 1000){
        return 5.0f;
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

int CudaRasterizer::Rasterizer::forwardJAX(
                    cudaStream_t stream,
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
    FORWARD::preprocessCUDA<<<num_blocks_gaussian, num_threads_gaussian, 0, stream>>>(
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
    getGaussianTileKeys<<<num_blocks_gaussian, num_threads_gaussian, 0, stream>>>(num_gaussians,
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
    getPerTileRanges<<<num_blocks_total, num_threads_total, 0, stream>>>(n_tile_gaussians,
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
    cudaFuncSetAttribute(FORWARD::gaussianRayRasterizeCUDA, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    FORWARD::gaussianRayRasterizeCUDA<<<num_blocks_render, num_threads_render, shmem_size, stream>>>(
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

    return n_tile_gaussians;
}

// // Forward rendering procedure for differentiable rasterization
// // of Gaussians.
// int CudaRasterizer::Rasterizer::forwardJAX(
// 	cudaStream_t stream,

// 	std::function<char* (size_t)> geometryBuffer,
// 	std::function<char* (size_t)> binningBuffer,
// 	std::function<char* (size_t)> imageBuffer,
// 	const int P, int D, int M,
// 	const float* background,
// 	const int width, int height,
// 	const float* means3D,
// 	const float* shs,
// 	const float* colors_precomp,
// 	const float* opacities,
// 	const float* scales,
// 	const float scale_modifier,
// 	const float* rotations,
// 	const float* cov3D_precomp,
// 	const float* viewmatrix,
// 	const float* projmatrix,
// 	const float* cam_pos,
// 	const float tan_fovx, float tan_fovy,
// 	const bool prefiltered,
// 	float* out_color,
// 	int* radii,
// 	bool debug)
// {
// 	const float focal_y = height / (2.0f * tan_fovy);
// 	const float focal_x = width / (2.0f * tan_fovx);

// 	size_t chunk_size = required<GeometryState>(P);
// 	char* chunkptr = geometryBuffer(chunk_size);
// 	GeometryState geomState = GeometryState::fromChunkJAX(stream, chunkptr, P);

// 	if (radii == nullptr)
// 	{
// 		radii = geomState.internal_radii;
// 	}

// 	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
// 	dim3 block(BLOCK_X, BLOCK_Y, 1);

// 	// Dynamically resize image-based auxiliary buffers during training
// 	size_t img_chunk_size = required<ImageState>(width * height);
// 	char* img_chunkptr = imageBuffer(img_chunk_size);
// 	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

// 	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
// 	{
// 		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
// 	}

// 	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
// 	CHECK_CUDA(FORWARD::preprocessJAX(
// 		stream,
// 		P, D, M,
// 		means3D,
// 		(glm::vec3*)scales,
// 		scale_modifier,
// 		(glm::vec4*)rotations,
// 		opacities,
// 		shs,
// 		geomState.clamped,
// 		cov3D_precomp,
// 		colors_precomp,
// 		viewmatrix, projmatrix,
// 		(glm::vec3*)cam_pos,
// 		width, height,
// 		focal_x, focal_y,
// 		tan_fovx, tan_fovy,
// 		radii,
// 		geomState.means2D,
// 		geomState.depths,
// 		geomState.cov3D,
// 		geomState.rgb,
// 		geomState.conic_opacity,
// 		tile_grid,
// 		geomState.tiles_touched,
// 		prefiltered
// 	), debug)

// 	// Compute prefix sum over full list of touched tile counts by Gaussians
// 	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
// 	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space,
// 											geomState.scan_size,
// 											geomState.tiles_touched,
// 											geomState.point_offsets, P,
// 											stream),
// 											debug)

// 	// Retrieve total number of Gaussian instances to launch and resize aux buffers
// 	cudaStreamSynchronize(stream);
// 	int num_rendered;
// 	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
// 	cudaStreamSynchronize(stream);

// 	size_t binning_chunk_size = required<BinningState>(num_rendered);
// 	char* binning_chunkptr = binningBuffer(binning_chunk_size);
// 	BinningState binningState = BinningState::fromChunkJAX(stream, binning_chunkptr, num_rendered);

// 	// For each instance to be rendered, produce adequate [ tile | depth ] key
// 	// and corresponding dublicated Gaussian indices to be sorted
// 	// duplicateWithKeys << <(P + 255) / 256, 256 >> > (
// 	duplicateWithKeys << <(P + 255) / 256, 256, 0, stream >> > (
// 		P,
// 		geomState.means2D,
// 		geomState.depths,
// 		geomState.point_offsets,
// 		binningState.point_list_keys_unsorted,
// 		binningState.point_list_unsorted,
// 		radii,
// 		tile_grid)
// 	CHECK_CUDA(, debug)

// 	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

// 	// Sort complete list of (duplicated) Gaussian indices by keys
// 	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
// 		binningState.list_sorting_space,
// 		binningState.sorting_size,
// 		binningState.point_list_keys_unsorted, binningState.point_list_keys,
// 		binningState.point_list_unsorted, binningState.point_list,
// 		num_rendered, 0, 32 + bit,
// 		stream
// 		),
// 		debug)

// 	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

// 	// Identify start and end of per-tile workloads in sorted list
// 	if (num_rendered > 0)
// 		// identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
// 		identifyTileRanges << <(num_rendered + 255) / 256, 256, 0, stream >> > (
// 			num_rendered,
// 			binningState.point_list_keys,
// 			imgState.ranges);
// 	CHECK_CUDA(, debug)

// 	// Let each tile blend its range of Gaussians independently in parallel
// 	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
// 	CHECK_CUDA(FORWARD::renderJAX(
// 		stream,
// 		tile_grid, block,
// 		imgState.ranges,
// 		binningState.point_list,
// 		width, height,
// 		geomState.means2D,
// 		feature_ptr,
// 		geomState.conic_opacity,
// 		imgState.accum_alpha,
// 		imgState.n_contrib,
// 		background,
// 		out_color), debug)

// 	return num_rendered;
// }
