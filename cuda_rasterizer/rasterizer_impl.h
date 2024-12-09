#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include <functional>

/*
  Memory helpers.
*/
// CUDA error checking function declaration
void cuda_check(cudaError_t code, const char *file, int line);

// Macro to use for checking CUDA errors
#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

// GpuMemoryPool class declaration
class GpuMemoryPool {
public:
    GpuMemoryPool() = default;
    ~GpuMemoryPool();

    // Delete copy and move constructors/assignments
    GpuMemoryPool(GpuMemoryPool const &) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool const &) = delete;
    GpuMemoryPool(GpuMemoryPool &&) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool &&) = delete;

    // Memory allocation and reset functions
    void *alloc(size_t size);
    void reset();

private:
    std::vector<void *> allocations_;  // Vector holding pointers to allocated memory
    std::vector<size_t> capacities_;   // Vector holding the sizes of allocations
    size_t next_idx_ = 0;              // Index to track the next allocation
};

/*
  Rasterizer class declaration.
*/
namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		// static int renderLaunchFunction(
        static int forwardJAX(
            cudaStream_t stream,
            int img_height, int img_width, int num_gaussians,
            float* means3d,  // (N, 3)
            float const* g_scales, // (N, 3) -- gsplat covariance parameterization
            float const* g_rots, // (N, 4)  -- gsplat covariance parameterization
            float* weights_log, // (N,)
            float* _camera_rays,  // (H*W, 3)
            float* rot, // (3, 3) // supply in rotation form not quaternion
            float* trans, // (3, )
            float* viewmatrix, float* projmatrix,
            float tan_fovx, float tan_fovy,
            float* zs_final,    // final gpu output
            float* est_alpha,   // final gpu output
            GpuMemoryPool &memory_pool
        );
	};
};

#endif
