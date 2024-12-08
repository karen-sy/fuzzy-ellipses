#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		// static int renderLaunchFunction(
        static int forwardJAX(
            int img_height, int img_width, int num_gaussians,
            float* means3d,  // (N, 3)
            float const* g_scales, // (N, 3) -- gsplat covariance parameterization
            float const* g_rots, // (N, 4)  -- gsplat covariance parameterization
            float* weights_log, // (N,)
            float* _camera_rays,  // (H*W, 3)
            float* rot, // (3, 3) // supply in rotation form not quaternion
            float* trans, // (3, )
            float* camera_rays, // intermediate gpu output
            float* viewmatrix, float* projmatrix,
            float tan_fovx, float tan_fovy,
            // float* cov3ds, // (N, 6) intermediate gpu output; gsplat covariance parameterization (init to 0)
            int* radii, // (N,) intermediate gpu output; init to 0
            uint32_t* tiles_touched, // (N,) intermediate gpu output; init to 0
            float* zs_final,    // final gpu output
            float* est_alpha,   // final gpu output
            GpuMemoryPool &memory_pool
        )
	};
};

#endif
