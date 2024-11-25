#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "thrust_primitives.cuh"  // namespace thrust_primitives
#include "blas_primitives.cuh"    // namespace blas

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

// #define DEBUG_MODE ;     // uncomment to enable some prints


////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation
// image partitioning
constexpr uint32_t TILE_HEIGHT = 8;  // image is partitioned into tiles, one per block
constexpr uint32_t TILE_WIDTH = 16;
constexpr uint32_t N_PIXELS_PER_BLOCK = TILE_HEIGHT * TILE_WIDTH;

// gaussians partitionning
constexpr int N_GAUSSIANS_PER_BATCH = 8;

namespace fmb {

__global__ void gaussian_ray_kernel(
                                int img_height,
                                int img_width,
                                int num_gaussians,
                                float const* prc_arr, // TODO add triu to generalize
                                float const* w_arr,
                                float const* meansI_arr,
                                float* camera_rays, // gpu array
                                float* camera_trans,    // gpu array
                                float const beta_2,
                                float const beta_3,
                                float* est_alpha_exp_factor,    // output
                                float* zs_final    // output
){
    // TODO rename variables eventually

    // shared input pixel (used interchangeably as "ray") buffers
    __shared__ float r_shmem[N_PIXELS_PER_BLOCK*3];
    __shared__ float t_shmem[3];    // extra for word alignment
    if ((threadIdx.x == 0) && (threadIdx.y == 0)){
        for (uint8_t vec_offset = 0; vec_offset < 3; vec_offset++){     // t \in R^3
            t_shmem[vec_offset] = camera_trans[vec_offset];
        }
    }

    // shared input gaussian buffers
    __shared__ float means_shmem[N_GAUSSIANS_PER_BATCH*3];  // (3,) for each gaussian
    __shared__ float prc_shmem[N_GAUSSIANS_PER_BATCH*9];  // linearized (9,) for each gaussian
    __shared__ float w_shmem[N_GAUSSIANS_PER_BATCH];  // (1,) for each gaussian

    // shared output pixel buffers
    __shared__ float est_alpha_exp_factor_shmem[N_PIXELS_PER_BLOCK];
    __shared__ float wgt_shmem[N_PIXELS_PER_BLOCK];  // atomicAdd over gaussians, then atomicAdd the wget_shmem into global mem wget
    __shared__ float zs_final_shmem[N_PIXELS_PER_BLOCK];  // divide by jnp.where(wgt == 0, 1, wgt) after kernel

    // Identify current pixel
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
    est_alpha_exp_factor_shmem[block_pixel_id] = 0.0f;
    wgt_shmem[block_pixel_id] = 0.0f;
    zs_final_shmem[block_pixel_id] = 0.0f;

    __syncthreads();

    // load value from shmem
    float* r = &r_shmem[block_pixel_id * 3];
    float* t = &t_shmem[0];
    float prev_batch_normalizer = -INFINITY;

    for (uint32_t gaussian_id_offset = 0; gaussian_id_offset < num_gaussians; gaussian_id_offset += N_GAUSSIANS_PER_BATCH){
        // each batch processes N_GAUSSIANS_PER_BATCH gaussians; sequentially iterate over batches
        float exp_batch[N_GAUSSIANS_PER_BATCH] = {0.0f};  // (1,) for each gaussian
        float z_batch[N_GAUSSIANS_PER_BATCH] = {0.0f};  // (1,) for each gaussian
        float exp_normalizer_batch = prev_batch_normalizer;

        //// Load global -> shared for gaussians
        for (uint32_t g_id_in_batch = 0; g_id_in_batch < N_GAUSSIANS_PER_BATCH; g_id_in_batch++){
            uint32_t gaussian_id = gaussian_id_offset + g_id_in_batch;
            if (gaussian_id >= num_gaussians){
                break;
            }
            #pragma unroll
            for (uint8_t vec_offset = 0; vec_offset < 3; vec_offset++){     // meansI \in R^3
                means_shmem[g_id_in_batch * 3 + vec_offset] = meansI_arr[gaussian_id * 3 + vec_offset];
            }
            #pragma unroll
            for (uint8_t vec_offset = 0; vec_offset < 9; vec_offset++){     // prc \in R^9
                prc_shmem[g_id_in_batch * 9 + vec_offset] = prc_arr[gaussian_id * 9 + vec_offset];
            }
            w_shmem[g_id_in_batch] = w_arr[gaussian_id];
        }

        // Do computations over gaussians in batch
        for (uint32_t g_id_in_batch = 0; g_id_in_batch < N_GAUSSIANS_PER_BATCH; g_id_in_batch++){
            uint32_t gaussian_id = gaussian_id_offset + g_id_in_batch;
            if (gaussian_id >= num_gaussians){
                break;
            }
            float* meansI = &means_shmem[g_id_in_batch * 3];
            float* prc = &prc_shmem[g_id_in_batch * 9];
            float w = w_shmem[g_id_in_batch];

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
            // (calculate the -exp(std) factor and sum it into est_alpha_exp_factor[pixel_id])
            float est_alpha_exp = exp(std);
            est_alpha_exp_factor_shmem[block_pixel_id] -= est_alpha_exp ;

            // compute the algebraic weights in the paper (Eq. 7)
            exp_batch[g_id_in_batch] = -z * beta_2 + beta_3 * std;
            z_batch[g_id_in_batch] = thrust_primitives::nan_to_num(z);
            exp_normalizer_batch = fmax(exp_normalizer_batch, exp_batch[g_id_in_batch]);
        }

        // correct exponentiation for numerical stability
        thrust_primitives::sub_scalar(exp_batch, exp_normalizer_batch, exp_batch, N_GAUSSIANS_PER_BATCH); // in-place subtraction

        // get mask for z > 0
        float stencil[N_GAUSSIANS_PER_BATCH];
        thrust_primitives::positive_mask_vec(z_batch, stencil, N_GAUSSIANS_PER_BATCH);

        // calculate w_intersection (into exp_batch)
        thrust_primitives::clipped_exp(exp_batch, exp_batch, N_GAUSSIANS_PER_BATCH);  // nan_to_num(exp(w_intersection)) in-place
        thrust_primitives::multiply_vec(exp_batch, stencil, exp_batch, N_GAUSSIANS_PER_BATCH);  // w_intersection

        // update normalization factor for weights for the pixel (sum over gaussian intersections)
        float correction_factor = exp(prev_batch_normalizer - exp_normalizer_batch); // correct the previous exponentiation in current zs_final with new normalizer
        float exp_batch_sum;
        thrust_primitives::sum_vec(exp_batch, &exp_batch_sum, N_GAUSSIANS_PER_BATCH);
        wgt_shmem[block_pixel_id] *= correction_factor;
        wgt_shmem[block_pixel_id] += exp_batch_sum;  // weighted sum of w_intersection

        // compute weighted (but unnormalized) z
        // zs_final_shmem[block_pixel_id] /= correction_factor;
        float z_batch_sum;
        thrust_primitives::multiply_vec(z_batch, exp_batch, z_batch, N_GAUSSIANS_PER_BATCH);  // z * w_intersection in-place
        thrust_primitives::sum_vec(z_batch, &z_batch_sum, N_GAUSSIANS_PER_BATCH);
        zs_final_shmem[block_pixel_id] *= correction_factor;
        zs_final_shmem[block_pixel_id] += z_batch_sum;  // weighted sum of z

        prev_batch_normalizer = exp_normalizer_batch;
        __syncthreads();
    }

    zs_final[global_pixel_id] = zs_final_shmem[block_pixel_id] / (wgt_shmem[block_pixel_id] + 1e-10f);
    est_alpha_exp_factor[global_pixel_id] = est_alpha_exp_factor_shmem[block_pixel_id];

}

void render_func_quat(
                    int img_height, int img_width, int num_gaussians,
                    float const* means,  // (N, 3)
                    float const* prec_full, // (N, 3, 3)
                    float const* weights_log, // (N,)
                    float* _camera_rays,  // (H*W, 3)
                    float* rot, // (3, 3) // supply in rotation form not quaternion
                    float* trans, // (3, )
                    float* est_alpha_exp_factor,  // intermediae gpu
                    float* camera_rays, // intermediate gpu output
                    float* zs_final,    // final gpu output
                    float* est_alpha,   // final gpu output
                    GpuMemoryPool &memory_pool
                    )
{
    int total_num_pixels = img_height * img_width;
    float beta_2 = 21.4;
    float beta_3 = 2.66;
    blas::matmul(total_num_pixels, 3, 3, _camera_rays, rot, camera_rays);  // _camera_rays @ rot

    // allocate threads. Each thread processes one pixel and (N_GAUSSIANS_PER_BLOCK/N_GAUSSIANS_THREAD) gaussians
    dim3 N_THREADS_PER_BLOCK(TILE_WIDTH, TILE_HEIGHT);  // align x, y order
    dim3 N_BLOCKS_PER_GRID(img_width/TILE_WIDTH, img_height/TILE_HEIGHT);

    #ifdef DEBUG_MODE
    printf("Launching kernel with (%u,%u) blocks per grid\n", N_BLOCKS_PER_GRID.x, N_BLOCKS_PER_GRID.y);
    printf("Launching kernel with (%u,%u)=%u threads per block\n", N_THREADS_PER_BLOCK.x, N_THREADS_PER_BLOCK.y, N_THREADS_PER_BLOCK.x*N_THREADS_PER_BLOCK.y);
    #endif

    // launch gaussian_ray kernel to fill in needed values for final calculations
    gaussian_ray_kernel<<<N_BLOCKS_PER_GRID, N_THREADS_PER_BLOCK>>>(
        img_height, img_width, num_gaussians,
        prec_full,
        weights_log,
        means,
        camera_rays,
        trans,
        beta_2,
        beta_3,
        est_alpha_exp_factor,
        zs_final
    );

    // finish processing intermediate results into zs_final, est_alpha
    auto f_one_minus_exp = [] __device__ (float x) {return 1 - exp(x);};  // 1 - exp(x)
    thrust::transform(thrust::device,est_alpha_exp_factor,
                    est_alpha_exp_factor + total_num_pixels,
                    est_alpha,
                    f_one_minus_exp);
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
    std::vector<float> prc;    // (9N, ) contiguous
    std::vector<float> weights;  // (N, )
    std::vector<float> camera_rays;  // (H*W*3, ) contiguous
    std::vector<float> camera_rot; // (3*3, )
    std::vector<float> camera_trans; // (3, )

    std::vector<float> _camera_rays_xfm;


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


Results run_config(Mode mode, Scene const &scene) {
    auto img_expected = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f)
        };

    img_expected.zs = read_data("../data/zs.bin", scene.n_pixels());
    img_expected.alphas = read_data("../data/alphas.bin", scene.n_pixels());

    auto means_gpu = GpuBuf<float>(scene.means);
    auto prc_gpu = GpuBuf<float>(scene.prc);
    auto weights_gpu = GpuBuf<float>(scene.weights);
    auto camera_rays_gpu = GpuBuf<float>(scene.camera_rays);
    auto camera_rot_gpu = GpuBuf<float>(scene.camera_rot);
    auto camera_trans_gpu = GpuBuf<float>(scene.camera_trans);

    // need to do cudamemset for these below
    auto zs_final_gpu = GpuBuf<float>(scene.height * scene.width);
    auto est_alpha_gpu = GpuBuf<float>(scene.height * scene.width);

    // intermediate values to compute in-kernel. Allocate gpu mem
    auto est_alpha_exp_factor = GpuBuf<float>(scene.height * scene.width);
    auto camera_rays_xfm_gpu = GpuBuf<float>(3 * scene.height * scene.width);

    auto memory_pool = GpuMemoryPool();

    // reset function for allocated memory
    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(zs_final_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            est_alpha_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            est_alpha_exp_factor.data, 0, scene.height * scene.width * sizeof(float)));
        memory_pool.reset();
    };

    #ifdef DEBUG_MODE
    printf("scene.height = %d\n", scene.height);
    printf("scene.width = %d\n", scene.width);
    printf("scene.n_gaussians = %d\n", scene.n_gaussians());
    printf("scene.means[0] = %f\n", scene.means[0]);
    printf("scene.prc[0] = %f\n", scene.prc[0]);
    printf("scene.weights[0] = %f\n", scene.weights[0]);
    printf("scene.camera_rays[0] = %f\n", scene.camera_rays[0]);
    printf("scene.camera_rot[0] = %f\n", scene.camera_rot[0]);
    printf("scene.camera_trans[0] = %f\n", scene.camera_trans[0]);
    #endif

    // kernel launch function
    auto f = [&]() {
        fmb::render_func_quat(
            scene.height, scene.width, scene.n_gaussians(),
            means_gpu.data,
            prc_gpu.data,
            weights_gpu.data,
            camera_rays_gpu.data,
            camera_rot_gpu.data,
            camera_trans_gpu.data,
            est_alpha_exp_factor.data,
            camera_rays_xfm_gpu.data,
            zs_final_gpu.data,
            est_alpha_gpu.data,
            memory_pool
            );
    };
    reset();
    f();

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

    if (max_diff > 3.5e-2) {
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

    double time_ms = benchmark_ms(1000.0, reset, f);

    return Results{
        true,
        max_diff,
        std::move(img_expected),
        std::move(img_actual),
        time_ms,
    };
}

Scene gen_ycb_box_150() {
    /*
        YCB box scene with 150 gaussians
    */
    auto scene = Scene{};

    auto w_h_g = read_int_data("../data/width_height_gaussians.bin", 3);
    int32_t width = w_h_g[0];
    int32_t height = w_h_g[1];
    int32_t n_gaussians = w_h_g[2];
    scene.width = width;
    scene.height = height;
    scene.means = read_data("../data/means.bin", 3 * n_gaussians);
    scene.prc = read_data("../data/precs.bin", 9 * n_gaussians);
    scene.weights = read_data("../data/weights.bin", n_gaussians);
    scene.camera_rays = read_data("../data/camera_rays.bin", 3 * width * height);  // TODO
    scene.camera_rot = read_data("../data/camera_rot.bin", 9);
    scene.camera_trans = read_data("../data/camera_trans.bin", 3);

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
    scenes.push_back({"ycb_box_150", Mode::TEST, gen_ycb_box_150()});
    scenes.push_back({"ycb_box_150", Mode::BENCHMARK, gen_ycb_box_150()});

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

        #ifdef DEBUG_MODE
        printf("max(actual zs) = %f\n", *std::max_element(results.image_actual.zs.begin(), results.image_actual.zs.end()));
        printf("max(actual alpha) = %f\n", *std::max_element(results.image_actual.alphas.begin(), results.image_actual.alphas.end()));
        #endif

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
            z_normalizer_actual);
        write_alphas_image(
            std::string("fmb_out/img") + std::to_string(i) + "_" + scene_test.name +
                "_alphas_cuda.bmp",
            results.image_actual,
            alpha_normalizer_actual);
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
