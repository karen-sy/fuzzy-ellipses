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

#include "primitives.cuh"

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
// CPU Reference Implementation


////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

constexpr uint16_t N_GAUSSIANS_PER_BLOCK = 64;
constexpr uint16_t N_PIXELS_PER_BLOCK = 16 * 16;

__device__ void matmul_331(  float* A,
                        float* B,
                        float* C
){
    C[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
    C[1] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
    C[2] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2];
}


__global__ void gaussian_ray_kernel(
                                int height,
                                int width,
                                int num_gaussians,
                                float* prc_arr,
                                float w_arr,
                                float* meansI_arr,
                                float* r_arr,
                                float* t_arr,
                                float beta_2,
                                float beta_3,
                                float* est_alpha_exp_factor,
                                float* wgt,
                                float* zs_final_unnormalized,

){
    // TODO rename variables eventually

    // shared input pixel (used interchangeably as "ray") buffers
    __shared__ float r_shmem[N_PIXELS_PER_BLOCK];
    __shared__ float t_shmem[N_PIXELS_PER_BLOCK];

    // shared input gaussian buffers
    __shared__ float means_shmem[N_PIXELS_PER_BLOCK];  // (3,) for each gaussian
    __shared__ float prc_shmem[N_PIXELS_PER_BLOCK];  // linearized (9,) for each gaussian
    __shared__ float w_shmem[N_PIXELS_PER_BLOCK];  // (1,) for each gaussian

    // shared output pixel buffers
    __shared__ float est_alpha_exp_factor_shmem[N_PIXELS_PER_BLOCK];
    __shared__ float wgt_shmem[N_PIXELS_PER_BLOCK];  // add over gaussians
    __shared__ float zs_final_unnormalized_shmem[N_PIXELS_PER_BLOCK];  // divide by jnp.where(wgt == 0, 1, wgt) after kernel

    // Identify current gaussian and pixel (TODO figure out mapping for gaussian-pixel intersection orders)
    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int pixel_id = 0; // TODO
    int gaussian_id = 0; // TODO
    float* r = r_shmem[pixel_id * 3];
    float* t = t_shmem[pixel_id * 3];
    float* meansI = means_shmem[gaussian_id * 3];
    float* prc = prc_shmem[gaussian_id * 9];
    float w = w_shmem[gaussian_id];

    // Load global -> shared for input buffers
    int global_pixel_id = 0;   // TODO
    r_shmem[pixel_id * 3] = r_arr[global_pixel_id * 3];
    t_shmem[pixel_id * 3] = t_arr[global_pixel_id * 3];
    means_shmem[gaussian_id * 3] = meansI_arr[gaussian_id * 3];
    prc_shmem[gaussian_id * 9] = prc_arr[gaussian_id * 9];
    w_shmem[gaussian_id] = w_arr[gaussian_id];

    // Initialize output pixel buffers to 0
    est_alpha_exp_factor_shmem[pixel_id] = 0;
    wgt_shmem[pixel_id] = 0;
    zs_final_unnormalized_shmem[pixel_id] = 0;

    __syncthreads();

    // shift the mean to be relative to ray start
    float* p;
    cudaMalloc(&p, 3 * sizeof(float));  // TODO: Workspace alloc
    thrust_prims::subtract_vec(meansI, t, p, 3);

    // compute \sigma^{-0.5} p, which is reused
    float* projp;
    cudaMalloc(&projp, 3 * sizeof(float));  // TODO: Workspace alloc
    matmul_331(prc, p, projp);

    // compute v^T \sigma^{-1} v
    float* vsv_sqrt;  // prc @ r
    cudaMalloc(&vsv_sqrt, 3 * sizeof(float));  // TODO: Workspace alloc
    matmul_331(r, projp, vsv_sqrt);
    float vsv;
    float* _vsv; cudaMalloc(&_vsv, 3*sizeof(float));  // TODO: Workspace alloc
    thrust_prims::pow2_vec(vsv_sqrt, _vsv, 3);
    thrust_prims::sum_vec(_vsv, vsv, 3);

    // compute p^T \sigma^{-1} v
    float psv;
    float* _psv; cudaMalloc(&_psv, 3*sizeof(float));  // TODO: Workspace alloc
    thrust_prims::multiply_vec(projp, vsv_sqrt, _psv, 3);
    thrust_prims::sum_vec(_psv, psv, 3);

    // distance to get maximum likelihood point for this gaussian
    // scale here is based on r!
    // if r = [x, y, 1], then depth. if ||r|| = 1, then distance
    float z = psv / vsv;

    // get the intersection point
    float v;
    thrust_prims::mul_scalar(r, z, v, 3);
    thrust_prims::subtract_vec(v, p, v, 3);

    // compute intersection's unnormalized Gaussian log likelihood
    float* _std; cudaMalloc(&_std, 3*sizeof(float));  // TODO: Workspace alloc
    matmul_331(prc, v, _std);
    thrust_prims::pow2_vec(_std, _std, 3);

    // multiply by weight
    float std;
    thrust_prims::sum_vec(_std, std, 3);
    std = -0.5 * std + w;

    // alpha is based on distance from all gaussians. (Eq. 8)
    // (calculate the -exp(std) factor and sum it into est_alpha_exp_factor[pixel_id])
    float est_alpha_exp = -exp(std);
    atomicAdd(est_alpha_exp_factor_shmem[pixel_id], est_alpha_exp);

    // compute the algebraic weights in the paper (Eq. 7)
    uint8_t sig = (uint8_t) (z > 0);
    float w_intersection = sig * exp(-z * beta_2 * beta * std) + 1e-20; // TODO stable_exp stuff

    // update normalization factor for weights for the pixel
    atomicAdd(wgt_shmem[pixel_id], w_intersection);  // wgt = w_intersection.sum(0)

    // compute weighted (but unnormalized) z
    atomicAdd(zs_final_unnormalized_shmem[pixel_id], w_intersection * z);  // TODO nan_to_num(zs)

    __syncthreads();

    //// Store back output shmems into global memory
    // Make note of which arrays are atomicAdd'ed into.
    est_alpha_exp_factor[global_pixel_id] += est_alpha_exp_factor_shmem[pixel_id];
    wgt[global_pixel_id] += wgt_shmem[pixel_id];
    zs_final_unnormalized[global_pixel_id] += zs_final_unnormalized_shmem[pixel_id];
}



void render_func_rays(
                    int height,
                    int width,
                    int num_gaussians,
                    float* means,  // (N, 3)
                    float* prec_full, // (N, 3, 3)
                    float* weights_log, // (N,)
                    float* camera_starts_rays,  // (H*W, 2, 3)
                    float beta_2, // float
                    float beta_3, // float
                    float* zs_final,
                    float* est_alpha

){
    // launch gaussian_ray kernel to fill in needed values for final calculations
    float* est_alpha_exp_factor; // TODO GPU allocate (from workspace) to pass into kernel
    float* wgt;
    float* zs_final_unnormalized;

    int N_BLOCKS_PER_GRID = (height*width*num_gaussians) / (N_PIXELS_PER_BLOCK * N_GAUSSIANS_PER_BLOCK);  // TODO make it a dim3
    int N_THREADS_PER_BLOCK = 0;  // TODO default to 256? make it depend on other params?

    gaussian_ray_kernel<<<N_BLOCKS_PER_GRID, N_THREADS_PER_BLOCK>>>(
        height, weight, num_gaussians,
        prec_full,
        weights_log,
        means,
        camera_starts_rays,  // TODO split this into r and t beforehand
        beta_2,
        beta_3,
        est_alpha_exp_factor,
        wgt,
        zs_final_unnormalized
    );

    // finish processing intermediate results into zs_final, est_alpha
    thrust::transform(thrust::device, zs_final_unnormalized,
                    zs_final_unnormalized + height*width,
                    wgt, zs_final, thrust::divides<float>());
    auto f_one_minus_exp = [] __device__ (float x) {return 1 - exp(x);};
    thrust::transform(thrust::device,est_alpha_exp_factor,
                    est_alpha_exp_factor + height*width,
                    est_alpha,
                    f_one_minus_exp);

}



void render_func_quat(float* means,  // (N, 3)
                    float* prec_full, // (N, 3, 3)
                    float* weights_log, // (N,)
                    float* camera_rays,  // (H*W, 3)
                    float* quat, // (4, )
                    float* trans, // (3, )
                    float beta_2, // float
                    float beta_3 // float
                    ){
    // launch render_func_rays kernel
}

////////////////////////////////////////////////////////////////////////////////
///          TESTING HARNESS                                                  ///
////////////////////////////////////////////////////////////////////////////////

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
    std::vector<float> circle_x;
    std::vector<float> circle_y;
    std::vector<float> circle_radius;
    std::vector<float> circle_red;
    std::vector<float> circle_green;
    std::vector<float> circle_blue;
    std::vector<float> circle_alpha;

    int32_t n_circle() const { return circle_x.size(); }
};

struct Image {
    int32_t width;
    int32_t height;
    std::vector<float> red;
    std::vector<float> green;
    std::vector<float> blue;
};

float max_abs_diff(Image const &a, Image const &b) {
    float max_diff = 0.0f;
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        float diff_red = std::abs(a.red.at(idx) - b.red.at(idx));
        float diff_green = std::abs(a.green.at(idx) - b.green.at(idx));
        float diff_blue = std::abs(a.blue.at(idx) - b.blue.at(idx));
        max_diff = std::max(max_diff, diff_red);
        max_diff = std::max(max_diff, diff_green);
        max_diff = std::max(max_diff, diff_blue);
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
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    render_cpu(
        scene.width,
        scene.height,
        scene.n_circle(),
        scene.circle_x.data(),
        scene.circle_y.data(),
        scene.circle_radius.data(),
        scene.circle_red.data(),
        scene.circle_green.data(),
        scene.circle_blue.data(),
        scene.circle_alpha.data(),
        img_expected.red.data(),
        img_expected.green.data(),
        img_expected.blue.data());

    auto circle_x_gpu = GpuBuf<float>(scene.circle_x);
    auto circle_y_gpu = GpuBuf<float>(scene.circle_y);
    auto circle_radius_gpu = GpuBuf<float>(scene.circle_radius);
    auto circle_red_gpu = GpuBuf<float>(scene.circle_red);
    auto circle_green_gpu = GpuBuf<float>(scene.circle_green);
    auto circle_blue_gpu = GpuBuf<float>(scene.circle_blue);
    auto circle_alpha_gpu = GpuBuf<float>(scene.circle_alpha);
    auto img_red_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_green_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_blue_gpu = GpuBuf<float>(scene.height * scene.width);

    auto memory_pool = GpuMemoryPool();

    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(img_red_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            img_green_gpu.data,
            0,
            scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(img_blue_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        memory_pool.reset();
    };

    auto f = [&]() {
        circles_gpu::launch_render(
            scene.width,
            scene.height,
            scene.n_circle(),
            circle_x_gpu.data,
            circle_y_gpu.data,
            circle_radius_gpu.data,
            circle_red_gpu.data,
            circle_green_gpu.data,
            circle_blue_gpu.data,
            circle_alpha_gpu.data,
            img_red_gpu.data,
            img_green_gpu.data,
            img_blue_gpu.data,
            memory_pool);
    };

    reset();
    f();

    auto img_actual = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    CUDA_CHECK(cudaMemcpy(
        img_actual.red.data(),
        img_red_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.green.data(),
        img_green_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.blue.data(),
        img_blue_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));

    float max_diff = max_abs_diff(img_expected, img_actual);

    if (max_diff > 1e-3) {
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

template <typename Rng>
Scene gen_random(Rng &rng, int32_t width, int32_t height, int32_t n_circle) {
    auto unif_0_1 = std::uniform_real_distribution<float>(0.0f, 1.0f);
    auto z_values = std::vector<float>();
    for (int32_t i = 0; i < n_circle; i++) {
        float z;
        for (;;) {
            z = unif_0_1(rng);
            z = std::max(z, unif_0_1(rng));
            if (z > 0.01) {
                break;
            }
        }
        // float z = std::max(unif_0_1(rng), unif_0_1(rng));
        z_values.push_back(z);
    }
    std::sort(z_values.begin(), z_values.end(), std::greater<float>());

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };
    auto color_idx_dist = std::uniform_int_distribution<int>(0, colors.size() - 1);
    auto alpha_dist = std::uniform_real_distribution<float>(0.0f, 0.5f);

    int32_t fog_interval = n_circle / 10;
    float fog_alpha = 0.2;

    auto scene = Scene{width, height};
    float base_radius_scale = 1.0f;
    int32_t i = 0;
    for (float z : z_values) {
        float max_radius = base_radius_scale / z;
        float radius = std::max(1.0f, unif_0_1(rng) * max_radius);
        float x = unif_0_1(rng) * (width + 2 * max_radius) - max_radius;
        float y = unif_0_1(rng) * (height + 2 * max_radius) - max_radius;
        int color_idx = color_idx_dist(rng);
        uint32_t color = colors[color_idx];
        scene.circle_x.push_back(x);
        scene.circle_y.push_back(y);
        scene.circle_radius.push_back(radius);
        scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
        scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
        scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
        scene.circle_alpha.push_back(alpha_dist(rng));
        i++;
        if (i % fog_interval == 0 && i + 1 < n_circle) {
            scene.circle_x.push_back(float(width - 1) / 2.0f);
            scene.circle_y.push_back(float(height - 1) / 2.0f);
            scene.circle_radius.push_back(float(std::max(width, height)));
            scene.circle_red.push_back(1.0f);
            scene.circle_green.push_back(1.0f);
            scene.circle_blue.push_back(1.0f);
            scene.circle_alpha.push_back(fog_alpha);
        }
    }

    return scene;
}

constexpr float PI = 3.14159265359f;

Scene gen_overlapping_opaque() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };

    int32_t n_circle = 20;
    int32_t n_ring = 4;
    float angle_range = PI;
    for (int32_t ring = 0; ring < n_ring; ring++) {
        float dist = 20.0f * (ring + 1);
        float saturation = float(ring + 1) / n_ring;
        float hue_shift = float(ring) / (n_ring - 1);
        for (int32_t i = 0; i < n_circle; i++) {
            float theta = angle_range * i / (n_circle - 1);
            float x = width / 2.0f - dist * std::cos(theta);
            float y = height / 2.0f - dist * std::sin(theta);
            scene.circle_x.push_back(x);
            scene.circle_y.push_back(y);
            scene.circle_radius.push_back(16.0f);
            auto color = colors[(i + ring * 2) % colors.size()];
            scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
            scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
            scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
            scene.circle_alpha.push_back(1.0f);
        }
    }

    return scene;
}

Scene gen_overlapping_transparent() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    float offset = 20.0f;
    float radius = 40.0f;
    scene.circle_x = std::vector<float>{
        (width - 1) / 2.0f - offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f - offset,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
    };
    scene.circle_radius = std::vector<float>{
        radius,
        radius,
        radius,
        radius,
    };
    // 0xd32360
    // 0x2874aa
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0x28) / 255.0f,
        float(0x28) / 255.0f,
        float(0xd3) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x74) / 255.0f,
        float(0x74) / 255.0f,
        float(0x23) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0xaa) / 255.0f,
        float(0xaa) / 255.0f,
        float(0x60) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        0.75f,
        0.75f,
        0.75f,
        0.75f,
    };
    return scene;
}

Scene gen_overlapping_transparent2() {
    int32_t width = 1024;
    int32_t height = 1024;

    auto scene = Scene{width, height};

    float offset = 160.0f;
    float radius = 80.0f;
    scene.circle_x = std::vector<float>{
        (width - 1) / 2.0f - offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f - offset,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
    };
    scene.circle_radius = std::vector<float>{
        radius,
        radius,
        radius,
        radius,
    };
    // 0xd32360
    // 0x2874aa
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0x28) / 255.0f,
        float(0x28) / 255.0f,
        float(0xd3) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x74) / 255.0f,
        float(0x74) / 255.0f,
        float(0x23) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0xaa) / 255.0f,
        float(0xaa) / 255.0f,
        float(0x60) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        0.75f,
        0.75f,
        0.75f,
        0.75f,
    };
    return scene;
}


Scene gen_simple() {
    /*
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    */
    int32_t width = 256;
    int32_t height = 256;
    auto scene = Scene{width, height};
    scene.circle_x = std::vector<float>{
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
    };
    scene.circle_radius = std::vector<float>{
        40.0f,
        40.0f,
        40.0f,
        40.0f,
    };
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0xcc) / 255.0f,
        float(0x20) / 255.0f,
        float(0x28) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x9f) / 255.0f,
        float(0x80) / 255.0f,
        float(0x74) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0x26) / 255.0f,
        float(0x20) / 255.0f,
        float(0xaa) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        1.0f,
        1.0f,
        1.0f,
        1.0f,
    };
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

uint8_t float_to_byte(float x) {
    if (x < 0) {
        return 0;
    } else if (x >= 1) {
        return 255;
    } else {
        return x * 255.0f;
    }
}

void write_image(std::string const &fname, Image const &img) {
    auto pixels = std::vector<uint8_t>(img.width * img.height * 3);
    for (int32_t idx = 0; idx < img.width * img.height; idx++) {
        float red = img.red.at(idx);
        float green = img.green.at(idx);
        float blue = img.blue.at(idx);
        // BMP stores pixels in BGR order
        pixels.at(idx * 3) = float_to_byte(blue);
        pixels.at(idx * 3 + 1) = float_to_byte(green);
        pixels.at(idx * 3 + 2) = float_to_byte(red);
    }
    write_bmp(fname, img.width, img.height, pixels);
}

Image compute_img_diff(Image const &a, Image const &b) {
    auto img_diff = Image{
        a.width,
        a.height,
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
    };
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        img_diff.red.at(idx) = std::abs(a.red.at(idx) - b.red.at(idx));
        img_diff.green.at(idx) = std::abs(a.green.at(idx) - b.green.at(idx));
        img_diff.blue.at(idx) = std::abs(a.blue.at(idx) - b.blue.at(idx));
    }
    return img_diff;
}

struct SceneTest {
    std::string name;
    Mode mode;
    Scene scene;
};

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    // create image output dir if not exists
    if (!std::filesystem::exists("circles_out/")) {
        if (std::filesystem::create_directory("circles_out/")) {
            std::cout << "Directory created: " << "circles_out/" << '\n';
        } else {
            std::cerr << "Failed to create directory: " << "circles_out/" << '\n';
        }
    } else {
        std::cout << "Directory already exists: " << "circles_out/" << '\n';
    }

    auto scenes = std::vector<SceneTest>();
    // scenes.push_back({"simple", Mode::TEST, gen_simple()});
    // scenes.push_back({"overlapping_opaque", Mode::TEST, gen_overlapping_opaque()});
    // scenes.push_back(
    //     {"overlapping_transparent", Mode::TEST, gen_overlapping_transparent()});
    // scenes.push_back(
    //     {"million_circles", Mode::BENCHMARK, gen_random(rng, 1024, 1024, 1'000'000)});

    int32_t fail_count = 0;

    int32_t count = 0;
    for (auto const &scene_test : scenes) {
        auto i = count++;
        printf("\nTesting scene '%s'\n", scene_test.name.c_str());
        auto results = run_config(scene_test.mode, scene_test.scene);
        write_image(
            std::string("circles_out/img") + std::to_string(i) + "_" + scene_test.name +
                "_cpu.bmp",
            results.image_expected);
        write_image(
            std::string("circles_out/img") + std::to_string(i) + "_" + scene_test.name +
                "_gpu.bmp",
            results.image_actual);
        if (!results.correct) {
            printf("  Result did not match expected image\n");
            printf("  Max absolute difference: %.2e\n", results.max_abs_diff);
            auto diff = compute_img_diff(results.image_expected, results.image_actual);
            write_image(
                std::string("circles_out/img") + std::to_string(i) + "_" + scene_test.name +
                    "_diff.bmp",
                diff);
            printf(
                "  (Wrote image diff to 'circles_out/img%d_%s_diff.bmp')\n",
                i,
                scene_test.name.c_str());
            fail_count++;
            continue;
        } else {
            printf("  OK\n");
        }
        if (scene_test.mode == Mode::BENCHMARK) {
            printf("  Time: %f ms (%f FPS)\n", results.time_ms, 1000.0f/results.time_ms);
        }
    }

    if (fail_count) {
        printf("\nCorrectness: %d tests failed\n", fail_count);
    } else {
        printf("\nCorrectness: All tests passed\n");
    }

    return 0;
}
