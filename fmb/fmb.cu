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

#include "thrust_primitives.cuh"
#include "blas_primitives.cuh"

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
// Optimized GPU Implementation
constexpr uint16_t NUM_THREADS = 256;

// image partitioning
constexpr uint8_t TILE_HEIGHT = 16;  // image is partitioned into tiles, one per block
constexpr uint8_t TILE_WIDTH = 16;
constexpr uint16_t N_PIXELS_PER_BLOCK = TILE_HEIGHT * TILE_WIDTH;

// gaussians partitionning
constexpr int N_GAUSSIANS_PER_THREAD = max(1, NUM_THREADS / N_PIXELS_PER_BLOCK);
constexpr uint16_t N_GAUSSIANS_PER_BLOCK = NUM_THREADS * N_GAUSSIANS_PER_THREAD; // for shmem alloc

namespace fmb {
__global__ void gaussian_ray_kernel(
                                int img_height,
                                int img_width,
                                int num_gaussians,
                                float* prc_arr,
                                float w_arr,
                                float* meansI_arr,
                                float* camera_starts_rays,
                                float* camera_trans,
                                float beta_2,
                                float beta_3,
                                float* est_alpha_exp_factor,    // output
                                float* wgt,                     // output
                                float* zs_final_unnormalized    // output

){
    // TODO rename variables eventually

    // shared input pixel (used interchangeably as "ray") buffers
    __shared__ float r_shmem[N_PIXELS_PER_BLOCK*3];
    __shared__ float t_shmem[8];    // extra for word alignment
    if ((threadIdx.x == 0) && (threadIdx.y == 0)){
        for (uint8_t vec_offset = 0; vec_offset < 3; vec_offset++){     // t \in R^3
            t_shmem[vec_offset] = camera_trans[vec_offset];
        }
    }

    // shared input gaussian buffers
    __shared__ float means_shmem[N_GAUSSIANS_PER_BLOCK*3];  // (3,) for each gaussian
    __shared__ float prc_shmem[N_GAUSSIANS_PER_BLOCK*9];  // linearized (9,) for each gaussian
    __shared__ float w_shmem[N_GAUSSIANS_PER_BLOCK];  // (1,) for each gaussian

    // shared output pixel buffers
    __shared__ float est_alpha_exp_factor_shmem[N_PIXELS_PER_BLOCK];
    __shared__ float wgt_shmem[N_PIXELS_PER_BLOCK];  // atomicAdd over gaussians, then atomicAdd the wget_shmem into global mem wget
    __shared__ float zs_final_unnormalized_shmem[N_PIXELS_PER_BLOCK];  // divide by jnp.where(wgt == 0, 1, wgt) after kernel

    // Identify current gaussian and pixel (TODO figure out mapping for gaussian-pixel intersection orders)
    int tile_offset_x = blockIdx.x * TILE_WIDTH;
    int tile_offset_y = blockIdx.y * TILE_HEIGHT;
    int block_pixel_id = threadIdx.y * TILE_WIDTH + threadIdx.x; // 0 to N_PIXELS_PER_BLOCK
    int global_pixel_id = (tile_offset_y + threadIdx.y) * img_width + (tile_offset_x + threadIdx.x);

    int gaussian_id_offset = blockIdx.z * N_GAUSSIANS_PER_BLOCK; // begins range of length N_GAUSSIANS_PER_THREAD

    // Load global -> shared for input buffers
    for (uint8_t vec_offset = 0; vec_offset < 3; vec_offset++){     // r, t \in R^3, passed in as [*r0, *t0, *r1, *t1, ...]
        r_shmem[block_pixel_id * 3 + vec_offset] = camera_starts_rays[global_pixel_id * 3 + vec_offset];
    }

    for (uint16_t g_offset = 0; g_offset < N_GAUSSIANS_PER_THREAD; g_offset++){ // each thread processes N_GAUSSIANS_PER_THREAD gaussians
        uint16_t gaussian_id = gaussian_id_offset + g_offset;
        for (uint8_t vec_offset = 0; vec_offset < 3; vec_offset++){     // meansI \in R^3
            means_shmem[gaussian_id * 3 + vec_offset] = meansI_arr[gaussian_id * 3 + vec_offset];
        }
        for (uint8_t vec_offset = 0; vec_offset < 9; vec_offset++){     // prc \in R^9
            prc_shmem[gaussian_id * 9 + vec_offset] = prc_arr[gaussian_id * 9 + vec_offset];
        }
        w_shmem[gaussian_id] = w_arr[gaussian_id];
    }

    // Initialize output buffers (size (height*width, )) to 0
    est_alpha_exp_factor_shmem[block_pixel_id] = 0;
    wgt_shmem[block_pixel_id] = 0;
    zs_final_unnormalized_shmem[block_pixel_id] = 0;

    __syncthreads();

    // load value from shmem
    float* r = &r_shmem[block_pixel_id * 3];
    float* t = &t_shmem[0];

    // each thread processes N_GAUSSIANS_PER_THREAD gaussians (which should be a relatively small number)
    for (uint16_t g_offset = 0; g_offset < N_GAUSSIANS_PER_THREAD; g_offset++){
        uint16_t gaussian_id = gaussian_id_offset + g_offset;
        float* meansI = &means_shmem[gaussian_id * 3];
        float* prc = &prc_shmem[gaussian_id * 9];
        float w = &w_shmem[gaussian_id];

        // shift the mean to be relative to ray start
        float p[3];
        thrust_prims::subtract_vec(meansI, t, p, 3);

        // compute \sigma^{-0.5} p, which is reused
        float projp[3];
        blas::matmul_331(prc, p, projp);

        // compute v^T \sigma^{-1} v
        float vsv_sqrt[3];  // prc @ r  (reuse)
        blas::matmul_331(r, projp, vsv_sqrt);
        float vsv;
        float _vsv[3];
        thrust_prims::pow2_vec(vsv_sqrt, _vsv, 3);
        thrust_prims::sum_vec(_vsv, vsv, 3);

        // compute p^T \sigma^{-1} v
        float _psv[3];
        float psv;
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
        float _std[3];
        blas::matmul_331(prc, v, _std);
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
    }
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
                    float* camera_starts_rays,  // (H*W, 3)
                    float* camera_trans, // (3, )
                    float beta_2, // float
                    float beta_3, // float
                    float* est_alpha_exp_factor,    // intermediate gpu (from kernel) output
                    float* wgt,     // intermediate gpu (from kernel) output
                    float* zs_final_unnormalized,
                    float* zs_final,    // final gpu (back to host) output
                    float* est_alpha,   // final gpu output
){

    // TODO it would be nice to reduce this number by culling gaussians with < threshold pdf value
    int total_num_intersections = height * width * num_gaussians;  // (# rays) x (# gaussians)

    // allocate threads. Each thread processes one pixel and (N_GAUSSIANS_PER_BLOCK/N_GAUSSIANS_THREAD) gaussians
    dim3 N_THREADS_PER_BLOCK(TILE_WIDTH, TILE_HEIGHT);  // align x, y order
    dim3 N_BLOCKS_PER_GRID(width/TILE_WIDTH, height/TILE_HEIGHT, N_GAUSSIANS_PER_BLOCK);

    printf("Launching kernel with (%d,%d,%d)=%d blocks per grid\n", N_BLOCKS_PER_GRID.x, N_BLOCKS_PER_GRID.y, N_BLOCKS_PER_GRID.z, N_BLOCKS_PER_GRID.x*N_BLOCKS_PER_GRID.y*N_BLOCKS_PER_GRID.z);
    printf("Launching kernel with (%d,%d,%d)=%d threads per block\n", N_THREADS_PER_BLOCK.x, N_THREADS_PER_BLOCK.y, N_THREADS_PER_BLOCK.z, N_THREADS_PER_BLOCK.x*N_THREADS_PER_BLOCK.y*N_THREADS_PER_BLOCK.z);

    // launch gaussian_ray kernel to fill in needed values for final calculations
    gaussian_ray_kernel<<<N_BLOCKS_PER_GRID, N_THREADS_PER_BLOCK>>>(
        height, weight, num_gaussians,
        prec_full,
        weights_log,
        means,
        camera_starts_rays,
        camera_trans,
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


void render_func_quat(
                    int img_height, int img_width, int num_gaussians,
                    float* means,  // (N, 3)
                    float* prec_full, // (N, 3, 3)
                    float* weights_log, // (N,)
                    float* _camera_rays,  // (H*W, 3)
                    float* rot, // (3, 3) // supply in rotation form not quaternion
                    float* trans, // (3, )
                    float beta_2, // float
                    float beta_3, // float
                    float* zs_final,    // final gpu output
                    float* est_alpha,   // final gpu output
                    GpuMemoryPool &memory_pool
                    ){
    // TODO assumes that everything passed in are GPU memory pointers (not host)
    float* est_alpha_exp_factor, // TODO GPU allocate (from workspace) to pass into kernel
    float* wgt,
    float* zs_final_unnormalized,
    /////////////////////////////

    float* camera_rays;
    cudaMalloc(camera_rays, img_height*img_width*3*sizeof(float)); // TODO use memory_pool
    blas::matmul(img_height*img_width, 3, 3, _camera_rays, rot, camera_rays);  // _camera_rays @ rot

    render_func_rays(height, width, num_gaussians, means, prec_full, weights_log,
                    camera_rays, trans, beta_2, beta_3, zs_final, est_alpha);
}
} // namespace fmb

////////////////////////////////////////////////////////////////////////////////
///          TESTING HARNESS                                                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
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
    std::vector<float> camera_starts_rays;  // (H*W*3, ) contiguous
    std::vector<float> camera_trans; // (3, )

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
    if (a.width != b.width || a.height != b.height || a.n_gaussians() != b.n_gaussians()) {
        return std::numeric_limits<float>::infinity();
    }

    float max_diff = 0.0f;
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        float diff_z = std::abs(a.z.at(idx) - b.z.at(idx));
        float diff_alpha = std::abs(a.alpha.at(idx) - b.alpha.at(idx));
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

    img_expected.zs = read_data('data/zs.bin', n_gaussians);
    img_expected.alphas = read_data('data/est_alpha.bin', n_gaussians);

    auto means_gpu = GpuBuf<float>(scene.means);
    auto prc_gpu = GpuBuf<float>(scene.prc);
    auto weights_gpu = GpuBuf<float>(scene.weights);
    auto camera_starts_rays_gpu = GpuBuf<float>(scene.camera_starts_rays);
    auto camera_trans_gpu = GpuBuf<float>(scene.camera_trans);
    auto beta_2_gpu = GpuBuf<float>(1);
    auto beta_3_gpu = GpuBuf<float>(1);
    auto zs_final_gpu = GpuBuf<float>(scene.height * scene.width);
    auto est_alpha_gpu = GpuBuf<float>(scene.height * scene.width);

    auto memory_pool = GpuMemoryPool();

    float beta_2 = 1.0;   // TODO
    float beta_3 = 1.0;

    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(zs_final_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            est_alpha_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(cudaMemset(beta_2_gpu.data, beta_2, sizeof(float)));
        CUDA_CHECK(cudaMemset(beta_3_gpu.data, beta_3, sizeof(float)));
        memory_pool.reset();
    };

    auto f = [&]() {
        fmb::render_func_quat(
            scene.width,
            scene.height,
            scene.n_gaussians(),
            means_gpu.data(),
            prc_gpu.data(),
            weights_gpu.data(),
            camera_starts_rays_gpu.data(),
            camera_trans.data(),
            beta_2_gpu.data()[0],
            beta_3_gpu.data()[0],
            zs_final_gpu.data(),
            est_alpha_gpu.data(),
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

constexpr float PI = 3.14159265359f;

Scene gen_simple() {
    /*
        Simple scene with 2 isotropic gaussians
    */
    int32_t width = 256;
    int32_t height = 256;
    int n_gaussians = 2;

    auto scene = Scene{width, height};

    scene.means = read_data('data/means.bin', 3 * n_gaussians);
    scene.prc = read_data('data/prc.bin', 9 * n_gaussians);
    scene.weights = read_data('data/weights.bin', n_gaussians);
    scene.camera_starts_rays = read_data('data/camera_starts_rays.bin', 3 * width * height);
    scene.camera_trans = read_data('data/camera_trans.bin', 3);

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
    scenes.push_back({"simple", Mode::TEST, gen_simple()});
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
