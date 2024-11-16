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
#include <thrust/sort.h>
#include <thrust/scan.h>

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

void render_cpu(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,
    float const *circle_y,
    float const *circle_radius,
    float const *circle_red,
    float const *circle_green,
    float const *circle_blue,
    float const *circle_alpha,
    float *img_red,
    float *img_green,
    float *img_blue) {

    // Initialize background to white
    for (int32_t pixel_idx = 0; pixel_idx < width * height; pixel_idx++) {
        img_red[pixel_idx] = 1.0f;
        img_green[pixel_idx] = 1.0f;
        img_blue[pixel_idx] = 1.0f;
    }

    // Render circles
    for (int32_t i = 0; i < n_circle; i++) {
        float c_x = circle_x[i];
        float c_y = circle_y[i];
        float c_radius = circle_radius[i];
        for (int32_t y = int32_t(c_y - c_radius); y <= int32_t(c_y + c_radius + 1.0f);
             y++) {
            for (int32_t x = int32_t(c_x - c_radius); x <= int32_t(c_x + c_radius + 1.0f);
                 x++) {
                float dx = x - c_x;
                float dy = y - c_y;
                if (!(0 <= x && x < width && 0 <= y && y < height &&
                      dx * dx + dy * dy < c_radius * c_radius)) {
                    continue;
                }
                int32_t pixel_idx = y * width + x;
                float pixel_red = img_red[pixel_idx];
                float pixel_green = img_green[pixel_idx];
                float pixel_blue = img_blue[pixel_idx];
                float pixel_alpha = circle_alpha[i];
                pixel_red =
                    circle_red[i] * pixel_alpha + pixel_red * (1.0f - pixel_alpha);
                pixel_green =
                    circle_green[i] * pixel_alpha + pixel_green * (1.0f - pixel_alpha);
                pixel_blue =
                    circle_blue[i] * pixel_alpha + pixel_blue * (1.0f - pixel_alpha);
                img_red[pixel_idx] = pixel_red;
                img_green[pixel_idx] = pixel_green;
                img_blue[pixel_idx] = pixel_blue;
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

constexpr int TILE_WIDTH = 16;
constexpr int TILE_HEIGHT = 16;

namespace circles_gpu {


__global__ void get_touched_tiles_kernel(uint32_t *touched_tiles,
                             int n_circle,
                             float const *circle_x,
                             float const *circle_y,
                             float const *circle_radius,
                             int width,
                             int height) {
    /*
        Get number of tiles that the circle's bounding box touches.
        (The exact number may be inexact; is an upper bound.)
    */

    uint32_t c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_idx >= n_circle) return;

    float x = circle_x[c_idx];
    float y = circle_y[c_idx];
    float r = circle_radius[c_idx];

    uint32_t x_min = x - r;
    uint32_t x_max = x + r;
    uint32_t y_min = y - r;
    uint32_t y_max = y + r;

    uint32_t x_min_tile = x_min / TILE_WIDTH;
    uint32_t x_max_tile = x_max / TILE_WIDTH;
    uint32_t y_min_tile = y_min / TILE_HEIGHT;
    uint32_t y_max_tile = y_max / TILE_HEIGHT;

    uint32_t num_touched_tiles = (x_max_tile - x_min_tile + 1) * (y_max_tile - y_min_tile + 1);

    touched_tiles[c_idx] = num_touched_tiles;
}

__global__ void get_tile_circle_keys(uint64_t *tile_circle_keys,
                             int n_circle,
                             int n_tile_circle,
                             float const *circle_x,
                             float const *circle_y,
                             float const *circle_radius,
                             int width,
                             int height,
                             uint32_t *touched_tiles,
                             uint32_t *psum_touched_tiles) {
    uint32_t c_idx = blockIdx.x * blockDim.x + threadIdx.x;  // circle idx
    if (c_idx >= n_circle) return;

    // circle info
    float x = circle_x[c_idx];
    float y = circle_y[c_idx];
    float r = circle_radius[c_idx];

    uint32_t x_min = x - r;
    uint32_t x_max = x + r;
    uint32_t y_min = y - r;
    uint32_t y_max = y + r;

    uint32_t x_min_tile = x_min / TILE_WIDTH;
    uint32_t x_max_tile = x_max / TILE_WIDTH;
    uint32_t y_min_tile = y_min / TILE_HEIGHT;
    uint32_t y_max_tile = y_max / TILE_HEIGHT;


    // tile-circle pair info
    uint32_t offset = (c_idx == 0) ? 0 : psum_touched_tiles[c_idx-1];  // offset in `n_tile_circle` array

    // for each tile-circle pair corresponding to the current circle,
    // create a key that is [tile | depth] and store in `tile_circle_keys`
    // at the appropriate offset.
    for (uint32_t x_tile = x_min_tile; x_tile <= x_max_tile; x_tile++){
        for (uint32_t y_tile = y_min_tile; y_tile <= y_max_tile; y_tile++){

            uint64_t key = y_tile * (width / TILE_WIDTH) + x_tile;
            key <<= 32;
            key |= c_idx;

            tile_circle_keys[offset] = key;
            offset++;
        }
    }
    // printf("Exiting get_tile_circle_keys \n");
}

__global__ void identify_tile_ranges(int n_tile_circle,
                                    uint64_t *tile_circle_keys,
                                    uint2 *ranges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tile_circle) return;

    uint64_t key = tile_circle_keys[idx];
    uint32_t tile_idx = key >> 32;

    if (idx == 0) {
        ranges[tile_idx].x = 0;
    }
    else {
        uint64_t prev_key = tile_circle_keys[idx-1];
        uint32_t prev_tile_idx = prev_key >> 32;

        if (tile_idx != prev_tile_idx) {
            ranges[prev_tile_idx].y = idx;
            ranges[tile_idx].x = idx;
        }
    }
    if (idx == n_tile_circle - 1) {
        ranges[tile_idx].y = n_tile_circle;
    }
}

__global__ void rasterize_tile_kernel(uint64_t *tile_circle_keys,
                             uint2 *tile_work_ranges,
                             int32_t width,
                             int32_t height,
                             uint32_t n_tile_circle,
                             float const *circle_x,
                             float const *circle_y,
                             float const *circle_radius,
                             float const *circle_red,
                             float const *circle_green,
                             float const *circle_blue,
                             float const *circle_alpha,
                             float *img_red,
                             float *img_green,
                             float *img_blue) {

    // partition shared memory block
    extern __shared__ float sh_mem[];

    float *shared_img_red = sh_mem;
    float *shared_img_green = &sh_mem[TILE_WIDTH * TILE_HEIGHT];
    float *shared_img_blue = &sh_mem[2 * TILE_WIDTH * TILE_HEIGHT];

    // identify tile/pixel location
    int linear_tile_idx = blockIdx.x;
    int pidx = threadIdx.x;  // pixel idx in TILE

    int x_tile = linear_tile_idx % (width / TILE_WIDTH);    // tile index in IMAGE
    int y_tile = linear_tile_idx / (width / TILE_WIDTH);    // tile index in IMAGE

    int x_pixel = x_tile * TILE_WIDTH + pidx % TILE_WIDTH;  // pixel idx in IMAGE
    int y_pixel = y_tile * TILE_HEIGHT + pidx / TILE_WIDTH; // pixel idx in IMAGE
    int linear_pixel_idx = y_pixel * width + x_pixel;
    if (x_pixel >= width || y_pixel >= height) return;

    // initialize shared memory value
    shared_img_red[pidx] = 1.0f;
    shared_img_green[pidx] = 1.0f;
    shared_img_blue[pidx] = 1.0f;

    // identify current workload of circles to render in tile
    uint32_t start_cidx = tile_work_ranges[blockIdx.x].x;
    uint32_t end_cidx = tile_work_ranges[blockIdx.x].y;    // exclusive range
    uint32_t num_circles_in_tile = end_cidx - start_cidx;

    // printf("Tile %d: start %d, end %d, num circles %d \n", linear_tile_idx, start_cidx, end_cidx, num_circles_in_tile);

    uint64_t prev_key = 0;
    // iterate over circles in tile in rendering order; if overlap with current pixel, blend.
    for (int c = 0; c < num_circles_in_tile; c++){
        // decode the tile-circle key
        uint32_t cidx = start_cidx + c;  // index into tile_circle_keys
        uint64_t key = tile_circle_keys[cidx];

        if (prev_key == key){
            continue;   // I'm not sure why there would ever be duplicate keys..
        }

        int circle_idx = key & 0xFFFFFFFF;

        // load circle information
        float x = circle_x[circle_idx];
        float y = circle_y[circle_idx];
        float r = circle_radius[circle_idx];

        float dx = x_pixel - x;
        float dy = y_pixel - y;

        // if pixel is within circle, blend
        if ((dx * dx + dy * dy) < r * r){
            // load rgba values
            float red = circle_red[circle_idx];
            float green = circle_green[circle_idx];
            float blue = circle_blue[circle_idx];
            float alpha = circle_alpha[circle_idx];

            shared_img_red[pidx] = red * alpha + shared_img_red[pidx] * (1.0f - alpha);
            shared_img_green[pidx] = green * alpha + shared_img_green[pidx] * (1.0f - alpha);
            shared_img_blue[pidx] = blue * alpha + shared_img_blue[pidx] * (1.0f - alpha);
        }

        prev_key = key;
    }

    // write back to global memory
    img_red[linear_pixel_idx] = shared_img_red[pidx];
    img_green[linear_pixel_idx] = shared_img_green[pidx];
    img_blue[linear_pixel_idx] = shared_img_blue[pidx];
}


void launch_render(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,      // pointer to GPU memory
    float const *circle_y,      // pointer to GPU memory
    float const *circle_radius, // pointer to GPU memory
    float const *circle_red,    // pointer to GPU memory
    float const *circle_green,  // pointer to GPU memory
    float const *circle_blue,   // pointer to GPU memory
    float const *circle_alpha,  // pointer to GPU memory
    float *img_red,             // pointer to GPU memory
    float *img_green,           // pointer to GPU memory
    float *img_blue,            // pointer to GPU memory
    GpuMemoryPool &memory_pool) {

    // sanity check
    assert(width % TILE_WIDTH == 0);
    assert(height % TILE_HEIGHT == 0);

    int num_blocks = (n_circle + 255) / 256;
    int num_threads = 256;

	// Compute prefix sum over full list of overlapped tile counts by circles
    // E.g., [2, 3, 1, 2, 1] -> [2, 5, 6, 8, 9]
    uint32_t *touched_tiles = reinterpret_cast<uint32_t *>(memory_pool.alloc(n_circle * sizeof(int)));
    uint32_t *psum_touched_tiles = reinterpret_cast<uint32_t *>(memory_pool.alloc(n_circle * sizeof(int)));
    get_touched_tiles_kernel<<<num_blocks, num_threads>>>(touched_tiles, n_circle, circle_x, circle_y, circle_radius, width, height);  // [2,3,1,2,1]
    thrust::inclusive_scan(thrust::device, touched_tiles, touched_tiles + n_circle, psum_touched_tiles);  // [2,5,6,8,9]

	// Retrieve total number of circle instances to launch and resize aux buffers
    uint32_t n_tile_circle;
    cudaMemcpy(&n_tile_circle, &psum_touched_tiles[n_circle - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    int num_blocks_tile_circle = (n_tile_circle + 255) / 256;
    int num_threads_tile_circle = 256;

	// For each tile-circle instance, encode into [ tile_id | circle_id ] key
    // ex) [2, 5, 6, 8, 9] -> [t0_0, t1_0, t2_1, t3_1, t4_1, t5_2, t6_3, t7_3, t7_4]
    // Where t[0-7] are specific, not necessarily unique tile indices for the tile-circle pair.
    uint64_t *tile_circle_keys = reinterpret_cast<uint64_t *>(memory_pool.alloc(n_tile_circle * sizeof(uint64_t)));
    get_tile_circle_keys<<<num_blocks, num_threads>>>(tile_circle_keys, n_circle, n_tile_circle,
                                                    circle_x, circle_y, circle_radius,
                                                    width, height,
                                                    touched_tiles,
                                                    psum_touched_tiles);


    // Sort tile-circle keys; result will be ordered by tile id, then by circle id (i.e. rendering order)
    thrust::sort(thrust::device, tile_circle_keys, tile_circle_keys + n_tile_circle); // in-place sort


    // Identify start and end indices of per-tile workloads in sorted tile-circle key list
    uint32_t total_num_tiles = (width / TILE_WIDTH) * (height / TILE_HEIGHT);
    uint2 *tile_work_ranges = reinterpret_cast<uint2 *>(memory_pool.alloc(total_num_tiles * sizeof(uint2)));
    identify_tile_ranges<<<num_blocks_tile_circle, num_threads_tile_circle>>>(n_tile_circle,
                                                        tile_circle_keys,
                                                        tile_work_ranges);

	// Rasterize the corresponding range of circles for each tile (and pixel), independently in parallel
    int num_threads_render = TILE_WIDTH * TILE_HEIGHT; // Do not change this without thought
    int num_blocks_render = (width * height) / num_threads_render;
    uint32_t shmem_byte_sizes = TILE_WIDTH * TILE_HEIGHT * 3 *sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(
        (&rasterize_tile_kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_byte_sizes)
    );
    rasterize_tile_kernel<<<num_blocks_render, num_threads_render, shmem_byte_sizes>>>(tile_circle_keys,
                                                        tile_work_ranges,
                                                        width, height,
                                                        n_tile_circle,
                                                        circle_x, circle_y, circle_radius,
                                                        circle_red, circle_green, circle_blue,
                                                        circle_alpha, img_red, img_green, img_blue);
}

} // namespace circles_gpu


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
    scenes.push_back({"simple", Mode::TEST, gen_simple()});
    scenes.push_back({"overlapping_opaque", Mode::TEST, gen_overlapping_opaque()});
    scenes.push_back(
        {"overlapping_transparent", Mode::TEST, gen_overlapping_transparent()});
    scenes.push_back(
        {"million_circles", Mode::BENCHMARK, gen_random(rng, 1024, 1024, 1'000'000)});

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
