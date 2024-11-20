#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

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


int main(int argc, char **argv) {
    std::string test_data_dir = "../data";

    int num_pixels = 480*480;
    int num_gaussians = 150;

    auto size_i = 256;
    auto size_j = 256;
    auto size_k = 256;

    auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i) + "x" +
        std::to_string(size_j) + "x" + std::to_string(size_k);
    auto a = read_data(path_prefix + "_a.bin", size_i * size_k);
    auto b = read_data(path_prefix + "_b.bin", size_k * size_j);
    auto c = read_data(path_prefix + "_c.bin", size_i * size_j);

    auto zs = read_data(test_data_dir + "/zs.bin", num_pixels);
    auto alphas = read_data(test_data_dir + "/alphas.bin", num_pixels);
    auto camera_rot = read_data(test_data_dir + "/camera_rot.bin", 3*3);
    auto camera_trans = read_data(test_data_dir + "/camera_trans.bin", 3);
    auto camera_rays = read_data(test_data_dir + "/camera_rays.bin", num_pixels*3);
    auto means = read_data(test_data_dir + "/means.bin", num_gaussians*3);
    auto precs = read_data(test_data_dir + "/precs.bin", num_gaussians*3*3);
    auto weights = read_data(test_data_dir + "/weights.bin", num_gaussians);


    printf("a[0] = %f\n", a[0]);
    printf("b[0] = %f\n", b[0]);
    printf("c[0] = %f\n", c[0]);
    printf("zs[0] = %f\n", zs[0]);
    printf("alphas[0] = %f\n", alphas[0]);
    printf("camera_rot[0] = %f\n", camera_rot[0]);
    printf("camera_trans[0] = %f\n", camera_trans[0]);
    printf("camera_rays[0] = %f\n", camera_rays[0]);
    printf("means[0] = %f\n", means[0]);
    printf("precs[0] = %f\n", precs[0]);
    printf("weights[0] = %f\n", weights[0]);

    float *a_gpu;
    float *b_gpu;
    float *c_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size_i * size_k * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        b_gpu,
        b.data(),
        size_k * size_j * sizeof(float),
        cudaMemcpyHostToDevice));


    return 0;
}
