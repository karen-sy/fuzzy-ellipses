#include <fstream>
#include <iostream>

#include "../fmb/blas_primitives.cuh"

// Helper function to check CUDA errors
void checkCudaError(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}


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

int main() {
    // Dimensions of the matrices
    // int M = 3; // Rows of A and C
    // int N = 4; // Columns of B and C
    // int K = 3; // Columns of A, Rows of B

    // // Allocate and initialize host matrices in row-major order
    // float h_A[M * K] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    // float h_B[K * N] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    // float h_C[N * M] = { 0 }; // Result matrix
    // float h_C_host[M * N] = { 0 }; // Result matrix

    int M = 480 * 480;
    int N = 3;
    int K = 3;

    std::vector<float> h_A = read_data("../data/camera_rays.bin", M*K);
    std::vector<float> h_B = read_data("../data/camera_rot.bin", K*N);
    float h_C[M * N] = {0.0f};
    std::vector<float> h_C_host = read_data("../data/camera_rays_xfm.bin", M * N);


    // // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void**)&d_A, M * K * sizeof(float)), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc((void**)&d_B, K * N * sizeof(float)), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc((void**)&d_C, M * N * sizeof(float)), "Failed to allocate device memory for C");

    // // Copy host matrices to device
    checkCudaError(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy B to device");

    // printf("before matmul");
    // // run cuBLAS matmul
    blas::matmul(M, N, K, d_A, d_B, d_C);
    blas::matmul(M, N, K, d_A, d_B, d_C); // run twice to make sure no cublas session issues

    // // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy C to host");

    printf("matmul done\n\n");
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < min(5, M); ++i) {   // just first five row/cols
        for (int j = 0; j < min(5, N); ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Compare with CPU result
    std::cout << "Reference matrix C:" << std::endl;
    for (int i = 0; i < min(5,M); ++i) {    // just first five row/cols
        for (int j = 0; j < min(5,N); ++j) {
            std::cout << h_C_host[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // RMSE
    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < N; ++j) {
            float diff = h_C_host[i * N + j] - h_C[i * N + j];
            mse += diff * diff;
            ref_mean_square += h_C[i * N + j] * h_C[i * N + j];
        }
    }
    mse /= M * N;
    ref_mean_square /= M * N;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);

    printf("size %d * %d * %d:\t", M, K, N);
    printf("correctness: %.02e relative RMSE\n", rel_rmse);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
