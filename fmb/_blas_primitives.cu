#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// Helper function to check CUDA errors
void checkCudaError(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Helper function to check cuBLAS errors
void checkCublasError(cublasStatus_t result, const char* msg) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << ": cuBLAS error code " << result << std::endl;
        exit(EXIT_FAILURE);
    }
}

void matmul_cpu_naive(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    for (int32_t i = 0; i < size_i; ++i) {
        for (int32_t j = 0; j < size_j; ++j) {
            float sum = 0.0;
            for (int32_t k = 0; k < size_k; ++k) {
                sum += a[i * size_k + k] * b[k * size_j + j];
            }
            c[i * size_j + j] = sum;
        }
    }
}

int main() {
    // Dimensions of the matrices
    int M = 3; // Rows of A and C
    int N = 4; // Columns of B and C
    int K = 3; // Columns of A, Rows of B

    // Allocate and initialize host matrices in row-major order
    float h_A[M * K] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float h_B[K * N] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    float h_C[N * M] = { 0 }; // Result matrix
    float h_C_host[M * N] = { 0 }; // Result matrix

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void**)&d_A, M * K * sizeof(float)), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc((void**)&d_B, K * N * sizeof(float)), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc((void**)&d_C, M * N * sizeof(float)), "Failed to allocate device memory for C");

    // Copy host matrices to device
    checkCudaError(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy B to device");

    // cuBLAS handle
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");

    // Set alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // A and B are in row-major -> use B^T @ A^T = C^T
    // (https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication)
    checkCublasError(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,CUBLAS_OP_N,
            N, M, K,                 // Dimensions of matrices
            &alpha,                  // Alpha
            d_B, N,                  // B (row-major) with leading dimension N
            d_A, K,                  // A (row-major) with leading dimension K
            &beta,                   // Beta
            d_C, N                   // C (row-major) with leading dimension N
        ),
        "Failed to execute cublasSgemm"
    );

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy C to host");

    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Compare with CPU result
    matmul_cpu_naive(M, N, K, h_A, h_B, h_C_host);
    std::cout << "Reference matrix C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C_host[i * N + j] << " ";
        }
        std::cout << std::endl;
    }



    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
