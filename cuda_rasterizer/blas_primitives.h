#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#ifndef BLAS_H
#define BLAS_H

// Helper function to check cuBLAS errors
inline void checkCublasError(cublasStatus_t result, const char* msg) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << ": cuBLAS error code " << result << std::endl;
        exit(EXIT_FAILURE);
    }
}

namespace blas {
__forceinline__ __device__ void matmul_331(  float* A,
                        float* B,
                        float* C
){
    /* 3x3 @ 3x1 matrix multiplication helper */
    C[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
    C[1] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
    C[2] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2];
}

__forceinline__ __device__ void matmul_333(  float* A,
                        float* B,
                        float* C
){
    /* 3x3 @ 3x3 matrix multiplication helper */
    C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];

    C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];

    C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}


inline void matmul(int m, int n, int k,
            float *A, float *B, float *C  // gpu arrays
            ) {
    /* Perform A@B = C, of dimensions (m,k)x(k,n)=(m,n) */

    // Set alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS handle
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");


    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // A and B are in row-major -> use B^T @ A^T = C^T
    // (https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication)
    checkCublasError(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,CUBLAS_OP_N,
            n,m,k,                 // Dimensions of matrices
            &alpha,                  // Alpha
            B, n,                  // B (row-major) with leading dimension N
            A, k,                  // A (row-major) with leading dimension K
            &beta,                   // Beta
            C, n                   // C (row-major) with leading dimension N
        ),
        "Failed to execute cublasSgemm"
    );

    // Cleanup
    cublasDestroy(handle);      // TODO assumes this function will not be called very frequently
}

} // namespace blas


#endif // BLAS_H
