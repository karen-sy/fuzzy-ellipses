#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// Helper function to check cuBLAS errors
void checkCublasError(cublasStatus_t result, const char* msg) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << ": cuBLAS error code " << result << std::endl;
        exit(EXIT_FAILURE);
    }
}

namespace blas {
__device__ void matmul_331(  float* A,
                        float* B,
                        float* C
){
    /* 3x3 @ 3x1 matrix multiplication helper */
    C[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
    C[1] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
    C[2] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2];
}

void matmul(int m, int n, int k,
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
