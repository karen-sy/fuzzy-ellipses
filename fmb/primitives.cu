#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#include "primitives.cuh"

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


////////////////////////////////////////////////////////////////////

constexpr int N = 5; 

void print_host_vec(float* c){
    for (int i = 0; i < N; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;
}

void print_vec(float* c, std::string name){
    std::cout << name << ": ";
    
    // Allocate memory on the host to copy back gpu mem
    float* C = new float[N];  
    CUDA_CHECK(cudaMemcpy(&C[0], &c[0], sizeof(float) * N, cudaMemcpyDeviceToHost));

    print_host_vec(C);

    // free memory on the host
    delete[] C;
}

void launch_operations(float* a, float* b, float* c){
    add_vec<<<1,1>>>(a, b, c, N);
    print_vec(c, "add_vec");
    
    subtract_vec<<<1, 1>>>(a, b, c, N);
    print_vec(c, "subtract_vec");

    multiply_vec<<<1,1>>>(a, b, c, N);
    print_vec(c, "multiply_vec");
    
    divide_vec<<<1,1>>>(a, b, c, N);
    print_vec(c, "divide_vec");

    negate_vec<<<1,1>>>(a, c, N);
    print_vec(c, "negate_vec (-a)");

    exp_vec<<<1,1>>>(a, c, N);
    print_vec(c, "exp_vec");
    
    
    float b_copy; // need to be on host for constant iterator
    cudaMemcpy(&b_copy, &b[0], sizeof(float), cudaMemcpyDeviceToHost);
    add_scalar<<<1,1>>>(a, b_copy, c, N);
    print_vec(c, "add_scalar (a + b[0])");
}


int main() {
    // Host vectors
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);

    for (float i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>((i + 1));
        h_B[i] = static_cast<float>(10.0f * (i + 1));
    } 

    // Device vectors
    thrust::device_vector<float> d_A(h_A.size());
    thrust::device_vector<float> d_B(h_B.size());
    thrust::device_vector<float> d_C(h_A.size()); // To store the result

    // Copy data from host to device
    thrust::copy(h_A.begin(), h_A.end(), d_A.begin());
    thrust::copy(h_B.begin(), h_B.end(), d_B.begin());

    // launch operations
    float* a = thrust::raw_pointer_cast(d_A.data());
    float* b = thrust::raw_pointer_cast(d_B.data());
    float* c = thrust::raw_pointer_cast(d_C.data());

    std::cout << "A: ";
    print_host_vec(h_A.data());
    std::cout << "B: ";
    print_host_vec(h_B.data());
    std::cout << "--------------------------------" << std::endl;

    launch_operations(a, b, c);

    return 0;
}