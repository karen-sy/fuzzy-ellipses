#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


#include "../fmb/thrust_primitives.cuh"

constexpr int N = 10;

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
    (cudaMemcpy(&C[0], &c[0], sizeof(float) * N, cudaMemcpyDeviceToHost));

    print_host_vec(C);

    // free memory on the host
    delete[] C;
}

__global__ void test_launch_operations(float* a, float* b,
                                float* c_add,
                                float* c_subtract,
                                float* c_multiply,
                                float* c_divide,
                                float* c_negate,
                                float* c_exp,
                                float* c_pow2,
                                float* c_add_scalar,
                                float* c_mul_scalar,
                                float* c_mask,
                                float* c_clipped_exp,
                                float b_copy,  // on host
                                float* c_sum
                                ){
    thrust_primitives::add_vec(a, b, c_add, N);

    thrust_primitives::subtract_vec(a, b, c_subtract, N);

    thrust_primitives::multiply_vec(a, b, c_multiply, N);

    thrust_primitives::divide_vec(a, b, c_divide, N);

    thrust_primitives::negate_vec(a, c_negate, N);

    thrust_primitives::exp_vec(a, c_exp, N);

    thrust_primitives::pow2_vec(a, c_pow2, N);

    thrust_primitives::add_scalar(a, b_copy, c_add_scalar, N);

    thrust_primitives::mul_scalar(a, b_copy, c_mul_scalar, N);

    thrust_primitives::positive_mask_vec(a, c_mask, N);

    thrust_primitives::clipped_exp(a, c_clipped_exp, N);

    thrust_primitives::sum_vec(a, c_sum, N);
}


int main() {
    // Host vectors
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);

    for (float i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>((i - 2));
        h_B[i] = static_cast<float>(10.0f * (i + 1));
    }

    // Device vectors
    thrust::device_vector<float> d_A(h_A.size());
    thrust::device_vector<float> d_B(h_B.size());
    thrust::device_vector<float> d_C(h_A.size() * 12); // To store the result

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

    // launch on-device thrust operations
    test_launch_operations<<<1,5>>>(a, b, c, &c[N], &c[2*N], &c[3*N], &c[4*N], &c[5*N], &c[6*N], &c[7*N], &c[8*N], &c[9*N], &c[10*N], h_B[0], &c[11*N]);

    print_vec(c, "add_vec");
    print_vec(&c[N], "subtract_vec");
    print_vec(&c[2*N], "multiply_vec");
    print_vec(&c[3*N], "divide_vec");
    print_vec(&c[4*N], "negate_vec");
    print_vec(&c[5*N], "exp_vec");
    print_vec(&c[6*N], "pow2_vec");
    print_vec(&c[7*N], "add_scalar (a + b[0])");
    print_vec(&c[8*N], "mul_scalar (a * b[0])");
    print_vec(&c[9*N], "positive_mask_vec (a)");
    print_vec(&c[10*N], "clipped_exp_vec");
    print_vec(&c[11*N], "sum_vec (a)");

    return 0;
}
