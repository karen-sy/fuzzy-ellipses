#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


__global__ void add_vec(float* a, float* b, float* c, const int N){
    thrust::transform(thrust::device, a, a + N, b, c, thrust::plus<float>());
}

__global__ void subtract_vec(float* a, float* b, float* c, const int N){
    thrust::transform(thrust::device, a, a + N, b, c, thrust::minus<float>());
}

__global__ void multiply_vec(float* a, float* b, float* c, const int N){
    thrust::transform(thrust::device, a, a + N, b, c, thrust::multiplies<float>());
}

__global__ void divide_vec(float* a, float* b, float* c, const int N){
    thrust::transform(thrust::device, a, a + N, b, c, thrust::divides<float>());
}

__global__ void negate_vec(float* a, float* c, const int N){
    thrust::transform(thrust::device, a, a + N, c, thrust::negate<float>());
}

__global__ void exp_vec(float* a, float* c, const int N){
    auto ff = [] __device__ (float x) {return exp(x);};
    thrust::transform(thrust::device, a, a + N, c, ff);
}

__global__ void add_scalar(float* a, float b, float* c, const int N){
    auto ff = [b] __device__ (float x) {return x + b;};
    thrust::transform(thrust::device, a, a + N, c, ff);
}
