#pragma once
#ifndef FORWARD_H
#define FORWARD_H

#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace thrust_primitives {

__forceinline__ __device__ float nan_to_num(float x){
    if (isnan(x)){
        return 0.0f;
    } if (isinf(x)){
        return copysign(FLT_MAX, x);
    }
    return x;
}

__forceinline__ __device__ void prefix_sum_vec(float* a, float* c, const int N){
    // store prefix sum of a in c
    thrust::inclusive_scan(thrust::device, a, a + N, c);
}

__forceinline__ __device__ void sum_vec(float* a, float* c, const int N, const float init=0.0f){
    // store sum of a in c
    *c = thrust::reduce(thrust::device, a, a + N, init);
}

__forceinline__ __device__ void add_vec(float* a, float* b, float* c, const int N){
    thrust::transform(thrust::device, a, a + N, b, c, thrust::plus<float>());
}

__forceinline__ __device__ void subtract_vec(float* a, float* b, float* c, const int N){
    thrust::transform(thrust::device, a, a + N, b, c, thrust::minus<float>());
}

__forceinline__ __device__ void multiply_vec(float* a, float* b, float* c, const int N){
    thrust::transform(thrust::device, a, a + N, b, c, thrust::multiplies<float>());
}

__forceinline__ __device__ void divide_vec(float* a, float* b, float* c, const int N){
    thrust::transform(thrust::device, a, a + N, b, c, thrust::divides<float>());
}

__forceinline__ __device__ void negate_vec(float* a, float* c, const int N){
    thrust::transform(thrust::device, a, a + N, c, thrust::negate<float>());
}

__forceinline__ __device__ void exp_vec(float* a, float* c, const int N){
    auto ff = [] __device__ (float x) {return exp(x);};
    thrust::transform(thrust::device, a, a + N, c, ff);
}

__forceinline__ __device__ void pow2_vec(float* a, float* c, const int N){
    auto ff = [] __device__ (float x) {return pow(x, 2);};
    thrust::transform(thrust::device, a, a + N, c, ff);
}

__forceinline__ __device__ void add_scalar(float* a, float b, float* c, const int N){
    auto ff = [b] __device__ (float x) {return x + b;};
    thrust::transform(thrust::device, a, a + N, c, ff);
}

__forceinline__ __device__ void sub_scalar(float* a, float b, float* c, const int N){
    auto ff = [b] __device__ (float x) {return x - b;};
    thrust::transform(thrust::device, a, a + N, c, ff);
}

__forceinline__ __device__ void mul_scalar(float* a, float b, float* c, const int N){
    auto ff = [b] __device__ (float x) {return x * b;};
    thrust::transform(thrust::device, a, a + N, c, ff);
}

__forceinline__ __device__ void positive_mask_vec(float* a, float* c, const int N){
    auto ff = [] __device__ (float x) {return x > 0 ? 1.0f : 0.0f;};
    thrust::transform(thrust::device, a, a + N, c, ff);
}

__forceinline__ __device__ void clipped_exp(float* a, float* c, const int N){
    auto ff = [] __device__ (float x) {return nan_to_num(exp(x));};
    thrust::transform(thrust::device, a, a + N, c, ff);
}

} // namespace thrust_primitives

#endif  // FORWARD_H
