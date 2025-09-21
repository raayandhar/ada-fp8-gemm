#pragma once

#include "../common/types.cuh"

namespace ada_fp8_gemm::sm89 {

// Forward declaration of our main kernel function
// This is where you'll implement the actual CUDA kernel
__global__ void fp8_gemm_square_kernel(
    const fp8_t* B,    // N x N matrix B
    const fp8_t* C,    // N x N matrix C  
    const fp8_t* D,    // N x N matrix D
    fp8_t* A,          // N x N output matrix A = BC + D
    int N              // Matrix dimension (should be 1024 for now)
);

// Host wrapper function that launches the kernel
void launch_fp8_gemm_square(
    const fp8_t* B,
    const fp8_t* C, 
    const fp8_t* D,
    fp8_t* A,
    int N,
    cudaStream_t stream = nullptr
);

}
