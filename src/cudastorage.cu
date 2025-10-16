#include <cuda_runtime.h>
#include <iostream>
extern "C" {
float *cudanew(size_t n) {
  if (n == 0) {
    return nullptr;
  }

  const size_t size_in_bytes = n * sizeof(float);
  float *device_ptr = nullptr;

  // Allocate memory on the GPU device
  cudaError_t err = cudaMalloc((void **)&device_ptr, size_in_bytes);

  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: Failed to allocate memory - "
              << cudaGetErrorString(err) << std::endl;
    return nullptr; // Return null on failure, a common C-style error signal
  }

  return device_ptr;
}
}
