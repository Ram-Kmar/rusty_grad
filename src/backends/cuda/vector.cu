#include <cuda_runtime.h>
#include <iostream>

// A simple helper function to check for CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

/**
 * CUDA Kernel: addVectors
 * This function runs on the GPU. Each thread executes this function.
 * * @param a Pointer to the first input array in device memory.
 * @param b Pointer to the second input array in device memory.
 * @param c Pointer to the output array in device memory.
 * @param n The total number of elements in the arrays.
 */
__global__ void addVectors(const int *a, const int *b, int *c, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}
int *new(const int n) {

  const size_t size = n * sizeof(int);
  int *d_a, *d_b, *d_c;
  checkCudaError(cudaMalloc((void **)&d_a, size), "cudaMalloc for a");
  checkCudaError(cudaMalloc((void **)&d_b, size), "cudaMalloc for b");
  checkCudaError(cudaMalloc((void **)&d_c, size), "cudaMalloc for c");
  return d_a, d_b, d_c;
}

extern "C" void add_vectors_wrapper(const int *h_a, const int *h_b, int *h_c,
                                    int n) {
  const size_t size = n * sizeof(int);

  int *d_a, *d_b, *d_c;
  checkCudaError(cudaMalloc((void **)&d_a, size), "cudaMalloc for a");
  checkCudaError(cudaMalloc((void **)&d_b, size), "cudaMalloc for b");
  checkCudaError(cudaMalloc((void **)&d_c, size), "cudaMalloc for c");

  checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice),
                 "cudaMemcpy from h_a to d_a");
  checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice),
                 "cudaMemcpy from h_b to d_b");

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

  checkCudaError(cudaGetLastError(), "Kernel launch");
  checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  checkCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost),
                 "cudaMemcpy from d_c to h_c");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
