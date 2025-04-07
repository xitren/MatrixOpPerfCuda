#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <xitren/math/matrix_alignment.hpp>

using namespace xitren::math;

template <typename T> struct data_parameters {
  std::size_t m;
  std::size_t k;
  std::size_t n;

  T alpha;
  T beta;

  T *A;
  T *B;
  T *C;
};

template <typename T> __global__ void gemm_v00(data_parameters<T> pars) {
  // Compute the row and column of C that this thread is responsible for.
  size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
  size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

  // Each thread compute
  // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
  // beta * C[C_row_idx, C_col_idx].
  if (C_row_idx < pars.m && C_col_idx < pars.n) {
    T sum{static_cast<T>(0)};
    for (size_t k_idx{0U}; k_idx < pars.k; ++k_idx) {
      sum += pars.A[C_row_idx * pars.k + k_idx] *
             pars.B[k_idx * pars.n + C_col_idx];
    }
    pars.C[C_row_idx * pars.n + C_col_idx] =
        pars.alpha * sum + pars.beta * pars.C[C_row_idx * pars.n + C_col_idx];
  }
}

template <typename T>
void launch_gemm_kernel_v00(data_parameters<T> pars, cudaStream_t stream) {
  dim3 const block_dim{32U, 32U, 1U};
  dim3 const grid_dim{
      (static_cast<unsigned int>(pars.m) + block_dim.x - 1U) / block_dim.x,
      (static_cast<unsigned int>(pars.n) + block_dim.y - 1U) / block_dim.y, 1U};
  gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(pars);
}

void print_device_info() {
  int device_id{0};
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  std::cout << "Device Name: " << device_prop.name << std::endl;
  float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                          (1 << 30)};
  std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
  float const peak_bandwidth{
      static_cast<float>(2.0f * device_prop.memoryClockRate *
                         (device_prop.memoryBusWidth / 8) / 1.0e6)};
  std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;
  std::cout << std::endl;
}

template <typename T> void copy_to_device(data_parameters<T> &pars) {
  cudaError_t err = cudaSuccess;
  const float *host_a = pars.A;
  const float *host_b = pars.B;
  const float *host_c = pars.C;

  // Allocate the device input vectors A, B and output vector C
  err = cudaMalloc((void **)&pars.A, pars.m * pars.k * sizeof(T));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&pars.B, pars.k * pars.n * sizeof(T));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&pars.C, pars.m * pars.n * sizeof(T));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors A and B in host memory to the device input
  // vectors
  err = cudaMemcpy(pars.A, host_a, pars.m * pars.k * sizeof(T),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(pars.B, host_b, pars.k * pars.n * sizeof(T),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(pars.C, host_c, pars.m * pars.n * sizeof(T),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template <typename T> void copy_to_host(data_parameters<T> &pars, T *host_c) {
  cudaError_t err = cudaSuccess;
  // Copy the host input vectors A and B in host memory to the device input
  // vectors
  err = cudaMemcpy(host_c, pars.C, pars.m * pars.n * sizeof(T),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free device global memory
  err = cudaFree(pars.A);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(pars.B);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(pars.C);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

struct measurement {
  std::size_t us;
  std::size_t cycles;
};

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value,
                                  bool>::type = true>
measurement profile_gemm(size_t m, size_t k, size_t n, T const *A_host,
                         T const *B_host, T *C_host) {
  constexpr size_t num_repeats = 10;
  constexpr size_t num_warmups = 10;
  constexpr unsigned int seed = 0U;

  T const alpha{static_cast<T>(1.0)};
  T const beta{static_cast<T>(0.0)};

  float const fp32_abs_tol{1.0e-3f};
  double const fp32_rel_tol{0.0e-4f};

  const size_t lda{k};
  const size_t ldb{n};
  const size_t ldc{n};

  // static_assert(lda >= k, "");
  // static_assert(ldb >= n, "");
  // static_assert(ldc >= n, "");
  cudaError_t err = cudaSuccess;

  std::cout << "Matrix Size: "
            << "M = " << m << " N = " << n << " K = " << k << std::endl;
  std::cout << "Matrix A: " << m << " x " << k << std::endl;
  std::cout << "Matrix B: " << k << " x " << n << std::endl;
  std::cout << "Matrix C: " << m << " x " << n << std::endl;
  std::cout << std::endl;

  cudaStream_t stream;
  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to run stream (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  data_parameters<T> params{m,    k,           n,           alpha,
                            beta, (T *)A_host, (T *)B_host, C_host};
  copy_to_device(params);

  launch_gemm_kernel_v00(params, stream);

  copy_to_host(params, C_host);

  err = cudaStreamDestroy(stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to destroy stream (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return measurement{0, 0};
}

int main(void) {
  print_device_info();

  // Print the vector length to be used, and compute its size
  constexpr size_t mmSize = 4096;
  constexpr size_t numElements = mmSize * mmSize;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vectors A, B and output vector C
  auto h_A = matrix_aligned<float, mmSize, mmSize,
                            optimization::naive>::get_rand_matrix(0.0, 1.0);
  auto h_B = matrix_aligned<float, mmSize, mmSize,
                            optimization::naive>::get_rand_matrix(0.0, 1.0);
  auto h_C = matrix_aligned<float, mmSize, mmSize,
                            optimization::naive>::get_zeros_matrix();

  // cuda_prep(numElements, h_A->data_, h_B->data_, h_C->data_);
  profile_gemm<float>(mmSize, mmSize, mmSize, h_A->data_, h_B->data_,
                      h_C->data_);

  printf("Done\n");
  return 0;
}