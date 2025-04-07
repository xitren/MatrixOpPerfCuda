#include <xitren/cuda/gemm_v00.cuh>
#include <xitren/math/matrix_alignment.hpp>

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

using namespace xitren::math;

void
print_device_info()
{
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Device Name: " << device_prop.name << std::endl;
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) / (1 << 30)};
    std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
    float const peak_bandwidth{static_cast<float>(2.0f * device_prop.memoryClockRate
                                                  * (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;
    std::cout << std::endl;
}

struct measurement {
    std::size_t us;
    std::size_t cycles;
};

// template <typename T,
//           typename std::enable_if<std::is_same<T, float>::value || std::is_same<T,
//           double>::value,
//                                   bool>::type
//           = true>
// measurement
// profile_gemm(size_t m, size_t k, size_t n, T const* A_host, T const* B_host, T* C_host)
// {
//     constexpr size_t       num_repeats = 10;
//     constexpr size_t       num_warmups = 10;
//     constexpr unsigned int seed        = 0U;

//     T const alpha{static_cast<T>(1.0)};
//     T const beta{static_cast<T>(0.0)};

//     float const  fp32_abs_tol{1.0e-3f};
//     double const fp32_rel_tol{0.0e-4f};

//     const size_t lda{k};
//     const size_t ldb{n};
//     const size_t ldc{n};

//     // static_assert(lda >= k, "");
//     // static_assert(ldb >= n, "");
//     // static_assert(ldc >= n, "");
//     cudaError_t err = cudaSuccess;

//     std::cout << "Matrix Size: "
//               << "M = " << m << " N = " << n << " K = " << k << std::endl;
//     std::cout << "Matrix A: " << m << " x " << k << std::endl;
//     std::cout << "Matrix B: " << k << " x " << n << std::endl;
//     std::cout << "Matrix C: " << m << " x " << n << std::endl;
//     std::cout << std::endl;

//     cudaStream_t stream;
//     err = cudaStreamCreate(&stream);
//     if (err != cudaSuccess) {
//         fprintf(stderr, "Failed to run stream (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }

//     data_parameters<T> params{m, k, n, alpha, beta, (T*)A_host, (T*)B_host, C_host};
//     copy_to_device(params);

//     launch_gemm_kernel_v00(params, stream);

//     copy_to_host(params, C_host);

//     err = cudaStreamDestroy(stream);
//     if (err != cudaSuccess) {
//         fprintf(stderr, "Failed to destroy stream (error code %s)!\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }
//     return measurement{0, 0};
// }

int
main(void)
{
    print_device_info();

    // Print the vector length to be used, and compute its size
    constexpr size_t mmSize      = 4096;
    constexpr size_t numElements = mmSize * mmSize;
    size_t           size        = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vectors A, B and output vector C
    auto h_A = matrix_aligned<float, mmSize, mmSize, optimization::cuda_naive>::get_rand_matrix(
        0.0, 1.0);
    auto h_B = matrix_aligned<float, mmSize, mmSize, optimization::cuda_naive>::get_rand_matrix(
        0.0, 1.0);
    auto h_C = matrix_aligned<float, mmSize, mmSize, optimization::cuda_naive>::get_zeros_matrix();

    std::cout << "Matrix Size: "
              << "M = " << mmSize << " N = " << mmSize << " K = " << mmSize << std::endl;
    std::cout << "Matrix A: " << mmSize << " x " << mmSize << std::endl;
    std::cout << "Matrix B: " << mmSize << " x " << mmSize << std::endl;
    std::cout << "Matrix C: " << mmSize << " x " << mmSize << std::endl;
    std::cout << std::endl;
    matrix_aligned<float, mmSize, mmSize, optimization::cuda_naive>::mult(*h_A, *h_B, *h_C);

    // cuda_prep(numElements, h_A->data_, h_B->data_, h_C->data_);
    // profile_gemm<float>(mmSize, mmSize, mmSize, h_A->data_, h_B->data_, h_C->data_);

    printf("Done\n");
    return 0;
}