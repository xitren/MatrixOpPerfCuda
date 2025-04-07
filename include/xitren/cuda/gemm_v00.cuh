#pragma once
#include <xitren/cuda/general.cuh>
#include <xitren/math/gemm_core.hpp>

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

using namespace xitren::math;

template <typename T>
__global__ void
gemm_v00(data_parameters<T> pars)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < pars.m && C_col_idx < pars.n) {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < pars.k; ++k_idx) {
            sum += pars.A[C_row_idx * pars.k + k_idx] * pars.B[k_idx * pars.n + C_col_idx];
        }
        pars.C[C_row_idx * pars.n + C_col_idx]
            = pars.alpha * sum + pars.beta * pars.C[C_row_idx * pars.n + C_col_idx];
    }
}

template <typename T>
void
launch_gemm_kernel_v00(data_parameters<T> pars, cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(pars.m) + block_dim.x - 1U) / block_dim.x,
                        (static_cast<unsigned int>(pars.n) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(pars);
}

namespace xitren {
namespace math {

template <std::uint_fast32_t Rows, std::uint_fast32_t Columns>
class gemm_core<Rows, Columns, float, optimization::cuda_naive>
    : gemm_core<Rows, Columns, float, optimization::naive> {

public:
    template <std::uint_fast32_t Other>
    static void
    mult(float const* a, float const* b, float* c) noexcept
    {
        constexpr size_t       num_repeats = 10;
        constexpr size_t       num_warmups = 10;
        constexpr unsigned int seed        = 0U;

        float const alpha{static_cast<float>(1.0)};
        float const beta{static_cast<float>(0.0)};

        float const  fp32_abs_tol{1.0e-3f};
        double const fp32_rel_tol{0.0e-4f};

        const size_t lda{Other};
        const size_t ldb{Columns};
        const size_t ldc{Columns};

        cudaError_t err = cudaSuccess;

        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to run stream (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        data_parameters<float> params{Rows, Other, Columns, alpha, beta, (float*)a, (float*)b, c};
        copy_to_device<float>(params);

        launch_gemm_kernel_v00<float>(params, stream);

        copy_to_host<float>(params, c);

        err = cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to destroy stream (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
};

}    // namespace math
}    // namespace xitren
