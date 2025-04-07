#pragma once
#include <xitren/cuda/gemm_utils.cuh>
#include <xitren/cuda/general.cuh>
#include <xitren/math/gemm_core.hpp>

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

using namespace xitren::math;

// GEMM kernel v03.
// Coalesced read and write from global memory.
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K,
          size_t THREAD_TILE_SIZE_Y>
__global__ void
gemm_v03(data_parameters<T> pars)
{
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
    size_t const     thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(pars.k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K};

    // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
    // Specifically, these values corresponds to
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X]
    T C_thread_results[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};

    for (size_t thread_block_tile_idx{0U}; thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx) {
        load_data_from_global_memory_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                                                      NUM_THREADS>(pars.A, pars.k, pars.B, pars.n, A_thread_block_tile,
                                                                   B_thread_block_tile, thread_block_tile_idx,
                                                                   thread_linear_idx, pars.m, pars.n, pars.k);
        __syncthreads();

#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
            size_t const B_thread_block_tile_row_idx{k_i};
            // B_val is cached in the register to alleviate the pressure on the
            // shared memory access.
            T const B_val{B_thread_block_tile[B_thread_block_tile_row_idx][thread_linear_idx % BLOCK_TILE_SIZE_X]};
#pragma unroll
            for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx) {
                size_t const A_thread_block_tile_row_idx{thread_linear_idx / BLOCK_TILE_SIZE_X * THREAD_TILE_SIZE_Y
                                                         + thread_tile_row_idx};
                size_t const A_thread_block_tile_col_idx{k_i};
                T const      A_val{A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx]};
                C_thread_results[thread_tile_row_idx] += A_val * B_val;
            }
        }
        __syncthreads();
    }

// Write the results to DRAM.
#pragma unroll
    for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx) {
        size_t const C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y
                               + thread_linear_idx / BLOCK_TILE_SIZE_X * THREAD_TILE_SIZE_Y + thread_tile_row_idx};
        size_t const C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X + thread_linear_idx % BLOCK_TILE_SIZE_X};
        if (C_row_idx < pars.m && C_col_idx < pars.n) {
            pars.C[C_row_idx * pars.n + C_col_idx] = pars.alpha * C_thread_results[thread_tile_row_idx]
                                                     + pars.beta * pars.C[C_row_idx * pars.n + C_col_idx];
        }
    }
}

template <typename T>
void
launch_gemm_kernel_v03(data_parameters<T> pars, cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    // Each thread computes THREAD_TILE_SIZE_Y values of C.
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};
    constexpr unsigned int NUM_THREADS_PER_BLOCK{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U, "");
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U, "");
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U, "");
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(pars.n) + BLOCK_TILE_SIZE_X - 1U) / BLOCK_TILE_SIZE_X,
                        (static_cast<unsigned int>(pars.m) + BLOCK_TILE_SIZE_Y - 1U) / BLOCK_TILE_SIZE_Y, 1U};
    gemm_v03<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(pars);
}

namespace xitren {
namespace math {

template <std::uint_fast32_t Rows, std::uint_fast32_t Columns, typename T>
class gemm_core<Rows, Columns, T, optimization::cuda_2d_block_1d_thread>
    : gemm_core<Rows, Columns, T, optimization::naive> {
    static_assert(std::is_same<T, float>() || std::is_same<T, double>(), "");

public:
    template <std::uint_fast32_t Other>
    static void
    mult(T const* a, T const* b, T* c) noexcept
    {
        T const alpha{static_cast<T>(1.0)};
        T const beta{static_cast<T>(0.0)};

        cudaError_t err = cudaSuccess;

        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to run stream (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        data_parameters<T> params{Rows, Other, Columns, alpha, beta, (T*)a, (T*)b, c};
        copy_to_device<T>(params);

        launch_gemm_kernel_v03<T>(params, stream);

        copy_to_host<T>(params, c);

        err = cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to destroy stream (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
};

}    // namespace math
}    // namespace xitren
