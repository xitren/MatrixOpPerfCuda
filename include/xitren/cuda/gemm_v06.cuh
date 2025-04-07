#pragma once
#include <xitren/cuda/gemm_utils.cuh>
#include <xitren/cuda/general.cuh>
#include <xitren/math/gemm_core.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

#include <iostream>

using namespace xitren::math;

// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
// https://github.com/NVIDIA/cutlass/blob/b7508e337938137a699e486d8997646980acfc58/media/docs/programming_guidelines.md

// GEMM kernel v06.
// Each thread in the block processes THREAD_TILE_SIZE_Y *
// THREAD_TILE_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K,
          size_t BLOCK_TILE_SKEW_SIZE_X, size_t BLOCK_TILE_SKEW_SIZE_Y, size_t WARP_TILE_SIZE_X,
          size_t WARP_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_X, size_t WMMA_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_K,
          size_t NUM_THREADS>
__global__ void
gemm_v06(data_parameters<T> pars)
{
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U, "");
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U, "");

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X];

    constexpr size_t NUM_WMMA_TILES_X{WARP_TILE_SIZE_X / WMMA_TILE_SIZE_X};
    static_assert(WARP_TILE_SIZE_X % WMMA_TILE_SIZE_X == 0U, "");
    constexpr size_t NUM_WMMA_TILES_Y{WARP_TILE_SIZE_Y / WMMA_TILE_SIZE_Y};
    static_assert(WARP_TILE_SIZE_Y % WMMA_TILE_SIZE_Y == 0U, "");
    constexpr size_t NUM_WMMA_TILES_K{BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K};
    static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0U, "");

    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::col_major>
        a_frags[NUM_WMMA_TILES_Y];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::row_major>
        b_frags[NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T>
        acc_frags[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T> c_frag;

// Make sure the accumulator starts from 0.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx) {
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx) {
            nvcuda::wmma::fill_fragment(acc_frags[wmma_tile_row_idx][wmma_tile_col_idx], static_cast<T>(0));
        }
    }

    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
    size_t const warp_linear_idx{thread_linear_idx / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(pars.k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K};

    for (size_t thread_block_tile_idx{0U}; thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx) {
        load_data_from_global_memory_to_shared_memory_transposed<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                                                 BLOCK_TILE_SIZE_K, NUM_THREADS, BLOCK_TILE_SKEW_SIZE_X,
                                                                 BLOCK_TILE_SKEW_SIZE_Y>(
            pars.A, pars.k, pars.B, pars.n, A_thread_block_tile_transposed, B_thread_block_tile, thread_block_tile_idx,
            thread_linear_idx, pars.m, pars.n, pars.k);
        __syncthreads();

// Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
// thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
// shared memory as A_thread_block_tile and B_thread_block_tile,
// respectively. This inner product is further decomposed to
// BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
// B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
// A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
// both A_thread_block_tile and B_thread_block_tile can be cached in the
// register.
#pragma unroll
        for (size_t k_i{0U}; k_i < NUM_WMMA_TILES_K; ++k_i) {
#pragma unroll
            for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx) {
                nvcuda::wmma::load_matrix_sync(
                    a_frags[wmma_tile_row_idx],
                    &A_thread_block_tile_transposed[k_i * WMMA_TILE_SIZE_K][warp_row_idx * WARP_TILE_SIZE_Y
                                                                            + wmma_tile_row_idx * WMMA_TILE_SIZE_Y],
                    BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y);
            }
#pragma unroll
            for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx) {
                nvcuda::wmma::load_matrix_sync(
                    b_frags[wmma_tile_col_idx],
                    &B_thread_block_tile[k_i * WMMA_TILE_SIZE_K]
                                        [warp_col_idx * WARP_TILE_SIZE_X + wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                    BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X);
            }
#pragma unroll
            for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx) {
#pragma unroll
                for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx) {
                    // Perform the matrix multiplication.
                    nvcuda::wmma::mma_sync(acc_frags[wmma_tile_row_idx][wmma_tile_col_idx], a_frags[wmma_tile_row_idx],
                                           b_frags[wmma_tile_col_idx], acc_frags[wmma_tile_row_idx][wmma_tile_col_idx]);
                }
            }
        }
        __syncthreads();
    }

// Write the results to DRAM.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx) {
#pragma unroll
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx) {
            // Load the fragment from global memory.
            nvcuda::wmma::load_matrix_sync(c_frag,
                                           &pars.C[(blockIdx.y * BLOCK_TILE_SIZE_Y + warp_row_idx * WARP_TILE_SIZE_Y
                                                    + wmma_tile_row_idx * WMMA_TILE_SIZE_Y)
                                                       * pars.n
                                                   + blockIdx.x * BLOCK_TILE_SIZE_X + warp_col_idx * WARP_TILE_SIZE_X
                                                   + wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                                           pars.n, nvcuda::wmma::mem_row_major);
            // Perform scaling and addition.
            for (size_t i{0}; i < c_frag.num_elements; ++i) {
                c_frag.x[i]
                    = pars.alpha * acc_frags[wmma_tile_row_idx][wmma_tile_col_idx].x[i] + pars.beta * c_frag.x[i];
            }
            // Store the fragment back to global memory.
            nvcuda::wmma::store_matrix_sync(&pars.C[(blockIdx.y * BLOCK_TILE_SIZE_Y + warp_row_idx * WARP_TILE_SIZE_Y
                                                     + wmma_tile_row_idx * WMMA_TILE_SIZE_Y)
                                                        * pars.n
                                                    + blockIdx.x * BLOCK_TILE_SIZE_X + warp_col_idx * WARP_TILE_SIZE_X
                                                    + wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                                            c_frag, pars.n, nvcuda::wmma::mem_row_major);
        }
    }
}

template <typename T>
void
launch_gemm_kernel_v06(data_parameters<T> pars, cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    constexpr unsigned int WARP_TILE_SIZE_X{32U};
    constexpr unsigned int WARP_TILE_SIZE_Y{64U};
    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U, "");
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U, "");

    // The skew size is used to avoid bank conflicts in shared memory.
    constexpr size_t BLOCK_TILE_SKEW_SIZE_X{16U};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_Y{16U};

    constexpr unsigned int WMMA_TILE_SIZE_X{16U};
    constexpr unsigned int WMMA_TILE_SIZE_Y{16U};
    constexpr unsigned int WMMA_TILE_SIZE_K{16U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_WARPS_X * NUM_WARPS_Y * 32U};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(pars.n) + BLOCK_TILE_SIZE_X - 1U) / BLOCK_TILE_SIZE_X,
                        (static_cast<unsigned int>(pars.m) + BLOCK_TILE_SIZE_Y - 1U) / BLOCK_TILE_SIZE_Y, 1U};
    gemm_v06<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, BLOCK_TILE_SKEW_SIZE_X, BLOCK_TILE_SKEW_SIZE_Y,
             WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_K,
             NUM_THREADS_PER_BLOCK><<<grid_dim, block_dim, 0U, stream>>>(pars);
}

namespace xitren {
namespace math {

template <std::uint_fast32_t Rows, std::uint_fast32_t Columns>
class gemm_core<Rows, Columns, float, optimization::cuda_2d_block_2d_warp_2d_thread_transpose_wmma>
    : gemm_core<Rows, Columns, float, optimization::naive> {
    static_assert(std::is_same<T, float>() || std::is_same<T, double>(), "");

public:
    template <std::uint_fast32_t Other>
    static void
    mult(float const* a, float const* b, float* c) noexcept
    {
        // constexpr size_t       num_repeats = 10;
        // constexpr size_t       num_warmups = 10;
        // constexpr unsigned int seed        = 0U;

        float const alpha{static_cast<float>(1.0)};
        float const beta{static_cast<float>(0.0)};

        // float const  fp32_abs_tol{1.0e-3f};
        // double const fp32_rel_tol{0.0e-4f};

        // const size_t lda{Other};
        // const size_t ldb{Columns};
        // const size_t ldc{Columns};

        cudaError_t err = cudaSuccess;

        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to run stream (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        data_parameters<float> params{Rows, Other, Columns, alpha, beta, (float*)a, (float*)b, c};
        copy_to_device<float>(params);

        launch_gemm_kernel_v06<float>(params, stream);

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
