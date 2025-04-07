#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

template <typename T>
struct data_parameters {
    std::size_t m;
    std::size_t k;
    std::size_t n;

    T alpha;
    T beta;

    T* A;
    T* B;
    T* C;
};

template <typename T>
void
copy_to_device(data_parameters<T>& pars)
{
    cudaError_t  err    = cudaSuccess;
    float const* host_a = pars.A;
    float const* host_b = pars.B;
    float const* host_c = pars.C;

    // Allocate the device input vectors A, B and output vector C
    err = cudaMalloc((void**)&pars.A, pars.m * pars.k * sizeof(T));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&pars.B, pars.k * pars.n * sizeof(T));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&pars.C, pars.m * pars.n * sizeof(T));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input
    // vectors
    err = cudaMemcpy(pars.A, host_a, pars.m * pars.k * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(pars.B, host_b, pars.k * pars.n * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(pars.C, host_c, pars.m * pars.n * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void
copy_to_host(data_parameters<T>& pars, T* host_c)
{
    cudaError_t err = cudaSuccess;
    // Copy the host input vectors A and B in host memory to the device input
    // vectors
    err = cudaMemcpy(host_c, pars.C, pars.m * pars.n * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n",
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