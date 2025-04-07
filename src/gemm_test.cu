#include <xitren/cuda/gemm_v00.cuh>
#include <xitren/math/matrix_alignment.hpp>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

using namespace xitren::math;
using namespace std::literals;

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

using time_type = std::chrono::microseconds;

struct measurement {
    std::size_t us;
    std::size_t cycles;
};

auto
measure(std::function<std::size_t(void)> callback)
{
    using time       = std::chrono::high_resolution_clock;
    using fsec       = std::chrono::duration<float>;
    auto      t0     = time::now();
    auto      cycles = callback();
    auto      t1     = time::now();
    fsec      fs     = t1 - t0;
    time_type d      = std::chrono::duration_cast<time_type>(fs);
    return measurement{static_cast<std::size_t>(d.count()), cycles};
}

auto
matrix_test(std::string name, measurement base, std::function<void(void)> callback)
{
    auto calc_time = measure([&]() -> std::size_t {
        std::size_t cnt{};
        time_type   period_{10000000us};
        auto        start_time{std::chrono::system_clock::now()};
        auto        last_time{std::chrono::system_clock::now()};
        while ((last_time - start_time) <= period_) {
            for (std ::size_t i{}; i < 1; i++, cnt++) {
                callback();
            }
            last_time = std::chrono::system_clock::now();
        }
        return cnt;
    });
    if (base.us == 0) {
        std::cout << name << "\tTime:\t" << calc_time.us << "\tCycles:\t" << calc_time.cycles
                  << "\tx1" << std::endl;
    } else {
        double const in{static_cast<double>(base.cycles) / static_cast<double>(base.us)};
        double const out{static_cast<double>(calc_time.cycles) / static_cast<double>(calc_time.us)};
        std::cout << name << "\tTime:\t" << calc_time.us << "\tCycles:\t" << calc_time.cycles
                  << "\tx" << static_cast<int>(out / in) << "."
                  << ((static_cast<int>(out * 10 / in)) % 10) << std::endl;
    }
    return calc_time;
}

template <class Type, std::size_t Size, optimization Optim>
auto
check(std::string name, measurement base)
{
    auto Aal = matrix_aligned<Type, Size, Size, Optim>::get_rand_matrix(0., 1.);
    auto Bal = matrix_aligned<Type, Size, Size, Optim>::get_rand_matrix(0., 1.);
    auto Cal = matrix_aligned<Type, Size, Size, Optim>::get_zeros_matrix();

    return matrix_test(name, base,
                       [&]() { matrix_aligned<Type, Size, Size, Optim>::mult(*Aal, *Bal, *Cal); });
}

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

    std::string str = "F ";
    check<float, mmSize, optimization::cuda_naive>(
        " "s + std::to_string(mmSize) + "\t" + str + "Naive   ", measurement{0, 0});
    // matrix_aligned<float, mmSize, mmSize, optimization::cuda_naive>::mult(*h_A, *h_B, *h_C);

    printf("Done\n");
    return 0;
}