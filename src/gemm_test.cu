#include <xitren/cuda/gemm_v00.cuh>
#include <xitren/cuda/gemm_v01.cuh>
#include <xitren/cuda/gemm_v02.cuh>
#include <xitren/cuda/gemm_v03.cuh>
#include <xitren/cuda/gemm_v04.cuh>
#include <xitren/cuda/gemm_v05.cuh>
// #include <xitren/cuda/gemm_v06.cuh>
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
    float const peak_bandwidth{
        static_cast<float>(2.0f * device_prop.memoryClockRate * (device_prop.memoryBusWidth / 8) / 1.0e6)};
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
        std::cout << name << "\tTime:\t" << calc_time.us << "\tCycles:\t" << calc_time.cycles << "\tx1" << std::endl;
    } else {
        double const in{static_cast<double>(base.cycles) / static_cast<double>(base.us)};
        double const out{static_cast<double>(calc_time.cycles) / static_cast<double>(calc_time.us)};
        std::cout << name << "\tTime:\t" << calc_time.us << "\tCycles:\t" << calc_time.cycles << "\tx"
                  << static_cast<int>(out / in) << "." << ((static_cast<int>(out * 10 / in)) % 10) << std::endl;
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

    return matrix_test(name, base, [&]() { matrix_aligned<Type, Size, Size, Optim>::mult(*Aal, *Bal, *Cal); });
}

template <class Type, std::size_t Size>
void
test_sized_matrix()
{
    constexpr std::size_t size = Size;
    std::string           str  = "X ";
    if constexpr (std::is_same<Type, double>()) {
        str = "D  ";
    }
    if constexpr (std::is_same<Type, float>()) {
        str = "F  ";
    }
    if constexpr (std::is_same<Type, std::int8_t>()) {
        str = "I8 ";
    }
    auto sz = std::to_string(size);

    auto base = check<Type, Size, optimization::naive>(
        " "s + sz + "\t" + str + "HOST naive                                     ", measurement{0, 0});
    check<Type, Size, optimization::cuda_naive>(
        " "s + sz + "\t" + str + "CUDA naive                                     ", base);
    check<Type, Size, optimization::cuda_coalesced>(
        " "s + sz + "\t" + str + "CUDA coalesced                                 ", base);
    check<Type, Size, optimization::cuda_2d_block>(
        " "s + sz + "\t" + str + "CUDA 2d_block                                  ", base);
    check<Type, Size, optimization::cuda_2d_block_1d_thread>(
        " "s + sz + "\t" + str + "CUDA 2d_block_1d_thread                        ", base);
    check<Type, Size, optimization::cuda_2d_block_2d_thread>(
        " "s + sz + "\t" + str + "CUDA 2d_block_2d_thread                        ", base);
    check<Type, Size, optimization::cuda_2d_block_2d_warp_2d_thread_transpose>(
        " "s + sz + "\t" + str + "CUDA 2d_block_2d_warp_2d_thread_transpose      ", base);
    // check<Type, Size, optimization::cuda_2d_block_2d_warp_2d_thread_transpose_wmma>(
    //     " "s + sz + "\t" + str + "CUDA 2d_block_2d_warp_2d_thread_transpose_wmma ", base);
}

int
main(void)
{
    print_device_info();

    test_sized_matrix<float, 128>();
    test_sized_matrix<double, 128>();
    test_sized_matrix<float, 256>();
    test_sized_matrix<double, 256>();
    test_sized_matrix<float, 512>();
    test_sized_matrix<double, 512>();
    test_sized_matrix<float, 1024>();
    test_sized_matrix<double, 1024>();
    test_sized_matrix<float, 2048>();
    test_sized_matrix<double, 2048>();
    test_sized_matrix<float, 4096>();
    test_sized_matrix<double, 4096>();
    test_sized_matrix<float, 8096>();
    test_sized_matrix<double, 8096>();
    test_sized_matrix<float, 16384>();
    test_sized_matrix<double, 16384>();

    printf("Done\n");
    return 0;
}