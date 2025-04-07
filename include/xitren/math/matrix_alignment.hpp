#pragma once

#include <xitren/math/gemm_core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

namespace xitren {
namespace math {

template <class Type, std::size_t Rows, std::size_t Columns, optimization Alg>
class matrix_aligned {
    using Core = gemm_core<Rows, Columns, Type, Alg>;

    static_assert(noexcept(Core::template mult<32>(nullptr, nullptr, nullptr)), "");

public:
    matrix_aligned()
    {
        data_ = (Type*)malloc(Rows * Columns * sizeof(Type));
        if (data_ == nullptr)
            throw std::bad_alloc{};    // ("failed to allocate largest problem size");
    }
    ~matrix_aligned() { free(data_); }

    template <std::size_t ColumnsOther>
    static void
    mult(matrix_aligned<Type, Rows, ColumnsOther, Alg> const&    a,
         matrix_aligned<Type, ColumnsOther, Columns, Alg> const& b,
         matrix_aligned<Type, Rows, Columns, Alg>&               c)
    {
        Core::template mult<ColumnsOther>(a.data_, b.data_, c.data_);
    }

    auto&
    get(std::size_t row, std::size_t column)
    {
        return data_[(row * Columns) + column];
    }

    static std::shared_ptr<matrix_aligned>
    get_rand_matrix(double max_val, double min_val)
    {
        std::shared_ptr<matrix_aligned> ret = std::make_shared<matrix_aligned>();
        for (std::uint32_t i = 0; i < Rows * Columns; ++i) {
            ret->data_[i] = rand() / (float)RAND_MAX;
        }
        return ret;
    }

    Type* data_{nullptr};

    static std::shared_ptr<matrix_aligned>
    get_zeros_matrix()
    {
        std::shared_ptr<matrix_aligned> ret = std::make_shared<matrix_aligned>();
        for (std::uint32_t i = 0; i < Rows * Columns; ++i) {
            ret->data_[i] = 0;
        }
        return ret;
    }

private:
};

}    // namespace math
}    // namespace xitren
