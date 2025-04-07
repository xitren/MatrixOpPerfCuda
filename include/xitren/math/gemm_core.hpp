#pragma once

#include <xitren/math/branchless.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <utility>
#include <vector>

namespace xitren {
namespace math {

enum class optimization { naive, cuda_naive };

template <std::uint_fast32_t Rows, std::uint_fast32_t Columns, typename Type, optimization Alg>
class gemm_core {
    static_assert(Alg == optimization::naive, "Falling to base gemm!");

public:
    template <std::uint_fast32_t Other>
    static void
    mult(Type const* a, Type const* b, Type* c) noexcept
    {
        for (std::uint_fast32_t i = 0; i < Rows; ++i) {
            for (std::uint_fast32_t j = 0; j < Columns; ++j) {
                auto const current = i * Columns + j;
                Type       cij     = c[current];                  /* cij = C[i][j] */
                for (std::uint_fast32_t k = 0; k < Other; k++) {
                    cij += a[i * Other + k] * b[k * Columns + j]; /* cij += A[i][k]*B[k][j] */
                }
                c[current] = cij;                                 /* C[i][j] = cij */
            }
        }
    }
};

}    // namespace math
}    // namespace xitren
