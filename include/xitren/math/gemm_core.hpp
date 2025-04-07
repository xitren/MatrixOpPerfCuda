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

enum class optimization { naive, blocked };

template <std::uint_fast32_t Rows, std::uint_fast32_t Columns, typename Type,
          optimization Alg>
class gemm_core {
  static_assert(Alg == optimization::naive, "Falling to base gemm!");

public:
  template <std::uint_fast32_t Other>
  static void mult(Type const *a, Type const *b, Type *c) noexcept {
    for (std::uint_fast32_t i = 0; i < Rows; ++i) {
      for (std::uint_fast32_t j = 0; j < Columns; ++j) {
        auto const current = i * Columns + j;
        Type cij = c[current]; /* cij = C[i][j] */
        for (std::uint_fast32_t k = 0; k < Other; k++) {
          cij += a[i * Other + k] *
                 b[k * Columns + j]; /* cij += A[i][k]*B[k][j] */
        }
        c[current] = cij; /* C[i][j] = cij */
      }
    }
  }

  static void add(Type const *a, Type const *b, Type *c) noexcept {
    for (std::uint_fast32_t i = 0; i < Rows; ++i) {
      for (std::uint_fast32_t j = 0; j < Columns; ++j) {
        auto const current = i * Columns + j;
        Type cij = a[current] + b[current]; /* cij += A[i][j] + B[i][j] */
        c[current] = cij;                   /* C[i][j] = cij */
      }
    }
  }

  static void sub(Type const *a, Type const *b, Type *c) noexcept {
    for (std::uint_fast32_t i = 0; i < Rows; ++i) {
      for (std::uint_fast32_t j = 0; j < Columns; ++j) {
        auto const current = i * Columns + j;
        Type cij = a[current] - b[current]; /* cij += A[i][j] - B[i][j] */
        c[current] = cij;                   /* C[i][j] = cij */
      }
    }
  }

  static void transpose(Type const *a, Type *c) noexcept {
    for (std::uint_fast32_t i = 0; i < Rows; ++i) {
      for (std::uint_fast32_t j = 0; j < Columns; ++j) {
        Type &cij = a[j * Columns + i]; /* cij += A[i][j] */
        c[i * Columns + j] = cij;       /* C[i][j] = cij */
      }
    }
  }

  static Type trace(Type const *a) noexcept {
    Type ret{};
    for (std::uint_fast32_t i = 0; i < Rows; ++i) {
      auto const current = i * Columns + i;
      Type &cij = a[current]; /* cij = A[i][i] */
      ret += cij;             /* ret += cij */
    }
    return ret;
  }

  static Type min(Type const *a) noexcept {
    Type ret{};
    for (std::uint_fast32_t i = 0; i < Rows; ++i) {
      for (std::uint_fast32_t j = 0; j < Columns; ++j) {
        auto const current = i * Columns + j;
        Type &cij = a[current]; /* cij = A[i][j] */
        ret = branchless_select(cij < ret, cij, ret);
      }
    }
    return ret;
  }

  static Type max(Type const *a) noexcept {
    Type ret{};
    for (std::uint_fast32_t i = 0; i < Rows; ++i) {
      for (std::uint_fast32_t j = 0; j < Columns; ++j) {
        auto const current = i * Columns + j;
        Type &cij = a[current]; /* cij = A[i][j] */
        ret = branchless_select(cij > ret, cij, ret);
      }
    }
    return ret;
  }
};

template <std::uint_fast32_t Rows, std::uint_fast32_t Columns, typename Type>
class gemm_core<Rows, Columns, Type, optimization::blocked>
    : gemm_core<Rows, Columns, Type, optimization::naive> {
  static constexpr std::uint_fast32_t blocksize = 32;
  static_assert(!(Rows % blocksize), "Should be dividable to blocksize!");
  static_assert(!(Columns % blocksize), "Should be dividable to blocksize!");

public:
  using gemm_core<Rows, Columns, Type, optimization::naive>::add;
  using gemm_core<Rows, Columns, Type, optimization::naive>::sub;
  using gemm_core<Rows, Columns, Type, optimization::naive>::transpose;
  using gemm_core<Rows, Columns, Type, optimization::naive>::trace;
  using gemm_core<Rows, Columns, Type, optimization::naive>::min;
  using gemm_core<Rows, Columns, Type, optimization::naive>::max;

  template <std::uint_fast32_t Other>
  static void mult(Type const *a, Type const *b, Type *c) noexcept {
    static_assert(!(Other % blocksize), "Should be dividable to blocksize!");
    for (std::uint_fast32_t si = 0; si < Rows; si += blocksize) {
      for (std::uint_fast32_t sj = 0; sj < Columns; sj += blocksize) {
        for (std::uint_fast32_t sk = 0; sk < Other; sk += blocksize) {
          do_block<Other>(si, sj, sk, a, b, c);
        }
      }
    }
  }

private:
  template <std::uint_fast32_t Other>
  static void do_block(const std::uint_fast32_t si, const std::uint_fast32_t sj,
                       const std::uint_fast32_t sk, Type const *a,
                       Type const *b, Type *c) noexcept {
    auto const last_si = si + blocksize;
    auto const last_sj = sj + blocksize;
    auto const last_sk = sk + blocksize;
    for (std::uint_fast32_t i = si; i < last_si; ++i) {
      for (std::uint_fast32_t j = sj; j < last_sj; ++j) {
        auto const current = i * Columns + j;
        Type cij = c[current]; /* cij = C[i][j] */
        for (std::uint_fast32_t k = sk; k < last_sk; ++k) {
          cij +=
              a[i * Other + k] * b[k * Columns + j]; /* cij+=A[i][k]*B[k][j] */
        }
        c[current] = cij; /* C[i][j] = cij */
      }
    }
  }
};

} // namespace math
} // namespace xitren
