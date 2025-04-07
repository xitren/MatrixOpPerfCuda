#pragma once

#include <algorithm>
#include <cstdint>

namespace xitren {
namespace math {

template <class Type>
static inline Type &branchless_select(int compare, Type &a, Type &b) {
  auto ptr_a = reinterpret_cast<std::uintptr_t>(&a);
  auto ptr_b = reinterpret_cast<std::uintptr_t>(&b);
  auto ptr_c = ((compare - 1) & (ptr_b ^ ptr_a)) ^ ptr_a;
  Type &c = *(reinterpret_cast<Type *>(ptr_c));
  return c;
}
} // namespace math
} // namespace xitren
