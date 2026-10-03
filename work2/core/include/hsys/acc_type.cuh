#ifndef HSYS_ACC_TYPE_CUH
#define HSYS_ACC_TYPE_CUH

#include <cuda_fp16.h>

namespace hsys {

// Тип для накопления суммы: для half копим во float, иначе быстро теряется точность
template <class AtomT>
struct acc_type {
  using type = AtomT;
};

template <>
struct acc_type<__half> {
  using type = float;
};

template <class AtomT>
using acc_type_t = typename acc_type<AtomT>::type;

}  // namespace hsys

#endif  // HSYS_ACC_TYPE_CUH
