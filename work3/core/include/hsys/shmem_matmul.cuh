#ifndef HSYS_SHMEM_MATMUL_CUH
#define HSYS_SHMEM_MATMUL_CUH

#include "hsys/matmul_strategy.cuh"

namespace hsys {

// Блочное умножение через разделяемую память (kernel_matmul_shmem)
template <class AtomT>
class ShmemMatmul : public MatmulStrategy<AtomT> {
 public:
  static constexpr unsigned block_size = 16;

 protected:
  void launch(
      MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) const override;
};

}  // namespace hsys

#endif  // HSYS_SHMEM_MATMUL_CUH
