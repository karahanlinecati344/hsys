#ifndef HSYS_NAIVE_MATMUL_CUH
#define HSYS_NAIVE_MATMUL_CUH

#include "matmul_strategy.cuh"

namespace hsys {

// Умножение без разделяемой памяти (kernel_matmul_naive)
template <class AtomT>
class NaiveMatmul : public MatmulStrategy<AtomT> {
 protected:
  void launch(
      MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) const override;
};

}  // namespace hsys

#endif  // HSYS_NAIVE_MATMUL_CUH
