#ifndef HSYS_WMMA_MATMUL_CUH
#define HSYS_WMMA_MATMUL_CUH

#include "hsys/matmul_strategy.cuh"

#include <cuda_fp16.h>

namespace hsys {

// Умножение half-матриц на тензорных ядрах (kernel_matmul_wmma).
// Накопление идёт во float, результат округляется до half
class WmmaMatmul : public MatmulStrategy<__half> {
 protected:
  void launch(
      MatrixView<__half> a, MatrixView<__half> b, MatrixView<__half> c) const override;
};

}  // namespace hsys

#endif  // HSYS_WMMA_MATMUL_CUH
