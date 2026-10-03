#include "hsys/acc_type.cuh"
#include "hsys/cuda_check.cuh"
#include "hsys/naive_matmul.cuh"

namespace hsys {

namespace {

constexpr unsigned kBlockDim = 16;

// каждая нить считает один элемент c(row, col), читая данные из глобальной памяти
template <class AtomT>
__global__ void kernel_matmul_naive(
    MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) {
  using acc_t = acc_type_t<AtomT>;

  const std::size_t row = blockIdx.y * static_cast<std::size_t>(blockDim.y) + threadIdx.y;
  const std::size_t col = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (row >= c.nrows() || col >= c.ncols()) return;

  acc_t sum = 0;
  for (std::size_t k = 0; k < a.ncols(); ++k) {
    sum += static_cast<acc_t>(a(row, k)) * static_cast<acc_t>(b(k, col));
  }
  c(row, col) = static_cast<AtomT>(sum);
}

}  // namespace

template <class AtomT>
void NaiveMatmul<AtomT>::launch(
    MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) const {
  const dim3 block(kBlockDim, kBlockDim);
  const dim3 grid(
      (c.ncols() + kBlockDim - 1) / kBlockDim, (c.nrows() + kBlockDim - 1) / kBlockDim);

  kernel_matmul_naive<<<grid, block>>>(a, b, c);
  cuda_check(cudaGetLastError(), "kernel_matmul_naive");
}

template <class AtomT>
std::shared_ptr<const MatmulStrategy<AtomT>> default_matmul() {
  static const auto strategy = std::make_shared<const NaiveMatmul<AtomT>>();
  return strategy;
}

template class NaiveMatmul<float>;
template class NaiveMatmul<double>;
template class NaiveMatmul<__half>;

template std::shared_ptr<const MatmulStrategy<float>> default_matmul();
template std::shared_ptr<const MatmulStrategy<double>> default_matmul();
template std::shared_ptr<const MatmulStrategy<__half>> default_matmul();

}  // namespace hsys
