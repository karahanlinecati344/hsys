#include "hsys/acc_type.cuh"
#include "hsys/cuda_check.cuh"
#include "hsys/shmem_matmul.cuh"

namespace hsys {

namespace {

// Блок нитей считает подматрицу c_sub размером kBlock x kBlock.
// По оси k идём кусками: каждая нить грузит по одному элементу из A и B
// в разделяемую память, затем блок считает частичные суммы из shared.
template <class AtomT, unsigned kBlock>
__global__ void kernel_matmul_shmem(
    MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) {
  using acc_t = acc_type_t<AtomT>;

  __shared__ AtomT a_sub[kBlock][kBlock];
  __shared__ AtomT b_sub[kBlock][kBlock];

  const unsigned tx = threadIdx.x;
  const unsigned ty = threadIdx.y;
  const std::size_t row = blockIdx.y * static_cast<std::size_t>(kBlock) + ty;
  const std::size_t col = blockIdx.x * static_cast<std::size_t>(kBlock) + tx;
  const std::size_t depth = a.ncols();
  const auto zero = static_cast<AtomT>(0.0F);

  acc_t sum = 0;
  for (std::size_t t = 0; t < depth; t += kBlock) {
    // за границами матрицы кладём нули, чтобы не портить сумму
    a_sub[ty][tx] = (row < a.nrows() && t + tx < depth) ? a(row, t + tx) : zero;
    b_sub[ty][tx] = (t + ty < depth && col < b.ncols()) ? b(t + ty, col) : zero;
    __syncthreads();

    for (unsigned k = 0; k < kBlock; ++k) {
      sum += static_cast<acc_t>(a_sub[ty][k]) * static_cast<acc_t>(b_sub[k][tx]);
    }
    __syncthreads();  // ждём всех, прежде чем перезаписать shared
  }

  if (row < c.nrows() && col < c.ncols()) c(row, col) = static_cast<AtomT>(sum);
}

}  // namespace

template <class AtomT>
void ShmemMatmul<AtomT>::launch(
    MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) const {
  const dim3 block(block_size, block_size);
  const dim3 grid((c.ncols() + block_size - 1) / block_size,
      (c.nrows() + block_size - 1) / block_size);

  kernel_matmul_shmem<AtomT, block_size><<<grid, block>>>(a, b, c);
  cuda_check(cudaGetLastError(), "kernel_matmul_shmem");
}

template class ShmemMatmul<float>;
template class ShmemMatmul<double>;
template class ShmemMatmul<__half>;

}  // namespace hsys
