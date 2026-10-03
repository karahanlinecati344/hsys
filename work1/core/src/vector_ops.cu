#include "hsys/vector_ops.cuh"

#include <stdexcept>

namespace hsys {

namespace {

constexpr unsigned kBlockSize = 256;

// одна нить считает один элемент результата
template <class AtomT>
__global__ void kernel_vecadd(
    VectorView<AtomT> a, VectorView<AtomT> b, VectorView<AtomT> out) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < out.size()) out[i] = a[i] + b[i];
}

}  // namespace

template <class AtomT>
void vecadd(const Vector<AtomT>& a, const Vector<AtomT>& b, Vector<AtomT>& out) {
  if (a.size() != b.size() || a.size() != out.size()) {
    throw std::invalid_argument("vecadd: sizes of vectors do not match");
  }
  if (out.size() == 0) return;

  const auto grid = static_cast<unsigned>((out.size() + kBlockSize - 1) / kBlockSize);
  kernel_vecadd<<<grid, kBlockSize>>>(a.view(), b.view(), out.view());
  cuda_check(cudaGetLastError(), "kernel_vecadd");
}

template void vecadd(const Vector<float>&, const Vector<float>&, Vector<float>&);
template void vecadd(const Vector<double>&, const Vector<double>&, Vector<double>&);
template void vecadd(const Vector<int>&, const Vector<int>&, Vector<int>&);

}  // namespace hsys
