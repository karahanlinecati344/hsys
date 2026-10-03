#include "hsys/cuda_check.cuh"
#include "hsys/vecred_strategy.cuh"

namespace hsys {

namespace {

constexpr unsigned kBlock = VecredStrategy<float>::block_size;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFU;

// Каждая нить проходит по массиву с шагом в размер сетки
// и копит свою сумму в регистре (сразу по два элемента за шаг)
template <class AtomT>
__device__ AtomT thread_sum(const VectorView<AtomT>& in) {
  const std::size_t stride = 2 * static_cast<std::size_t>(kBlock) * gridDim.x;
  std::size_t i = 2 * static_cast<std::size_t>(kBlock) * blockIdx.x + threadIdx.x;

  AtomT acc = 0;
  for (; i < in.size(); i += stride) {
    acc += in[i];
    if (i + kBlock < in.size()) acc += in[i + kBlock];
  }
  return acc;
}

// (1) без широковещания: дерево сложений в разделяемой памяти
template <class AtomT>
__global__ void kernel_vecred_nobr(VectorView<AtomT> in, VectorView<AtomT> out) {
  __shared__ AtomT partial[kBlock];

  const unsigned tid = threadIdx.x;
  partial[tid] = thread_sum(in);
  __syncthreads();

  // на каждом шаге первая половина нитей прибавляет к себе вторую половину
  for (unsigned s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    __syncthreads();
  }

  if (tid == 0) out[blockIdx.x] = partial[0];
}

// сумма внутри варпа через обмен регистрами, результат у нити с lane = 0
template <class AtomT>
__device__ AtomT warp_sum(AtomT value) {
  for (unsigned offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullMask, value, offset);
  }
  return value;
}

// (2) с широковещанием регистров: shared нужна только для сумм варпов
template <class AtomT>
__global__ void kernel_vecred_br(VectorView<AtomT> in, VectorView<AtomT> out) {
  constexpr unsigned kWarps = kBlock / kWarpSize;
  __shared__ AtomT warp_partial[kWarps];

  const unsigned tid = threadIdx.x;
  const unsigned lane = tid % kWarpSize;
  const unsigned warp = tid / kWarpSize;

  const AtomT acc = warp_sum(thread_sum(in));
  if (lane == 0) warp_partial[warp] = acc;
  __syncthreads();

  // суммы варпов складывает первый варп
  if (warp == 0) {
    const AtomT total = warp_sum(lane < kWarps ? warp_partial[lane] : AtomT(0));
    if (lane == 0) out[blockIdx.x] = total;
  }
}

}  // namespace

template <class AtomT>
void VecredShared<AtomT>::launch(
    VectorView<AtomT> in, VectorView<AtomT> out, unsigned grid) const {
  kernel_vecred_nobr<<<grid, kBlock>>>(in, out);
  cuda_check(cudaGetLastError(), "kernel_vecred_nobr");
}

template <class AtomT>
void VecredShuffle<AtomT>::launch(
    VectorView<AtomT> in, VectorView<AtomT> out, unsigned grid) const {
  kernel_vecred_br<<<grid, kBlock>>>(in, out);
  cuda_check(cudaGetLastError(), "kernel_vecred_br");
}

template class VecredShared<float>;
template class VecredShared<double>;
template class VecredShuffle<float>;
template class VecredShuffle<double>;

}  // namespace hsys
