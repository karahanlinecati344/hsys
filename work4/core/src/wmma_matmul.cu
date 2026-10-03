#include "hsys/cuda_check.cuh"
#include "hsys/wmma_matmul.cuh"

#include <mma.h>

namespace hsys {

namespace {

namespace wmma = nvcuda::wmma;

// размер фрагмента WMMA (m16n16k16)
constexpr unsigned kFrag = 16;

// блок из 8 варпов: 2 по вертикали и 4 по горизонтали,
// каждый варп считает свой кусок 16x16 матрицы C
constexpr unsigned kWarpsY = 2;
constexpr unsigned kWarpsX = 4;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kThreads = kWarpsY * kWarpsX * kWarpSize;

constexpr unsigned kTileM = kWarpsY * kFrag;  // 32 строки A и C
constexpr unsigned kTileN = kWarpsX * kFrag;  // 64 столбца B и C
constexpr unsigned kTileK = 2 * kFrag;  // 32 - шаг по общей размерности

__global__ void __launch_bounds__(kThreads)
    kernel_matmul_wmma(MatrixView<__half> a, MatrixView<__half> b, MatrixView<__half> c) {
  // тайлы A и B в разделяемой памяти (общие для всех варпов блока)
  __shared__ __align__(32) __half a_tile[kTileM][kTileK];
  __shared__ __align__(32) __half b_tile[kTileK][kTileN];
  // буфер, через который каждый варп выгружает свой результат
  __shared__ __align__(32) float c_stage[kWarpsY * kWarpsX][kFrag * kFrag];

  const unsigned tid = threadIdx.x;
  const unsigned warp = tid / kWarpSize;
  const unsigned lane = tid % kWarpSize;
  const unsigned warp_y = warp / kWarpsX;
  const unsigned warp_x = warp % kWarpsX;

  const std::size_t m = a.nrows();
  const std::size_t depth = a.ncols();
  const std::size_t n = b.ncols();
  const std::size_t tile_row = blockIdx.y * static_cast<std::size_t>(kTileM);
  const std::size_t tile_col = blockIdx.x * static_cast<std::size_t>(kTileN);
  const __half zero = __float2half(0.0F);

  wmma::fragment<wmma::accumulator, kFrag, kFrag, kFrag, float> acc;
  wmma::fill_fragment(acc, 0.0F);

  for (std::size_t t = 0; t < depth; t += kTileK) {
    // все нити блока вместе грузят тайлы, за границами - нули
    for (unsigned i = tid; i < kTileM * kTileK; i += kThreads) {
      const unsigned r = i / kTileK;
      const unsigned col = i % kTileK;
      const std::size_t gr = tile_row + r;
      const std::size_t gc = t + col;
      a_tile[r][col] = (gr < m && gc < depth) ? a(gr, gc) : zero;
    }
    for (unsigned i = tid; i < kTileK * kTileN; i += kThreads) {
      const unsigned r = i / kTileN;
      const unsigned col = i % kTileN;
      const std::size_t gr = t + r;
      const std::size_t gc = tile_col + col;
      b_tile[r][col] = (gr < depth && gc < n) ? b(gr, gc) : zero;
    }
    __syncthreads();

    for (unsigned kk = 0; kk < kTileK; kk += kFrag) {
      wmma::fragment<wmma::matrix_a, kFrag, kFrag, kFrag, __half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, kFrag, kFrag, kFrag, __half, wmma::row_major> b_frag;

      wmma::load_matrix_sync(a_frag, &a_tile[warp_y * kFrag][kk], kTileK);
      wmma::load_matrix_sync(b_frag, &b_tile[kk][warp_x * kFrag], kTileN);
      wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
    __syncthreads();
  }

  // фрагмент C хранится в регистрах варпа, выгружаем его через shared
  float* stage = c_stage[warp];
  wmma::store_matrix_sync(stage, acc, kFrag, wmma::mem_row_major);
  __syncwarp();

  const std::size_t out_row = tile_row + warp_y * kFrag;
  const std::size_t out_col = tile_col + warp_x * kFrag;
  for (unsigned i = lane; i < kFrag * kFrag; i += kWarpSize) {
    const std::size_t gr = out_row + i / kFrag;
    const std::size_t gc = out_col + i % kFrag;
    if (gr < m && gc < n) c(gr, gc) = __float2half(stage[i]);
  }
}

}  // namespace

void WmmaMatmul::launch(
    MatrixView<__half> a, MatrixView<__half> b, MatrixView<__half> c) const {
  const dim3 grid((c.ncols() + kTileN - 1) / kTileN, (c.nrows() + kTileM - 1) / kTileM);

  kernel_matmul_wmma<<<grid, kThreads>>>(a, b, c);
  cuda_check(cudaGetLastError(), "kernel_matmul_wmma");
}

}  // namespace hsys
