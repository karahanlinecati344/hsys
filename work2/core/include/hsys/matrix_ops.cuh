#ifndef HSYS_MATRIX_OPS_CUH
#define HSYS_MATRIX_OPS_CUH

#include "matrix.cuh"

namespace hsys {

// c = a * b, память под c уже выделена; кёрнел выбирает стратегия матрицы a
template <class AtomT>
void matmul(const Matrix<AtomT>& a, const Matrix<AtomT>& b, Matrix<AtomT>& c) {
  (*a.matmul())(a.view(), b.view(), c.view());
}

template <class AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& a, const Matrix<AtomT>& b) {
  Matrix<AtomT> c(a.nrows(), b.ncols(), a.matmul());
  matmul(a, b, c);
  return c;
}

}  // namespace hsys

#endif  // HSYS_MATRIX_OPS_CUH
