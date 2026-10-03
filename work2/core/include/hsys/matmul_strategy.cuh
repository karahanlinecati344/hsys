#ifndef HSYS_MATMUL_STRATEGY_CUH
#define HSYS_MATMUL_STRATEGY_CUH

#include "matrix_view.cuh"

#include <memory>
#include <stdexcept>

namespace hsys {

// Стратегия умножения матриц: каждая реализация запускает свой кёрнел
template <class AtomT>
class MatmulStrategy {
 public:
  MatmulStrategy() = default;
  MatmulStrategy(const MatmulStrategy&) = default;
  MatmulStrategy(MatmulStrategy&&) noexcept = default;
  MatmulStrategy& operator=(const MatmulStrategy&) = default;
  MatmulStrategy& operator=(MatmulStrategy&&) noexcept = default;
  virtual ~MatmulStrategy() = default;

  // c = a * b, память под c уже выделена
  void operator()(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) const {
    if (a.ncols() != b.nrows() || c.nrows() != a.nrows() || c.ncols() != b.ncols()) {
      throw std::invalid_argument("matmul: matrix sizes do not match");
    }
    if (c.size() == 0) return;
    launch(a, b, c);
  }

 protected:
  virtual void launch(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) const
      = 0;
};

// стратегия по умолчанию (наивный кёрнел), определена в naive_matmul.cu
template <class AtomT>
std::shared_ptr<const MatmulStrategy<AtomT>> default_matmul();

}  // namespace hsys

#endif  // HSYS_MATMUL_STRATEGY_CUH
