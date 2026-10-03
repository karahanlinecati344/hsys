#ifndef HSYS_MATRIX_CUH
#define HSYS_MATRIX_CUH

#include "hsys/data.cuh"
#include "matmul_strategy.cuh"
#include "matrix_view.cuh"

#include <memory>
#include <utility>

namespace hsys {

// Фасад: данные (общие, через shared_ptr) + собственное представление.
// Дополнительно хранит стратегию умножения, которую использует operator*
template <class AtomT>
class Matrix {
 public:
  using strategy_t = MatmulStrategy<AtomT>;
  using strategy_ptr = std::shared_ptr<const strategy_t>;

  Matrix(
      std::size_t nrows, std::size_t ncols, strategy_ptr matmul = default_matmul<AtomT>())
      : data_(std::make_shared<Data<AtomT>>(nrows * ncols))
      , view_(data_->data(), nrows, ncols)
      , matmul_(std::move(matmul)) {}

  std::size_t size() const {
    return view_.size();
  }

  std::size_t nrows() const {
    return view_.nrows();
  }

  std::size_t ncols() const {
    return view_.ncols();
  }

  Data<AtomT>& data() {
    return *data_;
  }

  const Data<AtomT>& data() const {
    return *data_;
  }

  MatrixView<AtomT>& view() {
    return view_;
  }

  const MatrixView<AtomT>& view() const {
    return view_;
  }

  const strategy_ptr& matmul() const {
    return matmul_;
  }

  void set_matmul(strategy_ptr matmul) {
    matmul_ = std::move(matmul);
  }

 private:
  std::shared_ptr<Data<AtomT>> data_;
  MatrixView<AtomT> view_;
  strategy_ptr matmul_;
};

}  // namespace hsys

#endif  // HSYS_MATRIX_CUH
