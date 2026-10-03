#ifndef HSYS_VECRED_STRATEGY_CUH
#define HSYS_VECRED_STRATEGY_CUH

#include "hsys/vector_view.cuh"

#include <algorithm>
#include <stdexcept>

namespace hsys {

// Стратегия редукции (суммы элементов) вектора.
// Схема в два прохода: 1) каждый блок считает свою частичную сумму в workspace;
// 2) один блок складывает частичные суммы в out[0].
template <class AtomT>
class VecredStrategy {
 public:
  static constexpr unsigned block_size = 256;
  static constexpr unsigned max_blocks = 1024;

  VecredStrategy() = default;
  VecredStrategy(const VecredStrategy&) = default;
  VecredStrategy(VecredStrategy&&) noexcept = default;
  VecredStrategy& operator=(const VecredStrategy&) = default;
  VecredStrategy& operator=(VecredStrategy&&) noexcept = default;
  virtual ~VecredStrategy() = default;

  // сколько блоков в первом проходе (= размер workspace)
  static unsigned grid_size(std::size_t n) {
    // каждая нить при загрузке сразу складывает минимум два элемента
    const std::size_t per_block = 2 * static_cast<std::size_t>(block_size);
    const std::size_t blocks = (n + per_block - 1) / per_block;
    return static_cast<unsigned>(std::clamp<std::size_t>(blocks, 1, max_blocks));
  }

  void operator()(
      VectorView<AtomT> in, VectorView<AtomT> workspace, VectorView<AtomT> out) const {
    const unsigned grid = grid_size(in.size());
    if (workspace.size() < grid || out.size() < 1) {
      throw std::invalid_argument("vecred: workspace or output is too small");
    }

    if (grid == 1) {
      launch(in, out, 1);
      return;
    }
    launch(in, workspace, grid);
    // второй проход - по первым grid элементам workspace
    launch(VectorView<AtomT>(&workspace[0], grid), out, 1);
  }

 protected:
  virtual void launch(VectorView<AtomT> in, VectorView<AtomT> out, unsigned grid) const
      = 0;
};

// (1) частичные суммы блока - в разделяемой памяти (kernel_vecred_nobr)
template <class AtomT>
class VecredShared : public VecredStrategy<AtomT> {
 protected:
  void launch(VectorView<AtomT> in, VectorView<AtomT> out, unsigned grid) const override;
};

// (2) редукция внутри варпа через __shfl_down_sync (kernel_vecred_br)
template <class AtomT>
class VecredShuffle : public VecredStrategy<AtomT> {
 protected:
  void launch(VectorView<AtomT> in, VectorView<AtomT> out, unsigned grid) const override;
};

}  // namespace hsys

#endif  // HSYS_VECRED_STRATEGY_CUH
