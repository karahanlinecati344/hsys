#ifndef HSYS_VECTOR_REDUCE_CUH
#define HSYS_VECTOR_REDUCE_CUH

#include "hsys/vector.cuh"
#include "vecred_strategy.cuh"

namespace hsys {

// out[0] = сумма элементов v; workspace и out выделены заранее
template <class AtomT>
void reduce(const Vector<AtomT>& v,
    Vector<AtomT>& workspace,
    Vector<AtomT>& out,
    const VecredStrategy<AtomT>& strategy) {
  strategy(v.view(), workspace.view(), out.view());
}

// удобная обёртка: сама выделяет память и возвращает результат на хост
template <class AtomT>
AtomT sum(const Vector<AtomT>& v, const VecredStrategy<AtomT>& strategy) {
  Vector<AtomT> workspace(VecredStrategy<AtomT>::grid_size(v.size()));
  Vector<AtomT> out(1);
  reduce(v, workspace, out, strategy);

  AtomT result{};
  out.data().copy_to_host(&result);
  return result;
}

}  // namespace hsys

#endif  // HSYS_VECTOR_REDUCE_CUH
