#ifndef HSYS_VECTOR_OPS_CUH
#define HSYS_VECTOR_OPS_CUH

#include "vector.cuh"

namespace hsys {

// out = a + b, память под out уже выделена (вызывает kernel_vecadd)
template <class AtomT>
void vecadd(const Vector<AtomT>& a, const Vector<AtomT>& b, Vector<AtomT>& out);

template <class AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& a, const Vector<AtomT>& b) {
  Vector<AtomT> out(a.size());
  vecadd(a, b, out);
  return out;
}

}  // namespace hsys

#endif  // HSYS_VECTOR_OPS_CUH
