#ifndef HSYS_VECTOR_VIEW_CUH
#define HSYS_VECTOR_VIEW_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

namespace hsys {

// Невладеющее представление вектора, передаётся в кёрнел по значению
template <class AtomT>
class VectorView {
 public:
  __host__ __device__ VectorView(AtomT* data, std::size_t size)
      : data_(data)
      , size_(size) {}

  __host__ __device__ std::size_t size() const {
    return size_;
  }

  __host__ __device__ AtomT& operator[](std::size_t n) {
    return data_[n];
  }

  __host__ __device__ const AtomT& operator[](std::size_t n) const {
    return data_[n];
  }

  __host__ __device__ AtomT& operator()(std::size_t i) {
    return data_[i];
  }

  __host__ __device__ const AtomT& operator()(std::size_t i) const {
    return data_[i];
  }

 private:
  AtomT* data_;
  std::size_t size_;
};

static_assert(std::is_trivially_copyable_v<VectorView<float>>);

}  // namespace hsys

#endif  // HSYS_VECTOR_VIEW_CUH
