#ifndef HSYS_VECTOR_CUH
#define HSYS_VECTOR_CUH

#include "data.cuh"
#include "vector_view.cuh"

#include <memory>

namespace hsys {

// Фасад: данные (общие, через shared_ptr) + собственное представление
template <class AtomT>
class Vector {
 public:
  explicit Vector(std::size_t size)
      : data_(std::make_shared<Data<AtomT>>(size))
      , view_(data_->data(), size) {}

  std::size_t size() const {
    return view_.size();
  }

  Data<AtomT>& data() {
    return *data_;
  }

  const Data<AtomT>& data() const {
    return *data_;
  }

  VectorView<AtomT>& view() {
    return view_;
  }

  const VectorView<AtomT>& view() const {
    return view_;
  }

 private:
  std::shared_ptr<Data<AtomT>> data_;
  VectorView<AtomT> view_;
};

}  // namespace hsys

#endif  // HSYS_VECTOR_CUH
