#ifndef HSYS_DATA_CUH
#define HSYS_DATA_CUH

#include "cuda_check.cuh"

#include <cstddef>
#include <utility>

namespace hsys {

// Владеет массивом в глобальной памяти устройства (RAII)
template <class AtomT>
class Data {
 public:
  explicit Data(std::size_t size)
      : size_(size) {
    if (size_ > 0) {
      cuda_check(cudaMalloc(&data_, size_ * sizeof(AtomT)), "cudaMalloc");
    }
  }

  Data(const Data& other)
      : Data(other.size_) {
    if (size_ > 0) {
      cuda_check(cudaMemcpy(data_, other.data_, bytes(), cudaMemcpyDeviceToDevice),
          "cudaMemcpy D2D");
    }
  }

  Data(Data&& other) noexcept
      : size_(std::exchange(other.size_, 0))
      , data_(std::exchange(other.data_, nullptr)) {}

  Data& operator=(const Data& other) {
    if (this != &other) {
      Data tmp(other);
      swap(tmp);
    }
    return *this;
  }

  Data& operator=(Data&& other) noexcept {
    if (this != &other) {
      Data tmp(std::move(other));
      swap(tmp);
    }
    return *this;
  }

  ~Data() {
    // из деструктора исключения не бросаем
    if (data_ != nullptr) cudaFree(data_);
  }

  AtomT* data() {
    return data_;
  }

  const AtomT* data() const {
    return data_;
  }

  std::size_t size() const {
    return size_;
  }

  void copy_to_host(AtomT* host_ptr) const {
    if (size_ == 0) return;
    cuda_check(
        cudaMemcpy(host_ptr, data_, bytes(), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  }

  void copy_from_host(const AtomT* host_ptr) {
    if (size_ == 0) return;
    cuda_check(
        cudaMemcpy(data_, host_ptr, bytes(), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  }

 private:
  std::size_t bytes() const {
    return size_ * sizeof(AtomT);
  }

  void swap(Data& other) noexcept {
    std::swap(size_, other.size_);
    std::swap(data_, other.data_);
  }

  std::size_t size_ = 0;
  AtomT* data_ = nullptr;
};

}  // namespace hsys

#endif  // HSYS_DATA_CUH
