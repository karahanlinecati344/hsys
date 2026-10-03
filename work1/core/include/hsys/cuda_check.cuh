#ifndef HSYS_CUDA_CHECK_CUH
#define HSYS_CUDA_CHECK_CUH

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace hsys {

// Бросает исключение, если вызов CUDA API завершился с ошибкой
inline void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

}  // namespace hsys

#endif  // HSYS_CUDA_CHECK_CUH
