#ifndef HSYS_EVENT_TIMER_CUH
#define HSYS_EVENT_TIMER_CUH

#include "cuda_check.cuh"

namespace hsys {

// Замер времени выполнения на GPU через CUDA Events
class EventTimer {
 public:
  EventTimer() {
    cuda_check(cudaEventCreate(&start_), "cudaEventCreate");
    cuda_check(cudaEventCreate(&stop_), "cudaEventCreate");
  }

  EventTimer(const EventTimer&) = delete;
  EventTimer& operator=(const EventTimer&) = delete;
  EventTimer(EventTimer&&) = delete;
  EventTimer& operator=(EventTimer&&) = delete;

  ~EventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() {
    cuda_check(cudaEventRecord(start_), "cudaEventRecord");
  }

  // останавливает таймер, возвращает время в секундах
  double stop() {
    cuda_check(cudaEventRecord(stop_), "cudaEventRecord");
    cuda_check(cudaEventSynchronize(stop_), "cudaEventSynchronize");
    float ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
    return static_cast<double>(ms) / 1000.0;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

}  // namespace hsys

#endif  // HSYS_EVENT_TIMER_CUH
