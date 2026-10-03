#include "hsys/event_timer.cuh"
#include "hsys/matrix_ops.cuh"
#include "hsys/naive_matmul.cuh"
#include "hsys/shmem_matmul.cuh"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

#include <memory>

namespace {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// общий код для обеих версий, отличается только стратегия
template <class StrategyT>
void bm_matmul(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto strategy = std::make_shared<const StrategyT>();

  const RowMatrixXf host = RowMatrixXf::Random(n, n);
  hsys::Matrix<float> a(n, n, strategy);
  hsys::Matrix<float> b(n, n, strategy);
  hsys::Matrix<float> c(n, n, strategy);
  a.data().copy_from_host(host.data());
  b.data().copy_from_host(host.data());

  hsys::EventTimer timer;
  hsys::matmul(a, b, c);  // прогрев

  for (auto _ : state) {
    timer.start();
    hsys::matmul(a, b, c);
    state.SetIterationTime(timer.stop());
  }
}

}  // namespace

BENCHMARK(bm_matmul<hsys::NaiveMatmul<float>>)
    ->Name("CUDA Matrix Multiplication (GPU, Naive, float)")
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_matmul<hsys::ShmemMatmul<float>>)
    ->Name("CUDA Matrix Multiplication (GPU, Shared, float)")
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
