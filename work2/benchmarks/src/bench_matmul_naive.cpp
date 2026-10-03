#include "hsys/event_timer.cuh"
#include "hsys/matrix_ops.cuh"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

namespace {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void bm_matmul_naive(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));

  // выделение и копирование - до начала замеров
  const RowMatrixXf host = RowMatrixXf::Random(n, n);
  hsys::Matrix<float> a(n, n);
  hsys::Matrix<float> b(n, n);
  hsys::Matrix<float> c(n, n);
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

void bm_matmul_eigen(benchmark::State& state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));

  const Eigen::MatrixXf a = Eigen::MatrixXf::Random(n, n);
  const Eigen::MatrixXf b = Eigen::MatrixXf::Random(n, n);
  Eigen::MatrixXf c(n, n);

  for (auto _ : state) {
    c.noalias() = a * b;
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
}

}  // namespace

// n = 16, 32, ..., 1024
BENCHMARK(bm_matmul_naive)
    ->Name("CUDA Matrix Multiplication (GPU, Naive, float)")
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_matmul_eigen)
    ->Name("Eigen Matrix Multiplication (CPU, float)")
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
