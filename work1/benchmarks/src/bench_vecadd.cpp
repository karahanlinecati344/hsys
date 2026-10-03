#include "hsys/event_timer.cuh"
#include "hsys/vector_ops.cuh"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

namespace {

void bm_vecadd_cuda(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));

  // выделение и заполнение памяти - вне замера
  const Eigen::VectorXf host = Eigen::VectorXf::Random(static_cast<Eigen::Index>(n));
  hsys::Vector<float> a(n);
  hsys::Vector<float> b(n);
  hsys::Vector<float> c(n);
  a.data().copy_from_host(host.data());
  b.data().copy_from_host(host.data());

  hsys::EventTimer timer;
  hsys::vecadd(a, b, c);  // прогрев

  for (auto _ : state) {
    timer.start();
    hsys::vecadd(a, b, c);
    state.SetIterationTime(timer.stop());
  }
}

void bm_vecadd_eigen(benchmark::State& state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));

  const Eigen::VectorXf a = Eigen::VectorXf::Random(n);
  const Eigen::VectorXf b = Eigen::VectorXf::Random(n);
  Eigen::VectorXf c(n);

  for (auto _ : state) {
    c.noalias() = a + b;
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
}

}  // namespace

// n = 8, 8^2, ..., 8^8
BENCHMARK(bm_vecadd_cuda)
    ->Name("CUDA Vector Addition (GPU)")
    ->RangeMultiplier(8)
    ->Range(8, 1 << 24)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_vecadd_eigen)
    ->Name("Eigen Vector Addition (CPU)")
    ->RangeMultiplier(8)
    ->Range(8, 1 << 24)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
