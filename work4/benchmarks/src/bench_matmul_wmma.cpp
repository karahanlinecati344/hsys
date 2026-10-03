#include "hsys/event_timer.cuh"
#include "hsys/matrix_ops.cuh"
#include "hsys/shmem_matmul.cuh"
#include "hsys/wmma_matmul.cuh"

#include <benchmark/benchmark.h>

#include <memory>
#include <random>
#include <vector>

namespace {

template <class AtomT>
std::vector<AtomT> random_host(std::size_t size) {
  std::mt19937 gen(size);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<AtomT> host(size);
  for (auto& x : host) x = static_cast<AtomT>(dist(gen));
  return host;
}

template <class AtomT, class StrategyT>
void bm_matmul(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto strategy = std::make_shared<const StrategyT>();

  const auto host = random_host<AtomT>(n * n);
  hsys::Matrix<AtomT> a(n, n, strategy);
  hsys::Matrix<AtomT> b(n, n, strategy);
  hsys::Matrix<AtomT> c(n, n, strategy);
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

// предыдущая версия operator* (работа 3)
BENCHMARK(bm_matmul<float, hsys::ShmemMatmul<float>>)
    ->Name("CUDA Matrix Multiplication (GPU, Shared, float)")
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_matmul<__half, hsys::WmmaMatmul>)
    ->Name("CUDA Matrix Multiplication (GPU, WMMA, half)")
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
