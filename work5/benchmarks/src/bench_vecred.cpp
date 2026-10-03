#include "hsys/event_timer.cuh"
#include "hsys/vector_reduce.cuh"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <exception>

namespace {

template <class StrategyT>
void bm_vecred(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const StrategyT strategy;

  try {
    // память выделяем и заполняем до замеров; для времени значения не важны
    hsys::Vector<float> v(n);
    hsys::Vector<float> workspace(StrategyT::grid_size(n));
    hsys::Vector<float> out(1);
    hsys::cuda_check(cudaMemset(v.data().data(), 0, n * sizeof(float)), "cudaMemset");

    hsys::EventTimer timer;
    hsys::reduce(v, workspace, out, strategy);  // прогрев

    for (auto _ : state) {
      timer.start();
      hsys::reduce(v, workspace, out, strategy);
      state.SetIterationTime(timer.stop());
    }
  } catch (const std::exception& e) {
    // на больших n может не хватить памяти видеокарты
    state.SkipWithError(e.what());
  }
}

}  // namespace

// n = 8 * 2^0, 8 * 2^1, ..., 8 * 2^28
BENCHMARK(bm_vecred<hsys::VecredShared<float>>)
    ->Name("CUDA Vector Reduction (GPU, nobr)")
    ->RangeMultiplier(2)
    ->Range(8, std::int64_t{8} << 28)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_vecred<hsys::VecredShuffle<float>>)
    ->Name("CUDA Vector Reduction (GPU, br)")
    ->RangeMultiplier(2)
    ->Range(8, std::int64_t{8} << 28)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
