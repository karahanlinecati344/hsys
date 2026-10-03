add_executable(work1_benchmarks src/bench_vecadd.cpp)

target_link_libraries(work1_benchmarks PRIVATE work1_core Eigen3::Eigen benchmark::benchmark)
