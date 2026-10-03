add_executable(work2_benchmarks src/bench_matmul_naive.cpp)

target_link_libraries(work2_benchmarks PRIVATE work2_core Eigen3::Eigen benchmark::benchmark)
