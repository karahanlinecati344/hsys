add_executable(work4_benchmarks src/bench_matmul_wmma.cpp)

target_link_libraries(work4_benchmarks PRIVATE work4_core benchmark::benchmark)
