add_executable(work5_benchmarks src/bench_vecred.cpp)

target_link_libraries(work5_benchmarks PRIVATE work5_core benchmark::benchmark)
