add_executable(work2_tests src/test_matmul_naive.cpp)

target_link_libraries(work2_tests PRIVATE work2_core Eigen3::Eigen GTest::gtest_main)

gtest_discover_tests(work2_tests DISCOVERY_MODE PRE_TEST)
