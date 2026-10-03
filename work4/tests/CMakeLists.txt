add_executable(work4_tests src/test_matmul_wmma.cpp)

target_link_libraries(work4_tests PRIVATE work4_core Eigen3::Eigen GTest::gtest_main)

gtest_discover_tests(work4_tests DISCOVERY_MODE PRE_TEST)
