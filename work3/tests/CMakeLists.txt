add_executable(work3_tests src/test_matmul_shmem.cpp)

target_link_libraries(work3_tests PRIVATE work3_core Eigen3::Eigen GTest::gtest_main)

gtest_discover_tests(work3_tests DISCOVERY_MODE PRE_TEST)
