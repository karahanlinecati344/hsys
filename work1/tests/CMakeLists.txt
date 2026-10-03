add_executable(work1_tests src/test_vector.cpp)

target_link_libraries(work1_tests PRIVATE work1_core Eigen3::Eigen GTest::gtest_main)

gtest_discover_tests(work1_tests DISCOVERY_MODE PRE_TEST)
