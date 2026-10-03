add_executable(work5_tests src/test_vecred.cpp)

target_link_libraries(work5_tests PRIVATE work5_core Eigen3::Eigen GTest::gtest_main)

gtest_discover_tests(work5_tests DISCOVERY_MODE PRE_TEST)
