cmake_minimum_required(VERSION 3.24)

project(hsys LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# sm_75 - минимальная архитектура, которую поддерживает CUDA 13
set(HSYS_CUDA_MIN_ARCH 75)

find_package(CUDAToolkit REQUIRED)

include(cmake/utils/cuda_setup_targets.cmake)
include(cmake/deps/deps_ops.cmake)
include(cmake/deps/eigen.cmake)
include(cmake/deps/gtest.cmake)
include(cmake/deps/gbench.cmake)

enable_testing()

add_subdirectory(work1)
add_subdirectory(work2)
add_subdirectory(work3)
add_subdirectory(work4)
add_subdirectory(work5)
