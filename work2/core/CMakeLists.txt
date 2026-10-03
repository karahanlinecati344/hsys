add_library(work2_core STATIC src/naive_matmul.cu)

target_include_directories(work2_core PUBLIC include)
target_link_libraries(work2_core PUBLIC work1_core)

cuda_setup_targets(
  TARGETS work2_core
  CUDA_MIN_ARCH ${HSYS_CUDA_MIN_ARCH}
)
