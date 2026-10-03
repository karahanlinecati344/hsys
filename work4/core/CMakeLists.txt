add_library(work4_core STATIC src/wmma_matmul.cu)

target_include_directories(work4_core PUBLIC include)
target_link_libraries(work4_core PUBLIC work3_core)

cuda_setup_targets(
  TARGETS work4_core
  CUDA_MIN_ARCH ${HSYS_CUDA_MIN_ARCH}
)
