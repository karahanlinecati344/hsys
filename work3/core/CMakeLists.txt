add_library(work3_core STATIC src/shmem_matmul.cu)

target_include_directories(work3_core PUBLIC include)
target_link_libraries(work3_core PUBLIC work2_core)

cuda_setup_targets(
  TARGETS work3_core
  CUDA_MIN_ARCH ${HSYS_CUDA_MIN_ARCH}
)
