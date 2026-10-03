add_library(work1_core STATIC src/vector_ops.cu)

target_include_directories(work1_core PUBLIC include)
target_link_libraries(work1_core PUBLIC CUDA::cudart)

cuda_setup_targets(
  TARGETS work1_core
  CUDA_MIN_ARCH ${HSYS_CUDA_MIN_ARCH}
)
